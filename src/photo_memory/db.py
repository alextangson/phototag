"""SQLite database for tracking photo processing progress."""

import sqlite3
from datetime import datetime, timezone

CURRENT_SCHEMA_VERSION = 3


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _get_schema_version(self) -> int:
        try:
            row = self.conn.execute("SELECT version FROM schema_version").fetchone()
            return row["version"] if row else 0
        except sqlite3.OperationalError:
            return 0

    def _set_schema_version(self, version: int):
        self.conn.execute("DELETE FROM schema_version")
        self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    def _migrate(self):
        version = self._get_schema_version()

        if version < 1:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS photos (
                    uuid TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'pending',
                    file_path TEXT,
                    date_taken TIMESTAMP,
                    gps_lat REAL,
                    gps_lon REAL,
                    phash TEXT,
                    ai_result TEXT,
                    tags TEXT,
                    description TEXT,
                    importance TEXT,
                    media_type TEXT,
                    processed_at TIMESTAMP,
                    error_msg TEXT
                );
                CREATE TABLE IF NOT EXISTS duplicates (
                    group_id INTEGER,
                    photo_uuid TEXT,
                    similarity REAL,
                    FOREIGN KEY (photo_uuid) REFERENCES photos(uuid)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    photos_processed INTEGER DEFAULT 0,
                    photos_skipped INTEGER DEFAULT 0,
                    photos_errored INTEGER DEFAULT 0,
                    stop_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
                CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
                CREATE INDEX IF NOT EXISTS idx_duplicates_group ON duplicates(group_id);
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
            """)
            self._set_schema_version(1)
            self.conn.commit()
            version = 1

        if version < 2:
            new_columns = [
                ("apple_labels", "TEXT"),
                ("face_cluster_ids", "TEXT"),
                ("named_faces", "TEXT"),
                ("source_app", "TEXT"),
                ("is_selfie", "INTEGER"),
                ("is_screenshot", "INTEGER"),
                ("is_live_photo", "INTEGER"),
                ("location_city", "TEXT"),
                ("location_state", "TEXT"),
                ("location_country", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                try:
                    self.conn.execute(f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            # Ensure schema_version table exists (for v1 DBs without it)
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            self._set_schema_version(2)
            self.conn.commit()
            version = 2

        if version < 3:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    location_city TEXT,
                    location_state TEXT,
                    photo_count INTEGER,
                    face_cluster_ids TEXT,
                    summary TEXT,
                    mood TEXT,
                    cover_photo_uuid TEXT
                );
                CREATE TABLE IF NOT EXISTS event_photos (
                    event_id TEXT,
                    photo_uuid TEXT,
                    PRIMARY KEY (event_id, photo_uuid)
                );
                CREATE TABLE IF NOT EXISTS people (
                    face_cluster_id TEXT PRIMARY KEY,
                    apple_name TEXT,
                    user_name TEXT,
                    photo_count INTEGER,
                    event_count INTEGER,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    co_appearances TEXT,
                    top_locations TEXT,
                    appearance_trend TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time);
                CREATE INDEX IF NOT EXISTS idx_event_photos_photo ON event_photos(photo_uuid);
            """)
            self._set_schema_version(3)
            self.conn.commit()
            version = 3

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def upsert_photo(self, uuid: str, **kwargs):
        existing = self.get_photo(uuid)
        if existing:
            if kwargs:
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                self.conn.execute(
                    f"UPDATE photos SET {sets} WHERE uuid = ?",
                    (*kwargs.values(), uuid),
                )
                self.conn.commit()
        else:
            cols = ["uuid"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO photos ({col_names}) VALUES ({placeholders})",
                (uuid, *kwargs.values()),
            )
            self.conn.commit()

    def get_photo(self, uuid: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM photos WHERE uuid = ?", (uuid,)).fetchone()
        return dict(row) if row else None

    def get_pending_photos(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM photos WHERE status = 'pending' ORDER BY date_taken ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_photo_status(self, uuid: str, status: str, error_msg: str | None = None):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE photos SET status = ?, processed_at = ?, error_msg = ? WHERE uuid = ?",
            (status, now, error_msg, uuid),
        )
        self.conn.commit()

    def update_photo_result(self, uuid: str, status: str, phash: str, ai_result: str,
                            tags: str, description: str, importance: str, media_type: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE photos SET status=?, phash=?, ai_result=?, tags=?, description=?,
               importance=?, media_type=?, processed_at=? WHERE uuid=?""",
            (status, phash, ai_result, tags, description, importance, media_type, now, uuid),
        )
        self.conn.commit()

    def start_run(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute("INSERT INTO runs (started_at) VALUES (?)", (now,))
        self.conn.commit()
        return cursor.lastrowid

    def end_run(self, run_id: int, photos_processed: int, photos_skipped: int,
                photos_errored: int, stop_reason: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE runs SET ended_at=?, photos_processed=?, photos_skipped=?,
               photos_errored=?, stop_reason=? WHERE run_id=?""",
            (now, photos_processed, photos_skipped, photos_errored, stop_reason, run_id),
        )
        self.conn.commit()

    def add_duplicate_pair(self, group_id: int, photo_uuid: str, similarity: float):
        self.conn.execute(
            "INSERT INTO duplicates (group_id, photo_uuid, similarity) VALUES (?, ?, ?)",
            (group_id, photo_uuid, similarity),
        )
        self.conn.commit()

    def get_duplicate_group(self, group_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM duplicates WHERE group_id = ?", (group_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_phashes(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT uuid, phash FROM photos WHERE phash IS NOT NULL AND status = 'done'"
        ).fetchall()
        return [dict(r) for r in rows]

    def reset_photos_for_reprocess(self) -> int:
        """Reset all 'done' photos back to 'pending' for reprocessing.
        Returns the number of photos reset.
        """
        cursor = self.conn.execute(
            "UPDATE photos SET status = 'pending', ai_result = NULL, tags = NULL, "
            "description = NULL, importance = NULL, processed_at = NULL "
            "WHERE status = 'done'"
        )
        self.conn.commit()
        return cursor.rowcount

    def upsert_event(self, event_id: str, **kwargs):
        existing = self.conn.execute(
            "SELECT event_id FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            self.conn.execute(
                f"UPDATE events SET {sets} WHERE event_id = ?",
                (*kwargs.values(), event_id),
            )
        else:
            cols = ["event_id"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO events ({col_names}) VALUES ({placeholders})",
                (event_id, *kwargs.values()),
            )
        self.conn.commit()

    def link_photos_to_event(self, event_id: str, photo_uuids: list[str]):
        for uuid in photo_uuids:
            self.conn.execute(
                "INSERT OR IGNORE INTO event_photos (event_id, photo_uuid) VALUES (?, ?)",
                (event_id, uuid),
            )
        self.conn.commit()

    def get_event_photos(self, event_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM event_photos WHERE event_id = ?", (event_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY start_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_person(self, face_cluster_id: str, **kwargs):
        existing = self.conn.execute(
            "SELECT face_cluster_id FROM people WHERE face_cluster_id = ?",
            (face_cluster_id,),
        ).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            self.conn.execute(
                f"UPDATE people SET {sets} WHERE face_cluster_id = ?",
                (*kwargs.values(), face_cluster_id),
            )
        else:
            cols = ["face_cluster_id"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO people ({col_names}) VALUES ({placeholders})",
                (face_cluster_id, *kwargs.values()),
            )
        self.conn.commit()

    def get_all_people(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM people ORDER BY photo_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_person_user_name(self, face_cluster_id: str, user_name: str):
        self.conn.execute(
            "UPDATE people SET user_name = ? WHERE face_cluster_id = ?",
            (user_name, face_cluster_id),
        )
        self.conn.commit()

    def get_done_photos_ordered(self) -> list[dict]:
        """Get all done photos ordered by date_taken ASC for aggregation."""
        rows = self.conn.execute(
            "SELECT * FROM photos WHERE status = 'done' AND date_taken IS NOT NULL "
            "ORDER BY date_taken ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_person_by_name(self, name: str) -> dict | None:
        """Find a person by user_name first, then apple_name."""
        row = self.conn.execute(
            "SELECT * FROM people WHERE user_name = ? LIMIT 1", (name,)
        ).fetchone()
        if row:
            return dict(row)
        row = self.conn.execute(
            "SELECT * FROM people WHERE apple_name = ? LIMIT 1", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_events_for_person(self, face_cluster_id: str) -> list[dict]:
        """Get all events whose face_cluster_ids JSON array contains fc_id, ordered ASC."""
        pattern = f'%"{face_cluster_id}"%'
        rows = self.conn.execute(
            "SELECT * FROM events WHERE face_cluster_ids LIKE ? ORDER BY start_time ASC",
            (pattern,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_events_in_year(self, year: int) -> list[dict]:
        """Get all events whose start_time falls in the given year, ordered ASC."""
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        rows = self.conn.execute(
            "SELECT * FROM events WHERE start_time >= ? AND start_time < ? "
            "ORDER BY start_time ASC",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_events_on_this_day(self, month: int, day: int) -> list[dict]:
        """Get events from any year whose start_time matches the given month+day.

        Returns events grouped by year (sorted year DESC, time ASC within year).
        """
        # SQLite strftime: %m = month, %d = day (zero-padded)
        pattern_month = f"{month:02d}"
        pattern_day = f"{day:02d}"
        rows = self.conn.execute(
            "SELECT * FROM events "
            "WHERE strftime('%m', start_time) = ? AND strftime('%d', start_time) = ? "
            "ORDER BY start_time DESC",
            (pattern_month, pattern_day),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_photos(
        self,
        face_cluster_ids: list[str] | None = None,
        year: int | None = None,
        city: str | None = None,
        text: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search done photos by combined filters (AND semantics)."""
        conditions = ["status = 'done'"]
        params: list = []

        if face_cluster_ids:
            or_parts = []
            for fc_id in face_cluster_ids:
                or_parts.append("face_cluster_ids LIKE ?")
                params.append(f'%"{fc_id}"%')
            conditions.append("(" + " OR ".join(or_parts) + ")")

        if year is not None:
            conditions.append("date_taken >= ? AND date_taken < ?")
            params.append(f"{year}-01-01")
            params.append(f"{year + 1}-01-01")

        if city:
            conditions.append("location_city = ?")
            params.append(city)

        if text:
            conditions.append("(description LIKE ? OR tags LIKE ?)")
            params.append(f"%{text}%")
            params.append(f"%{text}%")

        sql = (
            "SELECT * FROM photos WHERE "
            + " AND ".join(conditions)
            + " ORDER BY date_taken DESC LIMIT ?"
        )
        params.append(limit)
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def get_cleanup_candidates(self, classes: list[str]) -> list[dict]:
        """Get photos whose cleanup_class (stored in 'importance' column) matches."""
        placeholders = ",".join(["?"] * len(classes))
        rows = self.conn.execute(
            f"SELECT * FROM photos WHERE status = 'done' AND importance IN ({placeholders}) "
            "ORDER BY date_taken DESC",
            tuple(classes),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_fc_ids_for_name(self, name: str) -> list[str]:
        """Return all face_cluster_ids whose user_name OR apple_name matches the given name."""
        rows = self.conn.execute(
            "SELECT face_cluster_id FROM people "
            "WHERE user_name = ? OR apple_name = ? "
            "ORDER BY photo_count DESC",
            (name, name),
        ).fetchall()
        return [r["face_cluster_id"] for r in rows]

    def close(self):
        self.conn.close()

"""SQLite database for tracking photo processing progress."""

import sqlite3
from datetime import datetime, timezone


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
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
        """)
        self.conn.commit()

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

    def close(self):
        self.conn.close()

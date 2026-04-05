import json
import pytest
from photo_memory.db import Database


def test_init_creates_tables(tmp_db_path):
    db = Database(tmp_db_path)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "photos" in table_names
    assert "duplicates" in table_names
    assert "runs" in table_names
    db.close()


def test_upsert_photo(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1", file_path="/path/to/photo.jpg", date_taken="2024-01-01")
    row = db.get_photo("uuid-1")
    assert row["file_path"] == "/path/to/photo.jpg"
    assert row["status"] == "pending"
    db.close()


def test_get_pending_photos(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1")
    db.upsert_photo("uuid-2")
    db.upsert_photo("uuid-3")
    db.update_photo_status("uuid-1", "done")
    pending = db.get_pending_photos(limit=10)
    assert len(pending) == 2
    assert {p["uuid"] for p in pending} == {"uuid-2", "uuid-3"}
    db.close()


def test_update_photo_result(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1")
    tags = json.dumps(["人物/合照", "美食/餐厅"])
    db.update_photo_result(
        "uuid-1",
        status="done",
        phash="abcdef1234567890",
        ai_result='{"description": "test"}',
        tags=tags,
        description="test photo",
        importance="medium",
        media_type="photo",
    )
    row = db.get_photo("uuid-1")
    assert row["status"] == "done"
    assert row["phash"] == "abcdef1234567890"
    assert row["description"] == "test photo"
    db.close()


def test_start_and_end_run(tmp_db_path):
    db = Database(tmp_db_path)
    run_id = db.start_run()
    assert run_id == 1
    db.end_run(run_id, photos_processed=10, photos_skipped=2, photos_errored=1, stop_reason="completed")
    row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["photos_processed"] == 10
    assert row["stop_reason"] == "completed"
    db.close()


def test_add_duplicate_group(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1")
    db.upsert_photo("uuid-2")
    db.add_duplicate_pair(group_id=1, photo_uuid="uuid-1", similarity=0.95)
    db.add_duplicate_pair(group_id=1, photo_uuid="uuid-2", similarity=0.95)
    groups = db.get_duplicate_group(1)
    assert len(groups) == 2
    db.close()


def test_get_all_phashes(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1")
    db.upsert_photo("uuid-2")
    db.update_photo_result("uuid-1", status="done", phash="aabb", ai_result="{}",
                           tags="[]", description="", importance="low", media_type="photo")
    db.update_photo_result("uuid-2", status="done", phash="ccdd", ai_result="{}",
                           tags="[]", description="", importance="low", media_type="photo")
    phashes = db.get_all_phashes()
    assert len(phashes) == 2
    assert phashes[0]["uuid"] in ("uuid-1", "uuid-2")
    db.close()


def test_schema_version_table_exists(tmp_db_path):
    db = Database(tmp_db_path)
    row = db.execute("SELECT version FROM schema_version").fetchone()
    assert row is not None
    assert row["version"] >= 2
    db.close()


def test_new_photo_columns_exist(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-test",
        file_path="/test.jpg",
        apple_labels='["人","猫"]',
        face_cluster_ids='["fc_001"]',
        named_faces='["唐嘉鑫"]',
        source_app="相机",
        is_selfie=False,
        is_screenshot=False,
        is_live_photo=True,
        location_city="北京市",
        location_state="北京市",
        location_country="中国",
    )
    row = db.get_photo("uuid-test")
    assert row["apple_labels"] == '["人","猫"]'
    assert row["face_cluster_ids"] == '["fc_001"]'
    assert row["named_faces"] == '["唐嘉鑫"]'
    assert row["is_selfie"] == 0
    assert row["is_screenshot"] == 0
    assert row["is_live_photo"] == 1
    assert row["location_city"] == "北京市"
    assert row["location_country"] == "中国"
    db.close()


def test_migration_from_v1_adds_columns(tmp_db_path):
    """Simulate a v1 database and verify migration adds new columns."""
    import sqlite3
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript("""
        CREATE TABLE photos (
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
        CREATE TABLE duplicates (
            group_id INTEGER,
            photo_uuid TEXT,
            similarity REAL
        );
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            photos_processed INTEGER DEFAULT 0,
            photos_skipped INTEGER DEFAULT 0,
            photos_errored INTEGER DEFAULT 0,
            stop_reason TEXT
        );
    """)
    conn.execute("INSERT INTO photos (uuid, file_path) VALUES ('old-1', '/old.jpg')")
    conn.commit()
    conn.close()

    db = Database(tmp_db_path)
    row = db.get_photo("old-1")
    assert row["file_path"] == "/old.jpg"
    assert row["apple_labels"] is None
    assert row["location_city"] is None
    ver = db.execute("SELECT version FROM schema_version").fetchone()
    assert ver["version"] >= 2
    db.close()

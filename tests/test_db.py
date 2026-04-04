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

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
        named_faces='["张三"]',
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
    assert row["named_faces"] == '["张三"]'
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


def test_schema_v3_creates_events_tables(tmp_db_path):
    db = Database(tmp_db_path)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "events" in table_names
    assert "event_photos" in table_names
    assert "people" in table_names
    ver = db.execute("SELECT version FROM schema_version").fetchone()
    assert ver["version"] >= 3
    db.close()


def test_upsert_event_and_link_photos(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("photo-1", date_taken="2024-09-28T13:00:00")
    db.upsert_photo("photo-2", date_taken="2024-09-28T13:15:00")

    db.upsert_event(
        event_id="evt_001",
        start_time="2024-09-28T13:00:00",
        end_time="2024-09-28T13:30:00",
        location_city="大连市",
        photo_count=2,
        face_cluster_ids='["fc_001"]',
        summary="海边散步",
        mood="愉快",
        cover_photo_uuid="photo-1",
    )
    db.link_photos_to_event("evt_001", ["photo-1", "photo-2"])

    row = db.execute("SELECT * FROM events WHERE event_id = ?", ("evt_001",)).fetchone()
    assert row["photo_count"] == 2
    assert row["location_city"] == "大连市"

    photos = db.get_event_photos("evt_001")
    assert len(photos) == 2
    assert {p["photo_uuid"] for p in photos} == {"photo-1", "photo-2"}
    db.close()


def test_upsert_person(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_person(
        face_cluster_id="fc_001",
        apple_name="张三",
        user_name=None,
        photo_count=671,
        event_count=120,
        first_seen="2015-03-01T12:00:00",
        last_seen="2026-03-15T18:00:00",
        co_appearances='{"fc_002": 6}',
        top_locations='["大连", "深圳"]',
        appearance_trend="stable",
    )
    row = db.execute("SELECT * FROM people WHERE face_cluster_id = ?", ("fc_001",)).fetchone()
    assert row["apple_name"] == "张三"
    assert row["photo_count"] == 671
    assert row["appearance_trend"] == "stable"
    db.close()


def test_get_done_photos_for_aggregation(tmp_db_path):
    """get_done_photos_ordered returns processed photos sorted by date."""
    db = Database(tmp_db_path)
    db.upsert_photo("p-1", date_taken="2024-09-28T13:00:00")
    db.upsert_photo("p-2", date_taken="2024-09-28T14:00:00")
    db.upsert_photo("p-3", date_taken="2024-09-27T12:00:00")
    for uuid in ["p-1", "p-2", "p-3"]:
        db.update_photo_status(uuid, "done")

    rows = db.get_done_photos_ordered()
    assert len(rows) == 3
    assert [r["uuid"] for r in rows] == ["p-3", "p-1", "p-2"]
    db.close()


def test_get_person_by_name_finds_user_name(tmp_db_path):
    """get_person_by_name should find person by user_name or apple_name."""
    db = Database(tmp_db_path)
    db.upsert_person(face_cluster_id="fc_001", apple_name="张三", user_name=None,
                     photo_count=100, event_count=10, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_002", apple_name=None, user_name="李四",
                     photo_count=50, event_count=5, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")

    # Match by apple_name
    p = db.get_person_by_name("张三")
    assert p is not None
    assert p["face_cluster_id"] == "fc_001"

    # Match by user_name
    p = db.get_person_by_name("李四")
    assert p is not None
    assert p["face_cluster_id"] == "fc_002"

    # user_name takes priority when both match
    db.upsert_person(face_cluster_id="fc_003", apple_name="李四", user_name=None,
                     photo_count=10, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    p = db.get_person_by_name("李四")
    assert p["face_cluster_id"] == "fc_002"  # user_name still wins

    # No match
    assert db.get_person_by_name("王五") is None
    db.close()


def test_get_events_for_person_returns_events_containing_face(tmp_db_path):
    """get_events_for_person returns events where face_cluster_ids JSON contains the fc_id."""
    db = Database(tmp_db_path)
    db.upsert_event(
        event_id="evt_001", start_time="2024-01-01T12:00:00", end_time="2024-01-01T13:00:00",
        location_city="北京", location_state=None, photo_count=3,
        face_cluster_ids='["fc_001", "fc_002"]',
        summary="事件1", mood="愉快", cover_photo_uuid="p1",
    )
    db.upsert_event(
        event_id="evt_002", start_time="2024-02-01T12:00:00", end_time="2024-02-01T13:00:00",
        location_city="上海", location_state=None, photo_count=2,
        face_cluster_ids='["fc_002"]',
        summary="事件2", mood="平静", cover_photo_uuid="p2",
    )
    db.upsert_event(
        event_id="evt_003", start_time="2024-03-01T12:00:00", end_time="2024-03-01T13:00:00",
        location_city="大连", location_state=None, photo_count=1,
        face_cluster_ids='["fc_001"]',
        summary="事件3", mood="愉快", cover_photo_uuid="p3",
    )

    events = db.get_events_for_person("fc_001")
    assert len(events) == 2
    event_ids = [e["event_id"] for e in events]
    assert "evt_001" in event_ids
    assert "evt_003" in event_ids
    assert event_ids.index("evt_001") < event_ids.index("evt_003")

    events_fc2 = db.get_events_for_person("fc_002")
    assert len(events_fc2) == 2
    db.close()


def test_get_events_in_year_filters_by_start_time(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_event(
        event_id="evt_2023", start_time="2023-06-15T10:00:00", end_time="2023-06-15T11:00:00",
        location_city="A", location_state=None, photo_count=1, face_cluster_ids='[]',
        summary="s", mood="", cover_photo_uuid="p1",
    )
    db.upsert_event(
        event_id="evt_2024_jan", start_time="2024-01-15T10:00:00", end_time="2024-01-15T11:00:00",
        location_city="B", location_state=None, photo_count=1, face_cluster_ids='[]',
        summary="s", mood="", cover_photo_uuid="p2",
    )
    db.upsert_event(
        event_id="evt_2024_dec", start_time="2024-12-31T23:00:00", end_time="2024-12-31T23:30:00",
        location_city="C", location_state=None, photo_count=1, face_cluster_ids='[]',
        summary="s", mood="", cover_photo_uuid="p3",
    )

    events = db.get_events_in_year(2024)
    assert len(events) == 2
    assert {e["event_id"] for e in events} == {"evt_2024_jan", "evt_2024_dec"}
    assert events[0]["event_id"] == "evt_2024_jan"
    db.close()

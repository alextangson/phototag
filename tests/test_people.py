"""Tests for people graph."""
import json
from photo_memory.people import compute_person_stats, infer_appearance_trend


def _photo(uuid, date, face_ids, city=None):
    return {
        "uuid": uuid,
        "date_taken": date,
        "face_cluster_ids": json.dumps(face_ids),
        "location_city": city,
        "named_faces": json.dumps([]),
    }


def test_compute_person_stats_counts_photos_and_coappearance():
    """compute_person_stats aggregates per-person photo count and co-appearance."""
    photos = [
        _photo("p1", "2024-01-01T12:00:00", ["fc_001"]),
        _photo("p2", "2024-02-01T12:00:00", ["fc_001", "fc_002"]),
        _photo("p3", "2024-03-01T12:00:00", ["fc_001", "fc_002", "fc_003"]),
        _photo("p4", "2024-04-01T12:00:00", ["fc_002"]),
    ]
    stats = compute_person_stats(photos)

    fc1 = next(s for s in stats if s["face_cluster_id"] == "fc_001")
    assert fc1["photo_count"] == 3
    assert fc1["first_seen"] == "2024-01-01T12:00:00"
    assert fc1["last_seen"] == "2024-03-01T12:00:00"
    assert fc1["co_appearances"]["fc_002"] == 2
    assert fc1["co_appearances"]["fc_003"] == 1

    fc2 = next(s for s in stats if s["face_cluster_id"] == "fc_002")
    assert fc2["photo_count"] == 3
    assert fc2["co_appearances"]["fc_001"] == 2


def test_compute_person_stats_uses_apple_name():
    """When named_faces contains a name, associate it with the face cluster."""
    photos = [
        {
            "uuid": "p1", "date_taken": "2024-01-01T12:00:00",
            "face_cluster_ids": json.dumps(["fc_001"]),
            "named_faces": json.dumps(["唐嘉鑫"]),
            "location_city": None,
        },
    ]
    stats = compute_person_stats(photos)
    fc1 = next(s for s in stats if s["face_cluster_id"] == "fc_001")
    assert fc1["apple_name"] == "唐嘉鑫"


def test_compute_person_stats_top_locations():
    photos = [
        _photo("p1", "2024-01-01T12:00:00", ["fc_001"], city="大连"),
        _photo("p2", "2024-02-01T12:00:00", ["fc_001"], city="大连"),
        _photo("p3", "2024-03-01T12:00:00", ["fc_001"], city="深圳"),
    ]
    stats = compute_person_stats(photos)
    fc1 = next(s for s in stats if s["face_cluster_id"] == "fc_001")
    assert fc1["top_locations"][0] == "大连"
    assert "深圳" in fc1["top_locations"]


def test_infer_appearance_trend_increasing():
    """Photos in recent 6 months > in earlier period → increasing."""
    dates = [f"2026-0{m}-01T12:00:00" for m in range(1, 5)]
    trend = infer_appearance_trend(dates, reference_date="2026-04-05")
    assert trend == "increasing"


def test_infer_appearance_trend_one_time():
    """Single appearance → one_time."""
    trend = infer_appearance_trend(["2024-01-01T12:00:00"], reference_date="2026-04-05")
    assert trend == "one_time"


def test_infer_appearance_trend_stable():
    """Long history, steady presence → stable."""
    dates = [
        "2023-01-01T12:00:00", "2023-06-01T12:00:00",
        "2024-01-01T12:00:00", "2024-06-01T12:00:00",
        "2025-01-01T12:00:00", "2025-06-01T12:00:00",
        "2026-01-01T12:00:00",
    ]
    trend = infer_appearance_trend(dates, reference_date="2026-04-05")
    assert trend == "stable"


from unittest.mock import patch


def test_build_people_writes_to_db(tmp_path):
    from photo_memory.db import Database
    from photo_memory.people import build_people

    db = Database(str(tmp_path / "test.db"))
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='["唐嘉鑫"]',
                    location_city="大连")
    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    face_cluster_ids='["fc_001", "fc_002"]',
                    named_faces='["唐嘉鑫"]',
                    location_city="深圳")
    for uuid in ["p1", "p2"]:
        db.update_photo_status(uuid, "done")

    count = build_people(db)

    assert count == 2  # fc_001 and fc_002
    people = db.get_all_people()
    assert len(people) == 2
    fc1 = next(p for p in people if p["face_cluster_id"] == "fc_001")
    assert fc1["photo_count"] == 2
    assert fc1["apple_name"] == "唐嘉鑫"
    co = json.loads(fc1["co_appearances"])
    assert co["fc_002"] == 1
    db.close()


def test_build_people_preserves_user_name(tmp_path):
    """Rebuilding people should not overwrite user_name set by user."""
    from photo_memory.db import Database
    from photo_memory.people import build_people

    db = Database(str(tmp_path / "test.db"))
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='[]')
    db.update_photo_status("p1", "done")

    build_people(db)
    db.set_person_user_name("fc_001", "阿菁")

    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='[]')
    db.update_photo_status("p2", "done")
    build_people(db)

    row = db.execute("SELECT user_name FROM people WHERE face_cluster_id = ?", ("fc_001",)).fetchone()
    assert row["user_name"] == "阿菁"
    db.close()

"""Tests for daily memory push."""
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from photo_memory.daily import generate_on_this_day, generate_this_week, _extract_year


def test_extract_year():
    assert _extract_year("2024-03-15T10:00:00") == 2024
    assert _extract_year(None) == 0
    assert _extract_year("invalid") == 0


def test_generate_on_this_day_returns_none_when_no_events(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    result = generate_on_this_day(db, datetime(2024, 3, 15), {"host": "h", "model": "m", "timeout": 10})
    assert result is None
    db.close()


def test_generate_on_this_day_skips_current_year(tmp_path):
    """Events from the current year should not appear in On This Day."""
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    # Only event is from 2024, and we're looking at 2024-03-15
    db.upsert_event(
        event_id="e1", start_time="2024-03-15T10:00:00", end_time="2024-03-15T11:00:00",
        location_city="北京", location_state=None, photo_count=1,
        face_cluster_ids='[]', summary="今年的事", mood="愉快", cover_photo_uuid="p1",
    )
    result = generate_on_this_day(db, datetime(2024, 3, 15), {"host": "h", "model": "m", "timeout": 10})
    assert result is None
    db.close()


def test_generate_on_this_day_with_past_event(tmp_path):
    from photo_memory.db import Database
    from PIL import Image

    db = Database(str(tmp_path / "t.db"))

    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100), "blue").save(img_path, "JPEG")

    db.upsert_photo("p1", date_taken="2022-03-15T10:00:00", file_path=str(img_path))
    db.upsert_event(
        event_id="e1", start_time="2022-03-15T10:00:00", end_time="2022-03-15T11:00:00",
        location_city="大连", location_state=None, photo_count=1,
        face_cluster_ids='[]', summary="2年前在大连", mood="愉快", cover_photo_uuid="p1",
    )
    db.link_photos_to_event("e1", ["p1"])

    with patch("photo_memory.daily.generate_period_narrative", return_value="那年的回忆"):
        result = generate_on_this_day(db, datetime(2024, 3, 15), {"host": "h", "model": "m", "timeout": 10})

    assert result is not None
    assert "3月15日" in result
    assert "2年前" in result
    assert "那年的回忆" in result
    assert "data:image/jpeg;base64," in result
    db.close()


def test_generate_this_week_finds_events_across_week(tmp_path):
    from photo_memory.db import Database
    from PIL import Image

    db = Database(str(tmp_path / "t.db"))

    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100), "red").save(img_path, "JPEG")

    # Event on a Wednesday 2 years ago
    db.upsert_photo("p1", date_taken="2022-04-06T10:00:00", file_path=str(img_path))
    db.upsert_event(
        event_id="e1", start_time="2022-04-06T10:00:00", end_time="2022-04-06T11:00:00",
        location_city="深圳", location_state=None, photo_count=1,
        face_cluster_ids='[]', summary="深圳出差", mood="平静", cover_photo_uuid="p1",
    )
    db.link_photos_to_event("e1", ["p1"])

    # target_date is 2024-04-08 (Monday), week covers Apr 8-14
    # But our event is Apr 6 which is in a different week
    # Let's use a date where Apr 6 IS in the same week
    # 2024-04-06 is a Saturday. Week (Mon-Sun) = Apr 1-7
    with patch("photo_memory.daily.generate_period_narrative", return_value="本周回忆"):
        result = generate_this_week(db, datetime(2024, 4, 3), {"host": "h", "model": "m", "timeout": 10})

    # Apr 3 2024 is Wednesday, week = Apr 1-7. Event on Apr 6 (any year) should match.
    assert result is not None
    assert "This Week" in result
    assert "本周回忆" in result
    db.close()

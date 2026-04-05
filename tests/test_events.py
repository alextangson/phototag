"""Tests for event aggregation logic."""
import json
from datetime import datetime
from photo_memory.events import slice_into_events, _parse_date


def _photo(uuid, date, face_ids=None, city=None, source=None, scene="other"):
    return {
        "uuid": uuid,
        "date_taken": date,
        "face_cluster_ids": json.dumps(face_ids or []),
        "location_city": city,
        "source_app": source,
        "ai_result": json.dumps({"scene_category": scene}),
    }


def test_slice_single_event_within_30min():
    """Photos taken within 30 minutes should form one event."""
    photos = [
        _photo("p1", "2024-09-28T13:00:00"),
        _photo("p2", "2024-09-28T13:10:00"),
        _photo("p3", "2024-09-28T13:25:00"),
    ]
    events = slice_into_events(photos, gap_minutes=30)
    assert len(events) == 1
    assert len(events[0]["photos"]) == 3
    assert events[0]["start_time"] == "2024-09-28T13:00:00"
    assert events[0]["end_time"] == "2024-09-28T13:25:00"


def test_slice_multiple_events_on_time_gap():
    """Gap of 31+ minutes should split into separate events."""
    photos = [
        _photo("p1", "2024-09-28T13:00:00"),
        _photo("p2", "2024-09-28T13:15:00"),
        _photo("p3", "2024-09-28T14:00:00"),  # 45min gap
        _photo("p4", "2024-09-28T14:10:00"),
    ]
    events = slice_into_events(photos, gap_minutes=30)
    assert len(events) == 2
    assert len(events[0]["photos"]) == 2
    assert len(events[1]["photos"]) == 2


def test_slice_skips_photos_without_date():
    """Photos without date_taken should be skipped."""
    photos = [
        _photo("p1", "2024-09-28T13:00:00"),
        _photo("p2", None),
        _photo("p3", "2024-09-28T13:20:00"),
    ]
    events = slice_into_events(photos, gap_minutes=30)
    assert len(events) == 1
    assert len(events[0]["photos"]) == 2


def test_slice_empty_list():
    assert slice_into_events([], gap_minutes=30) == []


def test_parse_date_handles_iso_formats():
    assert _parse_date("2024-09-28T13:00:00") == datetime(2024, 9, 28, 13, 0, 0)
    assert _parse_date("2024-09-28T13:00:00+08:00") is not None
    assert _parse_date(None) is None
    assert _parse_date("invalid") is None

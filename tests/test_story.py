"""Tests for story generation."""
import json
from photo_memory.story import group_events_by_period, _pick_grouping_strategy


def _event(event_id, start_time, summary="s", location="北京", photo_count=1, mood="平静"):
    return {
        "event_id": event_id,
        "start_time": start_time,
        "end_time": start_time,
        "summary": summary,
        "location_city": location,
        "photo_count": photo_count,
        "mood": mood,
    }


def test_pick_grouping_strategy_single_event():
    assert _pick_grouping_strategy([_event("e1", "2024-01-01T12:00:00")]) == "none"


def test_pick_grouping_strategy_small_batch():
    """5 events in one month → no grouping needed."""
    events = [_event(f"e{i}", f"2024-01-{i:02d}T12:00:00") for i in range(1, 6)]
    assert _pick_grouping_strategy(events) == "none"


def test_pick_grouping_strategy_medium_spans_months():
    """20 events across 6 months → group by month."""
    events = []
    for m in range(1, 7):
        for d in [1, 10, 20]:
            events.append(_event(f"e{m}_{d}", f"2024-{m:02d}-{d:02d}T12:00:00"))
    # 18 events across 6 months
    assert _pick_grouping_strategy(events) == "month"


def test_pick_grouping_strategy_large_multi_year():
    """100 events across 3 years → group by quarter."""
    events = []
    for y in [2022, 2023, 2024]:
        for m in range(1, 13):
            for d in [5, 15, 25]:
                events.append(_event(f"e{y}_{m}_{d}", f"{y}-{m:02d}-{d:02d}T12:00:00"))
    # 108 events across 3 years
    assert _pick_grouping_strategy(events) == "quarter"


def test_group_events_by_period_none_returns_single_group():
    events = [_event("e1", "2024-01-01T12:00:00"), _event("e2", "2024-01-02T12:00:00")]
    groups = group_events_by_period(events, strategy="none")
    assert len(groups) == 1
    assert groups[0]["label"] == ""
    assert len(groups[0]["events"]) == 2


def test_group_events_by_period_month():
    events = [
        _event("e1", "2024-01-10T12:00:00"),
        _event("e2", "2024-01-25T12:00:00"),
        _event("e3", "2024-02-05T12:00:00"),
        _event("e4", "2024-03-15T12:00:00"),
    ]
    groups = group_events_by_period(events, strategy="month")
    assert len(groups) == 3
    assert groups[0]["label"] == "2024-01"
    assert len(groups[0]["events"]) == 2
    assert groups[1]["label"] == "2024-02"
    assert groups[2]["label"] == "2024-03"


def test_group_events_by_period_quarter():
    events = [
        _event("e1", "2024-01-10T12:00:00"),  # Q1
        _event("e2", "2024-03-25T12:00:00"),  # Q1
        _event("e3", "2024-04-05T12:00:00"),  # Q2
        _event("e4", "2024-11-15T12:00:00"),  # Q4
    ]
    groups = group_events_by_period(events, strategy="quarter")
    assert len(groups) == 3
    assert groups[0]["label"] == "2024 Q1"
    assert len(groups[0]["events"]) == 2
    assert groups[1]["label"] == "2024 Q2"
    assert groups[2]["label"] == "2024 Q4"

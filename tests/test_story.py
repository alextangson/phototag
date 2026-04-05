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


from unittest.mock import patch, MagicMock
from photo_memory.story import generate_period_narrative, _extract_json_object


def test_extract_json_object_direct():
    raw = '{"narrative": "hello"}'
    assert _extract_json_object(raw) == {"narrative": "hello"}


def test_extract_json_object_markdown_block():
    raw = '这是结果：\n```json\n{"narrative": "hello"}\n```'
    assert _extract_json_object(raw) == {"narrative": "hello"}


def test_extract_json_object_prose_wrap():
    raw = '好的 {"narrative": "hello"} 就这样'
    assert _extract_json_object(raw) == {"narrative": "hello"}


def test_extract_json_object_returns_none_on_junk():
    assert _extract_json_object("totally not json") is None


def test_generate_period_narrative_calls_llm():
    events = [
        _event("e1", "2024-01-10T12:00:00", summary="和朋友在海边散步", location="大连", mood="愉快"),
        _event("e2", "2024-01-25T12:00:00", summary="公司年会聚餐", location="大连", mood="热闹"),
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": '{"narrative": "一月在大连度过了放松的时光，和朋友散步看海，也参加了热闹的年会聚餐。"}'
    }

    with patch("photo_memory.story.requests.post", return_value=mock_response) as mock_post:
        narrative = generate_period_narrative(
            events, period_label="2024-01",
            host="http://localhost:11434", model="gemma4:e4b", timeout=60,
        )

    assert "大连" in narrative
    assert "年会" in narrative or "朋友" in narrative
    call_payload = mock_post.call_args[1]["json"]
    assert "和朋友在海边散步" in call_payload["prompt"]
    assert "公司年会聚餐" in call_payload["prompt"]


def test_generate_period_narrative_fallback_on_error():
    events = [_event("e1", "2024-01-10T12:00:00", summary="和朋友散步")]

    with patch("photo_memory.story.requests.post", side_effect=Exception("network down")):
        narrative = generate_period_narrative(
            events, period_label="2024-01",
            host="http://x", model="m", timeout=10,
        )

    assert "和朋友散步" in narrative


def test_generate_person_story_unknown_person_returns_error_message(tmp_path):
    from photo_memory.db import Database
    from photo_memory.story import generate_person_story

    db = Database(str(tmp_path / "t.db"))
    result = generate_person_story(
        db, "不存在的人",
        ollama_config={"host": "h", "model": "m", "timeout": 10},
    )
    assert "找不到" in result or "not found" in result.lower()
    db.close()


def test_generate_person_story_with_events(tmp_path):
    from photo_memory.db import Database
    from photo_memory.story import generate_person_story

    db = Database(str(tmp_path / "t.db"))
    db.upsert_person(
        face_cluster_id="fc_001", apple_name="张三", user_name=None,
        photo_count=3, event_count=2,
        first_seen="2024-01-01T12:00:00", last_seen="2024-03-01T12:00:00",
        co_appearances="{}", top_locations='["北京", "大连"]', appearance_trend="stable",
    )
    db.upsert_event(
        event_id="evt_1", start_time="2024-01-15T12:00:00", end_time="2024-01-15T13:00:00",
        location_city="北京", location_state=None, photo_count=2,
        face_cluster_ids='["fc_001"]',
        summary="和张三在颐和园散步", mood="愉快", cover_photo_uuid="p1",
    )
    db.upsert_event(
        event_id="evt_2", start_time="2024-02-20T12:00:00", end_time="2024-02-20T13:00:00",
        location_city="大连", location_state=None, photo_count=1,
        face_cluster_ids='["fc_001"]',
        summary="大连海边合影", mood="平静", cover_photo_uuid="p2",
    )

    with patch("photo_memory.story.generate_period_narrative",
               return_value="一段回忆叙事"):
        md = generate_person_story(
            db, "张三",
            ollama_config={"host": "h", "model": "m", "timeout": 10},
        )

    assert "# 和张三的回忆" in md
    assert "3" in md
    assert "2024" in md
    assert "一段回忆叙事" in md
    db.close()


def test_generate_year_story_no_events_returns_message(tmp_path):
    from photo_memory.db import Database
    from photo_memory.story import generate_year_story

    db = Database(str(tmp_path / "t.db"))
    result = generate_year_story(
        db, 2099,
        ollama_config={"host": "h", "model": "m", "timeout": 10},
    )
    assert "2099" in result
    assert "没有" in result or "暂无" in result
    db.close()


def test_generate_year_story_with_events(tmp_path):
    from photo_memory.db import Database
    from photo_memory.story import generate_year_story

    db = Database(str(tmp_path / "t.db"))
    db.upsert_event(
        event_id="e1", start_time="2024-03-15T10:00:00", end_time="2024-03-15T11:00:00",
        location_city="北京", location_state=None, photo_count=5,
        face_cluster_ids='["fc_001"]',
        summary="春天在颐和园", mood="愉快", cover_photo_uuid="p1",
    )
    db.upsert_event(
        event_id="e2", start_time="2024-08-10T10:00:00", end_time="2024-08-10T11:00:00",
        location_city="大连", location_state=None, photo_count=3,
        face_cluster_ids='["fc_002"]',
        summary="夏日海滨", mood="愉快", cover_photo_uuid="p2",
    )

    with patch("photo_memory.story.generate_period_narrative",
               return_value="这段时间发生了美好的事"):
        md = generate_year_story(
            db, 2024,
            ollama_config={"host": "h", "model": "m", "timeout": 10},
        )

    assert "# 2024 年度回忆" in md
    assert "这段时间发生了美好的事" in md
    assert "8" in md  # total photo count
    db.close()


def test_generate_relationship_story_frames_as_relationship(tmp_path):
    """Relationship story is like person story but with different framing."""
    from photo_memory.db import Database
    from photo_memory.story import generate_relationship_story

    db = Database(str(tmp_path / "t.db"))
    db.upsert_person(
        face_cluster_id="fc_001", apple_name=None, user_name="李四",
        photo_count=200, event_count=45,
        first_seen="2023-01-01T12:00:00", last_seen="2026-02-01T12:00:00",
        co_appearances="{}", top_locations='["北京"]', appearance_trend="stable",
    )
    db.upsert_event(
        event_id="evt_1", start_time="2023-06-15T12:00:00", end_time="2023-06-15T13:00:00",
        location_city="北京", location_state=None, photo_count=5,
        face_cluster_ids='["fc_001"]',
        summary="初次见面", mood="新鲜", cover_photo_uuid="p1",
    )

    with patch("photo_memory.story.generate_period_narrative",
               return_value="时光回溯"):
        md = generate_relationship_story(
            db, "李四",
            ollama_config={"host": "h", "model": "m", "timeout": 10},
        )

    assert "# 和李四在一起的日子" in md
    assert "天" in md or "年" in md  # duration mentioned
    db.close()

# M5: Story Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 L2 的 events + people 结构化数据转化为 Markdown 叙事，通过 `phototag story --person <name>` / `--year <year>` / `--relationship <name>` 生成用户可读的回忆故事。

**Architecture:** 新建 `story.py` 负责叙事生成。三种模式共享核心流程：解析目标 → 从 DB 查事件 → 按时间段分组 → 每段调 LLM 生成叙事 → 拼接成 Markdown。事件少时单次 LLM 调用；事件多时分层生成（按季度/月分组）避免超长上下文。CLI 新增 `story` 命令支持 `--output` 导出 .md / .json。

**Tech Stack:** Python 3.12, SQLite, Ollama API (gemma4:e4b), Click, pytest

---

### Task 1: DB 查询方法 — 人物解析 + 事件查询

**Files:**
- Modify: `src/photo_memory/db.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_db.py`:

```python
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
    # Results ordered by start_time ASC
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
    # Ordered ASC
    assert events[0]["event_id"] == "evt_2024_jan"
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py::test_get_person_by_name_finds_user_name tests/test_db.py::test_get_events_for_person_returns_events_containing_face tests/test_db.py::test_get_events_in_year_filters_by_start_time -v`
Expected: FAIL — methods do not exist

- [ ] **Step 3: 实现 DB 方法**

Add to `Database` class in `src/photo_memory/db.py` (before `close`):

```python
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
        # SQLite JSON pattern: face_cluster_ids stores '["fc_001", "fc_002"]'
        # Use LIKE with quoted fc_id for safe matching
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py -v`
Expected: ALL PASS (原有 14 + 3 新 = 17)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/db.py tests/test_db.py
git commit -m "feat(db): add query methods for story generation (person/year event lookup)"
```

---

### Task 2: 事件分组工具 — 按时间段切块

**Files:**
- Create: `src/photo_memory/story.py`
- Create: `tests/test_story.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_story.py`:

```python
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
    assert groups[0]["label"] == ""  # no period label for single group
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_story.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: 实现 story.py 基础**

Create `src/photo_memory/story.py`:

```python
"""Story generation: transform L2 events + people into Markdown narratives."""

import json
import logging
import re
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _pick_grouping_strategy(events: list[dict]) -> str:
    """Decide how to group events for narrative generation.

    Returns: 'none' | 'month' | 'quarter'

    Heuristic:
    - <= 10 events → 'none' (single LLM call)
    - <= 30 events AND spans <= 12 months → 'month'
    - otherwise → 'quarter'
    """
    if len(events) <= 10:
        return "none"

    dates = [_parse_date(e.get("start_time")) for e in events]
    dates = [d for d in dates if d]
    if not dates:
        return "none"

    span_days = (max(dates) - min(dates)).days
    span_months = span_days / 30

    if len(events) <= 30 and span_months <= 12:
        return "month"
    return "quarter"


def group_events_by_period(events: list[dict], strategy: str) -> list[dict]:
    """Group events into periods according to strategy.

    Returns a list of group dicts: [{"label": "2024-01", "events": [...]}, ...]
    Groups are sorted by chronological order.
    """
    if strategy == "none" or not events:
        return [{"label": "", "events": events}]

    buckets: dict[str, list[dict]] = {}
    for e in events:
        dt = _parse_date(e.get("start_time"))
        if dt is None:
            continue

        if strategy == "month":
            label = f"{dt.year}-{dt.month:02d}"
        elif strategy == "quarter":
            q = (dt.month - 1) // 3 + 1
            label = f"{dt.year} Q{q}"
        else:
            label = ""

        buckets.setdefault(label, []).append(e)

    return [{"label": label, "events": evs} for label, evs in sorted(buckets.items())]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_story.py -v`
Expected: 7/7 PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/story.py tests/test_story.py
git commit -m "feat(story): event grouping by period (month/quarter) with auto strategy"
```

---

### Task 3: 叙事段生成 — LLM 调用

**Files:**
- Modify: `src/photo_memory/story.py`
- Modify: `tests/test_story.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_story.py`:

```python
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
    # Verify prompt contains event summaries
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

    # Fallback: concatenated event summaries
    assert "和朋友散步" in narrative
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_story.py::test_generate_period_narrative_calls_llm tests/test_story.py::test_generate_period_narrative_fallback_on_error tests/test_story.py::test_extract_json_object_direct -v`
Expected: FAIL

- [ ] **Step 3: 实现 _extract_json_object 和 generate_period_narrative**

Append to `src/photo_memory/story.py`:

```python
def _extract_json_object(raw: str) -> dict | None:
    """Extract JSON object from raw LLM output (direct / markdown / prose wrap)."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


PERIOD_NARRATIVE_PROMPT = """你是一个温暖的回忆讲述者。基于以下事件列表，生成一段自然流畅的中文叙事。

时间段：{period_label}
事件数：{event_count}

事件列表：
{events_text}

返回严格 JSON 格式（不要其他文字）：
{{
  "narrative": "一段 100-200 字的中文叙事，串联这些事件，像在对老朋友讲述回忆。不要罗列事件，要有情感温度和时间流动感。"
}}"""


def generate_period_narrative(
    events: list[dict],
    period_label: str,
    host: str,
    model: str,
    timeout: int,
) -> str:
    """Generate a narrative paragraph for a period's events via LLM.

    Falls back to concatenated summaries if LLM call fails.
    """
    if not events:
        return ""

    events_text_lines = []
    for i, e in enumerate(events, 1):
        date = (e.get("start_time") or "")[:10]
        location = e.get("location_city") or "某地"
        summary = e.get("summary") or ""
        mood = e.get("mood") or ""
        mood_str = f"（{mood}）" if mood else ""
        events_text_lines.append(f"{i}. [{date}] {location}{mood_str}：{summary}")
    events_text = "\n".join(events_text_lines)

    prompt = PERIOD_NARRATIVE_PROMPT.format(
        period_label=period_label or "全部时间",
        event_count=len(events),
        events_text=events_text,
    )

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        data = _extract_json_object(raw)
        if data and "narrative" in data:
            return data["narrative"]
        logger.warning(f"Period narrative LLM returned unparseable output: {raw[:200]}")
    except Exception as e:
        logger.warning(f"Period narrative LLM failed: {e}")

    # Fallback: join event summaries
    return "；".join(e.get("summary") or "" for e in events if e.get("summary"))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_story.py -v`
Expected: ALL PASS (12/12)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/story.py tests/test_story.py
git commit -m "feat(story): generate_period_narrative via LLM with robust parsing"
```

---

### Task 4: 故事编排 — Person / Year / Relationship

**Files:**
- Modify: `src/photo_memory/story.py`
- Modify: `tests/test_story.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_story.py`:

```python
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

    # Markdown header with person name
    assert "# 和张三的回忆" in md
    # Stats summary
    assert "3" in md  # photo_count
    assert "2024" in md  # date range
    # At least one narrative body
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
    # Should mention 8 photos total
    assert "8" in md
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
    # Relationship story emphasizes duration
    assert "天" in md or "年" in md  # duration mentioned
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_story.py -v 2>&1 | tail -20`
Expected: FAIL — generate_person_story / generate_year_story / generate_relationship_story 不存在

- [ ] **Step 3: 实现三个顶层函数**

Append to `src/photo_memory/story.py`:

```python
def _render_story_markdown(title: str, stats_line: str, groups: list[dict],
                           ollama_config: dict) -> str:
    """Render a story Markdown document from grouped events + LLM narratives."""
    lines = [f"# {title}", "", stats_line, ""]

    for group in groups:
        label = group["label"]
        events = group["events"]
        if label:
            lines.append(f"## {label}")
            lines.append("")

        narrative = generate_period_narrative(
            events,
            period_label=label,
            host=ollama_config["host"],
            model=ollama_config["model"],
            timeout=ollama_config["timeout"],
        )
        lines.append(narrative)
        lines.append("")

    return "\n".join(lines)


def generate_person_story(db, person_name: str, ollama_config: dict) -> str:
    """Generate a person's photo timeline story as Markdown."""
    person = db.get_person_by_name(person_name)
    if not person:
        return f"# 找不到「{person_name}」\n\n请先用 `phototag people --name <fc_id> <name>` 命名该人物，或检查拼写。"

    events = db.get_events_for_person(person["face_cluster_id"])
    if not events:
        return f"# 和{person_name}的回忆\n\n暂无包含此人的事件记录。"

    first = (person["first_seen"] or "")[:10]
    last = (person["last_seen"] or "")[:10]
    photo_count = person["photo_count"] or 0
    event_count = len(events)

    title = f"和{person_name}的回忆"
    stats_line = (
        f"> 共 {photo_count} 张照片，{event_count} 个事件，"
        f"{first} ~ {last}"
    )

    strategy = _pick_grouping_strategy(events)
    groups = group_events_by_period(events, strategy=strategy)

    return _render_story_markdown(title, stats_line, groups, ollama_config)


def generate_year_story(db, year: int, ollama_config: dict) -> str:
    """Generate a year-in-review story as Markdown."""
    events = db.get_events_in_year(year)
    if not events:
        return f"# {year} 年度回忆\n\n这一年暂无事件记录。"

    photo_total = sum(e.get("photo_count") or 0 for e in events)
    cities = {e.get("location_city") for e in events if e.get("location_city")}

    title = f"{year} 年度回忆"
    stats_line = (
        f"> 共 {photo_total} 张照片，{len(events)} 个事件，"
        f"到过 {len(cities)} 个地方"
    )

    strategy = _pick_grouping_strategy(events)
    groups = group_events_by_period(events, strategy=strategy)

    return _render_story_markdown(title, stats_line, groups, ollama_config)


def generate_relationship_story(db, person_name: str, ollama_config: dict) -> str:
    """Generate a relationship-framed story (duration emphasized)."""
    person = db.get_person_by_name(person_name)
    if not person:
        return f"# 找不到「{person_name}」\n\n请先命名该人物。"

    events = db.get_events_for_person(person["face_cluster_id"])
    if not events:
        return f"# 和{person_name}在一起的日子\n\n暂无事件记录。"

    first_dt = _parse_date(person["first_seen"])
    last_dt = _parse_date(person["last_seen"])
    days = (last_dt - first_dt).days if (first_dt and last_dt) else 0
    years = days / 365

    photo_count = person["photo_count"] or 0

    title = f"和{person_name}在一起的日子"
    if years >= 1:
        stats_line = (
            f"> 从 {(person['first_seen'] or '')[:10]} 到 {(person['last_seen'] or '')[:10]}，"
            f"共 {days} 天（约 {years:.1f} 年），{photo_count} 张照片，{len(events)} 个事件"
        )
    else:
        stats_line = (
            f"> 从 {(person['first_seen'] or '')[:10]} 到 {(person['last_seen'] or '')[:10]}，"
            f"共 {days} 天，{photo_count} 张照片，{len(events)} 个事件"
        )

    strategy = _pick_grouping_strategy(events)
    groups = group_events_by_period(events, strategy=strategy)

    return _render_story_markdown(title, stats_line, groups, ollama_config)
```

- [ ] **Step 4: 运行 story 测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_story.py -v`
Expected: ALL PASS (17/17)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/story.py tests/test_story.py
git commit -m "feat(story): person/year/relationship story generators with Markdown output"
```

---

### Task 5: CLI `story` 命令

**Files:**
- Modify: `src/photo_memory/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_cli.py`:

```python
def test_story_year_command_outputs_markdown(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_event(
        event_id="e1", start_time="2024-05-15T12:00:00", end_time="2024-05-15T13:00:00",
        location_city="北京", location_state=None, photo_count=3,
        face_cluster_ids='[]', summary="颐和园", mood="愉快", cover_photo_uuid="p1",
    )
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.story.generate_period_narrative", return_value="春日回忆"):
        result = runner.invoke(main, ["--config", config_path, "story", "--year", "2024"])

    assert result.exit_code == 0
    assert "2024 年度回忆" in result.output
    assert "春日回忆" in result.output


def test_story_person_command_with_output_file(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_person(
        face_cluster_id="fc_001", apple_name="张三", user_name=None,
        photo_count=5, event_count=2,
        first_seen="2024-01-01T12:00:00", last_seen="2024-06-01T12:00:00",
        co_appearances="{}", top_locations="[]", appearance_trend="stable",
    )
    db.upsert_event(
        event_id="e1", start_time="2024-01-15T12:00:00", end_time="2024-01-15T13:00:00",
        location_city="北京", location_state=None, photo_count=3,
        face_cluster_ids='["fc_001"]', summary="颐和园", mood="愉快", cover_photo_uuid="p1",
    )
    db.close()

    output_file = tmp_path / "story.md"
    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.story.generate_period_narrative", return_value="一段回忆"):
        result = runner.invoke(main, [
            "--config", config_path, "story",
            "--person", "张三",
            "--output", str(output_file),
        ])

    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "和张三的回忆" in content
    assert "一段回忆" in content


def test_story_requires_one_mode(tmp_path, sample_config):
    """Running `phototag story` without --person/--year/--relationship should error."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "story"])

    assert result.exit_code != 0
    assert "需要指定" in result.output or "required" in result.output.lower()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py::test_story_year_command_outputs_markdown tests/test_cli.py::test_story_person_command_with_output_file tests/test_cli.py::test_story_requires_one_mode -v`
Expected: FAIL — story command 不存在

- [ ] **Step 3: 实现 CLI story 命令**

Add to imports section in `src/photo_memory/cli.py`:
```python
from photo_memory.story import (
    generate_person_story,
    generate_year_story,
    generate_relationship_story,
)
```

Add after the `people` command:

```python
@main.command()
@click.option("--person", "person_name", type=str, default=None,
              help="Generate a person's timeline story")
@click.option("--year", "year", type=int, default=None,
              help="Generate a year-in-review story")
@click.option("--relationship", "relationship_name", type=str, default=None,
              help="Generate a relationship-framed story")
@click.option("--output", "output_path", type=click.Path(), default=None,
              help="Write story to file instead of stdout (.md)")
@click.pass_context
def story(ctx, person_name, year, relationship_name, output_path):
    """Generate a Markdown story from L2 events and people."""
    modes = [person_name, year, relationship_name]
    mode_count = sum(1 for m in modes if m is not None)

    if mode_count == 0:
        click.echo("错误：需要指定 --person / --year / --relationship 之一", err=True)
        raise SystemExit(1)
    if mode_count > 1:
        click.echo("错误：--person / --year / --relationship 只能选一个", err=True)
        raise SystemExit(1)

    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    if person_name:
        markdown = generate_person_story(db, person_name, ollama_config=config["ollama"])
    elif year:
        markdown = generate_year_story(db, year, ollama_config=config["ollama"])
    else:
        markdown = generate_relationship_story(db, relationship_name, ollama_config=config["ollama"])

    db.close()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        click.echo(f"Story written to {output_path}")
    else:
        click.echo(markdown)
```

- [ ] **Step 4: 运行 CLI 测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 5: 全量回归**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/cli.py tests/test_cli.py
git commit -m "feat(cli): add story command for person/year/relationship narratives"
```

---

### Task 6: 真实数据验证

**Files:** 无代码改动

- [ ] **Step 1: 在 e2e 测试 DB 上生成年度故事**

Run: 
```bash
cd ~/photo-memory && .venv/bin/python3 -c "
import sys
sys.path.insert(0, 'src')
from photo_memory.db import Database
from photo_memory.story import generate_year_story

db = Database('/tmp/e2e_test.db')
md = generate_year_story(db, 2025, ollama_config={
    'host': 'http://localhost:11434', 'model': 'gemma4:e4b', 'timeout': 120,
})
print(md)
db.close()
"
```

Expected: 输出 Markdown 格式的 2025 年度回忆，包含 events 数据生成的叙事

- [ ] **Step 2: 生成人物故事**

```bash
cd ~/photo-memory && .venv/bin/python3 -c "
import sys
sys.path.insert(0, 'src')
from photo_memory.db import Database
from photo_memory.story import generate_person_story

db = Database('/tmp/e2e_test.db')
md = generate_person_story(db, '张三', ollama_config={
    'host': 'http://localhost:11434', 'model': 'gemma4:e4b', 'timeout': 120,
})
print(md)
db.close()
"
```

Expected: 输出"和张三的回忆"，包含事件时间线

- [ ] **Step 3: 主观评估**

对照 spec M5 的成功标准"自己用了一周不想删"：
- 叙事是否有温度？读着像回忆还是像流水账？
- 事件之间有连贯性吗？
- 有没有让你想起被遗忘的事？
- 有没有错或者尴尬的叙述？

记录一两个好例子和差例子，作为后续 prompt 优化的依据。

# M3 + M4: Layer 2 Association (Events + People) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 10000 张孤立照片聚合为事件、人物图谱，为叙事和搜索提供结构化语义数据。

**Architecture:** 新增 `events.py` 负责事件聚合（时间窗口切割 + 人脸信号增强 + LLM 摘要），`people.py` 负责人物图谱（统计 + 关系推断）。两个模块读 photos 表，写 events/event_photos/people 表。CLI 新增 `events` 和 `people` 子命令触发计算与展示。Place Memory 暂不做（数据覆盖 14% 较低，先验证 core 价值）。

**Tech Stack:** Python 3.12, SQLite, Ollama API (gemma4:e4b), Click, pytest

---

### Task 1: DB Schema v3 — events / event_photos / people 表

**Files:**
- Modify: `src/photo_memory/db.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_db.py`:

```python
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
        apple_name="唐嘉鑫",
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
    assert row["apple_name"] == "唐嘉鑫"
    assert row["photo_count"] == 671
    assert row["appearance_trend"] == "stable"
    db.close()


def test_get_done_photos_for_aggregation(tmp_db_path):
    """get_done_photos_ordered returns processed photos sorted by date."""
    db = Database(tmp_db_path)
    db.upsert_photo("p-1", date_taken="2024-09-28T13:00:00")
    db.upsert_photo("p-2", date_taken="2024-09-28T14:00:00")
    db.upsert_photo("p-3", date_taken="2024-09-27T12:00:00")
    # Mark all as done
    for uuid in ["p-1", "p-2", "p-3"]:
        db.update_photo_status(uuid, "done")

    rows = db.get_done_photos_ordered()
    assert len(rows) == 3
    assert [r["uuid"] for r in rows] == ["p-3", "p-1", "p-2"]
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py::test_schema_v3_creates_events_tables tests/test_db.py::test_upsert_event_and_link_photos tests/test_db.py::test_upsert_person tests/test_db.py::test_get_done_photos_for_aggregation -v`
Expected: FAIL — events/people tables 不存在, 方法不存在

- [ ] **Step 3: 实现 schema v3 迁移 + 方法**

修改 `src/photo_memory/db.py`:

1) 在文件顶部更新常量：
```python
CURRENT_SCHEMA_VERSION = 3
```

2) 在 `_migrate` 方法末尾（`if version < 2:` 块之后）追加 v3 迁移：
```python
        if version < 3:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    location_city TEXT,
                    location_state TEXT,
                    photo_count INTEGER,
                    face_cluster_ids TEXT,
                    summary TEXT,
                    mood TEXT,
                    cover_photo_uuid TEXT
                );
                CREATE TABLE IF NOT EXISTS event_photos (
                    event_id TEXT,
                    photo_uuid TEXT,
                    PRIMARY KEY (event_id, photo_uuid)
                );
                CREATE TABLE IF NOT EXISTS people (
                    face_cluster_id TEXT PRIMARY KEY,
                    apple_name TEXT,
                    user_name TEXT,
                    photo_count INTEGER,
                    event_count INTEGER,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    co_appearances TEXT,
                    top_locations TEXT,
                    appearance_trend TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time);
                CREATE INDEX IF NOT EXISTS idx_event_photos_photo ON event_photos(photo_uuid);
            """)
            self._set_schema_version(3)
            self.conn.commit()
            version = 3
```

3) 在 `Database` 类末尾添加方法：
```python
    def upsert_event(self, event_id: str, **kwargs):
        existing = self.conn.execute(
            "SELECT event_id FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            self.conn.execute(
                f"UPDATE events SET {sets} WHERE event_id = ?",
                (*kwargs.values(), event_id),
            )
        else:
            cols = ["event_id"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO events ({col_names}) VALUES ({placeholders})",
                (event_id, *kwargs.values()),
            )
        self.conn.commit()

    def link_photos_to_event(self, event_id: str, photo_uuids: list[str]):
        for uuid in photo_uuids:
            self.conn.execute(
                "INSERT OR IGNORE INTO event_photos (event_id, photo_uuid) VALUES (?, ?)",
                (event_id, uuid),
            )
        self.conn.commit()

    def get_event_photos(self, event_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM event_photos WHERE event_id = ?", (event_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY start_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_person(self, face_cluster_id: str, **kwargs):
        existing = self.conn.execute(
            "SELECT face_cluster_id FROM people WHERE face_cluster_id = ?",
            (face_cluster_id,),
        ).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            self.conn.execute(
                f"UPDATE people SET {sets} WHERE face_cluster_id = ?",
                (*kwargs.values(), face_cluster_id),
            )
        else:
            cols = ["face_cluster_id"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO people ({col_names}) VALUES ({placeholders})",
                (face_cluster_id, *kwargs.values()),
            )
        self.conn.commit()

    def get_all_people(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM people ORDER BY photo_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_person_user_name(self, face_cluster_id: str, user_name: str):
        self.conn.execute(
            "UPDATE people SET user_name = ? WHERE face_cluster_id = ?",
            (user_name, face_cluster_id),
        )
        self.conn.commit()

    def get_done_photos_ordered(self) -> list[dict]:
        """Get all done photos ordered by date_taken ASC for aggregation."""
        rows = self.conn.execute(
            "SELECT * FROM photos WHERE status = 'done' AND date_taken IS NOT NULL "
            "ORDER BY date_taken ASC"
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py -v`
Expected: ALL PASS（原有 10 个 + 4 个新测试 = 14 个）

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/db.py tests/test_db.py
git commit -m "feat(db): add schema v3 with events, event_photos, people tables"
```

---

### Task 2: 事件聚合核心逻辑 — 时间窗口切割

**Files:**
- Create: `src/photo_memory/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_events.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py -v`
Expected: FAIL — `photo_memory.events` 模块不存在

- [ ] **Step 3: 实现 slice_into_events**

Create `src/photo_memory/events.py`:

```python
"""Event aggregation: group photos into semantic events via time + face signals."""

import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _parse_date(s: str | None) -> datetime | None:
    """Parse ISO datetime string, return None on failure."""
    if not s:
        return None
    try:
        # Handle both naive and timezone-aware formats
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def slice_into_events(photos: list[dict], gap_minutes: int = 30) -> list[dict]:
    """Slice a time-ordered list of photos into events by time gap.

    Args:
        photos: list of photo row dicts, must be sorted by date_taken ASC.
                Photos without valid date_taken are skipped.
        gap_minutes: if adjacent photos are >gap_minutes apart, split into new event.

    Returns:
        list of event dicts with keys: photos, start_time, end_time
    """
    events = []
    current_photos = []
    current_start = None
    current_end = None

    for photo in photos:
        dt = _parse_date(photo.get("date_taken"))
        if dt is None:
            continue

        if not current_photos:
            current_photos = [photo]
            current_start = dt
            current_end = dt
            continue

        gap = dt - current_end
        if gap > timedelta(minutes=gap_minutes):
            # Close current event, start new one
            events.append({
                "photos": current_photos,
                "start_time": current_start.isoformat(),
                "end_time": current_end.isoformat(),
            })
            current_photos = [photo]
            current_start = dt
            current_end = dt
        else:
            current_photos.append(photo)
            current_end = dt

    if current_photos:
        events.append({
            "photos": current_photos,
            "start_time": current_start.isoformat(),
            "end_time": current_end.isoformat(),
        })

    return events
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py -v`
Expected: 5/5 PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/events.py tests/test_events.py
git commit -m "feat(events): time-window event slicing with configurable gap"
```

---

### Task 3: 事件元数据聚合 — 人脸 / 地点 / 封面

**Files:**
- Modify: `src/photo_memory/events.py`
- Modify: `tests/test_events.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_events.py`:

```python
from photo_memory.events import enrich_event_metadata


def test_enrich_event_aggregates_faces():
    """enrich_event_metadata should union face_cluster_ids across photos."""
    event = {
        "photos": [
            _photo("p1", "2024-09-28T13:00:00", face_ids=["fc_001", "fc_002"]),
            _photo("p2", "2024-09-28T13:10:00", face_ids=["fc_001", "fc_003"]),
        ],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:10:00",
    }
    enriched = enrich_event_metadata(event)
    assert set(enriched["face_cluster_ids"]) == {"fc_001", "fc_002", "fc_003"}
    assert enriched["photo_count"] == 2


def test_enrich_event_picks_majority_city():
    """enrich_event_metadata picks the most common location_city."""
    event = {
        "photos": [
            _photo("p1", "2024-09-28T13:00:00", city="大连市"),
            _photo("p2", "2024-09-28T13:10:00", city="大连市"),
            _photo("p3", "2024-09-28T13:20:00", city=None),
        ],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:20:00",
    }
    enriched = enrich_event_metadata(event)
    assert enriched["location_city"] == "大连市"


def test_enrich_event_city_none_when_no_data():
    event = {
        "photos": [
            _photo("p1", "2024-09-28T13:00:00", city=None),
            _photo("p2", "2024-09-28T13:10:00", city=None),
        ],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:10:00",
    }
    enriched = enrich_event_metadata(event)
    assert enriched["location_city"] is None


def test_enrich_event_picks_cover_photo():
    """Cover photo should be the first photo in the event (MVP behavior)."""
    event = {
        "photos": [
            _photo("p1", "2024-09-28T13:00:00"),
            _photo("p2", "2024-09-28T13:10:00"),
        ],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:10:00",
    }
    enriched = enrich_event_metadata(event)
    assert enriched["cover_photo_uuid"] == "p1"


def test_enrich_event_generates_event_id():
    event = {
        "photos": [_photo("p1", "2024-09-28T13:00:00")],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:00:00",
    }
    enriched = enrich_event_metadata(event)
    # event_id format: evt_YYYYMMDDHHMMSS_<first-uuid-prefix>
    assert enriched["event_id"].startswith("evt_20240928130000_")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py -v`
Expected: FAIL — enrich_event_metadata 不存在

- [ ] **Step 3: 实现 enrich_event_metadata**

Append to `src/photo_memory/events.py`:

```python
def enrich_event_metadata(event: dict) -> dict:
    """Enrich an event dict with aggregated metadata (faces, city, cover).

    Args:
        event: event dict from slice_into_events with 'photos', 'start_time', 'end_time'

    Returns:
        enriched event dict with additional keys: event_id, photo_count,
        face_cluster_ids (list), location_city, cover_photo_uuid
    """
    photos = event["photos"]

    # Union of all face cluster IDs
    face_ids = set()
    for p in photos:
        raw = p.get("face_cluster_ids")
        if raw:
            try:
                face_ids.update(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

    # Majority city (ignore None)
    from collections import Counter
    cities = [p.get("location_city") for p in photos if p.get("location_city")]
    majority_city = Counter(cities).most_common(1)[0][0] if cities else None

    # Cover photo: first photo (MVP; could be smarter later)
    cover_uuid = photos[0]["uuid"]

    # Event ID: evt_<start_compact>_<first-uuid-prefix>
    start_dt = _parse_date(event["start_time"])
    start_compact = start_dt.strftime("%Y%m%d%H%M%S") if start_dt else "unknown"
    uuid_prefix = cover_uuid[:8]
    event_id = f"evt_{start_compact}_{uuid_prefix}"

    return {
        **event,
        "event_id": event_id,
        "photo_count": len(photos),
        "face_cluster_ids": sorted(face_ids),
        "location_city": majority_city,
        "cover_photo_uuid": cover_uuid,
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py -v`
Expected: ALL PASS (10/10)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/events.py tests/test_events.py
git commit -m "feat(events): enrich events with face union, majority city, cover photo"
```

---

### Task 4: 事件摘要生成 — LLM 调用

**Files:**
- Modify: `src/photo_memory/events.py`
- Modify: `tests/test_events.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_events.py`:

```python
from unittest.mock import patch, MagicMock


def test_summarize_event_with_narratives():
    """summarize_event should call Ollama with event context and return summary."""
    from photo_memory.events import summarize_event
    event = {
        "event_id": "evt_001",
        "photos": [
            {
                "uuid": "p1",
                "date_taken": "2024-09-28T13:00:00",
                "ai_result": json.dumps({"narrative": "海边合影", "emotional_tone": "轻松"}),
            },
            {
                "uuid": "p2",
                "date_taken": "2024-09-28T13:10:00",
                "ai_result": json.dumps({"narrative": "看日落", "emotional_tone": "平静"}),
            },
        ],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:10:00",
        "location_city": "大连市",
        "face_cluster_ids": ["fc_001"],
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps({
            "summary": "和朋友在大连海边散步并看日落",
            "mood": "愉快轻松",
        })
    }

    with patch("photo_memory.events.requests.post", return_value=mock_response) as mock_post:
        result = summarize_event(event, host="http://localhost:11434",
                                 model="gemma4:e4b", timeout=60)

    assert result["summary"] == "和朋友在大连海边散步并看日落"
    assert result["mood"] == "愉快轻松"
    # Verify prompt contains narratives
    call_payload = mock_post.call_args[1]["json"]
    assert "海边合影" in call_payload["prompt"]
    assert "大连市" in call_payload["prompt"]


def test_summarize_event_fallback_on_error():
    """If LLM fails, return a fallback summary from available data."""
    from photo_memory.events import summarize_event
    event = {
        "event_id": "evt_002",
        "photos": [
            {"uuid": "p1", "date_taken": "2024-09-28T13:00:00",
             "ai_result": json.dumps({"narrative": "海边"})},
        ],
        "start_time": "2024-09-28T13:00:00",
        "end_time": "2024-09-28T13:00:00",
        "location_city": "大连市",
        "face_cluster_ids": [],
    }

    with patch("photo_memory.events.requests.post", side_effect=Exception("network down")):
        result = summarize_event(event, host="http://x", model="m", timeout=10)

    assert "summary" in result
    assert result["summary"]  # non-empty fallback
    assert "mood" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py::test_summarize_event_with_narratives tests/test_events.py::test_summarize_event_fallback_on_error -v`
Expected: FAIL

- [ ] **Step 3: 实现 summarize_event**

Append to `src/photo_memory/events.py`:

```python
import requests

EVENT_SUMMARY_PROMPT = """基于以下事件中多张照片的描述信息，生成一段简洁的事件摘要。

事件信息：
- 时间：{start_time} ~ {end_time}
- 地点：{location}
- 照片数：{photo_count}

照片描述（按时间顺序）：
{narratives}

返回严格 JSON 格式（不要其他文字）：
{{
  "summary": "一段 30-60 字的中文事件摘要，描述这是什么事件、发生了什么",
  "mood": "整体情绪基调（愉快/紧张/平静/庄重/等）"
}}"""


def summarize_event(event: dict, host: str, model: str, timeout: int) -> dict:
    """Generate a summary and mood for an event via LLM.

    Falls back to a rule-based summary if LLM call fails.
    """
    narratives = []
    for i, p in enumerate(event["photos"][:20], 1):  # cap at 20 narratives
        try:
            ai = json.loads(p.get("ai_result") or "{}")
            narr = ai.get("narrative", "")
            if narr:
                narratives.append(f"{i}. {narr}")
        except (json.JSONDecodeError, TypeError):
            pass

    narratives_text = "\n".join(narratives) if narratives else "(无详细描述)"
    location = event.get("location_city") or "未知地点"

    prompt = EVENT_SUMMARY_PROMPT.format(
        start_time=event["start_time"],
        end_time=event["end_time"],
        location=location,
        photo_count=len(event["photos"]),
        narratives=narratives_text,
    )

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        data = json.loads(raw)
        if "summary" in data and "mood" in data:
            return {"summary": data["summary"], "mood": data["mood"]}
    except Exception as e:
        logger.warning(f"Event summary LLM failed for {event.get('event_id')}: {e}")

    # Fallback: use first narrative + location
    first_narr = narratives[0][3:] if narratives else "一组照片"
    return {
        "summary": f"在{location}的一段时光，{first_narr}",
        "mood": "",
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py -v`
Expected: ALL PASS (12/12)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/events.py tests/test_events.py
git commit -m "feat(events): LLM-based event summary with rule-based fallback"
```

---

### Task 5: 事件聚合 orchestrator + CLI `events` 命令

**Files:**
- Modify: `src/photo_memory/events.py`
- Modify: `src/photo_memory/cli.py`
- Modify: `tests/test_events.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 写 orchestrator 失败测试**

Append to `tests/test_events.py`:

```python
def test_build_events_writes_to_db(tmp_path):
    """build_events orchestrator reads done photos, slices, enriches, writes to DB."""
    from photo_memory.db import Database
    from photo_memory.events import build_events

    db = Database(str(tmp_path / "test.db"))
    db.upsert_photo("p1", date_taken="2024-09-28T13:00:00",
                    face_cluster_ids='["fc_001"]', location_city="大连市",
                    ai_result=json.dumps({"narrative": "海边"}))
    db.upsert_photo("p2", date_taken="2024-09-28T13:15:00",
                    face_cluster_ids='["fc_001"]', location_city="大连市",
                    ai_result=json.dumps({"narrative": "沙滩"}))
    db.upsert_photo("p3", date_taken="2024-09-28T18:00:00",  # gap > 30min
                    face_cluster_ids='[]', location_city="大连市",
                    ai_result=json.dumps({"narrative": "晚餐"}))
    for uuid in ["p1", "p2", "p3"]:
        db.update_photo_status(uuid, "done")

    with patch("photo_memory.events.summarize_event",
               return_value={"summary": "测试摘要", "mood": "愉快"}):
        count = build_events(db, ollama_config={"host": "h", "model": "m", "timeout": 10},
                             gap_minutes=30)

    assert count == 2  # Two events due to time gap
    events = db.get_all_events()
    assert len(events) == 2
    # Verify event_photos link exists
    first_event_id = events[0]["event_id"]
    linked = db.get_event_photos(first_event_id)
    assert len(linked) >= 1
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py::test_build_events_writes_to_db -v`
Expected: FAIL — build_events 不存在

- [ ] **Step 3: 实现 build_events orchestrator**

Append to `src/photo_memory/events.py`:

```python
def build_events(db, ollama_config: dict, gap_minutes: int = 30) -> int:
    """Build events from done photos: slice, enrich, summarize, persist.

    Args:
        db: Database instance
        ollama_config: dict with 'host', 'model', 'timeout'
        gap_minutes: time gap threshold for event boundary

    Returns:
        number of events created
    """
    photos = db.get_done_photos_ordered()
    if not photos:
        logger.info("No done photos to aggregate")
        return 0

    logger.info(f"Aggregating {len(photos)} photos into events...")
    raw_events = slice_into_events(photos, gap_minutes=gap_minutes)
    logger.info(f"Sliced into {len(raw_events)} events")

    for event in raw_events:
        enriched = enrich_event_metadata(event)
        summary_data = summarize_event(
            enriched,
            host=ollama_config["host"],
            model=ollama_config["model"],
            timeout=ollama_config["timeout"],
        )

        db.upsert_event(
            event_id=enriched["event_id"],
            start_time=enriched["start_time"],
            end_time=enriched["end_time"],
            location_city=enriched.get("location_city"),
            location_state=None,
            photo_count=enriched["photo_count"],
            face_cluster_ids=json.dumps(enriched["face_cluster_ids"], ensure_ascii=False),
            summary=summary_data["summary"],
            mood=summary_data["mood"],
            cover_photo_uuid=enriched["cover_photo_uuid"],
        )

        photo_uuids = [p["uuid"] for p in enriched["photos"]]
        db.link_photos_to_event(enriched["event_id"], photo_uuids)

    return len(raw_events)
```

- [ ] **Step 4: 写 CLI 失败测试**

Append to `tests/test_cli.py`:

```python
def test_events_command_builds_and_lists(tmp_path, sample_config):
    """`phototag events` should build events and print a summary."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database
    import json

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-09-28T13:00:00",
                    face_cluster_ids='[]', ai_result=json.dumps({"narrative": "test"}))
    db.update_photo_status("p1", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.events.summarize_event",
               return_value={"summary": "测试", "mood": ""}):
        result = runner.invoke(main, ["--config", config_path, "events"])

    assert result.exit_code == 0
    assert "event" in result.output.lower() or "事件" in result.output
```

- [ ] **Step 5: 运行 CLI 测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py::test_events_command_builds_and_lists -v`
Expected: FAIL — events 命令不存在

- [ ] **Step 6: 实现 CLI events 命令**

在 `src/photo_memory/cli.py` 中添加 import：
```python
from photo_memory.events import build_events
```

在 `reprocess` 命令后添加：
```python
@main.command()
@click.option("--gap-minutes", default=30, help="Time gap (minutes) to split events")
@click.pass_context
def events(ctx, gap_minutes):
    """Build events from done photos and list them."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    count = build_events(db, ollama_config=config["ollama"], gap_minutes=gap_minutes)
    click.echo(f"Built {count} events.")

    all_events = db.get_all_events()
    click.echo(f"\n最近 10 个事件:")
    for e in all_events[:10]:
        city = e["location_city"] or "未知"
        click.echo(f"  [{e['start_time'][:10]}] {city} ({e['photo_count']} 张) — {e['summary'] or '(无摘要)'}")

    db.close()
```

- [ ] **Step 7: 运行全部相关测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_events.py tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/events.py src/photo_memory/cli.py tests/test_events.py tests/test_cli.py
git commit -m "feat(cli): add events command to build and list aggregated events"
```

---

### Task 6: 人物图谱统计 — 核心统计逻辑

**Files:**
- Create: `src/photo_memory/people.py`
- Create: `tests/test_people.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_people.py`:

```python
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

    # fc_001: 3 photos, co-appears with fc_002 (2x), fc_003 (1x)
    fc1 = next(s for s in stats if s["face_cluster_id"] == "fc_001")
    assert fc1["photo_count"] == 3
    assert fc1["first_seen"] == "2024-01-01T12:00:00"
    assert fc1["last_seen"] == "2024-03-01T12:00:00"
    assert fc1["co_appearances"]["fc_002"] == 2
    assert fc1["co_appearances"]["fc_003"] == 1

    # fc_002: 3 photos
    fc2 = next(s for s in stats if s["face_cluster_id"] == "fc_002")
    assert fc2["photo_count"] == 3
    assert fc2["co_appearances"]["fc_001"] == 2


def test_compute_person_stats_uses_apple_name():
    """When named_faces contains a name and only one face cluster, associate them."""
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
    dates = [f"2026-0{m}-01T12:00:00" for m in range(1, 5)]  # 2026-01 to 2026-04
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_people.py -v`
Expected: FAIL — photo_memory.people 模块不存在

- [ ] **Step 3: 实现 people.py**

Create `src/photo_memory/people.py`:

```python
"""People graph: aggregate per-person stats and infer relationships."""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def compute_person_stats(photos: list[dict]) -> list[dict]:
    """Aggregate per-person statistics across photos.

    Args:
        photos: list of photo row dicts (must include face_cluster_ids, date_taken,
                location_city, named_faces)

    Returns:
        list of person stat dicts ordered by photo_count DESC
    """
    # Index: face_cluster_id -> list of (photo_uuid, date, city, face_ids_in_photo, named)
    person_index = defaultdict(list)

    for photo in photos:
        raw_faces = photo.get("face_cluster_ids")
        if not raw_faces:
            continue
        try:
            face_ids = json.loads(raw_faces)
        except (json.JSONDecodeError, TypeError):
            continue
        if not face_ids:
            continue

        try:
            named = json.loads(photo.get("named_faces") or "[]")
        except (json.JSONDecodeError, TypeError):
            named = []

        for fc_id in face_ids:
            person_index[fc_id].append({
                "uuid": photo["uuid"],
                "date": photo.get("date_taken"),
                "city": photo.get("location_city"),
                "other_face_ids": [f for f in face_ids if f != fc_id],
                "named": named,
            })

    stats = []
    for fc_id, appearances in person_index.items():
        dates = [a["date"] for a in appearances if a["date"]]
        dates.sort()

        co_counter = Counter()
        for a in appearances:
            for other in a["other_face_ids"]:
                co_counter[other] += 1

        city_counter = Counter(a["city"] for a in appearances if a["city"])
        top_locations = [c for c, _ in city_counter.most_common(5)]

        # Apple name: if a single name appears in photos where this face is the only named face
        # Simple heuristic: take most common named face when exactly one cluster is present
        apple_name = None
        for a in appearances:
            if len(a["named"]) == 1 and len(appearances) > 0:
                apple_name = a["named"][0]
                break

        trend = infer_appearance_trend(dates)

        stats.append({
            "face_cluster_id": fc_id,
            "apple_name": apple_name,
            "photo_count": len(appearances),
            "first_seen": dates[0] if dates else None,
            "last_seen": dates[-1] if dates else None,
            "co_appearances": dict(co_counter),
            "top_locations": top_locations,
            "appearance_trend": trend,
        })

    stats.sort(key=lambda s: s["photo_count"], reverse=True)
    return stats


def infer_appearance_trend(dates: list[str], reference_date: str | None = None) -> str:
    """Classify appearance trend from a list of dates.

    Returns: 'increasing' | 'stable' | 'decreasing' | 'one_time'
    """
    parsed = [d for d in (_parse_date(s) for s in dates) if d]
    if len(parsed) <= 1:
        return "one_time"

    # Make all dates naive for comparison
    parsed = [d.replace(tzinfo=None) for d in parsed]
    parsed.sort()

    ref = _parse_date(reference_date) if reference_date else datetime.now()
    if ref:
        ref = ref.replace(tzinfo=None)
    else:
        ref = datetime.now()

    six_months_ago = ref - timedelta(days=180)
    recent = sum(1 for d in parsed if d >= six_months_ago)
    earlier = len(parsed) - recent

    span_days = (parsed[-1] - parsed[0]).days
    if span_days > 365 and earlier > 0:
        return "stable"
    if recent > earlier * 2 and recent >= 2:
        return "increasing"
    if earlier > recent * 2 and earlier >= 2:
        return "decreasing"
    return "stable"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_people.py -v`
Expected: ALL PASS (6/6)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/people.py tests/test_people.py
git commit -m "feat(people): compute per-person stats and appearance trend inference"
```

---

### Task 7: 人物图谱 orchestrator + 事件关联

**Files:**
- Modify: `src/photo_memory/people.py`
- Modify: `tests/test_people.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_people.py`:

```python
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
    # co_appearances stored as JSON
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

    # First build
    build_people(db)
    db.set_person_user_name("fc_001", "阿菁")

    # Add another photo and rebuild
    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='[]')
    db.update_photo_status("p2", "done")
    build_people(db)

    row = db.execute("SELECT user_name FROM people WHERE face_cluster_id = ?", ("fc_001",)).fetchone()
    assert row["user_name"] == "阿菁"
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_people.py::test_build_people_writes_to_db tests/test_people.py::test_build_people_preserves_user_name -v`
Expected: FAIL

- [ ] **Step 3: 实现 build_people**

Append to `src/photo_memory/people.py`:

```python
def build_people(db) -> int:
    """Build people graph from done photos and persist to DB.

    Preserves user_name set by the user when upserting.

    Returns:
        number of unique people processed
    """
    photos = db.get_done_photos_ordered()
    if not photos:
        return 0

    stats = compute_person_stats(photos)

    # Count events per person (requires events table populated; OK if empty)
    event_count_by_person = _count_events_per_person(db)

    for s in stats:
        fc_id = s["face_cluster_id"]
        # Preserve existing user_name
        existing = db.execute(
            "SELECT user_name FROM people WHERE face_cluster_id = ?", (fc_id,)
        ).fetchone()
        user_name = existing["user_name"] if existing else None

        db.upsert_person(
            face_cluster_id=fc_id,
            apple_name=s["apple_name"],
            user_name=user_name,
            photo_count=s["photo_count"],
            event_count=event_count_by_person.get(fc_id, 0),
            first_seen=s["first_seen"],
            last_seen=s["last_seen"],
            co_appearances=json.dumps(s["co_appearances"], ensure_ascii=False),
            top_locations=json.dumps(s["top_locations"], ensure_ascii=False),
            appearance_trend=s["appearance_trend"],
        )

    return len(stats)


def _count_events_per_person(db) -> dict[str, int]:
    """Count how many events each face_cluster_id appears in."""
    counts = defaultdict(int)
    try:
        rows = db.execute("SELECT face_cluster_ids FROM events").fetchall()
    except Exception:
        return dict(counts)

    for row in rows:
        raw = row["face_cluster_ids"]
        if not raw:
            continue
        try:
            face_ids = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for fc_id in face_ids:
            counts[fc_id] += 1
    return dict(counts)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_people.py -v`
Expected: ALL PASS (8/8)

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/people.py tests/test_people.py
git commit -m "feat(people): build_people orchestrator persisting stats to DB"
```

---

### Task 8: CLI `people` 命令 — 列表 + 命名

**Files:**
- Modify: `src/photo_memory/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_cli.py`:

```python
def test_people_command_builds_and_lists(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='["唐嘉鑫"]')
    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='["唐嘉鑫"]')
    for uuid in ["p1", "p2"]:
        db.update_photo_status(uuid, "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "people"])

    assert result.exit_code == 0
    assert "fc_001" in result.output or "唐嘉鑫" in result.output
    assert "2" in result.output  # photo count


def test_people_name_command_sets_user_name(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]', named_faces='[]')
    db.update_photo_status("p1", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        # First build people
        runner.invoke(main, ["--config", config_path, "people"])
        # Then name
        result = runner.invoke(main, ["--config", config_path, "people", "--name", "fc_001", "阿菁"])

    assert result.exit_code == 0
    db = Database(db_path)
    row = db.execute("SELECT user_name FROM people WHERE face_cluster_id = ?", ("fc_001",)).fetchone()
    assert row["user_name"] == "阿菁"
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py::test_people_command_builds_and_lists tests/test_cli.py::test_people_name_command_sets_user_name -v`
Expected: FAIL — people 命令不存在

- [ ] **Step 3: 实现 CLI people 命令**

在 `src/photo_memory/cli.py` 添加 import：
```python
from photo_memory.people import build_people
```

在 `events` 命令后添加：
```python
@main.command()
@click.option("--name", "name_args", nargs=2, type=str, default=None,
              help="Assign a user name: --name <face_cluster_id> <name>")
@click.option("--min-photos", default=10, help="Minimum photo count to display")
@click.pass_context
def people(ctx, name_args, min_photos):
    """Build and list the people graph. Use --name to assign a name."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    if name_args:
        fc_id, user_name = name_args
        db.set_person_user_name(fc_id, user_name)
        click.echo(f"Named {fc_id} as {user_name}")
        db.close()
        return

    count = build_people(db)
    click.echo(f"Processed {count} unique people.\n")

    all_people = db.get_all_people()
    click.echo("高频人物:")
    shown = 0
    for p in all_people:
        if p["photo_count"] < min_photos:
            continue
        name = p["user_name"] or p["apple_name"] or "(未命名)"
        fc_short = p["face_cluster_id"][:12]
        first = (p["first_seen"] or "")[:7]
        last = (p["last_seen"] or "")[:7]
        trend = p["appearance_trend"] or ""
        click.echo(f"  {fc_short}... {name} — {p['photo_count']} 张, {first} ~ {last} [{trend}]")
        shown += 1
        if shown >= 20:
            break

    unnamed = [p for p in all_people if not p["user_name"] and not p["apple_name"] and p["photo_count"] >= min_photos]
    if unnamed:
        click.echo(f"\n发现 {len(unnamed)} 位高频未命名人物。运行以下命令命名：")
        for p in unnamed[:3]:
            click.echo(f"  phototag people --name {p['face_cluster_id']} \"名字\"")

    db.close()
```

- [ ] **Step 4: 运行 CLI 测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 5: 运行全量回归测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/ -v --tb=short`
Expected: ALL PASS (预计 ~80+ 测试)

- [ ] **Step 6: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/cli.py tests/test_cli.py
git commit -m "feat(cli): add people command for listing and naming face clusters"
```

---

### Task 9: 真实数据验证

**Files:** 无代码改动，仅手动验证

- [ ] **Step 1: 在真实照片库上运行**

```bash
cd ~/photo-memory
.venv/bin/phototag events --gap-minutes 30
```

Expected: 打印出事件列表，每个事件包含日期、地点、照片数、摘要

- [ ] **Step 2: 人物图谱验证**

```bash
.venv/bin/phototag people --min-photos 20
```

Expected: 列出高频人物，至少包含 "唐嘉鑫"（671 张），显示未命名人物提示

- [ ] **Step 3: 命名测试**

找一个未命名的 fc_id，执行：
```bash
.venv/bin/phototag people --name <fc_id> "测试名"
.venv/bin/phototag people --min-photos 20
```

Expected: 该 fc_id 的 name 显示为"测试名"，重新运行 `people` 不会覆盖它

- [ ] **Step 4: Spot-check 事件质量**

```bash
sqlite3 ~/.phototag/progress.db "SELECT event_id, start_time, location_city, photo_count, summary FROM events ORDER BY photo_count DESC LIMIT 10;"
```

手动检查 10 个大事件：
- 边界是否合理（不把两件事混为一个）
- summary 是否有意义
- location_city 是否正确

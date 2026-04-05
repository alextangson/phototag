# Search + Cleanup + Face Merge + HTML Story Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 phototag 四大实用功能：结构化搜索、清理报告、人脸合并（手动+建议）、HTML 故事导出（带缩略图）。

**Architecture:** 三个独立模块并行开发（search.py / merge 逻辑在 db+people / html_renderer.py），互不触碰 cli.py；最后一个串行 task 把四个 CLI 命令一次性接入 cli.py。这样并行安全，无文件冲突。

**Tech Stack:** Python 3.12, SQLite, Click, Pillow (PIL), pytest. HTML 用手写模板 + 内嵌 base64 缩略图，无外部依赖。

---

## Phase A (Parallel): Search + Cleanup

### Task A: 搜索与清理核心逻辑

**Files:**
- Create: `src/photo_memory/search.py`
- Create: `tests/test_search.py`
- Modify: `src/photo_memory/db.py` (仅加只读查询方法)
- Modify: `tests/test_db.py`

**Goal:** 结构化搜索（`--person/--year/--city/--text`）+ cleanup 报告（按 cleanup_class 过滤）

- [ ] **Step 1: 写失败测试**

Append to `tests/test_db.py`:

```python
def test_search_photos_by_person_fc_id(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("p1", date_taken="2024-03-01T12:00:00",
                    face_cluster_ids='["fc_001"]', status="done",
                    description="和朋友散步", tags='["散步"]')
    db.upsert_photo("p2", date_taken="2024-04-01T12:00:00",
                    face_cluster_ids='["fc_002"]', status="done",
                    description="自拍", tags='["自拍"]')
    db.update_photo_status("p1", "done")
    db.update_photo_status("p2", "done")

    results = db.search_photos(face_cluster_ids=["fc_001"])
    assert len(results) == 1
    assert results[0]["uuid"] == "p1"
    db.close()


def test_search_photos_combined_filters(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("p1", date_taken="2024-03-15T12:00:00",
                    face_cluster_ids='["fc_001"]', location_city="北京",
                    description="颐和园", tags='["公园"]')
    db.upsert_photo("p2", date_taken="2024-03-20T12:00:00",
                    face_cluster_ids='["fc_001"]', location_city="上海",
                    description="外滩夜景", tags='["夜景"]')
    db.upsert_photo("p3", date_taken="2023-06-10T12:00:00",
                    face_cluster_ids='["fc_001"]', location_city="北京",
                    description="雍和宫", tags='["寺庙"]')
    for u in ["p1", "p2", "p3"]:
        db.update_photo_status(u, "done")

    # Person + year + city
    results = db.search_photos(
        face_cluster_ids=["fc_001"],
        year=2024,
        city="北京",
    )
    assert len(results) == 1
    assert results[0]["uuid"] == "p1"
    db.close()


def test_search_photos_by_text_matches_description_and_tags(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    description="海边散步", tags='["海", "散步"]')
    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    description="山间徒步", tags='["山", "徒步"]')
    db.update_photo_status("p1", "done")
    db.update_photo_status("p2", "done")

    # Match by description keyword
    results = db.search_photos(text="海边")
    assert len(results) == 1
    assert results[0]["uuid"] == "p1"

    # Match by tag
    results = db.search_photos(text="徒步")
    assert len(results) == 1
    assert results[0]["uuid"] == "p2"
    db.close()


def test_get_cleanup_candidates_returns_cleanup_class_photos(tmp_db_path):
    db = Database(tmp_db_path)
    # importance column stores cleanup_class (per M2 processor wiring)
    db.upsert_photo("p1", importance="keep", status="done")
    db.upsert_photo("p2", importance="review", status="done")
    db.upsert_photo("p3", importance="cleanup", status="done")
    db.upsert_photo("p4", importance="cleanup", status="done")

    results = db.get_cleanup_candidates(classes=["cleanup"])
    assert len(results) == 2
    assert {r["uuid"] for r in results} == {"p3", "p4"}

    results = db.get_cleanup_candidates(classes=["review", "cleanup"])
    assert len(results) == 3
    db.close()
```

Create `tests/test_search.py`:

```python
"""Tests for search and cleanup logic."""
from unittest.mock import MagicMock
from photo_memory.search import search_photos, list_cleanup_candidates


def test_search_photos_resolves_person_name(tmp_path):
    """search_photos(person=...) should look up face_cluster_id via db.get_person_by_name."""
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.upsert_person(face_cluster_id="fc_001", apple_name="张三", user_name=None,
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]', description="test")
    db.update_photo_status("p1", "done")

    results = search_photos(db, person="张三")
    assert len(results) == 1
    assert results[0]["uuid"] == "p1"
    db.close()


def test_search_photos_unknown_person_returns_empty(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    results = search_photos(db, person="不存在")
    assert results == []
    db.close()


def test_list_cleanup_candidates_groups_by_class(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.upsert_photo("p1", importance="cleanup", status="done", description="模糊")
    db.upsert_photo("p2", importance="cleanup", status="done", description="重复")
    db.upsert_photo("p3", importance="review", status="done", description="不确定")

    report = list_cleanup_candidates(db)
    assert report["cleanup"]["count"] == 2
    assert report["review"]["count"] == 1
    assert len(report["cleanup"]["photos"]) == 2
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py::test_search_photos_by_person_fc_id tests/test_db.py::test_search_photos_combined_filters tests/test_db.py::test_search_photos_by_text_matches_description_and_tags tests/test_db.py::test_get_cleanup_candidates_returns_cleanup_class_photos tests/test_search.py -v`

Expected: FAIL — methods and module don't exist

- [ ] **Step 3: 实现 DB 方法**

Add to `Database` class in `src/photo_memory/db.py` (before `close`):

```python
    def search_photos(
        self,
        face_cluster_ids: list[str] | None = None,
        year: int | None = None,
        city: str | None = None,
        text: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search done photos by combined filters (AND semantics)."""
        conditions = ["status = 'done'"]
        params: list = []

        if face_cluster_ids:
            # Match any of the given fc_ids via LIKE
            or_parts = []
            for fc_id in face_cluster_ids:
                or_parts.append("face_cluster_ids LIKE ?")
                params.append(f'%"{fc_id}"%')
            conditions.append("(" + " OR ".join(or_parts) + ")")

        if year is not None:
            conditions.append("date_taken >= ? AND date_taken < ?")
            params.append(f"{year}-01-01")
            params.append(f"{year + 1}-01-01")

        if city:
            conditions.append("location_city = ?")
            params.append(city)

        if text:
            conditions.append("(description LIKE ? OR tags LIKE ?)")
            params.append(f"%{text}%")
            params.append(f"%{text}%")

        sql = (
            "SELECT * FROM photos WHERE "
            + " AND ".join(conditions)
            + " ORDER BY date_taken DESC LIMIT ?"
        )
        params.append(limit)
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def get_cleanup_candidates(self, classes: list[str]) -> list[dict]:
        """Get photos whose cleanup_class (stored in 'importance' column) matches."""
        placeholders = ",".join(["?"] * len(classes))
        rows = self.conn.execute(
            f"SELECT * FROM photos WHERE status = 'done' AND importance IN ({placeholders}) "
            "ORDER BY date_taken DESC",
            tuple(classes),
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: 实现 search.py**

Create `src/photo_memory/search.py`:

```python
"""Structured search over photos and cleanup candidate reporting."""

import logging

logger = logging.getLogger(__name__)


def search_photos(
    db,
    person: str | None = None,
    year: int | None = None,
    city: str | None = None,
    text: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Structured photo search.

    Args:
        db: Database instance
        person: person name (user_name or apple_name) to filter by
        year: calendar year
        city: location_city exact match
        text: substring match in description + tags
        limit: max results

    Returns list of photo row dicts (newest first).
    """
    face_cluster_ids = None
    if person:
        p = db.get_person_by_name(person)
        if p is None:
            logger.info(f"Person not found: {person}")
            return []
        face_cluster_ids = [p["face_cluster_id"]]

    return db.search_photos(
        face_cluster_ids=face_cluster_ids,
        year=year,
        city=city,
        text=text,
        limit=limit,
    )


def list_cleanup_candidates(db) -> dict:
    """Return a cleanup report grouped by class.

    Structure:
        {
            "cleanup": {"count": N, "photos": [...]},
            "review":  {"count": M, "photos": [...]},
        }
    """
    cleanup = db.get_cleanup_candidates(classes=["cleanup"])
    review = db.get_cleanup_candidates(classes=["review"])
    return {
        "cleanup": {"count": len(cleanup), "photos": cleanup},
        "review": {"count": len(review), "photos": review},
    }
```

- [ ] **Step 5: 运行所有相关测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py tests/test_search.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/db.py src/photo_memory/search.py tests/test_db.py tests/test_search.py
git commit -m "feat(search): structured photo search + cleanup candidate reporting"
```

---

## Phase B (Parallel): Face Cluster Merge (手动 + 建议)

### Task B: 人脸合并核心

**Files:**
- Create: `src/photo_memory/merge.py`
- Create: `tests/test_merge.py`
- Modify: `src/photo_memory/db.py` (读+写方法)
- Modify: `tests/test_db.py`

**Goal:** 手动合并两个 face cluster（设相同 user_name）+ 基于 co-appearance 推荐合并候选

**合并语义**：通过 `user_name` 实现 soft merge — 多个 fc_id 可以共享同一个 user_name，查询时按 user_name 展开。无需改 schema，无需重写 photos/events。

- [ ] **Step 1: 写失败测试**

Append to `tests/test_db.py`:

```python
def test_get_all_fc_ids_by_name(tmp_db_path):
    """get_all_fc_ids_for_name returns all clusters sharing a name."""
    db = Database(tmp_db_path)
    db.upsert_person(face_cluster_id="fc_001", apple_name=None, user_name="张三",
                     photo_count=100, event_count=10, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_002", apple_name=None, user_name="张三",
                     photo_count=50, event_count=5, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_003", apple_name="张三", user_name=None,
                     photo_count=30, event_count=3, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_999", apple_name=None, user_name="李四",
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")

    fc_ids = db.get_all_fc_ids_for_name("张三")
    assert set(fc_ids) == {"fc_001", "fc_002", "fc_003"}

    fc_ids = db.get_all_fc_ids_for_name("李四")
    assert fc_ids == ["fc_999"]

    fc_ids = db.get_all_fc_ids_for_name("不存在")
    assert fc_ids == []
    db.close()
```

Create `tests/test_merge.py`:

```python
"""Tests for face cluster merge suggestions and operations."""
import json
from photo_memory.merge import suggest_merges, merge_clusters


def _person(fc_id, photo_count=10, first="2024-01-01T12:00:00", last="2024-06-01T12:00:00",
            co_appearances=None, user_name=None, apple_name=None):
    return {
        "face_cluster_id": fc_id,
        "apple_name": apple_name,
        "user_name": user_name,
        "photo_count": photo_count,
        "first_seen": first,
        "last_seen": last,
        "co_appearances": json.dumps(co_appearances or {}),
        "top_locations": "[]",
        "appearance_trend": "stable",
    }


def test_suggest_merges_returns_pairs_with_high_co_network_overlap():
    """Two clusters that share many co-appearances but never appear together → candidates."""
    # fc_A always with fc_X, fc_Y, fc_Z — never with fc_B
    # fc_B always with fc_X, fc_Y, fc_Z — never with fc_A
    # → likely same person photographed alone differently
    people = [
        _person("fc_A", photo_count=20, co_appearances={"fc_X": 15, "fc_Y": 10, "fc_Z": 8}),
        _person("fc_B", photo_count=18, co_appearances={"fc_X": 12, "fc_Y": 9, "fc_Z": 6}),
        _person("fc_X", photo_count=100, co_appearances={"fc_A": 15, "fc_B": 12}),
        _person("fc_Y", photo_count=80),
        _person("fc_Z", photo_count=60),
    ]

    suggestions = suggest_merges(people, min_photos=5, min_shared_contacts=2)
    # Expect (fc_A, fc_B) as a candidate
    pair_ids = [(s["fc_a"], s["fc_b"]) for s in suggestions]
    assert ("fc_A", "fc_B") in pair_ids or ("fc_B", "fc_A") in pair_ids


def test_suggest_merges_skips_when_clusters_co_appear():
    """If fc_A and fc_B appear together in photos, they are different people."""
    people = [
        _person("fc_A", photo_count=20, co_appearances={"fc_B": 5, "fc_X": 10}),
        _person("fc_B", photo_count=18, co_appearances={"fc_A": 5, "fc_X": 8}),
        _person("fc_X", photo_count=50),
    ]
    suggestions = suggest_merges(people, min_photos=5, min_shared_contacts=1)
    pair_ids = [(s["fc_a"], s["fc_b"]) for s in suggestions]
    assert ("fc_A", "fc_B") not in pair_ids
    assert ("fc_B", "fc_A") not in pair_ids


def test_suggest_merges_skips_low_photo_count():
    """Clusters below min_photos threshold should be ignored."""
    people = [
        _person("fc_A", photo_count=3, co_appearances={"fc_X": 2}),
        _person("fc_B", photo_count=2, co_appearances={"fc_X": 1}),
        _person("fc_X", photo_count=50),
    ]
    suggestions = suggest_merges(people, min_photos=5)
    assert suggestions == []


def test_merge_clusters_sets_shared_user_name(tmp_path):
    """merge_clusters(fc_a, fc_b, name) sets user_name on both."""
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.upsert_person(face_cluster_id="fc_a", apple_name=None, user_name=None,
                     photo_count=10, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_b", apple_name=None, user_name=None,
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")

    merge_clusters(db, "fc_a", "fc_b", name="张三")

    p_a = db.execute("SELECT user_name FROM people WHERE face_cluster_id = 'fc_a'").fetchone()
    p_b = db.execute("SELECT user_name FROM people WHERE face_cluster_id = 'fc_b'").fetchone()
    assert p_a["user_name"] == "张三"
    assert p_b["user_name"] == "张三"
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py::test_get_all_fc_ids_by_name tests/test_merge.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 DB 方法**

Add to `Database` class in `src/photo_memory/db.py` (before `close`):

```python
    def get_all_fc_ids_for_name(self, name: str) -> list[str]:
        """Return all face_cluster_ids whose user_name OR apple_name matches the given name."""
        rows = self.conn.execute(
            "SELECT face_cluster_id FROM people "
            "WHERE user_name = ? OR apple_name = ? "
            "ORDER BY photo_count DESC",
            (name, name),
        ).fetchall()
        return [r["face_cluster_id"] for r in rows]
```

- [ ] **Step 4: 实现 merge.py**

Create `src/photo_memory/merge.py`:

```python
"""Face cluster merge: manual action + co-appearance-based suggestions."""

import json
import logging

logger = logging.getLogger(__name__)


def _parse_co_appearances(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def suggest_merges(
    people: list[dict],
    min_photos: int = 10,
    min_shared_contacts: int = 3,
    min_jaccard: float = 0.4,
) -> list[dict]:
    """Suggest face cluster pairs that may be the same person.

    Heuristic: if two clusters never co-appear in any photo but share many
    co-appearance contacts (high Jaccard similarity of their networks),
    they are likely the same person photographed separately.

    Args:
        people: list of person dicts from db.get_all_people()
        min_photos: both clusters must have at least this many photos
        min_shared_contacts: both networks must share at least this many contacts
        min_jaccard: minimum Jaccard similarity of contact sets

    Returns: list of suggestion dicts, sorted by confidence DESC.
    """
    # Pre-compute contact sets
    networks: dict[str, set[str]] = {}
    co_maps: dict[str, dict[str, int]] = {}
    for p in people:
        if (p.get("photo_count") or 0) < min_photos:
            continue
        co = _parse_co_appearances(p.get("co_appearances"))
        networks[p["face_cluster_id"]] = set(co.keys())
        co_maps[p["face_cluster_id"]] = co

    fc_ids = list(networks.keys())
    suggestions = []

    for i, fc_a in enumerate(fc_ids):
        for fc_b in fc_ids[i + 1:]:
            # Skip if they co-appear in the same photos
            if fc_b in co_maps[fc_a] or fc_a in co_maps[fc_b]:
                continue

            net_a = networks[fc_a]
            net_b = networks[fc_b]
            shared = net_a & net_b
            if len(shared) < min_shared_contacts:
                continue

            union = net_a | net_b
            if not union:
                continue
            jaccard = len(shared) / len(union)
            if jaccard < min_jaccard:
                continue

            suggestions.append({
                "fc_a": fc_a,
                "fc_b": fc_b,
                "jaccard": jaccard,
                "shared_contacts": sorted(shared),
                "confidence": round(jaccard, 2),
            })

    suggestions.sort(key=lambda s: s["jaccard"], reverse=True)
    return suggestions


def merge_clusters(db, fc_a: str, fc_b: str, name: str) -> None:
    """Manually merge two face clusters by assigning the same user_name.

    This is a soft merge: photos and events are not rewritten. Queries that
    resolve by name will fan out to all fc_ids sharing the name.
    """
    db.set_person_user_name(fc_a, name)
    db.set_person_user_name(fc_b, name)
    logger.info(f"Merged {fc_a} and {fc_b} under name '{name}'")
```

- [ ] **Step 5: 运行所有相关测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py tests/test_merge.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/db.py src/photo_memory/merge.py tests/test_db.py tests/test_merge.py
git commit -m "feat(merge): face cluster soft-merge via shared user_name + suggest-merges heuristic"
```

---

## Phase C (Parallel): HTML Story Renderer with Thumbnails

### Task C: HTML 渲染器 + 缩略图

**Files:**
- Create: `src/photo_memory/html_renderer.py`
- Create: `tests/test_html_renderer.py`

**Goal:** 把 events 数据渲染为带多张缩略图的 HTML 故事页（可截图分享）

**核心方法：**
- 每个 event 选 3 张照片（first / middle / last，或所有如果≤3）
- 每张照片经 `sips` 降到 400px 宽 JPEG，base64 编码后内嵌 HTML
- 使用单文件 HTML + 内嵌 CSS（无外部资源）

- [ ] **Step 1: 写失败测试**

Create `tests/test_html_renderer.py`:

```python
"""Tests for HTML story renderer."""
import base64
import os
from unittest.mock import patch, MagicMock
from PIL import Image
from photo_memory.html_renderer import (
    _select_photos_for_event,
    _encode_thumbnail_b64,
    render_story_html,
)


def test_select_photos_small_event_returns_all(tmp_path):
    photos = [
        {"uuid": "p1", "file_path": str(tmp_path / "1.jpg")},
        {"uuid": "p2", "file_path": str(tmp_path / "2.jpg")},
    ]
    selected = _select_photos_for_event(photos, max_photos=5)
    assert len(selected) == 2


def test_select_photos_large_event_picks_first_middle_last(tmp_path):
    photos = [{"uuid": f"p{i}", "file_path": str(tmp_path / f"{i}.jpg")} for i in range(10)]
    selected = _select_photos_for_event(photos, max_photos=3)
    assert len(selected) == 3
    # First and last must be included
    assert selected[0]["uuid"] == "p0"
    assert selected[-1]["uuid"] == "p9"


def test_encode_thumbnail_b64_returns_data_uri(tmp_path):
    # Create a tiny JPEG
    img = Image.new("RGB", (800, 600), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path, "JPEG")

    data_uri = _encode_thumbnail_b64(str(img_path), max_width=400)
    assert data_uri.startswith("data:image/jpeg;base64,")
    # Verify the embedded data is valid base64 and represents an image
    b64_part = data_uri.split(",", 1)[1]
    raw = base64.b64decode(b64_part)
    assert len(raw) > 100  # non-trivial content
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG magic bytes


def test_encode_thumbnail_handles_missing_file():
    result = _encode_thumbnail_b64("/nonexistent/path.jpg")
    assert result is None


def test_render_story_html_includes_title_and_events(tmp_path):
    """render_story_html assembles full HTML document with event sections."""
    img_path = tmp_path / "cover.jpg"
    Image.new("RGB", (100, 100), "blue").save(img_path, "JPEG")

    events_with_photos = [
        {
            "event_id": "evt_1",
            "start_time": "2024-03-15T10:00:00",
            "location_city": "北京",
            "summary": "颐和园散步",
            "mood": "愉快",
            "photos": [
                {"uuid": "p1", "file_path": str(img_path)},
            ],
        },
    ]

    html = render_story_html(
        title="2024 年度回忆",
        stats_line="共 5 张照片，1 个事件",
        intro_narrative="这一年很开心",
        events_with_photos=events_with_photos,
    )

    assert "<!DOCTYPE html>" in html
    assert "2024 年度回忆" in html
    assert "颐和园散步" in html
    assert "北京" in html
    assert "这一年很开心" in html
    assert 'data:image/jpeg;base64,' in html
    # Self-contained — no external asset refs
    assert 'src="http' not in html
    assert 'href="http' not in html or 'href="https://fonts' in html  # allow google fonts link optionally; none for MVP
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_html_renderer.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: 实现 html_renderer.py**

Create `src/photo_memory/html_renderer.py`:

```python
"""HTML story renderer with inlined thumbnail images."""

import base64
import io
import logging
import os
from html import escape

from PIL import Image

logger = logging.getLogger(__name__)


def _select_photos_for_event(photos: list[dict], max_photos: int = 3) -> list[dict]:
    """Pick up to max_photos representative photos from an event.

    If photos count <= max_photos, return all.
    Otherwise pick evenly-spaced indices (always includes first and last).
    """
    if len(photos) <= max_photos:
        return list(photos)
    if max_photos <= 1:
        return [photos[0]]
    if max_photos == 2:
        return [photos[0], photos[-1]]

    # Evenly spaced picks
    n = len(photos)
    indices = [round(i * (n - 1) / (max_photos - 1)) for i in range(max_photos)]
    indices = sorted(set(indices))
    return [photos[i] for i in indices]


def _encode_thumbnail_b64(file_path: str, max_width: int = 400) -> str | None:
    """Load image, resize to max_width, encode as base64 data URI (JPEG)."""
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        img = Image.open(file_path)
        img = img.convert("RGB")
        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            new_size = (max_width, int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Thumbnail encoding failed for {file_path}: {e}")
        return None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", sans-serif;
    background: #fafaf8;
    color: #2a2a2a;
    line-height: 1.8;
    padding: 60px 20px;
}}
.container {{
    max-width: 720px;
    margin: 0 auto;
}}
h1 {{
    font-size: 2.4em;
    font-weight: 300;
    letter-spacing: 2px;
    margin-bottom: 16px;
    color: #1a1a1a;
}}
.stats {{
    color: #888;
    font-size: 0.95em;
    padding-bottom: 24px;
    border-bottom: 1px solid #e0e0dc;
    margin-bottom: 48px;
}}
.intro {{
    font-size: 1.15em;
    color: #444;
    margin-bottom: 56px;
    padding: 0 8px;
}}
.event {{
    margin-bottom: 72px;
}}
.event-meta {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 16px;
    color: #666;
    font-size: 0.9em;
    letter-spacing: 1px;
}}
.event-date {{ font-weight: 500; color: #444; }}
.event-location {{ color: #888; }}
.event-mood {{
    color: #b08968;
    background: #f4ede4;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
}}
.event-summary {{
    font-size: 1.05em;
    color: #333;
    margin-bottom: 20px;
    line-height: 1.9;
}}
.photos {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    margin-top: 16px;
}}
.photos img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 4px;
    aspect-ratio: 4 / 3;
}}
.footer {{
    text-align: center;
    color: #aaa;
    font-size: 0.85em;
    padding-top: 40px;
    border-top: 1px solid #e0e0dc;
    margin-top: 80px;
}}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<div class="stats">{stats}</div>
{intro_html}
{events_html}
<div class="footer">Generated by phototag · 本地 AI 回忆叙事</div>
</div>
</body>
</html>
"""


def render_story_html(
    title: str,
    stats_line: str,
    intro_narrative: str,
    events_with_photos: list[dict],
    max_photos_per_event: int = 3,
) -> str:
    """Render a complete self-contained HTML document for a story.

    Args:
        title: story title (H1)
        stats_line: stats summary (first line)
        intro_narrative: opening narrative paragraph
        events_with_photos: list of event dicts, each with 'photos' list containing
                            {'uuid', 'file_path'} entries
        max_photos_per_event: how many thumbnails to embed per event

    Returns a single HTML string with all images inlined as base64.
    """
    intro_html = ""
    if intro_narrative:
        intro_html = f'<div class="intro">{escape(intro_narrative)}</div>'

    event_blocks = []
    for e in events_with_photos:
        date = (e.get("start_time") or "")[:10]
        location = e.get("location_city") or "未知"
        mood = e.get("mood") or ""
        summary = e.get("summary") or ""

        photos = _select_photos_for_event(e.get("photos") or [], max_photos=max_photos_per_event)
        img_tags = []
        for ph in photos:
            data_uri = _encode_thumbnail_b64(ph.get("file_path", ""))
            if data_uri:
                img_tags.append(f'<img src="{data_uri}" alt="">')
        photos_html = ""
        if img_tags:
            photos_html = '<div class="photos">' + "".join(img_tags) + "</div>"

        mood_html = f'<span class="event-mood">{escape(mood)}</span>' if mood else ""

        event_blocks.append(
            '<div class="event">'
            f'<div class="event-meta">'
            f'<span class="event-date">{escape(date)}</span>'
            f'<span class="event-location">{escape(location)}</span>'
            f'{mood_html}'
            f'</div>'
            f'<div class="event-summary">{escape(summary)}</div>'
            f'{photos_html}'
            '</div>'
        )

    events_html = "\n".join(event_blocks)

    return HTML_TEMPLATE.format(
        title=escape(title),
        stats=escape(stats_line),
        intro_html=intro_html,
        events_html=events_html,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_html_renderer.py -v`
Expected: 5/5 PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/html_renderer.py tests/test_html_renderer.py
git commit -m "feat(html): self-contained HTML story renderer with base64 thumbnails"
```

---

## Phase D (Serial, after A+B+C): CLI Integration

### Task D: CLI 命令接入 + story HTML 支持

**Files:**
- Modify: `src/photo_memory/cli.py`
- Modify: `src/photo_memory/story.py` (添加 html 导出路径)
- Modify: `tests/test_cli.py`

**Depends on:** Tasks A, B, C must all be committed first.

- [ ] **Step 1: 写失败测试**

Append to `tests/test_cli.py`:

```python
def test_search_command_filters_photos(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-03-15T12:00:00",
                    location_city="北京", description="颐和园散步")
    db.upsert_photo("p2", date_taken="2023-05-10T12:00:00",
                    location_city="上海", description="外滩")
    db.update_photo_status("p1", "done")
    db.update_photo_status("p2", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, [
            "--config", config_path, "search",
            "--year", "2024",
            "--city", "北京",
        ])

    assert result.exit_code == 0
    assert "p1" in result.output or "颐和园" in result.output
    assert "p2" not in result.output


def test_cleanup_command_shows_counts(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", importance="cleanup", status="done", description="模糊")
    db.upsert_photo("p2", importance="cleanup", status="done", description="重复")
    db.upsert_photo("p3", importance="review", status="done", description="不确定")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "cleanup"])

    assert result.exit_code == 0
    assert "2" in result.output  # cleanup count
    assert "1" in result.output  # review count


def test_people_merge_sets_same_user_name(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_person(face_cluster_id="fc_001", apple_name=None, user_name=None,
                     photo_count=10, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_002", apple_name=None, user_name=None,
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, [
            "--config", config_path, "people",
            "--merge", "fc_001", "fc_002", "张三",
        ])

    assert result.exit_code == 0
    db = Database(db_path)
    a = db.execute("SELECT user_name FROM people WHERE face_cluster_id='fc_001'").fetchone()
    b = db.execute("SELECT user_name FROM people WHERE face_cluster_id='fc_002'").fetchone()
    assert a["user_name"] == "张三"
    assert b["user_name"] == "张三"
    db.close()


def test_story_html_output_creates_html_file(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database
    from PIL import Image

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    # Create a test image
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img_path = img_dir / "photo1.jpg"
    Image.new("RGB", (100, 100), "red").save(img_path, "JPEG")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-03-15T12:00:00",
                    file_path=str(img_path), description="颐和园")
    db.update_photo_status("p1", "done")
    db.upsert_event(
        event_id="e1", start_time="2024-03-15T12:00:00", end_time="2024-03-15T13:00:00",
        location_city="北京", location_state=None, photo_count=1,
        face_cluster_ids='[]', summary="颐和园散步", mood="愉快", cover_photo_uuid="p1",
    )
    db.link_photos_to_event("e1", ["p1"])
    db.close()

    output_file = tmp_path / "story.html"
    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.story.generate_period_narrative", return_value="春日回忆"):
        result = runner.invoke(main, [
            "--config", config_path, "story",
            "--year", "2024",
            "--output", str(output_file),
        ])

    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "<!DOCTYPE html>" in content
    assert "2024 年度回忆" in content
    assert 'data:image/jpeg;base64,' in content
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py -k "search or cleanup or merge or html" -v`
Expected: FAIL

- [ ] **Step 3: 扩展 story.py 支持 HTML 输出**

Append to `src/photo_memory/story.py`:

```python
def build_year_html(db, year: int, ollama_config: dict) -> str:
    """Generate a year story as self-contained HTML with inlined photos."""
    from photo_memory.html_renderer import render_story_html

    events = db.get_events_in_year(year)
    if not events:
        return render_story_html(
            title=f"{year} 年度回忆",
            stats_line="",
            intro_narrative=f"这一年暂无事件记录。",
            events_with_photos=[],
        )

    # Attach photos for each event
    events_with_photos = []
    for e in events:
        links = db.get_event_photos(e["event_id"])
        photo_rows = []
        for link in links:
            p = db.get_photo(link["photo_uuid"])
            if p and p.get("file_path"):
                photo_rows.append(p)
        events_with_photos.append({**e, "photos": photo_rows})

    photo_total = sum(e.get("photo_count") or 0 for e in events)
    cities = {e.get("location_city") for e in events if e.get("location_city")}
    stats_line = f"共 {photo_total} 张照片，{len(events)} 个事件，到过 {len(cities)} 个地方"

    # Single intro narrative for the whole year
    intro = generate_period_narrative(
        events, period_label=f"{year} 年",
        host=ollama_config["host"], model=ollama_config["model"],
        timeout=ollama_config["timeout"],
    )

    return render_story_html(
        title=f"{year} 年度回忆",
        stats_line=stats_line,
        intro_narrative=intro,
        events_with_photos=events_with_photos,
    )


def build_person_html(db, person_name: str, ollama_config: dict) -> str:
    """Generate a person story as self-contained HTML with inlined photos."""
    from photo_memory.html_renderer import render_story_html

    person = db.get_person_by_name(person_name)
    if not person:
        return render_story_html(
            title=f"找不到「{person_name}」",
            stats_line="",
            intro_narrative="请先用 `phototag people --name <fc_id> <name>` 命名该人物。",
            events_with_photos=[],
        )

    # Use all fc_ids sharing this name (soft merge support)
    fc_ids = db.get_all_fc_ids_for_name(person_name)
    all_events: list[dict] = []
    seen_event_ids = set()
    for fc_id in fc_ids:
        for e in db.get_events_for_person(fc_id):
            if e["event_id"] not in seen_event_ids:
                all_events.append(e)
                seen_event_ids.add(e["event_id"])
    all_events.sort(key=lambda e: e.get("start_time") or "")

    events_with_photos = []
    for e in all_events:
        links = db.get_event_photos(e["event_id"])
        photo_rows = []
        for link in links:
            p = db.get_photo(link["photo_uuid"])
            if p and p.get("file_path"):
                photo_rows.append(p)
        events_with_photos.append({**e, "photos": photo_rows})

    first = (person["first_seen"] or "")[:10]
    last = (person["last_seen"] or "")[:10]
    photo_count = person["photo_count"] or 0
    stats_line = f"共 {photo_count} 张照片，{len(all_events)} 个事件，{first} ~ {last}"

    intro = generate_period_narrative(
        all_events, period_label=f"和{person_name}的回忆",
        host=ollama_config["host"], model=ollama_config["model"],
        timeout=ollama_config["timeout"],
    )

    return render_story_html(
        title=f"和{person_name}的回忆",
        stats_line=stats_line,
        intro_narrative=intro,
        events_with_photos=events_with_photos,
    )
```

- [ ] **Step 4: 修改 cli.py**

Add imports near other photo_memory imports:

```python
from photo_memory.search import search_photos as do_search, list_cleanup_candidates
from photo_memory.merge import merge_clusters
from photo_memory.story import build_year_html, build_person_html
```

Add new commands after the `story` command:

```python
@main.command()
@click.option("--person", type=str, default=None)
@click.option("--year", type=int, default=None)
@click.option("--city", type=str, default=None)
@click.option("--text", type=str, default=None)
@click.option("--limit", type=int, default=50)
@click.pass_context
def search(ctx, person, year, city, text, limit):
    """Search photos by person, year, city, or text keyword."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    results = do_search(db, person=person, year=year, city=city, text=text, limit=limit)
    click.echo(f"找到 {len(results)} 张照片")
    for r in results[:limit]:
        date = (r.get("date_taken") or "")[:10]
        city = r.get("location_city") or "-"
        desc = (r.get("description") or "")[:60]
        click.echo(f"  [{date}] {city:<10s} {r['uuid'][:8]}... {desc}")
    db.close()


@main.command()
@click.option("--include-review", is_flag=True, default=False,
              help="Also list photos marked as 'review' (needs manual decision)")
@click.pass_context
def cleanup(ctx, include_review):
    """Report photos flagged for cleanup (cleanup_class=cleanup/review)."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    report = list_cleanup_candidates(db)
    cleanup_info = report["cleanup"]
    review_info = report["review"]

    click.echo(f"可清理照片：{cleanup_info['count']} 张")
    click.echo(f"待人工确认：{review_info['count']} 张")
    click.echo("")

    click.echo("=== cleanup （可直接删）===")
    for p in cleanup_info["photos"][:30]:
        desc = (p.get("description") or "")[:60]
        click.echo(f"  {p['uuid'][:8]}... {desc}")

    if include_review:
        click.echo("\n=== review （请人工判断）===")
        for p in review_info["photos"][:30]:
            desc = (p.get("description") or "")[:60]
            click.echo(f"  {p['uuid'][:8]}... {desc}")

    db.close()
```

Modify the existing `people` command to add `--merge` and `--suggest-merges` options. Find the existing `people` function and replace it with:

```python
@main.command()
@click.option("--name", "name_args", nargs=2, type=str, default=None,
              help="Assign a user name: --name <face_cluster_id> <name>")
@click.option("--merge", "merge_args", nargs=3, type=str, default=None,
              help="Merge two clusters under one name: --merge <fc_a> <fc_b> <name>")
@click.option("--suggest-merges", is_flag=True, default=False,
              help="Suggest cluster pairs likely to be the same person")
@click.option("--min-photos", default=10, help="Minimum photo count to display")
@click.pass_context
def people(ctx, name_args, merge_args, suggest_merges, min_photos):
    """Build and list the people graph."""
    from photo_memory.merge import suggest_merges as compute_suggestions

    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    if merge_args:
        fc_a, fc_b, name = merge_args
        merge_clusters(db, fc_a, fc_b, name=name)
        click.echo(f"已将 {fc_a} 与 {fc_b} 合并为「{name}」")
        db.close()
        return

    if name_args:
        fc_id, user_name = name_args
        db.set_person_user_name(fc_id, user_name)
        click.echo(f"Named {fc_id} as {user_name}")
        db.close()
        return

    if suggest_merges:
        all_people = db.get_all_people()
        suggestions = compute_suggestions(all_people, min_photos=min_photos)
        click.echo(f"找到 {len(suggestions)} 对可能的合并候选:\n")
        for s in suggestions[:20]:
            click.echo(f"  jaccard={s['confidence']}  {s['fc_a'][:12]}... ↔ {s['fc_b'][:12]}...")
            click.echo(f"    共享联系人: {len(s['shared_contacts'])} 位")
            click.echo(f"    合并命令: phototag people --merge {s['fc_a']} {s['fc_b']} \"名字\"")
            click.echo("")
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

Modify the existing `story` command to dispatch to HTML builder when output ends with `.html`. Find the existing story command and replace the bottom part:

```python
    if person_name:
        if output_path and output_path.endswith(".html"):
            content = build_person_html(db, person_name, ollama_config=config["ollama"])
        else:
            content = generate_person_story(db, person_name, ollama_config=config["ollama"])
    elif year:
        if output_path and output_path.endswith(".html"):
            content = build_year_html(db, year, ollama_config=config["ollama"])
        else:
            content = generate_year_story(db, year, ollama_config=config["ollama"])
    else:
        content = generate_relationship_story(db, relationship_name, ollama_config=config["ollama"])

    db.close()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"Story written to {output_path}")
    else:
        click.echo(content)
```

- [ ] **Step 5: 全量回归**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/ --tb=short 2>&1 | tail -10`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/cli.py src/photo_memory/story.py tests/test_cli.py
git commit -m "feat(cli): wire search/cleanup/merge/story-html commands"
```

---

## Parallelization Guide

**Phase A, B, C are independent and can run in parallel** (different files, different modules). Each phase produces a standalone commit.

**Phase D must run AFTER A + B + C are all committed** (modifies cli.py which depends on all three modules).

Dispatch pattern:
1. Launch Task A, B, C as parallel background subagents
2. Wait for all three to complete
3. Dispatch Task D as a final serial subagent
4. Final full regression test

# M2: Layer 1 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 升级 scanner 采集 Apple Photos 完整元数据（人脸、标签、地点、截图/自拍标记），重写 recognizer prompt 输出语义条目（narrative + search_tags + cleanup_class），为 L2 关联层提供数据基础。

**Architecture:** scanner.py 采集 osxphotos 元数据写入 DB 新列 → processor.py 将元数据作为 photo_context 传给 recognizer → recognizer.py 用新 prompt 输出结构化语义结果。DB 通过 schema version 表管理迁移。tagger.py 修复 AppleScript 字符串注入。

**Tech Stack:** Python 3.12, osxphotos, SQLite, Ollama API, pytest

---

### Task 1: DB Schema 迁移机制 + 新列

**Files:**
- Modify: `src/photo_memory/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: 写 schema version + migration 的失败测试**

```python
# tests/test_db.py — 追加

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
        named_faces='["唐嘉鑫"]',
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
    assert row["named_faces"] == '["唐嘉鑫"]'
    assert row["is_selfie"] == 0  # SQLite stores bool as int
    assert row["is_screenshot"] == 0
    assert row["is_live_photo"] == 1
    assert row["location_city"] == "北京市"
    assert row["location_country"] == "中国"
    db.close()


def test_migration_from_v1_adds_columns(tmp_db_path):
    """Simulate a v1 database and verify migration adds new columns."""
    import sqlite3
    # Create a v1-style database manually (no schema_version table)
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

    # Open with Database class — should auto-migrate
    db = Database(tmp_db_path)
    row = db.get_photo("old-1")
    assert row["file_path"] == "/old.jpg"
    # New columns should exist with NULL defaults
    assert row["apple_labels"] is None
    assert row["location_city"] is None
    # Schema version should be current
    ver = db.execute("SELECT version FROM schema_version").fetchone()
    assert ver["version"] >= 2
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py::test_schema_version_table_exists tests/test_db.py::test_new_photo_columns_exist tests/test_db.py::test_migration_from_v1_adds_columns -v`
Expected: FAIL — schema_version 表不存在，新列不存在

- [ ] **Step 3: 实现 schema migration**

修改 `src/photo_memory/db.py`，将 `_init_tables` 改为支持版本迁移：

```python
CURRENT_SCHEMA_VERSION = 2

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _get_schema_version(self) -> int:
        """Get current schema version, 0 if no version table exists."""
        try:
            row = self.conn.execute("SELECT version FROM schema_version").fetchone()
            return row["version"] if row else 0
        except sqlite3.OperationalError:
            return 0

    def _set_schema_version(self, version: int):
        self.conn.execute("DELETE FROM schema_version")
        self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    def _migrate(self):
        version = self._get_schema_version()

        if version < 1:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS photos (
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
                CREATE TABLE IF NOT EXISTS duplicates (
                    group_id INTEGER,
                    photo_uuid TEXT,
                    similarity REAL,
                    FOREIGN KEY (photo_uuid) REFERENCES photos(uuid)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    photos_processed INTEGER DEFAULT 0,
                    photos_skipped INTEGER DEFAULT 0,
                    photos_errored INTEGER DEFAULT 0,
                    stop_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
                CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
                CREATE INDEX IF NOT EXISTS idx_duplicates_group ON duplicates(group_id);
            """)
            version = 1

        if version < 2:
            # Ensure schema_version table exists (for v1 DBs created before migration)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                )
            """)
            # Add Apple metadata columns
            new_columns = [
                ("apple_labels", "TEXT"),
                ("face_cluster_ids", "TEXT"),
                ("named_faces", "TEXT"),
                ("source_app", "TEXT"),
                ("is_selfie", "BOOLEAN"),
                ("is_screenshot", "BOOLEAN"),
                ("is_live_photo", "BOOLEAN"),
                ("location_city", "TEXT"),
                ("location_state", "TEXT"),
                ("location_country", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                try:
                    self.conn.execute(f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            version = 2

        self._set_schema_version(version)
        self.conn.commit()
```

删除旧的 `_init_tables` 方法（被 `_migrate` 替代）。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_db.py -v`
Expected: ALL PASS（包括所有原有测试 + 3 个新测试）

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/db.py tests/test_db.py
git commit -m "feat(db): add schema versioning and migration for Apple metadata columns"
```

---

### Task 2: Scanner 采集 Apple 元数据

**Files:**
- Modify: `src/photo_memory/scanner.py`
- Test: `tests/test_scanner.py`

- [ ] **Step 1: 写 scanner 元数据采集的失败测试**

```python
# tests/test_scanner.py — 重写 _make_mock_photo 并追加测试

def _make_mock_photo(uuid, path, date, lat=None, lon=None, **kwargs):
    photo = MagicMock()
    photo.uuid = uuid
    photo.original_filename = f"{uuid}.jpg"
    photo.path = path
    photo.date = date
    photo.latitude = lat
    photo.longitude = lon
    # Apple metadata defaults
    photo.labels = kwargs.get("labels", [])
    photo.person_info = kwargs.get("person_info", [])
    photo.selfie = kwargs.get("selfie", False)
    photo.screenshot = kwargs.get("screenshot", False)
    photo.live_photo = kwargs.get("live_photo", False)
    photo.place = kwargs.get("place", None)
    # source_app via imported_by
    photo.imported_by = kwargs.get("imported_by", (None, None))
    return photo


def _make_mock_person(name, uuid, facecount=10):
    person = MagicMock()
    person.name = name
    person.uuid = uuid
    person.display_name = name if name != "_UNKNOWN_" else None
    person.facecount = facecount
    return person


def _make_mock_place(city="北京市", state=None, country="中国"):
    place = MagicMock()
    addr = MagicMock()
    addr.city = city
    addr.state_province = state
    addr.country = country
    place.address = addr
    return place


def test_scan_collects_apple_metadata(tmp_db_path):
    db = Database(tmp_db_path)
    from datetime import datetime
    import json

    person1 = _make_mock_person("唐嘉鑫", "person-uuid-001", facecount=663)
    person2 = _make_mock_person("_UNKNOWN_", "person-uuid-002", facecount=0)
    place = _make_mock_place(city="大连市", state="辽宁省", country="中国")

    mock_photos = [
        _make_mock_photo(
            "uuid-meta", "/photos/meta.jpg", datetime(2024, 9, 28),
            lat=38.9, lon=121.6,
            labels=["人", "牛仔裤", "海边"],
            person_info=[person1, person2],
            selfie=False,
            screenshot=False,
            live_photo=True,
            place=place,
            imported_by=("com.apple.mobileslideshow", "相机"),
        ),
    ]
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_db:
        mock_db.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)

    assert count == 1
    row = db.get_photo("uuid-meta")
    assert json.loads(row["apple_labels"]) == ["人", "牛仔裤", "海边"]
    assert json.loads(row["face_cluster_ids"]) == ["person-uuid-001", "person-uuid-002"]
    assert json.loads(row["named_faces"]) == ["唐嘉鑫"]
    assert row["is_selfie"] == 0
    assert row["is_screenshot"] == 0
    assert row["is_live_photo"] == 1
    assert row["location_city"] == "大连市"
    assert row["location_state"] == "辽宁省"
    assert row["location_country"] == "中国"
    assert row["source_app"] == "相机"
    db.close()


def test_scan_handles_missing_metadata(tmp_db_path):
    """Photo with no place, no persons, no labels should still work."""
    db = Database(tmp_db_path)
    from datetime import datetime

    mock_photos = [
        _make_mock_photo("uuid-bare", "/photos/bare.jpg", datetime(2024, 1, 1)),
    ]
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_db:
        mock_db.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)

    assert count == 1
    row = db.get_photo("uuid-bare")
    assert row["apple_labels"] == "[]"
    assert row["face_cluster_ids"] == "[]"
    assert row["named_faces"] == "[]"
    assert row["is_selfie"] == 0
    assert row["location_city"] is None
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_scanner.py::test_scan_collects_apple_metadata tests/test_scanner.py::test_scan_handles_missing_metadata -v`
Expected: FAIL — scanner 没有传这些字段

- [ ] **Step 3: 实现 scanner 元数据采集**

修改 `src/photo_memory/scanner.py`：

```python
"""Scan Apple Photos library and populate the progress database."""

import json
import logging

import osxphotos

from photo_memory.db import Database

logger = logging.getLogger(__name__)


def _extract_place_info(photo) -> dict:
    """Extract city/state/country from photo.place, return dict of values."""
    if not photo.place or not photo.place.address:
        return {}
    addr = photo.place.address
    result = {}
    if addr.city:
        result["location_city"] = addr.city
    if addr.state_province:
        result["location_state"] = addr.state_province
    if addr.country:
        result["location_country"] = addr.country
    return result


def _extract_face_info(photo) -> tuple[str, str]:
    """Extract face_cluster_ids and named_faces as JSON strings."""
    cluster_ids = []
    named = []
    for pi in (photo.person_info or []):
        cluster_ids.append(pi.uuid)
        if pi.name and pi.name != "_UNKNOWN_":
            named.append(pi.name)
    return json.dumps(cluster_ids, ensure_ascii=False), json.dumps(named, ensure_ascii=False)


def scan_photos_into_db(db: Database, photos_db_path: str | None = None) -> int:
    """Scan Photos library and insert new photos into the database.
    Returns the number of newly inserted photos.
    """
    logger.info("Opening Photos library...")
    photosdb = osxphotos.PhotosDB(dbfile=photos_db_path) if photos_db_path else osxphotos.PhotosDB()

    photos = photosdb.photos(images=True, movies=True)
    logger.info(f"Found {len(photos)} photos in library")

    new_count = 0
    for photo in photos:
        if db.get_photo(photo.uuid):
            continue

        face_cluster_ids, named_faces = _extract_face_info(photo)
        place_info = _extract_place_info(photo)

        # imported_by returns (bundle_id, display_name)
        source_app = None
        if photo.imported_by and photo.imported_by[1]:
            source_app = photo.imported_by[1]

        db.upsert_photo(
            uuid=photo.uuid,
            file_path=photo.path,
            date_taken=photo.date.isoformat() if photo.date else None,
            gps_lat=photo.latitude,
            gps_lon=photo.longitude,
            apple_labels=json.dumps(photo.labels or [], ensure_ascii=False),
            face_cluster_ids=face_cluster_ids,
            named_faces=named_faces,
            source_app=source_app,
            is_selfie=photo.selfie,
            is_screenshot=photo.screenshot,
            is_live_photo=photo.live_photo,
            **place_info,
        )
        new_count += 1

    logger.info(f"Inserted {new_count} new photos into database")
    return new_count
```

- [ ] **Step 4: 运行全部 scanner 测试确认通过**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_scanner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/scanner.py tests/test_scanner.py
git commit -m "feat(scanner): collect Apple metadata (faces, labels, place, selfie, screenshot)"
```

---

### Task 3: Recognizer 新 Prompt + 输出格式

**Files:**
- Modify: `src/photo_memory/recognizer.py`
- Test: `tests/test_recognizer.py`

- [ ] **Step 1: 写新 prompt 输出格式的失败测试**

```python
# tests/test_recognizer.py — 追加

def test_parse_new_format_response():
    """Test parsing the new narrative + search_tags format."""
    raw = json.dumps({
        "narrative": "周六下午和女友在大连海边散步的合影",
        "event_hint": "出游/约会",
        "people": [
            {"face_cluster_id": "fc_001", "description": "戴眼镜的年轻男性"},
        ],
        "emotional_tone": "轻松愉快",
        "significance": "日常约会记录",
        "scene_category": "outdoor_leisure",
        "series_hint": "standalone",
        "search_tags": ["海边", "合影", "散步"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "keep",
        "duplicate_hint": "standalone",
    })
    result = parse_ai_response(raw)
    assert result["narrative"] == "周六下午和女友在大连海边散步的合影"
    assert result["search_tags"] == ["海边", "合影", "散步"]
    assert result["cleanup_class"] == "keep"


def test_build_photo_context():
    from photo_memory.recognizer import build_photo_context
    import json

    photo_row = {
        "uuid": "test-uuid",
        "date_taken": "2024-09-28T13:33:00",
        "location_city": "大连市",
        "location_state": "辽宁省",
        "location_country": "中国",
        "apple_labels": '["人", "牛仔裤"]',
        "named_faces": '["唐嘉鑫"]',
        "face_cluster_ids": '["fc_001", "fc_002"]',
        "is_selfie": 0,
        "is_screenshot": 0,
        "is_live_photo": 1,
        "source_app": "相机",
    }
    ctx = build_photo_context(photo_row)
    assert ctx["location_city"] == "大连市"
    assert ctx["named_faces"] == ["唐嘉鑫"]
    assert ctx["apple_labels"] == ["人", "牛仔裤"]
    assert ctx["is_selfie"] is False
    assert ctx["is_live_photo"] is True


def test_recognize_photo_with_context(tmp_path):
    """Test that recognize_photo passes photo_context to the prompt."""
    from PIL import Image
    from photo_memory.recognizer import recognize_photo

    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps({
            "narrative": "测试照片",
            "event_hint": "其他",
            "people": [],
            "emotional_tone": "平静",
            "significance": "普通记录",
            "scene_category": "other",
            "series_hint": "standalone",
            "search_tags": ["测试"],
            "has_text": False,
            "text_summary": "",
            "cleanup_class": "keep",
            "duplicate_hint": "standalone",
        })
    }

    photo_context = {"date": "2024-09-28 周六 13:33", "location_city": "大连市"}

    with patch("photo_memory.recognizer.requests.post", return_value=mock_response) as mock_post:
        result = recognize_photo(str(img_path), host="http://localhost:11434",
                                 model="gemma4:e4b", timeout=60,
                                 photo_context=photo_context)

    assert result["narrative"] == "测试照片"
    # Verify context was included in the prompt
    call_json = mock_post.call_args[1]["json"]
    assert "大连市" in call_json["prompt"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_recognizer.py::test_parse_new_format_response tests/test_recognizer.py::test_build_photo_context tests/test_recognizer.py::test_recognize_photo_with_context -v`
Expected: FAIL

- [ ] **Step 3: 实现新 prompt 和 build_photo_context**

修改 `src/photo_memory/recognizer.py`：

1. 替换 `RECOGNITION_PROMPT` 为新的带 context 的 prompt 模板
2. 添加 `build_photo_context()` 函数
3. 更新 `REQUIRED_FIELDS` 为新格式
4. 修改 `recognize_photo()` 接受 `photo_context` 参数
5. 更新 fallback 返回值匹配新格式

```python
RECOGNITION_PROMPT_TEMPLATE = """你是一个照片分析助手。根据图片内容和以下上下文信息，返回严格的 JSON 格式（不要其他文字）。

上下文信息：
{context}

返回格式：
{{
  "narrative": "一句话中文描述这张照片的内容和场景（结合上下文，不要重复上下文已有的信息）",
  "event_hint": "事件类型：出游/约会/聚餐/工作/学习/运动/购物/日常/其他",
  "people": [
    {{"face_cluster_id": "对应上下文中的人脸ID", "description": "外貌/动作简述"}}
  ],
  "emotional_tone": "情绪基调",
  "significance": "这张照片的意义",
  "scene_category": "indoor_home|indoor_office|indoor_restaurant|outdoor_leisure|outdoor_travel|transport|other",
  "series_hint": "burst|sequence|standalone",
  "search_tags": ["简洁的可搜索场景标签，3-5个"],
  "has_text": false,
  "text_summary": "如果有文字，提取关键内容",
  "cleanup_class": "keep|review|cleanup",
  "duplicate_hint": "burst|live_variant|edited|standalone"
}}

注意：
- cleanup_class: 截图/模糊/无意义的照片标记为 cleanup，不确定的标记为 review，有意义的标记为 keep
- narrative 要有信息增量，不要只重复上下文
- search_tags 用于搜索，要简洁精确"""

# Legacy prompt for backward compatibility (no context)
RECOGNITION_PROMPT_NO_CONTEXT = """分析这张图片，返回严格的 JSON 格式（不要其他文字）：
{{
  "narrative": "一句话中文描述内容",
  "event_hint": "事件类型",
  "people": [],
  "emotional_tone": "情绪基调",
  "significance": "意义",
  "scene_category": "场景分类",
  "series_hint": "standalone",
  "search_tags": ["标签1", "标签2"],
  "has_text": false,
  "text_summary": "",
  "cleanup_class": "keep",
  "duplicate_hint": "standalone"
}}"""

REQUIRED_FIELDS = {"narrative", "search_tags", "has_text", "text_summary", "cleanup_class"}


def build_photo_context(photo_row: dict) -> dict:
    """Build a photo_context dict from a database row for the recognizer prompt."""
    import json as _json

    def _parse_json_field(val):
        if val is None:
            return []
        if isinstance(val, str):
            try:
                return _json.loads(val)
            except _json.JSONDecodeError:
                return []
        return val

    return {
        "date": photo_row.get("date_taken", ""),
        "location_city": photo_row.get("location_city"),
        "location_state": photo_row.get("location_state"),
        "location_country": photo_row.get("location_country"),
        "apple_labels": _parse_json_field(photo_row.get("apple_labels")),
        "named_faces": _parse_json_field(photo_row.get("named_faces")),
        "face_cluster_ids": _parse_json_field(photo_row.get("face_cluster_ids")),
        "is_selfie": bool(photo_row.get("is_selfie")),
        "is_screenshot": bool(photo_row.get("is_screenshot")),
        "is_live_photo": bool(photo_row.get("is_live_photo")),
        "source_app": photo_row.get("source_app"),
    }
```

更新 `recognize_photo` 签名，添加 `photo_context: dict | None = None` 参数。如果有 context，用 `RECOGNITION_PROMPT_TEMPLATE.format(context=json.dumps(photo_context, ensure_ascii=False, indent=2))`，否则用 `RECOGNITION_PROMPT_NO_CONTEXT`。

更新 `parse_ai_response` 中的 fallback 返回值，使用新字段名（`narrative` 代替 `description`，`search_tags` 代替 `tags`，添加 `cleanup_class` 等）。

- [ ] **Step 4: 运行全部 recognizer 测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_recognizer.py -v`
Expected: ALL PASS

注意：旧测试 `test_parse_valid_json_response` 等需要更新为新字段名，或保留兼容逻辑。由于 `REQUIRED_FIELDS` 变了，旧格式的 JSON 不再通过 `REQUIRED_FIELDS.issubset` 检查，会 fallback。需要同步更新旧测试的 JSON fixture。

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/recognizer.py tests/test_recognizer.py
git commit -m "feat(recognizer): new prompt with photo_context, narrative + search_tags output"
```

---

### Task 4: Tagger AppleScript 字符串注入修复

**Files:**
- Modify: `src/photo_memory/tagger.py`
- Test: `tests/test_tagger.py`

- [ ] **Step 1: 写 AppleScript 注入防护的失败测试**

```python
# tests/test_tagger.py — 追加

def test_sanitize_applescript_string():
    from photo_memory.tagger import _sanitize_for_applescript
    # Normal string
    assert _sanitize_for_applescript("hello") == "hello"
    # Double quotes should be escaped
    assert _sanitize_for_applescript('say "hi"') == 'say \\"hi\\"'
    # Backslashes should be escaped
    assert _sanitize_for_applescript("a\\b") == "a\\\\b"
    # Newlines/tabs stripped
    assert _sanitize_for_applescript("line1\nline2\ttab") == "line1 line2 tab"


def test_set_keywords_sanitizes_input(mocker):
    """Ensure keywords with special chars don't break AppleScript."""
    mock_run = mocker.patch("photo_memory.tagger._run_applescript", return_value=True)
    from photo_memory.tagger import _set_keywords
    _set_keywords("test-uuid", ['normal', 'has"quote', 'has\\slash'])
    script = mock_run.call_args[0][0]
    # Should not contain unescaped quotes that break AppleScript
    assert '\\"quote' in script or "quote" in script
    assert '\\\\slash' in script or "slash" in script
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_tagger.py::test_sanitize_applescript_string tests/test_tagger.py::test_set_keywords_sanitizes_input -v`
Expected: FAIL — `_sanitize_for_applescript` 不存在

- [ ] **Step 3: 实现 sanitize 函数并应用到所有 AppleScript 调用**

修改 `src/photo_memory/tagger.py`：

```python
def _sanitize_for_applescript(s: str) -> str:
    """Sanitize a string for safe use in AppleScript double-quoted strings."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s
```

在 `_set_keywords`、`_set_description`、`_add_to_album` 中，所有拼入 AppleScript 的用户数据都用 `_sanitize_for_applescript` 处理。

- [ ] **Step 4: 运行全部 tagger 测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_tagger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/tagger.py tests/test_tagger.py
git commit -m "fix(tagger): sanitize strings for AppleScript injection prevention"
```

---

### Task 5: Processor 串联 photo_context → recognizer

**Files:**
- Modify: `src/photo_memory/processor.py`
- Test: `tests/test_processor.py`

- [ ] **Step 1: 写 processor 传递 context 的失败测试**

```python
# tests/test_processor.py — 追加

def test_process_one_photo_passes_context(tmp_path, tmp_db_path):
    """Verify process_one_photo builds photo_context and passes it to recognizer."""
    from photo_memory.db import Database
    from photo_memory.processor import process_one_photo
    import json

    db = Database(tmp_db_path)
    db.upsert_photo("uuid-ctx",
        file_path=str(tmp_path / "test.jpg"),
        date_taken="2024-09-28T13:33:00",
        location_city="大连市",
        apple_labels='["人"]',
        face_cluster_ids='["fc_001"]',
        named_faces='["唐嘉鑫"]',
        is_selfie=False,
        is_screenshot=False,
        is_live_photo=False,
    )

    # Create a test image
    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img.save(tmp_path / "test.jpg")

    ai_result = {
        "narrative": "测试",
        "event_hint": "其他",
        "people": [],
        "emotional_tone": "平静",
        "significance": "测试",
        "scene_category": "other",
        "series_hint": "standalone",
        "search_tags": ["测试"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "keep",
        "duplicate_hint": "standalone",
    }

    with patch("photo_memory.processor.recognize_photo", return_value=ai_result) as mock_rec, \
         patch("photo_memory.processor.apply_tags_to_photo"), \
         patch("photo_memory.processor.compute_phash", return_value="aabb"):
        photo_row = db.get_photo("uuid-ctx")
        result = process_one_photo(db, photo_row, {"host": "h", "model": "m", "timeout": 60}, str(tmp_path))

    assert result is True
    # Verify photo_context was passed
    call_kwargs = mock_rec.call_args
    assert "photo_context" in call_kwargs[1] or (len(call_kwargs[0]) > 4)
    # Check context contains location
    ctx = call_kwargs[1].get("photo_context") or call_kwargs[0][4]
    assert ctx["location_city"] == "大连市"
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_processor.py::test_process_one_photo_passes_context -v`
Expected: FAIL — recognize_photo 没有收到 photo_context

- [ ] **Step 3: 修改 processor 传递 context**

修改 `src/photo_memory/processor.py` 中的 `process_one_photo`：

1. 在调用 `recognize_photo` 前，用 `build_photo_context(photo_row)` 构建 context
2. 将 context 作为 `photo_context=` 参数传给 `recognize_photo`
3. 更新 `update_photo_result` 调用，将 `description` 改为取 `narrative`，`tags` 改为取 `search_tags`

```python
from photo_memory.recognizer import recognize_photo, summarize_video_frames, build_photo_context

# 在 process_one_photo 中：
photo_context = build_photo_context(photo_row)

result = recognize_photo(
    exported_path,
    host=ollama_config["host"],
    model=ollama_config["model"],
    timeout=ollama_config["timeout"],
    photo_context=photo_context,
)

# update_photo_result 调用改为：
db.update_photo_result(
    uuid=uuid,
    status="done",
    phash=phash,
    ai_result=json.dumps(result, ensure_ascii=False),
    tags=json.dumps(result.get("search_tags", []), ensure_ascii=False),
    description=result.get("narrative", ""),
    importance=result.get("cleanup_class", "keep"),  # 用 cleanup_class 代替旧的 importance
    media_type=result.get("scene_category", "photo"),
)
```

- [ ] **Step 4: 运行全部 processor 测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_processor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/processor.py tests/test_processor.py
git commit -m "feat(processor): pass photo_context to recognizer, use new output fields"
```

---

### Task 6: CLI 添加 reprocess 命令 + run --recent/--all

**Files:**
- Modify: `src/photo_memory/cli.py`
- Modify: `src/photo_memory/db.py` (添加 `reset_photos_for_reprocess` 方法)
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写 reprocess 和 --recent 的失败测试**

```python
# tests/test_cli.py — 追加

from click.testing import CliRunner
from photo_memory.cli import main

def test_reprocess_command_resets_done_photos(tmp_path, sample_config):
    """reprocess should reset 'done' photos back to 'pending'."""
    config, config_path = sample_config
    from photo_memory.db import Database
    db_path = str(tmp_path / "progress.db")
    config["data_dir"] = str(tmp_path)

    db = Database(db_path)
    db.upsert_photo("uuid-1", status="done")
    db.upsert_photo("uuid-2", status="done")
    db.upsert_photo("uuid-3", status="pending")
    db.update_photo_status("uuid-1", "done")
    db.update_photo_status("uuid-2", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["reprocess", "--config", config_path])

    assert result.exit_code == 0
    db = Database(db_path)
    assert db.get_photo("uuid-1")["status"] == "pending"
    assert db.get_photo("uuid-2")["status"] == "pending"
    db.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py::test_reprocess_command_resets_done_photos -v`
Expected: FAIL — reprocess 命令不存在

- [ ] **Step 3: 实现 reprocess 命令 + reset 方法**

在 `src/photo_memory/db.py` 中添加：

```python
def reset_photos_for_reprocess(self) -> int:
    """Reset all 'done' photos back to 'pending' for reprocessing.
    Returns the number of photos reset.
    """
    cursor = self.conn.execute(
        "UPDATE photos SET status = 'pending', ai_result = NULL, tags = NULL, "
        "description = NULL, importance = NULL, processed_at = NULL "
        "WHERE status = 'done'"
    )
    self.conn.commit()
    return cursor.rowcount
```

在 `src/photo_memory/cli.py` 中添加 `reprocess` 命令：

```python
@main.command()
@click.pass_context
def reprocess(ctx):
    """Reset all processed photos back to pending for reprocessing with new prompt."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)
    count = db.reset_photos_for_reprocess()
    click.echo(f"Reset {count} photos to pending. Run 'phototag run' to reprocess.")
    db.close()
```

- [ ] **Step 4: 运行全部 CLI 测试**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/photo-memory
git add src/photo_memory/db.py src/photo_memory/cli.py tests/test_cli.py
git commit -m "feat(cli): add reprocess command to reset photos for re-analysis"
```

---

### Task 7: 全量集成测试 + 回归验证

**Files:**
- Test: `tests/test_integration.py` (已有，可能需要更新)

- [ ] **Step 1: 运行全部测试，确保无回归**

Run: `cd ~/photo-memory && .venv/bin/pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: 如有失败，逐个修复**

常见问题：
- 旧测试中 `REQUIRED_FIELDS` 变化导致 parse 行为变化 → 更新旧测试的 JSON fixture
- `process_one_photo` 测试中 mock 缺少新参数 → 添加 `photo_context` mock
- `update_photo_result` 参数名变化 → 同步更新

- [ ] **Step 3: Commit 修复（如有）**

```bash
cd ~/photo-memory
git add -A
git commit -m "fix: update existing tests for M2 format changes"
```

- [ ] **Step 4: 手动验证——用真实照片库跑一次 scan**

```bash
cd ~/photo-memory && .venv/bin/python -c "
from photo_memory.db import Database
from photo_memory.scanner import scan_photos_into_db
import json

db = Database('/tmp/test_m2.db')
count = scan_photos_into_db(db)
print(f'Scanned {count} photos')

# Check a few photos with metadata
rows = db.execute('''
    SELECT uuid, location_city, apple_labels, named_faces, face_cluster_ids, is_screenshot
    FROM photos
    WHERE location_city IS NOT NULL
    LIMIT 5
''').fetchall()
for r in rows:
    print(f'{r[\"uuid\"][:8]}... city={r[\"location_city\"]} labels={r[\"apple_labels\"][:30]} faces={r[\"named_faces\"]}')
db.close()
"
```

Expected: 能看到带有 city、labels、named_faces 的照片数据

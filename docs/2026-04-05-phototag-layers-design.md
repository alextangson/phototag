# phototag Layers Design Spec

**Date**: 2026-04-05
**Status**: Draft
**Author**: @alextangson + Claude

## Product Thesis

把 Apple Photos 的沉睡数据变成可感知的人生叙事。

Apple 有优秀的基建（人脸聚类、GPS 反向地理、物体标签），但没有 AI 上层应用。phototag 在用户自己的设备上，用本地 LLM 把这些数据转化为有意义的理解。

## Non-Goals

- 不做 Apple 已经做好的事（物体标签、人脸检测、关键词搜索）
- 不做创作者素材管理工具（不同产品、不同用户）
- 不依赖云端服务
- 不在 AppleScript 写回能力上建立核心价值（平台风险）

## Architecture Overview

```
Layer 1 — 感知             Layer 2 — 关联              Layer 3 — 叙事
Apple metadata + LLM  →   event + people + place   →  story generation

开源 CLI                   开源 CLI                    开源: Markdown/JSON
                                                      商业: HTML + 订阅推送
```

核心资产是 SQLite 中的语义数据，不是写回 Apple Photos 的能力。

---

## Layer 1: Perception (重构)

### Problem

当前 Layer 1 让 LLM 盲猜照片内容，输出扁平的关键词标签。这与 Apple 原生 ML 标签高度重叠，且 LLM 缺少上下文导致推理质量低。

### Design

**输入：Apple 元数据 + 图片**

Scanner 采集以下元数据，存入 DB 并传给 recognizer：

```python
photo_context = {
    "date": "2024-09-28 周六 13:33",
    "location_city": "大连市",
    "location_state": "辽宁省",
    "location_country": "中国",
    "location_street": "滨海路",        # 如果有
    "device": "iPhone 14 Pro Max",
    "source_app": "相机",               # imported_by_display_name
    "apple_labels": ["人", "牛仔裤"],    # Apple ML 已识别的物体标签
    "named_faces": ["唐嘉鑫"],          # Apple 已命名的人脸
    "unnamed_face_count": 1,            # 未命名人脸数
    "face_cluster_ids": ["fc_001", "fc_002"],  # 人脸聚类ID列表
    "is_selfie": False,
    "is_screenshot": False,
    "is_live_photo": True,
    "albums": ["JX💗WJ"],              # 用户已有相册归属
    "nearby_photos_count": 8,           # 前后 10 分钟内的照片数
}
```

**输出：语义条目（非标签）**

```python
ai_result = {
    "narrative": "周六下午和女友在大连海边散步的合影，两人穿着休闲",
    "event_hint": "出游/约会",
    "people": [
        {
            "face_cluster_id": "fc_001",
            "description": "戴眼镜的年轻男性",
            "is_photographer_likely": False,
        },
        {
            "face_cluster_id": "fc_002",
            "description": "长发女性，亲密互动",
            "is_photographer_likely": False,
        },
    ],
    "emotional_tone": "轻松愉快",
    "significance": "日常约会记录",
    "scene_category": "outdoor_leisure",
    "series_hint": "burst",  # burst | sequence | standalone
}
```

**不再输出：** tags, colors, quality_notes, objects, animals, location_type, time_of_day（这些 Apple 元数据已覆盖或无用）

### First-Run Strategy

- `phototag run` 默认只处理最近 6 个月的照片（按 date_taken DESC）
- `phototag run --all` 全量处理
- 处理完首批 500 张后自动打印一份简要摘要（人物统计、事件数、地点数）

### DB Schema Changes

photos 表新增列：

```sql
ALTER TABLE photos ADD COLUMN apple_labels TEXT;      -- JSON array
ALTER TABLE photos ADD COLUMN face_cluster_ids TEXT;   -- JSON array
ALTER TABLE photos ADD COLUMN named_faces TEXT;        -- JSON array
ALTER TABLE photos ADD COLUMN source_app TEXT;
ALTER TABLE photos ADD COLUMN is_selfie BOOLEAN;
ALTER TABLE photos ADD COLUMN is_screenshot BOOLEAN;
ALTER TABLE photos ADD COLUMN location_city TEXT;
ALTER TABLE photos ADD COLUMN location_state TEXT;
ALTER TABLE photos ADD COLUMN location_country TEXT;
```

### Validation Criteria

随机抽 20 张已处理照片，人工检查：
- narrative 准确描述了照片内容：>= 70%
- event_hint 分类合理：>= 80%
- people 描述与实际人物匹配：>= 75%

---

## Layer 2: Association

### Problem

10000 张照片是 10000 个孤立的点。用户想要的是"事件"、"人物关系"、"地点记忆" — 即点与点之间的线。

### Design: Event Aggregation

不依赖 GPS（86% 照片没有），使用多信号聚合：

**Step 1: 时间窗口切割**
```
相邻照片间隔 > 30 min → 断开为新事件
```

**Step 2: 事件内信号增强**
- 人脸聚类 ID 重合度 → 同一群人 = 同一事件
- source_app 一致性 → 全是微信 = 社交分享事件
- scene_category 一致性 → 全是 outdoor = 同一场外出
- GPS 相近（如果有）→ 加分项

**Step 3: 事件摘要生成**
对每个事件，取其中照片的 narrative 列表，让 LLM 生成一段事件级摘要：
```python
event = {
    "event_id": "evt_20240928_001",
    "date_range": ["2024-09-28 13:00", "2024-09-28 17:30"],
    "location": "大连市",
    "photo_count": 15,
    "people": ["fc_001", "fc_002"],
    "summary": "和女友在大连海边度过的一个下午，逛了滨海路，在海边拍了很多合影",
    "mood": "愉快轻松",
    "cover_photo_uuid": "xxx",  # 选 Apple score 最高的照片
}
```

### Design: People Graph

**Step 1: 统计**
```python
person = {
    "face_cluster_id": "fc_002",
    "apple_name": "阿菁",           # 如果用户已命名
    "suggested_name": None,          # Layer 2 不猜名字
    "photo_count": 200,
    "event_count": 45,
    "date_range": ["2023-01", "2026-02"],
    "co_appears_with": {
        "fc_001": 180,               # 和 fc_001 共同出现 180 次
    },
    "top_locations": ["大连", "深圳", "北京"],
    "appearance_trend": "increasing", # increasing | stable | decreasing | one_time
}
```

**Step 2: 关系推断**
- co_appears 频率高 + 时间跨度长 → 亲密关系
- 只在特定时间段出现 → 阶段性人物（同事、旅伴）
- appearance_trend = increasing → 新进入生活的重要人物

**Step 3: 命名引导**
对于未命名但出现频率高的人脸聚类，提示用户：
```
phototag people
→ 发现 3 位高频出现但未命名的人物：
  face_002: 200 张照片, 2023-01 至今, 可能是亲密关系
  face_015: 45 张照片, 2024-03 ~ 2024-12, 可能是同事
  face_023: 30 张照片, 2025-01 至今, 可能是朋友
  运行 phototag people --name face_002 "阿菁" 来命名
```

### Design: Place Memory

```python
place = {
    "city": "大连",
    "visit_count": 3,              # 去过几次
    "total_photos": 45,
    "date_range": ["2024-09", "2025-07"],
    "events": ["evt_20240928_001", ...],
    "people": ["fc_001", "fc_002"],
    "summary": "去过3次大连，都是和女友一起，主要在滨海路和老虎滩",
}
```

### DB Schema

```sql
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    location_city TEXT,
    location_state TEXT,
    photo_count INTEGER,
    face_cluster_ids TEXT,       -- JSON array
    summary TEXT,
    mood TEXT,
    cover_photo_uuid TEXT,
    FOREIGN KEY (cover_photo_uuid) REFERENCES photos(uuid)
);

CREATE TABLE event_photos (
    event_id TEXT,
    photo_uuid TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (photo_uuid) REFERENCES photos(uuid)
);

CREATE TABLE people (
    face_cluster_id TEXT PRIMARY KEY,
    apple_name TEXT,
    user_name TEXT,
    photo_count INTEGER,
    event_count INTEGER,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    co_appearances TEXT,         -- JSON: {face_id: count}
    top_locations TEXT,          -- JSON array
    appearance_trend TEXT
);
```

### Validation Criteria

- 事件聚合：手动检查 20 个事件，>= 80% 边界合理（不把两件事混为一个）
- 人物图谱：核心人物（出现 > 50 次）全部被识别
- 关系推断：亲密关系/同事/朋友分类准确率 >= 70%

---

## Layer 3: Narrative

### Problem

Layer 2 输出结构化数据（事件、人物、地点），但用户想看到的是"故事"。

### Design: CLI Commands (Open Source)

```bash
# 人物时间线
phototag story --person "阿菁"
→ 输出 Markdown：
  # 和阿菁的回忆
  ## 2023年1月 - 初次出现
  在xxx拍了第一张合影...
  ## 2023年10月 - 云南之旅
  一起去了玉溪...
  ...

# 年度回忆
phototag story --year 2024
→ 输出 Markdown：
  # 2024 年度回忆
  这一年你拍了 3200 张照片，去了 8 个城市...
  ## 重要时刻
  ...

# 关系回忆线
phototag story --relationship "阿菁"
→ 输出 Markdown：
  # 和阿菁在一起的日子
  从 2023 年 1 月到现在，共 500 天...

# 导出为文件
phototag story --person "阿菁" --output story.md
phototag story --person "阿菁" --output story.json
```

### Narrative Generation Strategy

不是一次性让 LLM 写整个故事（上下文窗口放不下），而是分层生成：

1. 从 events 表按时间排序取事件列表
2. 按季度/月分组，每组让 LLM 生成一段叙事
3. 最后让 LLM 生成开头和总结，串联全文

### Commercial Extension (Future)

- `phototag generate --template "couple" --person "阿菁" --output story.html`
- 精美 HTML 模板，内嵌照片缩略图
- 可导出 PDF
- 订阅制：每月自动生成回忆邮件

不在当前 scope 内，此处仅记录方向。

---

## Open Source vs Commercial Boundary

```
phototag CLI (Apache-2.0)
├── Layer 1: Apple metadata + LLM perception
├── Layer 2: Event aggregation + people graph + place memory
├── Layer 3: Markdown/JSON narrative generation
└── All data in local SQLite, user owns everything

phototag Pro (future, commercial)
├── HTML/PDF memoir template engine
├── Monthly auto-generated memory reports
├── Shareable story pages
└── Hosted service (optional)
```

Principle: **Engine open source, experience commercial.** But the engine itself must be valuable enough — `phototag story` output should already be emotionally compelling in plain Markdown.

---

## Milestones

| Milestone | Scope | Success Criteria | Est. Effort |
|-----------|-------|------------------|-------------|
| **M2** | Layer 1 重构: 新 prompt + Apple 元数据传入 + 新输出格式 | 20 张照片 narrative 准确率 >= 70% | ~1 week |
| **M3** | Layer 2 事件聚合: 时间+人脸+来源多信号 | 20 个事件 80% 边界合理 | ~1 week |
| **M4** | Layer 2 人物图谱: 统计 + 关系推断 + 命名引导 | 核心人物全部识别 | ~1 week |
| **M5** | Layer 3 叙事: `phototag story` 命令 | **自己看了想发朋友圈** | ~1 week |
| **M6** | 社交验证: 发布到社交媒体 | > 5 人问"这怎么做的" | ~1 day |

**M5 是生死线。** 如果生成的回忆录不能让自己感动，不继续往下做。

---

## Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 本地模型能力不足 | narrative 质量低，L2 垃圾进垃圾出 | 模型可插拔；M2 验证时如果不达标，切换更大模型或接入 API |
| Apple 发布类似功能 | L3 叙事层被替代 | 核心资产是结构化语义数据，不是呈现层；Apple 不会做个人化叙事 |
| osxphotos/AppleScript 接口变更 | macOS 升级后 break | 核心价值在 SQLite 数据，写回 Apple Photos 是可选增强 |
| 只有自己在用 | 无社区，无增长 | M5/M6 快速验证；如果无反响则定位为个人工具，不追求增长 |
| 处理速度太慢 | 首次体验差 | 默认只处理近 6 个月；首批 500 张后即出摘要 |

---

## Implementation Order

M2 (Layer 1 重构) 是第一个实施目标，包含以下改动：

1. **scanner.py** — 采集 Apple 元数据（labels, face_cluster_ids, named_faces, source_app, selfie, screenshot, place info）
2. **db.py** — 新增 photos 表列
3. **recognizer.py** — 重写 prompt，传入 photo_context，输出新格式
4. **processor.py** — 串联 scanner 元数据 → recognizer
5. **tagger.py** — 简化，只写 narrative 到 description，不再写关键词（Apple 的更好）
6. **cli.py** — `run` 命令默认处理最近 6 个月，支持 `--all`

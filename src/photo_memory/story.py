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


def build_year_html(db, year: int, ollama_config: dict) -> str:
    """Generate a year story as self-contained HTML with inlined photos."""
    from photo_memory.html_renderer import render_story_html

    events = db.get_events_in_year(year)
    if not events:
        return render_story_html(
            title=f"{year} 年度回忆",
            stats_line="",
            intro_narrative="这一年暂无事件记录。",
            events_with_photos=[],
        )

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

    fc_ids = db.get_all_fc_ids_for_name(person_name)
    all_events = []
    seen = set()
    for fc_id in fc_ids:
        for e in db.get_events_for_person(fc_id):
            if e["event_id"] not in seen:
                all_events.append(e)
                seen.add(e["event_id"])
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

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

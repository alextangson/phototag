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

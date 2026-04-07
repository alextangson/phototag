"""Daily memory push: 'On This Day' and 'This Week' recaps."""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

from photo_memory.html_renderer import render_story_html, _select_photos_for_event
from photo_memory.story import generate_period_narrative, _attach_person_names

logger = logging.getLogger(__name__)


def generate_on_this_day(db, target_date: datetime, ollama_config: dict) -> str | None:
    """Generate an 'On This Day' HTML page for the given date.

    Finds events from the same month+day in all previous years.
    Returns HTML string, or None if no events found.
    """
    month, day = target_date.month, target_date.day
    current_year = target_date.year

    events = db.get_events_on_this_day(month, day)
    if not events:
        return None

    # Filter to previous years only (not today's events)
    past_events = [e for e in events if _extract_year(e.get("start_time")) < current_year]
    if not past_events:
        return None

    _attach_person_names(db, past_events)

    # Attach photos to each event
    events_with_photos = []
    for e in past_events:
        links = db.get_event_photos(e["event_id"])
        photo_rows = []
        for link in links:
            p = db.get_photo(link["photo_uuid"])
            if p and p.get("file_path") and os.path.isfile(p["file_path"]):
                photo_rows.append(p)
        if photo_rows:  # Only include events that have accessible photos
            events_with_photos.append({**e, "photos": photo_rows})

    if not events_with_photos:
        return None

    # Build title and stats
    date_str = f"{month}月{day}日"
    years = sorted(set(_extract_year(e.get("start_time")) for e in events_with_photos), reverse=True)
    years_ago = [f"{current_year - y}年前" for y in years]
    title = f"回忆 · {date_str}"
    stats_line = f"{', '.join(years_ago)}的今天"

    # Generate intro narrative
    intro = generate_period_narrative(
        events_with_photos,
        period_label=f"{date_str}的回忆",
        host=ollama_config["host"],
        model=ollama_config["model"],
        timeout=ollama_config["timeout"],
    )

    return render_story_html(
        title=title,
        stats_line=stats_line,
        intro_narrative=intro,
        events_with_photos=events_with_photos,
        max_photos_per_event=5,
    )


def generate_this_week(db, target_date: datetime, ollama_config: dict) -> str | None:
    """Generate a 'This Week In Your Life' HTML for the week containing target_date.

    Covers all events from the same calendar week across all years.
    """
    # Get the week's date range (Monday to Sunday)
    weekday = target_date.weekday()  # 0=Mon
    week_start = target_date - timedelta(days=weekday)
    week_end = week_start + timedelta(days=6)

    start_month, start_day = week_start.month, week_start.day
    end_month, end_day = week_end.month, week_end.day

    current_year = target_date.year

    # Collect events for each day in the week across all years
    all_events = []
    seen = set()
    for offset in range(7):
        d = week_start + timedelta(days=offset)
        day_events = db.get_events_on_this_day(d.month, d.day)
        for e in day_events:
            yr = _extract_year(e.get("start_time"))
            if yr < current_year and e["event_id"] not in seen:
                all_events.append(e)
                seen.add(e["event_id"])

    if not all_events:
        return None

    _attach_person_names(db, all_events)

    # Attach photos
    events_with_photos = []
    for e in all_events:
        links = db.get_event_photos(e["event_id"])
        photo_rows = []
        for link in links:
            p = db.get_photo(link["photo_uuid"])
            if p and p.get("file_path") and os.path.isfile(p["file_path"]):
                photo_rows.append(p)
        if photo_rows:
            events_with_photos.append({**e, "photos": photo_rows})

    if not events_with_photos:
        return None

    # Sort by year DESC
    events_with_photos.sort(key=lambda e: e.get("start_time", ""), reverse=True)

    title = f"This Week In Your Life · {start_month}月{start_day}日 ~ {end_month}月{end_day}日"
    years = sorted(set(_extract_year(e.get("start_time")) for e in events_with_photos), reverse=True)
    stats_line = f"跨越 {len(years)} 个年份的回忆，共 {len(events_with_photos)} 个事件"

    intro = generate_period_narrative(
        events_with_photos[:15],  # cap for LLM context
        period_label="本周回忆",
        host=ollama_config["host"],
        model=ollama_config["model"],
        timeout=ollama_config["timeout"],
    )

    return render_story_html(
        title=title,
        stats_line=stats_line,
        intro_narrative=intro,
        events_with_photos=events_with_photos,
        max_photos_per_event=5,
    )


def run_daily_push(db, ollama_config: dict, output_dir: str) -> str | None:
    """Generate today's memory push and save to output_dir.

    Strategy:
    - Try 'On This Day' first (most specific, most magical)
    - Fallback to 'This Week' (broader, more likely to find events)
    - Returns the output file path, or None if nothing generated
    """
    today = datetime.now()
    os.makedirs(output_dir, exist_ok=True)

    date_str = today.strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, f"{date_str}.html")

    # Skip if already generated today
    if os.path.isfile(output_path):
        logger.info(f"Daily memory already exists: {output_path}")
        return output_path

    # Try On This Day
    html = generate_on_this_day(db, today, ollama_config)
    mode = "On This Day"

    # Fallback to This Week
    if not html:
        html = generate_this_week(db, today, ollama_config)
        mode = "This Week"

    if not html:
        logger.info("No memory events found for today or this week")
        return None

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Daily memory generated ({mode}): {output_path}")

    # macOS notification
    _notify_macos(mode, output_path)

    return output_path


def _notify_macos(mode: str, file_path: str):
    """Send a macOS notification about the new memory."""
    try:
        title = "phototag · 今日回忆"
        body = f"📷 {mode} — 点击查看"
        script = (
            f'display notification "{body}" with title "{title}" '
            f'sound name "Glass"'
        )
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)

        # Also open in browser
        subprocess.run(["open", file_path], capture_output=True, timeout=5)
    except Exception as e:
        logger.warning(f"Notification failed: {e}")


def _extract_year(date_str: str | None) -> int:
    if not date_str:
        return 0
    try:
        return datetime.fromisoformat(date_str).year
    except (ValueError, TypeError):
        return 0

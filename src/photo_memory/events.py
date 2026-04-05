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

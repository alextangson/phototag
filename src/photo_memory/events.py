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


def enrich_event_metadata(event: dict) -> dict:
    """Enrich an event dict with aggregated metadata (faces, city, cover).

    Args:
        event: event dict from slice_into_events with 'photos', 'start_time', 'end_time'

    Returns:
        enriched event dict with additional keys: event_id, photo_count,
        face_cluster_ids (list), location_city, cover_photo_uuid
    """
    from collections import Counter

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

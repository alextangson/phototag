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
    # Index: face_cluster_id -> list of appearance records
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

        # Simple heuristic: take first non-empty named entry from appearances
        apple_name = None
        for a in appearances:
            if a["named"]:
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

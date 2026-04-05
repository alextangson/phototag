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

    Heuristic: two clusters that never co-appear in any photo but share many
    co-appearance contacts (high Jaccard similarity of their networks) are
    likely the same person photographed separately.

    Returns list of suggestion dicts sorted by confidence DESC.
    """
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

    Soft merge: photos and events are not rewritten.
    """
    db.set_person_user_name(fc_a, name)
    db.set_person_user_name(fc_b, name)
    logger.info(f"Merged {fc_a} and {fc_b} under name '{name}'")

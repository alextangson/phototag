"""Structured search over photos and cleanup candidate reporting."""

import logging

logger = logging.getLogger(__name__)


def search_photos(
    db,
    person: str | None = None,
    year: int | None = None,
    city: str | None = None,
    text: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Structured photo search.

    Returns list of photo row dicts (newest first).
    """
    face_cluster_ids = None
    if person:
        p = db.get_person_by_name(person)
        if p is None:
            logger.info(f"Person not found: {person}")
            return []
        face_cluster_ids = [p["face_cluster_id"]]

    return db.search_photos(
        face_cluster_ids=face_cluster_ids,
        year=year,
        city=city,
        text=text,
        limit=limit,
    )


def list_cleanup_candidates(db) -> dict:
    """Return a cleanup report grouped by class.

    Structure:
        {
            "cleanup": {"count": N, "photos": [...]},
            "review":  {"count": M, "photos": [...]},
        }
    """
    cleanup = db.get_cleanup_candidates(classes=["cleanup"])
    review = db.get_cleanup_candidates(classes=["review"])
    return {
        "cleanup": {"count": len(cleanup), "photos": cleanup},
        "review": {"count": len(review), "photos": review},
    }

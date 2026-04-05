"""Scan Apple Photos library and populate the progress database."""

import json
import logging

import osxphotos

from photo_memory.db import Database

logger = logging.getLogger(__name__)


def _extract_place_info(photo) -> dict:
    """Extract city/state/country from photo.place or GPS reverse geocoding."""
    # Try Apple's place data first
    if photo.place and photo.place.address:
        addr = photo.place.address
        result = {}
        if addr.city:
            result["location_city"] = addr.city
        if addr.state_province:
            result["location_state"] = addr.state_province
        if addr.country:
            result["location_country"] = addr.country
        if result:
            return result

    # Fallback: reverse geocode from GPS coordinates
    if photo.latitude and photo.longitude:
        return _reverse_geocode(photo.latitude, photo.longitude)

    return {}


# Module-level cache for reverse_geocoder (loaded once on first use)
_rg_module = None


def _reverse_geocode(lat: float, lon: float) -> dict:
    """Reverse geocode GPS coordinates using offline reverse_geocoder library."""
    global _rg_module
    try:
        if _rg_module is None:
            import reverse_geocoder as rg
            _rg_module = rg
        results = _rg_module.search([(lat, lon)], mode=2, verbose=False)
        if results:
            r = results[0]
            result = {}
            if r.get("name"):
                result["location_city"] = r["name"]
            if r.get("admin1"):
                result["location_state"] = r["admin1"]
            if r.get("cc"):
                # Convert country code to country name
                result["location_country"] = r["cc"]
            return result
    except Exception as e:
        logger.warning(f"Reverse geocoding failed for ({lat}, {lon}): {e}")
    return {}


def _extract_face_info(photo) -> tuple[str, str]:
    """Extract face_cluster_ids and named_faces as JSON strings."""
    cluster_ids = []
    named = []
    for pi in (photo.person_info or []):
        cluster_ids.append(pi.uuid)
        if pi.name and pi.name != "_UNKNOWN_":
            named.append(pi.name)
    return json.dumps(cluster_ids, ensure_ascii=False), json.dumps(named, ensure_ascii=False)


def scan_photos_into_db(db: Database, photos_db_path: str | None = None) -> int:
    """Scan Photos library and insert new photos into the database.
    Returns the number of newly inserted photos.
    """
    logger.info("Opening Photos library...")
    photosdb = osxphotos.PhotosDB(dbfile=photos_db_path) if photos_db_path else osxphotos.PhotosDB()

    photos = photosdb.photos(images=True, movies=True)
    logger.info(f"Found {len(photos)} photos in library")

    new_count = 0
    for photo in photos:
        if db.get_photo(photo.uuid):
            continue

        face_cluster_ids, named_faces = _extract_face_info(photo)
        place_info = _extract_place_info(photo)

        source_app = None
        if photo.imported_by and photo.imported_by[1]:
            source_app = photo.imported_by[1]

        db.upsert_photo(
            uuid=photo.uuid,
            file_path=photo.path,
            date_taken=photo.date.isoformat() if photo.date else None,
            gps_lat=photo.latitude,
            gps_lon=photo.longitude,
            apple_labels=json.dumps(photo.labels or [], ensure_ascii=False),
            face_cluster_ids=face_cluster_ids,
            named_faces=named_faces,
            source_app=source_app,
            is_selfie=photo.selfie,
            is_screenshot=photo.screenshot,
            is_live_photo=photo.live_photo,
            **place_info,
        )
        new_count += 1

    logger.info(f"Inserted {new_count} new photos into database")
    return new_count

"""Scan Apple Photos library and populate the progress database."""

import logging

import osxphotos

from photo_memory.db import Database

logger = logging.getLogger(__name__)


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
        db.upsert_photo(
            uuid=photo.uuid,
            file_path=photo.path,
            date_taken=photo.date.isoformat() if photo.date else None,
            gps_lat=photo.latitude,
            gps_lon=photo.longitude,
        )
        new_count += 1

    logger.info(f"Inserted {new_count} new photos into database")
    return new_count

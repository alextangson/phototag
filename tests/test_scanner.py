import pytest
from unittest.mock import patch, MagicMock
from photo_memory.scanner import scan_photos_into_db
from photo_memory.db import Database


def _make_mock_photo(uuid, path, date, lat=None, lon=None):
    photo = MagicMock()
    photo.uuid = uuid
    photo.original_filename = f"{uuid}.jpg"
    photo.path = path
    photo.date = date
    photo.latitude = lat
    photo.longitude = lon
    return photo


def test_scan_inserts_new_photos(tmp_db_path):
    db = Database(tmp_db_path)
    from datetime import datetime
    mock_photos = [
        _make_mock_photo("uuid-1", "/photos/1.jpg", datetime(2024, 1, 1), 30.0, 120.0),
        _make_mock_photo("uuid-2", "/photos/2.jpg", datetime(2024, 6, 15)),
    ]
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_db:
        mock_db.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)
    assert count == 2
    assert db.get_photo("uuid-1")["file_path"] == "/photos/1.jpg"
    assert db.get_photo("uuid-2")["status"] == "pending"
    db.close()


def test_scan_skips_already_existing(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1", file_path="/old.jpg")
    from datetime import datetime
    mock_photos = [
        _make_mock_photo("uuid-1", "/photos/1.jpg", datetime(2024, 1, 1)),
        _make_mock_photo("uuid-2", "/photos/2.jpg", datetime(2024, 6, 15)),
    ]
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_db:
        mock_db.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)
    assert count == 1
    assert db.get_photo("uuid-1")["file_path"] == "/old.jpg"  # not overwritten
    db.close()

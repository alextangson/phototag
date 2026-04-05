import pytest
from unittest.mock import patch, MagicMock
from photo_memory.scanner import scan_photos_into_db
from photo_memory.db import Database


def _make_mock_photo(uuid, path, date, lat=None, lon=None, **kwargs):
    photo = MagicMock()
    photo.uuid = uuid
    photo.original_filename = f"{uuid}.jpg"
    photo.path = path
    photo.date = date
    photo.latitude = lat
    photo.longitude = lon
    photo.labels = kwargs.get("labels", [])
    photo.person_info = kwargs.get("person_info", [])
    photo.selfie = kwargs.get("selfie", False)
    photo.screenshot = kwargs.get("screenshot", False)
    photo.live_photo = kwargs.get("live_photo", False)
    photo.place = kwargs.get("place", None)
    photo.imported_by = kwargs.get("imported_by", (None, None))
    return photo


def _make_mock_person(name, uuid, facecount=10):
    person = MagicMock()
    person.name = name
    person.uuid = uuid
    person.display_name = name if name != "_UNKNOWN_" else None
    person.facecount = facecount
    return person


def _make_mock_place(city="北京市", state=None, country="中国"):
    place = MagicMock()
    addr = MagicMock()
    addr.city = city
    addr.state_province = state
    addr.country = country
    place.address = addr
    return place


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


def test_scan_collects_apple_metadata(tmp_db_path):
    db = Database(tmp_db_path)
    from datetime import datetime
    import json

    person1 = _make_mock_person("唐嘉鑫", "person-uuid-001", facecount=663)
    person2 = _make_mock_person("_UNKNOWN_", "person-uuid-002", facecount=0)
    place = _make_mock_place(city="大连市", state="辽宁省", country="中国")

    mock_photos = [
        _make_mock_photo(
            "uuid-meta", "/photos/meta.jpg", datetime(2024, 9, 28),
            lat=38.9, lon=121.6,
            labels=["人", "牛仔裤", "海边"],
            person_info=[person1, person2],
            selfie=False,
            screenshot=False,
            live_photo=True,
            place=place,
            imported_by=("com.apple.mobileslideshow", "相机"),
        ),
    ]
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_db:
        mock_db.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)

    assert count == 1
    row = db.get_photo("uuid-meta")
    assert json.loads(row["apple_labels"]) == ["人", "牛仔裤", "海边"]
    assert json.loads(row["face_cluster_ids"]) == ["person-uuid-001", "person-uuid-002"]
    assert json.loads(row["named_faces"]) == ["唐嘉鑫"]
    assert row["is_selfie"] == 0
    assert row["is_screenshot"] == 0
    assert row["is_live_photo"] == 1
    assert row["location_city"] == "大连市"
    assert row["location_state"] == "辽宁省"
    assert row["location_country"] == "中国"
    assert row["source_app"] == "相机"
    db.close()


def test_scan_handles_missing_metadata(tmp_db_path):
    db = Database(tmp_db_path)
    from datetime import datetime

    mock_photos = [
        _make_mock_photo("uuid-bare", "/photos/bare.jpg", datetime(2024, 1, 1)),
    ]
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_db:
        mock_db.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)

    assert count == 1
    row = db.get_photo("uuid-bare")
    assert row["apple_labels"] == "[]"
    assert row["face_cluster_ids"] == "[]"
    assert row["named_faces"] == "[]"
    assert row["is_selfie"] == 0
    assert row["location_city"] is None
    db.close()

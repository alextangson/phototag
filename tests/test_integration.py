"""Integration smoke test — runs the full pipeline on mock data."""
import json
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from photo_memory.config import load_config
from photo_memory.db import Database
from photo_memory.scanner import scan_photos_into_db
from photo_memory.processor import process_batch
from photo_memory.dedup import find_duplicate_groups
from photo_memory.load_monitor import LoadMonitor, LoadDecision


@pytest.fixture
def full_setup(tmp_path):
    """Set up a complete test environment with fake photos."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Create 3 images: 2 red (will be duplicates), 1 blue
    for i, color in enumerate(["red", "blue", "red"]):
        img = Image.new("RGB", (100, 100), color=color)
        img.save(img_dir / f"photo_{i}.jpg")

    db = Database(str(tmp_path / "progress.db"))

    from datetime import datetime
    mock_photos = []
    for i in range(3):
        p = MagicMock()
        p.uuid = f"test-uuid-{i}"
        p.path = str(img_dir / f"photo_{i}.jpg")
        p.original_filename = f"photo_{i}.jpg"
        p.date = datetime(2024, 1, i + 1)
        p.latitude = 30.0 + i
        p.longitude = 120.0 + i
        p.labels = []
        p.person_info = []
        p.selfie = False
        p.screenshot = False
        p.live_photo = False
        p.place = None
        p.imported_by = (None, None)
        mock_photos.append(p)

    return db, mock_photos, img_dir, tmp_path


def test_full_pipeline(full_setup):
    db, mock_photos, img_dir, tmp_path = full_setup

    # Step 1: Scan
    with patch("photo_memory.scanner.osxphotos.PhotosDB") as mock_pdb:
        mock_pdb.return_value.photos.return_value = mock_photos
        count = scan_photos_into_db(db)
    assert count == 3

    # Step 2: Process
    ai_result = {
        "narrative": "测试照片",
        "event_hint": "其他",
        "people": [],
        "emotional_tone": "平静",
        "significance": "测试",
        "scene_category": "other",
        "series_hint": "standalone",
        "search_tags": ["人物", "合照"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "keep",
        "duplicate_hint": "standalone",
    }

    mock_monitor = MagicMock()
    mock_monitor.check.return_value = LoadDecision.CONTINUE
    mock_monitor.is_past_deadline.return_value = False

    tmp_dir = str(tmp_path / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with patch("photo_memory.processor.recognize_photo", return_value=ai_result), \
         patch("photo_memory.processor.apply_tags_to_photo"):
        stats = process_batch(db, mock_monitor, check_interval=10,
                              ollama_config={"host": "h", "model": "m", "timeout": 60},
                              tmp_dir=tmp_dir, end_hour=7)

    assert stats["processed"] == 3
    assert stats["stop_reason"] == "completed"

    # Step 3: Dedup
    phashes = db.get_all_phashes()
    assert len(phashes) == 3
    groups = find_duplicate_groups(phashes, threshold=5)
    # photo_0 and photo_2 are both red -> same hash -> 1 duplicate group
    assert len(groups) >= 1

    db.close()

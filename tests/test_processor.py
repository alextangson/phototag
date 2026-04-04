import json
import os
import pytest
from unittest.mock import patch, MagicMock, ANY
from photo_memory.processor import process_batch, process_one_photo
from photo_memory.db import Database
from photo_memory.load_monitor import LoadDecision


def test_process_one_photo_success(tmp_db_path, tmp_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1", file_path="/photos/1.jpg")

    from PIL import Image
    fake_img = tmp_path / "uuid-1.jpg"
    Image.new("RGB", (100, 100), "red").save(fake_img)

    ai_result = {
        "description": "一只猫",
        "tags": ["宠物/猫"],
        "media_type": "photo",
        "scene": "indoor",
        "importance": "low",
        "has_text": False,
        "text_summary": "",
    }

    with patch("photo_memory.processor.export_photo", return_value=str(fake_img)), \
         patch("photo_memory.processor.recognize_photo", return_value=ai_result), \
         patch("photo_memory.processor.apply_tags_to_photo"), \
         patch("photo_memory.processor.compute_phash", return_value="abcdef1234567890"):
        success = process_one_photo(db, db.get_photo("uuid-1"),
                                    ollama_config={"host": "h", "model": "m", "timeout": 60},
                                    tmp_dir=str(tmp_path))

    assert success is True
    row = db.get_photo("uuid-1")
    assert row["status"] == "done"
    assert row["phash"] == "abcdef1234567890"
    assert "一只猫" in row["description"]
    db.close()


def test_process_one_photo_ai_error(tmp_db_path, tmp_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1", file_path="/photos/1.jpg")

    from PIL import Image
    fake_img = tmp_path / "uuid-1.jpg"
    Image.new("RGB", (100, 100), "red").save(fake_img)

    with patch("photo_memory.processor.export_photo", return_value=str(fake_img)), \
         patch("photo_memory.processor.recognize_photo", side_effect=Exception("Ollama down")), \
         patch("photo_memory.processor.compute_phash", return_value="aabb"):
        success = process_one_photo(db, db.get_photo("uuid-1"),
                                    ollama_config={"host": "h", "model": "m", "timeout": 60},
                                    tmp_dir=str(tmp_path))

    assert success is False
    row = db.get_photo("uuid-1")
    assert row["status"] == "error"
    assert "Ollama down" in row["error_msg"]
    db.close()


def test_process_batch_respects_load_pause(tmp_db_path):
    db = Database(tmp_db_path)
    for i in range(5):
        db.upsert_photo(f"uuid-{i}", file_path=f"/photos/{i}.jpg")

    mock_monitor = MagicMock()
    mock_monitor.check.side_effect = [LoadDecision.CONTINUE, LoadDecision.PAUSE]
    mock_monitor.is_past_deadline.return_value = False

    with patch("photo_memory.processor.process_one_photo", return_value=True) as mock_proc:
        stats = process_batch(db, mock_monitor, check_interval=2,
                              ollama_config={"host": "h", "model": "m", "timeout": 60},
                              tmp_dir="/tmp", end_hour=7)

    assert mock_proc.call_count == 2
    assert stats["processed"] == 2
    assert stats["stop_reason"] == "load_limit"
    db.close()


def test_process_batch_respects_deadline(tmp_db_path):
    db = Database(tmp_db_path)
    db.upsert_photo("uuid-1", file_path="/photos/1.jpg")

    mock_monitor = MagicMock()
    mock_monitor.check.return_value = LoadDecision.CONTINUE
    mock_monitor.is_past_deadline.return_value = True

    with patch("photo_memory.processor.process_one_photo", return_value=True):
        stats = process_batch(db, mock_monitor, check_interval=10,
                              ollama_config={"host": "h", "model": "m", "timeout": 60},
                              tmp_dir="/tmp", end_hour=7)

    assert stats["stop_reason"] == "time_limit"
    db.close()

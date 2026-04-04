import json
import os
import pytest
from unittest.mock import patch, MagicMock, ANY
from photo_memory.processor import process_batch, process_one_photo
from photo_memory.db import Database
from photo_memory.load_monitor import LoadDecision
import shutil


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


def test_process_one_video_success(tmp_db_path, tmp_path):
    db = Database(tmp_db_path)
    db.upsert_photo("vid-1", file_path="/videos/1.mov")

    # Create a fake video file (just needs the extension)
    fake_vid = tmp_path / "vid-1.mov"
    fake_vid.touch()

    # Create fake frames
    from PIL import Image
    frames_dir = tmp_path / "frames_vid-1"
    frames_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (100, 100), "green").save(frames_dir / f"frame_{i+1:04d}.jpg")
    frame_paths = sorted([str(f) for f in frames_dir.glob("frame_*.jpg")])

    ai_result = {
        "description": "风景视频",
        "tags": ["自然", "湖泊"],
        "people_count": 0,
        "animals": [],
        "objects": [],
        "location_type": "outdoor",
        "activity": "",
        "mood": "平静",
        "time_of_day": "白天",
        "media_type": "video_frame",
        "scene_type": "B-roll",
        "importance": "medium",
        "has_text": False,
        "text_summary": "",
        "colors": ["蓝"],
        "quality_notes": "清晰",
    }

    with patch("photo_memory.processor.export_photo", return_value=str(fake_vid)), \
         patch("photo_memory.processor._is_video", return_value=True), \
         patch("photo_memory.processor.extract_frames", return_value=frame_paths), \
         patch("photo_memory.processor.extract_audio", return_value=str(tmp_path / "audio.wav")), \
         patch("photo_memory.processor.transcribe_audio", return_value={"text": "测试语音", "language": "zh"}), \
         patch("photo_memory.processor.recognize_photo", return_value=ai_result), \
         patch("photo_memory.processor.apply_tags_to_photo"), \
         patch("photo_memory.processor.compute_phash", return_value="1234567890abcdef"), \
         patch("photo_memory.processor.summarize_video_frames") as mock_summarize, \
         patch("shutil.rmtree"):
        mock_summarize.return_value = {**ai_result, "text_summary": "测试语音", "tags": ["自然", "湖泊", "有语音"]}
        success = process_one_photo(db, db.get_photo("vid-1"),
                                    ollama_config={"host": "h", "model": "m", "timeout": 60},
                                    tmp_dir=str(tmp_path))

    assert success is True
    row = db.get_photo("vid-1")
    assert row["status"] == "done"
    mock_summarize.assert_called_once()
    db.close()

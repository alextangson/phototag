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
        "narrative": "一只猫",
        "search_tags": ["宠物/猫"],
        "scene_category": "photo",
        "cleanup_class": "keep",
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


def test_export_photo_converts_heic(tmp_path):
    """HEIC files should be converted to JPEG via sips."""
    from photo_memory.processor import export_photo

    heic_file = tmp_path / "test.heic"
    heic_file.write_bytes(b"fake heic data")

    with patch("photo_memory.processor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = export_photo("uuid-heic", str(heic_file), str(tmp_path))

    assert result.endswith(".jpg")
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "sips"
    assert "jpeg" in call_args


def test_export_photo_heic_failure_returns_none(tmp_path):
    """If sips fails, export_photo should return None."""
    import subprocess as sp
    from photo_memory.processor import export_photo

    heic_file = tmp_path / "bad.heic"
    heic_file.write_bytes(b"fake")

    with patch("photo_memory.processor.subprocess.run", side_effect=sp.CalledProcessError(1, "sips")):
        result = export_photo("uuid-bad", str(heic_file), str(tmp_path))

    assert result is None


def test_export_photo_copies_jpeg(tmp_path):
    """Non-HEIC files should just be copied."""
    from photo_memory.processor import export_photo
    from PIL import Image

    jpg_file = tmp_path / "src" / "test.jpg"
    jpg_file.parent.mkdir()
    Image.new("RGB", (10, 10), "red").save(jpg_file)

    result = export_photo("uuid-jpg", str(jpg_file), str(tmp_path))
    assert result.endswith(".jpg")
    assert os.path.exists(result)


def test_process_one_photo_passes_context(tmp_path, tmp_db_path):
    """Verify process_one_photo builds photo_context and passes it to recognizer."""
    from photo_memory.db import Database
    from photo_memory.processor import process_one_photo
    from unittest.mock import patch
    import json

    db = Database(tmp_db_path)
    db.upsert_photo("uuid-ctx",
        file_path=str(tmp_path / "test.jpg"),
        date_taken="2024-09-28T13:33:00",
        location_city="大连市",
        apple_labels='["人"]',
        face_cluster_ids='["fc_001"]',
        named_faces='{"fc_001": "唐嘉鑫"}',
        is_selfie=False,
        is_screenshot=False,
        is_live_photo=False,
    )

    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img.save(tmp_path / "test.jpg")

    ai_result = {
        "narrative": "测试",
        "event_hint": "其他",
        "people": [],
        "emotional_tone": "平静",
        "significance": "测试",
        "scene_category": "other",
        "series_hint": "standalone",
        "search_tags": ["测试"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "keep",
        "duplicate_hint": "standalone",
    }

    with patch("photo_memory.processor.recognize_photo", return_value=ai_result) as mock_rec, \
         patch("photo_memory.processor.apply_tags_to_photo"), \
         patch("photo_memory.processor.compute_phash", return_value="aabb"):
        photo_row = db.get_photo("uuid-ctx")
        result = process_one_photo(db, photo_row, {"host": "h", "model": "m", "timeout": 60}, str(tmp_path))

    assert result is True
    # Verify photo_context was passed as keyword argument
    call_kwargs = mock_rec.call_args[1]
    assert "photo_context" in call_kwargs
    ctx = call_kwargs["photo_context"]
    assert ctx["location_city"] == "大连市"
    assert ctx["named_faces"] == ["唐嘉鑫"]
    db.close()

import json
import pytest
from unittest.mock import patch, MagicMock
from photo_memory.recognizer import recognize_photo, parse_ai_response, RECOGNITION_PROMPT, summarize_video_frames


def test_parse_valid_json_response():
    raw = json.dumps({
        "description": "一张猫的照片",
        "tags": ["宠物/猫"],
        "media_type": "photo",
        "scene": "indoor",
        "importance": "low",
        "has_text": False,
        "text_summary": "",
    })
    result = parse_ai_response(raw)
    assert result["description"] == "一张猫的照片"
    assert result["tags"] == ["宠物/猫"]
    assert result["importance"] == "low"


def test_parse_json_embedded_in_markdown():
    raw = """Here's the analysis:
```json
{"description": "一张风景照", "tags": ["风景/自然"], "media_type": "photo", "scene": "outdoor", "importance": "medium", "has_text": false, "text_summary": ""}
```"""
    result = parse_ai_response(raw)
    assert result["description"] == "一张风景照"


def test_parse_invalid_json_returns_fallback():
    raw = "This is not valid JSON at all"
    result = parse_ai_response(raw)
    assert result["description"] == raw[:200]
    assert result["tags"] == ["其他"]
    assert result["importance"] == "low"


def test_recognize_photo_calls_ollama(tmp_path):
    # Create a small test image
    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps({
            "description": "测试照片",
            "tags": ["人物/自拍"],
            "media_type": "photo",
            "scene": "indoor",
            "importance": "medium",
            "has_text": False,
            "text_summary": "",
        })
    }
    with patch("photo_memory.recognizer.requests.post", return_value=mock_response) as mock_post:
        result = recognize_photo(
            str(img_path),
            host="http://localhost:11434",
            model="gemma4:e4b",
            timeout=60,
        )
    assert result["description"] == "测试照片"
    mock_post.assert_called_once()
    call_json = mock_post.call_args[1]["json"]
    assert call_json["model"] == "gemma4:e4b"
    assert "images" in call_json


def test_recognize_photo_retries_on_bad_json(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    good_response = {
        "description": "重试成功",
        "tags": ["其他"],
        "media_type": "photo",
        "scene": "outdoor",
        "importance": "low",
        "has_text": False,
        "text_summary": "",
    }
    mock_resp_bad = MagicMock()
    mock_resp_bad.status_code = 200
    mock_resp_bad.json.return_value = {"response": "not json"}

    mock_resp_good = MagicMock()
    mock_resp_good.status_code = 200
    mock_resp_good.json.return_value = {"response": json.dumps(good_response)}

    with patch("photo_memory.recognizer.requests.post", side_effect=[mock_resp_bad, mock_resp_good]):
        result = recognize_photo(str(img_path), host="http://localhost:11434",
                                 model="gemma4:e4b", timeout=60)
    assert result["description"] == "重试成功"


def test_summarize_video_frames():
    frames = [
        {"description": "风景画面", "tags": ["自然", "湖泊"], "scene_type": "B-roll",
         "people_count": 0, "animals": [], "objects": ["山"], "location_type": "outdoor",
         "activity": "", "mood": "平静", "time_of_day": "白天", "media_type": "video_frame",
         "importance": "medium", "has_text": False, "text_summary": "", "colors": ["蓝"], "quality_notes": "清晰"},
        {"description": "湖面", "tags": ["自然", "水面", "倒影"], "scene_type": "B-roll",
         "people_count": 0, "animals": [], "objects": ["湖"], "location_type": "outdoor",
         "activity": "", "mood": "平静", "time_of_day": "白天", "media_type": "video_frame",
         "importance": "medium", "has_text": False, "text_summary": "", "colors": ["蓝"], "quality_notes": "清晰"},
    ]
    result = summarize_video_frames(frames, transcript="")
    assert "自然" in result["tags"]
    assert "湖泊" in result["tags"]
    assert "水面" in result["tags"]  # merged from frame 2
    assert result["scene_type"] == "B-roll"
    assert result["importance"] == "medium"


def test_summarize_video_frames_with_transcript():
    frames = [
        {"description": "人在说话", "tags": ["人物", "口播"], "scene_type": "口播",
         "people_count": 1, "animals": [], "objects": [], "location_type": "indoor",
         "activity": "说话", "mood": "平静", "time_of_day": "室内无法判断", "media_type": "video_frame",
         "importance": "medium", "has_text": False, "text_summary": "", "colors": [], "quality_notes": "清晰"},
    ]
    result = summarize_video_frames(frames, transcript="今天聊一下AI创业的几个关键点")
    assert "有语音" in result["tags"]
    assert "AI创业" in result["text_summary"]
    assert result["importance"] == "high"  # has meaningful speech

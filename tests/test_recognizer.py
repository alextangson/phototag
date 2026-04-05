import json
import pytest
from unittest.mock import patch, MagicMock
from photo_memory.recognizer import recognize_photo, parse_ai_response, summarize_video_frames


def _make_new_format_json(**overrides):
    """Helper to create a valid new-format AI response."""
    base = {
        "narrative": "测试照片",
        "event_hint": "其他",
        "people": [],
        "emotional_tone": "平静",
        "significance": "普通记录",
        "scene_category": "other",
        "series_hint": "standalone",
        "search_tags": ["测试"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "keep",
        "duplicate_hint": "standalone",
    }
    base.update(overrides)
    return base


def test_parse_valid_json_response():
    raw = json.dumps(_make_new_format_json(
        narrative="一张猫的照片",
        search_tags=["宠物", "猫"],
        cleanup_class="keep",
    ))
    result = parse_ai_response(raw)
    assert result["narrative"] == "一张猫的照片"
    assert result["search_tags"] == ["宠物", "猫"]
    assert result["cleanup_class"] == "keep"


def test_parse_json_embedded_in_markdown():
    data = _make_new_format_json(narrative="一张风景照", search_tags=["风景", "自然"])
    raw = f"""Here's the analysis:
```json
{json.dumps(data)}
```"""
    result = parse_ai_response(raw)
    assert result["narrative"] == "一张风景照"


def test_parse_invalid_json_returns_fallback():
    raw = "This is not valid JSON at all"
    result = parse_ai_response(raw)
    assert result["narrative"] == raw[:200]
    assert result["search_tags"] == ["其他"]
    assert result["cleanup_class"] == "review"


def test_recognize_photo_calls_ollama(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps(_make_new_format_json(narrative="测试照片"))
    }
    with patch("photo_memory.recognizer.requests.post", return_value=mock_response) as mock_post:
        result = recognize_photo(str(img_path), host="http://localhost:11434",
                                 model="gemma4:e4b", timeout=60)
    assert result["narrative"] == "测试照片"
    mock_post.assert_called_once()
    call_json = mock_post.call_args[1]["json"]
    assert call_json["model"] == "gemma4:e4b"
    assert "images" in call_json


def test_recognize_photo_retries_on_bad_json(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    good_response = _make_new_format_json(narrative="重试成功")
    mock_resp_bad = MagicMock()
    mock_resp_bad.status_code = 200
    mock_resp_bad.json.return_value = {"response": "not json"}

    mock_resp_good = MagicMock()
    mock_resp_good.status_code = 200
    mock_resp_good.json.return_value = {"response": json.dumps(good_response)}

    with patch("photo_memory.recognizer.requests.post", side_effect=[mock_resp_bad, mock_resp_good]):
        result = recognize_photo(str(img_path), host="http://localhost:11434",
                                 model="gemma4:e4b", timeout=60)
    assert result["narrative"] == "重试成功"


def test_summarize_video_frames():
    frames = [
        _make_new_format_json(
            narrative="风景画面",
            search_tags=["自然", "湖泊"],
            scene_category="outdoor_travel",
        ),
        _make_new_format_json(
            narrative="湖面",
            search_tags=["自然", "水面", "倒影"],
            scene_category="outdoor_travel",
        ),
    ]
    result = summarize_video_frames(frames, transcript="")
    assert "自然" in result["search_tags"]
    assert "湖泊" in result["search_tags"]
    assert "水面" in result["search_tags"]
    assert result["scene_category"] == "outdoor_travel"
    assert result["cleanup_class"] == "keep"


def test_summarize_video_frames_with_transcript():
    frames = [
        _make_new_format_json(
            narrative="人在说话",
            search_tags=["人物", "口播"],
            scene_category="indoor_home",
        ),
    ]
    result = summarize_video_frames(frames, transcript="今天聊一下AI创业的几个关键点")
    assert "有语音" in result["search_tags"]
    assert "AI创业" in result["text_summary"]
    assert result["cleanup_class"] == "keep"


# === New tests for Task 3 ===

def test_parse_new_format_response():
    raw = json.dumps({
        "narrative": "周六下午和女友在大连海边散步的合影",
        "event_hint": "出游/约会",
        "people": [
            {"face_cluster_id": "fc_001", "description": "戴眼镜的年轻男性"},
        ],
        "emotional_tone": "轻松愉快",
        "significance": "日常约会记录",
        "scene_category": "outdoor_leisure",
        "series_hint": "standalone",
        "search_tags": ["海边", "合影", "散步"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "keep",
        "duplicate_hint": "standalone",
    })
    result = parse_ai_response(raw)
    assert result["narrative"] == "周六下午和女友在大连海边散步的合影"
    assert result["search_tags"] == ["海边", "合影", "散步"]
    assert result["cleanup_class"] == "keep"


def test_build_photo_context():
    from photo_memory.recognizer import build_photo_context

    photo_row = {
        "uuid": "test-uuid",
        "date_taken": "2024-09-28T13:33:00",
        "location_city": "大连市",
        "location_state": "辽宁省",
        "location_country": "中国",
        "apple_labels": '["人", "牛仔裤"]',
        "named_faces": '["唐嘉鑫"]',
        "face_cluster_ids": '["fc_001", "fc_002"]',
        "is_selfie": 0,
        "is_screenshot": 0,
        "is_live_photo": 1,
        "source_app": "相机",
    }
    ctx = build_photo_context(photo_row)
    assert ctx["location_city"] == "大连市"
    assert ctx["named_faces"] == ["唐嘉鑫"]
    assert ctx["apple_labels"] == ["人", "牛仔裤"]
    assert ctx["is_selfie"] is False
    assert ctx["is_live_photo"] is True


def test_recognize_photo_with_context(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps(_make_new_format_json(narrative="带上下文的测试"))
    }

    photo_context = {"date": "2024-09-28 周六 13:33", "location_city": "大连市"}

    with patch("photo_memory.recognizer.requests.post", return_value=mock_response) as mock_post:
        result = recognize_photo(str(img_path), host="http://localhost:11434",
                                 model="gemma4:e4b", timeout=60,
                                 photo_context=photo_context)

    assert result["narrative"] == "带上下文的测试"
    call_json = mock_post.call_args[1]["json"]
    assert "大连市" in call_json["prompt"]

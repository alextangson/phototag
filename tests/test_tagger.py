import pytest
from unittest.mock import patch, MagicMock
from photo_memory.tagger import get_album_name, get_special_album, apply_tags_to_photo


def test_get_album_name_extracts_first_level():
    assert get_album_name("人物/合照") == "AI-人物"
    assert get_album_name("美食/餐厅") == "AI-美食"
    assert get_album_name("其他") == "AI-其他"


def test_get_special_album_important_screenshot():
    result = {
        "tags": ["截屏/手机截屏"],
        "importance": "high",
        "media_type": "screenshot",
    }
    assert get_special_album(result) == "AI-重要截图"


def test_get_special_album_low_importance_chat():
    result = {
        "tags": ["聊天记录/微信"],
        "importance": "low",
        "media_type": "screenshot",
    }
    assert get_special_album(result) == "AI-截图待清理"


def test_get_special_album_normal_photo():
    result = {
        "tags": ["风景/自然"],
        "importance": "medium",
        "media_type": "photo",
    }
    assert get_special_album(result) is None


def test_apply_tags_calls_expected_operations():
    """Test that apply_tags attempts to set keywords, description, and albums."""
    result = {
        "description": "测试照片",
        "tags": ["人物/合照", "美食/餐厅"],
        "importance": "medium",
        "media_type": "photo",
    }
    with patch("photo_memory.tagger._set_keywords") as mock_kw, \
         patch("photo_memory.tagger._set_description") as mock_desc, \
         patch("photo_memory.tagger._add_to_album") as mock_album:
        apply_tags_to_photo("test-uuid", result)

    mock_kw.assert_called_once_with("test-uuid", ["人物/合照", "美食/餐厅"])
    mock_desc.assert_called_once_with("test-uuid", "测试照片")
    # Should be called for AI-人物 and AI-美食
    album_names = {call.args[1] for call in mock_album.call_args_list}
    assert "AI-人物" in album_names
    assert "AI-美食" in album_names


def test_sanitize_applescript_string():
    from photo_memory.tagger import _sanitize_for_applescript
    assert _sanitize_for_applescript("hello") == "hello"
    assert _sanitize_for_applescript('say "hi"') == 'say \\"hi\\"'
    assert _sanitize_for_applescript("a\\b") == "a\\\\b"
    assert _sanitize_for_applescript("line1\nline2\ttab") == "line1 line2 tab"

"""Tests for HTML story renderer."""
import base64
from PIL import Image
from photo_memory.html_renderer import (
    _select_photos_for_event,
    _encode_thumbnail_b64,
    render_story_html,
)


def test_select_photos_small_event_returns_all(tmp_path):
    photos = [
        {"uuid": "p1", "file_path": str(tmp_path / "1.jpg")},
        {"uuid": "p2", "file_path": str(tmp_path / "2.jpg")},
    ]
    selected = _select_photos_for_event(photos, max_photos=5)
    assert len(selected) == 2


def test_select_photos_large_event_picks_first_middle_last(tmp_path):
    photos = [{"uuid": f"p{i}", "file_path": str(tmp_path / f"{i}.jpg")} for i in range(10)]
    selected = _select_photos_for_event(photos, max_photos=3)
    assert len(selected) == 3
    assert selected[0]["uuid"] == "p0"
    assert selected[-1]["uuid"] == "p9"


def test_encode_thumbnail_b64_returns_data_uri(tmp_path):
    img = Image.new("RGB", (800, 600), "red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path, "JPEG")

    data_uri = _encode_thumbnail_b64(str(img_path), max_width=400)
    assert data_uri.startswith("data:image/jpeg;base64,")
    b64_part = data_uri.split(",", 1)[1]
    raw = base64.b64decode(b64_part)
    assert len(raw) > 100
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_encode_thumbnail_handles_missing_file():
    result = _encode_thumbnail_b64("/nonexistent/path.jpg")
    assert result is None


def test_render_story_html_includes_title_and_events(tmp_path):
    img_path = tmp_path / "cover.jpg"
    Image.new("RGB", (100, 100), "blue").save(img_path, "JPEG")

    events_with_photos = [
        {
            "event_id": "evt_1",
            "start_time": "2024-03-15T10:00:00",
            "location_city": "北京",
            "summary": "颐和园散步",
            "mood": "愉快",
            "photos": [
                {"uuid": "p1", "file_path": str(img_path)},
            ],
        },
    ]

    html = render_story_html(
        title="2024 年度回忆",
        stats_line="共 5 张照片，1 个事件",
        intro_narrative="这一年很开心",
        events_with_photos=events_with_photos,
    )

    assert "<!DOCTYPE html>" in html
    assert "2024 年度回忆" in html
    assert "颐和园散步" in html
    assert "北京" in html
    assert "这一年很开心" in html
    assert 'data:image/jpeg;base64,' in html
    # Self-contained — no external http image refs
    assert 'src="http' not in html

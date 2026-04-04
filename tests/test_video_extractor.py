# tests/test_video_extractor.py
import os
import pytest
from unittest.mock import patch, MagicMock
from photo_memory.video_extractor import (
    extract_frames, extract_audio, transcribe_audio, get_video_duration,
)


def test_extract_frames_calls_ffmpeg(tmp_path):
    """Test that extract_frames calls ffmpeg with correct arguments."""
    with patch("photo_memory.video_extractor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        # Create fake frame files so glob finds them
        for i in range(3):
            (tmp_path / f"frame_{i+1:04d}.jpg").touch()

        frames = extract_frames("/fake/video.mp4", str(tmp_path), fps=0.2)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "/fake/video.mp4" in args
    assert len(frames) == 3


def test_extract_frames_returns_empty_on_failure(tmp_path):
    """Test graceful failure when ffmpeg fails."""
    import subprocess
    with patch("photo_memory.video_extractor.subprocess.run",
               side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error")):
        frames = extract_frames("/fake/video.mp4", str(tmp_path))
    assert frames == []


def test_extract_audio_success(tmp_path):
    out_path = str(tmp_path / "audio.wav")
    with patch("photo_memory.video_extractor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = extract_audio("/fake/video.mp4", out_path)
    assert result == out_path


def test_extract_audio_failure(tmp_path):
    import subprocess
    with patch("photo_memory.video_extractor.subprocess.run",
               side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
        result = extract_audio("/fake/video.mp4", str(tmp_path / "audio.wav"))
    assert result is None


def test_transcribe_audio():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Hello world", "language": "en"}
    with patch("photo_memory.video_extractor._get_whisper_model", return_value=mock_model):
        result = transcribe_audio("/fake/audio.wav")
    assert result["text"] == "Hello world"
    assert result["language"] == "en"


def test_transcribe_audio_empty():
    """Test when there's no speech in the audio."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "", "language": "en"}
    with patch("photo_memory.video_extractor._get_whisper_model", return_value=mock_model):
        result = transcribe_audio("/fake/audio.wav")
    assert result["text"] == ""


def test_get_video_duration():
    with patch("photo_memory.video_extractor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="30.5\n")
        duration = get_video_duration("/fake/video.mp4")
    assert duration == 30.5

# src/photo_memory/video_extractor.py
"""Video analysis: frame extraction via FFmpeg + audio transcription via Whisper."""

import logging
import os
import subprocess
import glob

logger = logging.getLogger(__name__)

# Lazy-loaded whisper model
_whisper_model = None


def _get_whisper_model(model_size: str = "base"):
    """Lazy-load whisper model to avoid loading on import."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def extract_frames(video_path: str, output_dir: str, fps: float = 0.2) -> list[str]:
    """Extract frames from video using FFmpeg.

    Args:
        video_path: path to video file
        output_dir: directory to save frames
        fps: frames per second to extract (0.2 = 1 frame every 5 seconds)

    Returns:
        sorted list of extracted frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%04d.jpg")

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={fps},scale=512:-1",
                "-q:v", "2",
                pattern,
                "-y",
            ],
            capture_output=True, timeout=120, check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg frame extraction failed: {e.stderr[:200] if e.stderr else ''}")
        return []
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timed out for {video_path}")
        return []

    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def extract_audio(video_path: str, output_path: str) -> str | None:
    """Extract audio from video as 16kHz mono WAV for Whisper.

    Returns output_path on success, None on failure.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                output_path,
                "-y",
            ],
            capture_output=True, timeout=60, check=True,
        )
        return output_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Audio extraction failed for {video_path}: {e}")
        return None


def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    """Transcribe audio using Whisper.

    Returns:
        {"text": str, "language": str}
    """
    try:
        model = _get_whisper_model(model_size)
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        logger.info(f"Transcribed {len(text)} chars, language: {language}")
        return {"text": text, "language": language}
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return {"text": "", "language": "unknown"}


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

"""Main processing loop: export -> hash -> recognize -> tag."""

ACTUAL_MIN_TIMEOUT = 180  # Hard minimum for Ollama vision requests

import json
import logging
import os
import subprocess

from photo_memory.db import Database
from photo_memory.dedup import compute_phash
from photo_memory.load_monitor import LoadDecision, LoadMonitor
from photo_memory.recognizer import recognize_photo, summarize_video_frames, build_photo_context
from photo_memory.tagger import apply_tags_to_photo
from photo_memory.video_extractor import extract_frames, extract_audio, transcribe_audio

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def _is_video(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS


HEIC_EXTENSIONS = {".heic", ".heif"}


def export_photo(uuid: str, file_path: str, tmp_dir: str) -> str | None:
    """Export a photo to a temporary directory.

    For iCloud photos, osxphotos can trigger download. For now we just
    check if the file exists at file_path, and if so copy it to tmp_dir.
    HEIC/HEIF files are converted to JPEG via macOS sips.
    If not accessible, return None.
    """
    if not file_path or not os.path.isfile(file_path):
        logger.warning(f"Photo file not accessible: {file_path}")
        return None

    import shutil
    ext = os.path.splitext(file_path)[1].lower() or ".jpg"

    if ext in HEIC_EXTENSIONS:
        dst = os.path.join(tmp_dir, f"{uuid}.jpg")
        try:
            subprocess.run(
                ["sips", "-s", "format", "jpeg", file_path, "--out", dst],
                capture_output=True, timeout=30, check=True,
            )
            return dst
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"HEIC conversion failed for {uuid}: {e}")
            return None
    else:
        dst = os.path.join(tmp_dir, f"{uuid}{ext}")
        shutil.copy2(file_path, dst)
        return dst


def process_one_photo(db: Database, photo_row: dict, ollama_config: dict,
                      tmp_dir: str) -> bool:
    """Process a single photo: export, hash, recognize, tag.
    Returns True on success, False on error.
    """
    uuid = photo_row["uuid"]
    db.update_photo_status(uuid, "processing")

    exported_path = None
    try:
        exported_path = export_photo(uuid, photo_row.get("file_path", ""), tmp_dir)
        if not exported_path:
            db.update_photo_status(uuid, "error", error_msg="Export failed - file not accessible")
            return False

        if exported_path and _is_video(exported_path):
            # Video processing: extract frames + audio
            import shutil
            frames_dir = os.path.join(tmp_dir, f"frames_{uuid}")
            frames = extract_frames(exported_path, frames_dir)

            if not frames:
                db.update_photo_status(uuid, "error", error_msg="Frame extraction failed")
                return False

            # Compute phash from first frame
            phash = compute_phash(frames[0])

            # Recognize each frame
            frame_results = []
            for frame_path in frames[:6]:  # max 6 frames to avoid overloading
                try:
                    frame_result = recognize_photo(
                        frame_path,
                        host=ollama_config["host"],
                        model=ollama_config["model"],
                        timeout=ollama_config["timeout"],
                    )
                    frame_results.append(frame_result)
                except Exception as e:
                    logger.warning(f"Frame recognition failed: {e}")

            # Transcribe audio
            audio_path = os.path.join(tmp_dir, f"audio_{uuid}.wav")
            transcript = ""
            audio_file = extract_audio(exported_path, audio_path)
            if audio_file:
                whisper_result = transcribe_audio(audio_file)
                transcript = whisper_result.get("text", "")
                # Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

            # Summarize
            result = summarize_video_frames(frame_results, transcript)

            # Clean up frames
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
        else:
            # Photo processing
            phash = compute_phash(exported_path)
            photo_context = build_photo_context(photo_row)
            actual_timeout = max(ollama_config.get("timeout", 180), ACTUAL_MIN_TIMEOUT)
            logger.info(f"Processing {uuid[:8]} timeout={actual_timeout}")
            result = recognize_photo(
                exported_path,
                host=ollama_config["host"],
                model=ollama_config["model"],
                timeout=actual_timeout,
                photo_context=photo_context,
            )

        try:
            apply_tags_to_photo(uuid, result)
        except Exception as e:
            logger.warning(f"Tag write-back failed for {uuid}: {e}")

        db.update_photo_result(
            uuid=uuid,
            status="done",
            phash=phash,
            ai_result=json.dumps(result, ensure_ascii=False),
            tags=json.dumps(result.get("search_tags", []), ensure_ascii=False),
            description=result.get("narrative", ""),
            importance=result.get("cleanup_class", "keep"),
            media_type=result.get("scene_category", "photo"),
        )
        return True

    except Exception as e:
        logger.error(f"Error processing {uuid}: {e}")
        db.update_photo_status(uuid, "error", error_msg=str(e))
        return False
    finally:
        if exported_path and os.path.exists(exported_path):
            os.remove(exported_path)


def process_batch(db: Database, monitor: LoadMonitor, check_interval: int,
                  ollama_config: dict, tmp_dir: str, end_hour: int,
                  start_hour: int = 1) -> dict:
    """Process a batch of pending photos with adaptive load control.
    Returns stats dict with processed, skipped, errored, stop_reason.
    """
    stats = {"processed": 0, "skipped": 0, "errored": 0, "stop_reason": "completed"}

    if monitor.is_past_deadline(end_hour, start_hour):
        stats["stop_reason"] = "time_limit"
        return stats

    decision = monitor.check()
    if decision == LoadDecision.STOP:
        stats["stop_reason"] = "load_limit"
        return stats

    while True:
        pending = db.get_pending_photos(limit=check_interval)
        if not pending:
            stats["stop_reason"] = "completed"
            break

        for photo_row in pending:
            if monitor.is_past_deadline(end_hour, start_hour):
                stats["stop_reason"] = "time_limit"
                return stats

            success = process_one_photo(db, photo_row, ollama_config, tmp_dir)
            if success:
                stats["processed"] += 1
            else:
                stats["errored"] += 1

        decision = monitor.check()
        if decision == LoadDecision.STOP:
            stats["stop_reason"] = "load_limit"
            break
        elif decision == LoadDecision.PAUSE:
            logger.info("System load high, stopping this session")
            stats["stop_reason"] = "load_limit"
            break

    return stats

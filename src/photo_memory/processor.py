"""Main processing loop: export -> hash -> recognize -> tag."""

import json
import logging
import os

from photo_memory.db import Database
from photo_memory.dedup import compute_phash
from photo_memory.load_monitor import LoadDecision, LoadMonitor
from photo_memory.recognizer import recognize_photo
from photo_memory.tagger import apply_tags_to_photo

logger = logging.getLogger(__name__)


def export_photo(uuid: str, file_path: str, tmp_dir: str) -> str | None:
    """Export a photo to a temporary directory.

    For iCloud photos, osxphotos can trigger download. For now we just
    check if the file exists at file_path, and if so copy it to tmp_dir.
    If not accessible, return None.
    """
    if not file_path or not os.path.isfile(file_path):
        logger.warning(f"Photo file not accessible: {file_path}")
        return None

    import shutil
    ext = os.path.splitext(file_path)[1] or ".jpg"
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

        phash = compute_phash(exported_path)

        result = recognize_photo(
            exported_path,
            host=ollama_config["host"],
            model=ollama_config["model"],
            timeout=ollama_config["timeout"],
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
            tags=json.dumps(result.get("tags", []), ensure_ascii=False),
            description=result.get("description", ""),
            importance=result.get("importance", "low"),
            media_type=result.get("media_type", "photo"),
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
                  ollama_config: dict, tmp_dir: str, end_hour: int) -> dict:
    """Process a batch of pending photos with adaptive load control.
    Returns stats dict with processed, skipped, errored, stop_reason.
    """
    stats = {"processed": 0, "skipped": 0, "errored": 0, "stop_reason": "completed"}

    if monitor.is_past_deadline(end_hour):
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
            if monitor.is_past_deadline(end_hour):
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

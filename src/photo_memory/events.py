"""Event aggregation: group photos into semantic events via time + face signals."""

import json
import logging
import re

import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _extract_json_object(raw: str) -> dict | None:
    """Extract a JSON object from raw LLM output, tolerating markdown/prose wrap.

    Returns the parsed dict, or None if no valid object found.
    """
    if not raw:
        return None
    # Direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Markdown code block: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Greedy {...} match for wrapping prose
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _parse_date(s: str | None) -> datetime | None:
    """Parse ISO datetime string, return None on failure."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def slice_into_events(photos: list[dict], gap_minutes: int = 30) -> list[dict]:
    """Slice a time-ordered list of photos into events by time gap.

    Args:
        photos: list of photo row dicts, must be sorted by date_taken ASC.
                Photos without valid date_taken are skipped.
        gap_minutes: if adjacent photos are >gap_minutes apart, split into new event.

    Returns:
        list of event dicts with keys: photos, start_time, end_time
    """
    events = []
    current_photos = []
    current_start = None
    current_end = None

    for photo in photos:
        dt = _parse_date(photo.get("date_taken"))
        if dt is None:
            continue

        if not current_photos:
            current_photos = [photo]
            current_start = dt
            current_end = dt
            continue

        gap = dt - current_end
        if gap > timedelta(minutes=gap_minutes):
            events.append({
                "photos": current_photos,
                "start_time": current_start.isoformat(),
                "end_time": current_end.isoformat(),
            })
            current_photos = [photo]
            current_start = dt
            current_end = dt
        else:
            current_photos.append(photo)
            current_end = dt

    if current_photos:
        events.append({
            "photos": current_photos,
            "start_time": current_start.isoformat(),
            "end_time": current_end.isoformat(),
        })

    return events


def enrich_event_metadata(event: dict) -> dict:
    """Enrich an event dict with aggregated metadata (faces, city, cover).

    Args:
        event: event dict from slice_into_events with 'photos', 'start_time', 'end_time'

    Returns:
        enriched event dict with additional keys: event_id, photo_count,
        face_cluster_ids (list), location_city, cover_photo_uuid
    """
    from collections import Counter

    photos = event["photos"]

    # Union of all face cluster IDs
    face_ids = set()
    for p in photos:
        raw = p.get("face_cluster_ids")
        if raw:
            try:
                face_ids.update(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

    # Majority city (ignore None)
    cities = [p.get("location_city") for p in photos if p.get("location_city")]
    majority_city = Counter(cities).most_common(1)[0][0] if cities else None

    # Cover photo: first photo (MVP; could be smarter later)
    cover_uuid = photos[0]["uuid"]

    # Event ID: evt_<start_compact>_<first-uuid-prefix>
    start_dt = _parse_date(event["start_time"])
    start_compact = start_dt.strftime("%Y%m%d%H%M%S") if start_dt else "unknown"
    uuid_prefix = cover_uuid[:8]
    event_id = f"evt_{start_compact}_{uuid_prefix}"

    return {
        **event,
        "event_id": event_id,
        "photo_count": len(photos),
        "face_cluster_ids": sorted(face_ids),
        "location_city": majority_city,
        "cover_photo_uuid": cover_uuid,
    }


EVENT_SUMMARY_PROMPT = """基于以下事件中多张照片的描述信息，生成一段简洁的事件摘要。

事件信息：
- 时间：{start_time} ~ {end_time}
- 地点：{location}
- 在场人物：{people}
- 照片数：{photo_count}

照片描述（按时间顺序）：
{narratives}

严禁事项：
- 不要凭空推测医疗/诊疗/护理/治疗等场景，除非描述中明确出现「医院」「诊所」「病房」「医生」等词
- 美容/理发/美发/美甲/spa 等生活服务场景，不要误读为医疗或照护
- 如果描述模糊或只有「室内空间」「防护装备」等宽泛信息，就保持泛化（"某个室内场合"），不要具体化为敏感场景
- 如果提供了「在场人物」，摘要中请用真实人名而不是"朋友"等泛称

返回严格 JSON 格式（不要其他文字）：
{{
  "summary": "一段 30-60 字的中文事件摘要，描述这是什么事件、发生了什么",
  "mood": "整体情绪基调（愉快/紧张/平静/庄重/等）"
}}"""


def summarize_event(event: dict, host: str, model: str, timeout: int) -> dict:
    """Generate a summary and mood for an event via LLM.

    Falls back to a rule-based summary if LLM call fails.
    """
    narratives = []
    for i, p in enumerate(event["photos"][:20], 1):  # cap at 20 narratives
        try:
            ai = json.loads(p.get("ai_result") or "{}")
            narr = ai.get("narrative", "")
            if narr:
                narratives.append(f"{i}. {narr}")
        except (json.JSONDecodeError, TypeError):
            pass

    narratives_text = "\n".join(narratives) if narratives else "(无详细描述)"
    location = event.get("location_city") or "未知地点"
    person_names = event.get("person_names") or []
    people_str = "、".join(person_names) if person_names else "（未命名或无人）"

    prompt = EVENT_SUMMARY_PROMPT.format(
        start_time=event["start_time"],
        end_time=event["end_time"],
        location=location,
        people=people_str,
        photo_count=len(event["photos"]),
        narratives=narratives_text,
    )

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=(10, max(timeout, 120)),
            headers={"Connection": "close"},
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        data = _extract_json_object(raw)
        if data and "summary" in data:
            return {
                "summary": data["summary"],
                "mood": data.get("mood", ""),
            }
        logger.warning(
            f"Event summary LLM returned unparseable output for {event.get('event_id')}: {raw[:200]}"
        )
    except Exception as e:
        logger.warning(f"Event summary LLM failed for {event.get('event_id')}: {e}")

    # Fallback: use first narrative + location
    first_narr = narratives[0][3:] if narratives else "一组照片"
    return {
        "summary": f"在{location}的一段时光，{first_narr}",
        "mood": "",
    }


def build_events(db, ollama_config: dict, gap_minutes: int = 30) -> int:
    """Build events from done photos: slice, enrich, summarize, persist.

    Args:
        db: Database instance
        ollama_config: dict with 'host', 'model', 'timeout'
        gap_minutes: time gap threshold for event boundary

    Returns:
        number of events created
    """
    photos = db.get_done_photos_ordered()
    if not photos:
        logger.info("No done photos to aggregate")
        return 0

    logger.info(f"Aggregating {len(photos)} photos into events...")
    raw_events = slice_into_events(photos, gap_minutes=gap_minutes)
    logger.info(f"Sliced into {len(raw_events)} events")

    # Build fc_id → name map once for this run
    name_map: dict[str, str] = {}
    try:
        all_people = db.get_all_people()
        for p in all_people:
            name = p.get("user_name") or p.get("apple_name")
            if name:
                name_map[p["face_cluster_id"]] = name
    except Exception as e:
        logger.warning(f"Could not load people name map: {e}")

    for event in raw_events:
        enriched = enrich_event_metadata(event)
        # Resolve names for this event's face clusters
        person_names = list(dict.fromkeys(
            name_map[fc_id] for fc_id in enriched["face_cluster_ids"] if fc_id in name_map
        ))
        enriched["person_names"] = person_names

        summary_data = summarize_event(
            enriched,
            host=ollama_config["host"],
            model=ollama_config["model"],
            timeout=ollama_config["timeout"],
        )

        db.upsert_event(
            event_id=enriched["event_id"],
            start_time=enriched["start_time"],
            end_time=enriched["end_time"],
            location_city=enriched.get("location_city"),
            location_state=None,
            photo_count=enriched["photo_count"],
            face_cluster_ids=json.dumps(enriched["face_cluster_ids"], ensure_ascii=False),
            summary=summary_data["summary"],
            mood=summary_data["mood"],
            cover_photo_uuid=enriched["cover_photo_uuid"],
        )

        photo_uuids = [p["uuid"] for p in enriched["photos"]]
        db.link_photos_to_event(enriched["event_id"], photo_uuids)

    return len(raw_events)

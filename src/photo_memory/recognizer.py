"""Ollama vision API integration for photo recognition."""

import base64
import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

RECOGNITION_PROMPT_TEMPLATE = """你是一个照片分析助手。根据图片内容和以下上下文信息，返回严格的 JSON 格式（不要其他文字）。

上下文信息：
{context}

返回格式：
{{
  "narrative": "一句话中文描述这张照片的内容和场景（结合上下文，不要重复上下文已有的信息）",
  "event_hint": "事件类型：出游/约会/聚餐/工作/学习/运动/购物/日常/其他",
  "people": [
    {{"face_cluster_id": "对应上下文中的人脸ID", "description": "外貌/动作简述"}}
  ],
  "emotional_tone": "情绪基调",
  "significance": "这张照片的意义",
  "scene_category": "indoor_home|indoor_office|indoor_restaurant|outdoor_leisure|outdoor_travel|transport|other",
  "series_hint": "burst|sequence|standalone",
  "search_tags": ["简洁的可搜索场景标签，3-5个"],
  "has_text": false,
  "text_summary": "如果有文字，提取关键内容",
  "cleanup_class": "keep|review|cleanup",
  "duplicate_hint": "burst|live_variant|edited|standalone"
}}

注意：
- cleanup_class: 截图/模糊/无意义的照片标记为 cleanup，不确定的标记为 review，有意义的标记为 keep
- narrative 要有信息增量，不要只重复上下文
- search_tags 用于搜索，要简洁精确"""

RECOGNITION_PROMPT_NO_CONTEXT = """分析这张图片，返回严格的 JSON 格式（不要其他文字）：
{{
  "narrative": "一句话中文描述内容",
  "event_hint": "事件类型",
  "people": [],
  "emotional_tone": "情绪基调",
  "significance": "意义",
  "scene_category": "场景分类",
  "series_hint": "standalone",
  "search_tags": ["标签1", "标签2"],
  "has_text": false,
  "text_summary": "",
  "cleanup_class": "keep",
  "duplicate_hint": "standalone"
}}"""

REQUIRED_FIELDS = {"narrative", "search_tags", "has_text", "text_summary", "cleanup_class"}


def parse_ai_response(raw: str) -> dict:
    """Parse AI response, extracting JSON even if wrapped in markdown."""
    # Try direct JSON parse
    try:
        data = json.loads(raw)
        if REQUIRED_FIELDS.issubset(data.keys()):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if REQUIRED_FIELDS.issubset(data.keys()):
                return data
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if REQUIRED_FIELDS.issubset(data.keys()):
                return data
        except json.JSONDecodeError:
            pass

    # Fallback
    logger.warning(f"Failed to parse AI response as JSON: {raw[:100]}...")
    return {
        "narrative": raw[:200],
        "event_hint": "其他",
        "people": [],
        "emotional_tone": "",
        "significance": "",
        "scene_category": "other",
        "series_hint": "standalone",
        "search_tags": ["其他"],
        "has_text": False,
        "text_summary": "",
        "cleanup_class": "review",
        "duplicate_hint": "standalone",
    }


def build_photo_context(photo_row: dict) -> dict:
    """Build a photo_context dict from a database row for the recognizer prompt."""
    def _parse_json_field(val):
        if val is None:
            return []
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return []
        return val

    # named_faces is now stored as {fc_id: name} dict; extract name list for LLM context.
    # Backward-compat: legacy format was a flat array of names.
    raw_named = photo_row.get("named_faces")
    parsed_named = _parse_json_field(raw_named)
    if isinstance(parsed_named, dict):
        named_list = list(parsed_named.values())
    elif isinstance(parsed_named, list):
        named_list = parsed_named
    else:
        named_list = []

    return {
        "date": photo_row.get("date_taken", ""),
        "location_city": photo_row.get("location_city"),
        "location_state": photo_row.get("location_state"),
        "location_country": photo_row.get("location_country"),
        "apple_labels": _parse_json_field(photo_row.get("apple_labels")),
        "named_faces": named_list,
        "face_cluster_ids": _parse_json_field(photo_row.get("face_cluster_ids")),
        "is_selfie": bool(photo_row.get("is_selfie")),
        "is_screenshot": bool(photo_row.get("is_screenshot")),
        "is_live_photo": bool(photo_row.get("is_live_photo")),
        "source_app": photo_row.get("source_app"),
    }


def recognize_photo(image_path: str, host: str, model: str, timeout: int,
                    max_retries: int = 1, photo_context: dict | None = None) -> dict:
    """Send image to Ollama for recognition, return parsed result."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    if photo_context:
        prompt = RECOGNITION_PROMPT_TEMPLATE.format(
            context=json.dumps(photo_context, ensure_ascii=False, indent=2)
        )
    else:
        prompt = RECOGNITION_PROMPT_NO_CONTEXT

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    session = requests.Session()

    for attempt in range(max_retries + 1):
        response = session.post(
            f"{host}/api/generate",
            json=payload,
            timeout=(10, max(timeout, 120)),  # (connect, read) — enforce min 120s read
            headers={"Connection": "close"},  # prevent pool reuse with stale timeout
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        result = parse_ai_response(raw)

        if result["search_tags"] != ["其他"] or attempt == max_retries:
            return result

        logger.info(f"Retry {attempt + 1}: AI returned unparseable response, retrying...")

    return result


def summarize_video_frames(frame_results: list[dict], transcript: str = "") -> dict:
    """Summarize multiple frame analysis results into one video-level result.

    Merges search_tags, picks the most common scene_category,
    and incorporates transcript if available.
    """
    if not frame_results:
        return parse_ai_response("")  # fallback

    all_tags = []
    for r in frame_results:
        all_tags.extend(r.get("search_tags", []))
    merged_tags = list(dict.fromkeys(all_tags))

    from collections import Counter
    scene_cats = [r.get("scene_category", "other") for r in frame_results]
    most_common_scene = Counter(scene_cats).most_common(1)[0][0]

    narrative = frame_results[0].get("narrative", "")
    if transcript:
        narrative += f"（语音内容：{transcript[:100]}）"
        merged_tags.append("有语音")

    cleanup_class = "keep"
    if transcript and len(transcript) > 10:
        cleanup_class = "keep"

    return {
        "narrative": narrative,
        "event_hint": frame_results[0].get("event_hint", "其他"),
        "people": frame_results[0].get("people", []),
        "emotional_tone": frame_results[0].get("emotional_tone", ""),
        "significance": frame_results[0].get("significance", ""),
        "scene_category": most_common_scene,
        "series_hint": "standalone",
        "search_tags": merged_tags[:20],
        "has_text": any(r.get("has_text", False) for r in frame_results),
        "text_summary": transcript[:500] if transcript else "",
        "cleanup_class": cleanup_class,
        "duplicate_hint": "standalone",
    }

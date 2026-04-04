"""Ollama vision API integration for photo recognition."""

import base64
import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

RECOGNITION_PROMPT = """分析这张图片，返回严格的 JSON 格式（不要其他文字）：
{
  "description": "一句话中文描述内容",
  "tags": ["自由标签1", "自由标签2", "..."],
  "people_count": 0,
  "animals": [],
  "objects": [],
  "location_type": "indoor|outdoor|vehicle|studio|其他",
  "activity": "描述正在进行的活动",
  "mood": "开心|平静|正式|热闹|安静|紧张|其他",
  "time_of_day": "白天|夜晚|黄昏|清晨|室内无法判断",
  "media_type": "photo|screenshot|document|video_frame",
  "scene_type": "日常记录|B-roll|口播|屏幕录制|产品展示|活动现场|风景|人像|美食|其他",
  "importance": "high|medium|low",
  "has_text": false,
  "text_summary": "如果has_text为true，提取关键文字信息",
  "colors": ["主要颜色"],
  "quality_notes": "清晰|模糊|过曝|偏暗|正常"
}
tags 不限于固定分类，自由发挥，尽可能多地描述内容特征，中文标签。"""

REQUIRED_FIELDS = {"description", "tags", "media_type", "importance", "has_text", "text_summary"}


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
        "description": raw[:200],
        "tags": ["其他"],
        "people_count": 0,
        "animals": [],
        "objects": [],
        "location_type": "其他",
        "activity": "",
        "mood": "",
        "time_of_day": "",
        "media_type": "photo",
        "scene_type": "其他",
        "importance": "low",
        "has_text": False,
        "text_summary": "",
        "colors": [],
        "quality_notes": "正常",
    }


def recognize_photo(image_path: str, host: str, model: str, timeout: int,
                    max_retries: int = 1) -> dict:
    """Send image to Ollama for recognition, return parsed result."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": RECOGNITION_PROMPT,
        "images": [image_b64],
        "stream": False,
    }

    for attempt in range(max_retries + 1):
        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        result = parse_ai_response(raw)

        if result["tags"] != ["其他"] or attempt == max_retries:
            return result

        logger.info(f"Retry {attempt + 1}: AI returned unparseable response, retrying...")

    return result


def summarize_video_frames(frame_results: list[dict], transcript: str = "") -> dict:
    """Summarize multiple frame analysis results into one video-level result.

    Merges tags, picks the most common values for categorical fields,
    and incorporates transcript if available.
    """
    if not frame_results:
        return parse_ai_response("")  # fallback

    # Merge all tags (deduplicate)
    all_tags = []
    for r in frame_results:
        all_tags.extend(r.get("tags", []))
    merged_tags = list(dict.fromkeys(all_tags))  # preserve order, dedupe

    # Pick most common scene_type
    from collections import Counter
    scene_types = [r.get("scene_type", "其他") for r in frame_results]
    most_common_scene = Counter(scene_types).most_common(1)[0][0]

    # Merge objects and animals
    all_objects = []
    all_animals = []
    for r in frame_results:
        all_objects.extend(r.get("objects", []))
        all_animals.extend(r.get("animals", []))
    merged_objects = list(dict.fromkeys(all_objects))
    merged_animals = list(dict.fromkeys(all_animals))

    # Max people count across frames
    max_people = max((r.get("people_count", 0) for r in frame_results), default=0)

    # Use first frame's description as base, append transcript summary
    description = frame_results[0].get("description", "")
    if transcript:
        description += f"（语音内容：{transcript[:100]}）"
        # Add transcript-derived tags
        merged_tags.append("有语音")

    # Determine importance — videos with speech tend to be more important
    importance = "medium"
    if transcript and len(transcript) > 10:
        importance = "high"

    return {
        "description": description,
        "tags": merged_tags[:20],  # cap at 20 tags
        "people_count": max_people,
        "animals": merged_animals,
        "objects": merged_objects,
        "location_type": frame_results[0].get("location_type", "其他"),
        "activity": frame_results[0].get("activity", ""),
        "mood": frame_results[0].get("mood", ""),
        "time_of_day": frame_results[0].get("time_of_day", ""),
        "media_type": "video_frame",
        "scene_type": most_common_scene,
        "importance": importance,
        "has_text": any(r.get("has_text", False) for r in frame_results),
        "text_summary": transcript[:500] if transcript else "",
        "colors": frame_results[0].get("colors", []),
        "quality_notes": frame_results[0].get("quality_notes", "正常"),
    }

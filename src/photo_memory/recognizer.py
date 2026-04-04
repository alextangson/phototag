"""Ollama vision API integration for photo recognition."""

import base64
import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

RECOGNITION_PROMPT = """分析这张照片，返回严格的 JSON 格式（不要其他文字）：
{
  "description": "一句话中文描述照片内容",
  "tags": ["层级1/层级2", "层级1/层级2"],
  "media_type": "photo|screenshot|document|video_frame",
  "scene": "indoor|outdoor|screenshot|document",
  "importance": "high|medium|low",
  "has_text": true/false,
  "text_summary": "如果 has_text=true，提取关键文字信息，否则为空"
}

标签使用层级格式 "大类/小类"，从以下体系中选择：
- 人物/自拍、人物/合照、人物/证件照、人物/活动
- 风景/自然、风景/城市、风景/海边、风景/山景
- 美食/餐厅、美食/自制、美食/甜点、美食/饮品
- 旅行/国内、旅行/海外、旅行/酒店、旅行/交通
- 宠物/猫、宠物/狗、宠物/其他
- 生活/家居、生活/购物、生活/健身、生活/娱乐
- 工作/会议、工作/白板、工作/代码、工作/办公
- 截屏/手机截屏、截屏/电脑截屏、截屏/游戏
- 聊天记录/微信、聊天记录/iMessage、聊天记录/钉钉、聊天记录/飞书、聊天记录/其他
- 文档/扫描件、文档/名片、文档/收据、文档/证件、文档/二维码
- 活动/生日、活动/婚礼、活动/聚餐、活动/演出、活动/毕业
- 其他
可以同时输出多个标签。如果不确定，使用 "其他"。"""

REQUIRED_FIELDS = {"description", "tags", "media_type", "scene", "importance", "has_text", "text_summary"}


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
        "media_type": "photo",
        "scene": "outdoor",
        "importance": "low",
        "has_text": False,
        "text_summary": "",
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

"""New prompt template for M1.5 eval — tests narrative + search + OCR output."""

NEW_PROMPT_TEMPLATE = """你是一个照片分析助手。根据图片内容和以下已知信息，返回严格的 JSON 格式（不要其他文字）。

已知信息：
- 拍摄时间：{date}
- 地点：{location}
- 设备：{device}
- 来源App：{source_app}
- Apple已识别标签：{apple_labels}
- 已知人脸：{named_faces}
- 未命名人脸数：{unnamed_face_count}
- 是否自拍：{is_selfie}
- 是否截屏：{is_screenshot}
- 用户相册：{albums}

返回格式：
{{
  "narrative": "一句话中文叙事，描述这个时刻的故事，不只描述画面，要结合已知信息推理场景含义",
  "event_hint": "事件分类，如：出游/聚餐/工作/日常/庆祝/运动/学习/其他",
  "people": [
    {{
      "description": "对画面中人物的外貌描述",
      "role_guess": "可能的角色：朋友/同事/家人/恋人/路人/自己"
    }}
  ],
  "emotional_tone": "情感基调",
  "significance": "这个时刻的重要性判断及原因",
  "scene_category": "outdoor_leisure|indoor_social|work|food|travel|selfie|screenshot|document|other",
  "search_tags": ["简洁的可搜索中文场景标签，5-8个"],
  "has_text": false,
  "text_summary": "如果图片中有文字，提取关键文字内容",
  "cleanup_class": "keep|review|cleanup",
  "series_hint": "burst|live_variant|edited|standalone"
}}"""


def build_prompt(photo_context: dict) -> str:
    """Fill the prompt template with photo context."""
    return NEW_PROMPT_TEMPLATE.format(
        date=photo_context.get("date", "未知"),
        location=photo_context.get("location", "未知"),
        device=photo_context.get("device", "未知"),
        source_app=photo_context.get("source_app", "未知"),
        apple_labels=photo_context.get("apple_labels", "无"),
        named_faces=photo_context.get("named_faces", "无"),
        unnamed_face_count=photo_context.get("unnamed_face_count", 0),
        is_selfie=photo_context.get("is_selfie", False),
        is_screenshot=photo_context.get("is_screenshot", False),
        albums=photo_context.get("albums", "无"),
    )

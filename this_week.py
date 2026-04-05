#!/usr/bin/env python3
"""This Week In Your Life — show photos from the same week in previous years."""

import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import osxphotos
from PIL import Image, ImageOps
import requests

DAYS_RANGE = 3


def scan_this_week_photos() -> dict[int, list]:
    """Find photos from the same week (±DAYS_RANGE days) in previous years."""
    print("Opening Photos library...")
    db = osxphotos.PhotosDB()
    photos = db.photos(images=True, movies=False)

    now = datetime.now()
    current_year = now.year
    target_start = now - timedelta(days=DAYS_RANGE)
    target_end = now + timedelta(days=DAYS_RANGE)

    by_year = {}
    for p in photos:
        if not p.date or p.date.year >= current_year:
            continue
        if not p.path or not os.path.isfile(p.path):
            continue
        d = p.date
        try:
            compare_date = d.replace(year=current_year)
        except ValueError:
            continue
        if target_start <= compare_date <= target_end:
            year = d.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(p)

    for year in sorted(by_year):
        print(f"  {year}: {len(by_year[year])} photos")
    return by_year


def cluster_into_events(photos: list, time_gap_hours: float = 2.0) -> list[dict]:
    """Cluster photos into events by time proximity."""
    if not photos:
        return []
    sorted_photos = sorted(photos, key=lambda p: p.date)
    events = []
    current_event = [sorted_photos[0]]

    for photo in sorted_photos[1:]:
        prev = current_event[-1]
        gap = (photo.date - prev.date).total_seconds() / 3600
        if gap > time_gap_hours:
            events.append(current_event)
            current_event = [photo]
        else:
            current_event.append(photo)
    events.append(current_event)

    result = []
    for i, event_photos in enumerate(events):
        if len(event_photos) > 12:
            step = len(event_photos) / 12
            selected = [event_photos[int(j * step)] for j in range(12)]
        else:
            selected = event_photos

        dates = [p.date for p in event_photos]
        locations = set()
        for p in event_photos:
            if p.place and p.place.name:
                locations.add(p.place.name)

        face_names = set()
        for p in event_photos:
            for pi in p.person_info:
                if pi.name and not pi.name.startswith("_"):
                    face_names.add(pi.name)

        date_str = min(dates).strftime("%Y-%m-%d")
        title_parts = [date_str]
        if locations:
            title_parts.append(", ".join(locations))
        if face_names:
            title_parts.append(", ".join(face_names))
        title_parts.append(f"{len(event_photos)} 张照片")
        fallback_title = " · ".join(title_parts)

        result.append({
            "id": f"evt_{min(dates).strftime('%Y%m%d')}_{i}",
            "date_range": (min(dates), max(dates)),
            "photo_count": len(event_photos),
            "selected_photos": selected,
            "all_photos": event_photos,
            "locations": list(locations),
            "people": list(face_names),
            "fallback_title": fallback_title,
            "title": fallback_title,
            "description": "",
        })
    return result


def photo_to_thumbnail_b64(photo_path: str, max_size: int = 400) -> str:
    """Convert a photo to a base64-encoded JPEG thumbnail."""
    try:
        img = Image.open(photo_path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠ Thumbnail failed for {photo_path}: {e}")
        return ""


def generate_event_title_llm(event: dict, ollama_host: str = "http://localhost:11434",
                              model: str = "gemma4:e4b") -> tuple[str, str]:
    """Try to generate an event title via LLM. Returns (title, description)."""
    photo = event["selected_photos"][0]
    try:
        with open(photo.path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return event["fallback_title"], ""

    date_range = event["date_range"]
    date_str = date_range[0].strftime("%Y年%m月%d日")
    loc_str = ", ".join(event["locations"]) if event["locations"] else "未知地点"
    people_str = ", ".join(event["people"]) if event["people"] else "未知"

    prompt = f"""这是一组 {event['photo_count']} 张照片中的第一张。
拍摄日期：{date_str}
地点：{loc_str}
出现的人：{people_str}

请用中文给这组照片起一个简短的标题（10字以内）和一句话描述（30字以内）。
返回 JSON：{{"title": "标题", "description": "描述"}}
不要其他文字。"""

    try:
        resp = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": model, "prompt": prompt, "images": [image_b64], "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        match = re.search(r"\{[\s\S]*?\}", raw)
        if match:
            data = json.loads(match.group(0))
            return data.get("title", event["fallback_title"]), data.get("description", "")
    except Exception as e:
        print(f"  ⚠ LLM failed for event {event['id']}: {e}")
    return event["fallback_title"], ""


def render_html(all_events: dict[int, list[dict]], output_path: str = "this_week.html"):
    """Render all events into a single self-contained HTML file."""
    cards_html = ""
    event_counter = 0

    for year in sorted(all_events.keys(), reverse=True):
        events = all_events[year]
        if not events:
            continue
        cards_html += f'<h2 class="year-header">{year}年</h2>\n'

        for event in events:
            event_counter += 1
            date_range = event["date_range"]
            date_str = date_range[0].strftime("%m月%d日")
            if date_range[0].date() != date_range[1].date():
                date_str += f" - {date_range[1].strftime('%m月%d日')}"

            thumbs_html = ""
            for photo in event["selected_photos"]:
                b64 = photo_to_thumbnail_b64(photo.path)
                if b64:
                    thumbs_html += f'<img src="data:image/jpeg;base64,{b64}" alt="" loading="lazy">\n'

            desc = event.get("description", "")
            desc_html = f'<p class="event-desc">{desc}</p>' if desc else ""

            meta_parts = []
            if event["locations"]:
                meta_parts.append(f'📍 {", ".join(event["locations"])}')
            if event["people"]:
                meta_parts.append(f'👤 {", ".join(event["people"])}')
            meta_parts.append(f'📷 {event["photo_count"]} 张照片')
            meta_html = " · ".join(meta_parts)

            cards_html += f"""
<div class="event-card" id="event-{event_counter}">
  <div class="event-header">
    <span class="event-date">{date_str}</span>
    <h3 class="event-title">{event['title']}</h3>
  </div>
  {desc_html}
  <p class="event-meta">{meta_html}</p>
  <div class="photo-grid">
    {thumbs_html}
  </div>
  <div class="feedback-bar">
    <button onclick="feedback({event_counter}, 'useful')" class="btn-useful">👍 有用</button>
    <button onclick="feedback({event_counter}, 'not_useful')" class="btn-not-useful">👎 没用</button>
    <span id="fb-{event_counter}" class="fb-status"></span>
  </div>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>This Week In Your Life</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
         background: #fafafa; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 8px; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 30px; }}
  .year-header {{ font-size: 18px; font-weight: 600; color: #666; margin: 30px 0 15px;
                  padding-bottom: 8px; border-bottom: 1px solid #ddd; }}
  .event-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
                 padding: 16px; margin-bottom: 16px; }}
  .event-header {{ margin-bottom: 8px; }}
  .event-date {{ font-size: 13px; color: #888; }}
  .event-title {{ font-size: 16px; font-weight: 500; margin-top: 4px; }}
  .event-desc {{ font-size: 14px; color: #555; margin-bottom: 8px; }}
  .event-meta {{ font-size: 12px; color: #999; margin-bottom: 12px; }}
  .photo-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                 gap: 6px; margin-bottom: 12px; }}
  .photo-grid img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px;
                     cursor: pointer; transition: transform 0.2s; }}
  .photo-grid img:hover {{ transform: scale(1.05); }}
  .feedback-bar {{ display: flex; gap: 8px; align-items: center; }}
  .feedback-bar button {{ padding: 6px 14px; border: 1px solid #ddd; border-radius: 16px;
                          background: #fff; cursor: pointer; font-size: 13px; }}
  .feedback-bar button:hover {{ background: #f5f5f5; }}
  .feedback-bar button.selected {{ border-color: #333; background: #f0f0f0; }}
  .fb-status {{ font-size: 12px; color: #999; }}
  .lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
               background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; }}
  .lightbox.active {{ display: flex; }}
  .lightbox img {{ max-width: 90vw; max-height: 90vh; object-fit: contain; }}
  .lightbox-close {{ position: fixed; top: 20px; right: 20px; color: #fff; font-size: 30px;
                     cursor: pointer; z-index: 1001; }}
</style>
</head>
<body>
<h1>This Week In Your Life</h1>
<p class="subtitle">历史上的今天这周 · {datetime.now().strftime('%m月%d日')} ± {DAYS_RANGE}天</p>
{cards_html}
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <span class="lightbox-close">&times;</span>
  <img id="lightbox-img" src="" alt="">
</div>
<script>
const feedbackData = {{}};
function feedback(eventId, value) {{
  feedbackData[eventId] = value;
  document.getElementById('fb-' + eventId).textContent = value === 'useful' ? '已标记有用' : '已标记没用';
  const card = document.getElementById('event-' + eventId);
  card.querySelectorAll('button').forEach(b => b.classList.remove('selected'));
  const btn = value === 'useful' ? card.querySelector('.btn-useful') : card.querySelector('.btn-not-useful');
  btn.classList.add('selected');
  localStorage.setItem('this_week_feedback', JSON.stringify(feedbackData));
}}
try {{
  const saved = JSON.parse(localStorage.getItem('this_week_feedback') || '{{}}');
  Object.entries(saved).forEach(([id, val]) => {{
    feedbackData[id] = val;
    const el = document.getElementById('fb-' + id);
    if (el) el.textContent = val === 'useful' ? '已标记有用' : '已标记没用';
    const card = document.getElementById('event-' + id);
    if (card) {{
      const btn = val === 'useful' ? card.querySelector('.btn-useful') : card.querySelector('.btn-not-useful');
      if (btn) btn.classList.add('selected');
    }}
  }});
}} catch(e) {{}}
document.querySelectorAll('.photo-grid img').forEach(img => {{
  img.addEventListener('click', function(e) {{
    e.stopPropagation();
    document.getElementById('lightbox-img').src = this.src;
    document.getElementById('lightbox').classList.add('active');
  }});
}});
function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeLightbox();
}});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"\n✅ HTML saved to {output_path}")
    print(f"   {event_counter} events across {len(all_events)} years")


def main():
    use_llm = "--llm" in sys.argv

    by_year = scan_this_week_photos()
    if not by_year:
        print("No photos found for this week in previous years.")
        return

    all_events = {}
    for year in sorted(by_year.keys()):
        photos = by_year[year]
        events = cluster_into_events(photos)
        print(f"\n{year}: {len(events)} events")

        for event in events:
            print(f"  📅 {event['fallback_title']}")
            if use_llm:
                print(f"    🤖 Generating title...")
                title, desc = generate_event_title_llm(event)
                event["title"] = title
                event["description"] = desc
                print(f"    → {title}")
            print(f"    📷 Generating {len(event['selected_photos'])} thumbnails...")

        all_events[year] = events

    render_html(all_events)
    print(f"\n🎉 Done! Open this_week.html in your browser.")


if __name__ == "__main__":
    main()

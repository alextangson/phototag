# "This Week In Your Life" — 48h Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-page HTML that shows "历史上的今天这周" — photos from the same week in previous years, clustered into events, with optional LLM-generated titles.

**Architecture:** One standalone Python script (`this_week.py`) that: (1) reads Apple Photos via osxphotos, (2) filters to ±3 days same week in past years, (3) clusters into events by time, (4) optionally calls Ollama for event titles, (5) renders a self-contained HTML file with embedded base64 thumbnails. No DB, no production code changes.

**Tech Stack:** Python, osxphotos, Pillow (thumbnails), requests (Ollama, optional), Jinja2-free HTML string templating

**Success Criteria:** Open the HTML, see a photo, say "我忘了这件事". If no such reaction, rethink phototag direction.

---

## File Structure

| File | Purpose |
|------|---------|
| Create: `this_week.py` | Main script: scan → cluster → render |
| Output: `this_week.html` | Generated HTML (gitignored) |
| Output: `this_week_feedback.json` | User 👍/👎 feedback (gitignored) |

---

### Task 1: Photo scanning and filtering

**Files:**
- Create: `this_week.py` (first part)

- [ ] **Step 1: Write the photo scanning and filtering logic**

```python
#!/usr/bin/env python3
"""This Week In Your Life — show photos from the same week in previous years."""

import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import osxphotos
from PIL import Image

DAYS_RANGE = 3  # ±3 days from today


def scan_this_week_photos() -> dict[int, list]:
    """Find photos from the same week (±DAYS_RANGE days) in previous years."""
    print("Opening Photos library...")
    db = osxphotos.PhotosDB()
    photos = db.photos(images=True, movies=False)

    now = datetime.now()
    current_year = now.year
    # Target month-day range
    target_start = now - timedelta(days=DAYS_RANGE)
    target_end = now + timedelta(days=DAYS_RANGE)

    by_year = {}
    for p in photos:
        if not p.date or p.date.year >= current_year:
            continue
        if not p.path or not os.path.isfile(p.path):
            continue

        d = p.date
        # Check if this photo's month-day falls within ±DAYS_RANGE of today
        # Create a comparison date in the current year
        try:
            compare_date = d.replace(year=current_year)
        except ValueError:
            # Handle Feb 29 in non-leap years
            continue

        if target_start <= compare_date <= target_end:
            year = d.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(p)

    for year in sorted(by_year):
        print(f"  {year}: {len(by_year[year])} photos")

    return by_year
```

- [ ] **Step 2: Verify it runs**

```bash
cd /Users/macmini/photo-memory
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from this_week import scan_this_week_photos
result = scan_this_week_photos()
print(f'Found photos in {len(result)} years')
"
```

Expected: prints year counts matching our earlier survey.

- [ ] **Step 3: Commit**

```bash
git add this_week.py
git commit -m "feat: add this-week photo scanner"
```

---

### Task 2: Event clustering

**Files:**
- Modify: `this_week.py` (add clustering)

- [ ] **Step 1: Add event clustering function**

Append to `this_week.py`:

```python
def cluster_into_events(photos: list, time_gap_hours: float = 2.0) -> list[dict]:
    """Cluster photos into events by time proximity.

    Photos within time_gap_hours of each other belong to the same event.
    No GPS available for most photos, so time-only clustering.
    """
    if not photos:
        return []

    # Sort by date
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

    # Build event dicts
    result = []
    for i, event_photos in enumerate(events):
        # Sample 6-12 representative photos (evenly spaced by time)
        if len(event_photos) > 12:
            step = len(event_photos) / 12
            selected = [event_photos[int(j * step)] for j in range(12)]
        else:
            selected = event_photos

        # Collect metadata
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

        # Generate fallback title from metadata
        date_str = min(dates).strftime("%Y-%m-%d")
        loc_str = ", ".join(locations) if locations else ""
        people_str = ", ".join(face_names) if face_names else ""

        title_parts = [date_str]
        if loc_str:
            title_parts.append(loc_str)
        if people_str:
            title_parts.append(f"with {people_str}")
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
            "title": fallback_title,  # Will be overwritten by LLM if available
            "description": "",
        })

    return result
```

- [ ] **Step 2: Verify clustering works**

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from this_week import scan_this_week_photos, cluster_into_events
by_year = scan_this_week_photos()
for year, photos in sorted(by_year.items()):
    events = cluster_into_events(photos)
    print(f'{year}: {len(events)} events')
    for e in events:
        print(f'  {e[\"fallback_title\"]}')
"
```

- [ ] **Step 3: Commit**

```bash
git add this_week.py
git commit -m "feat: add time-based event clustering"
```

---

### Task 3: Thumbnail generation

**Files:**
- Modify: `this_week.py` (add thumbnail function)

- [ ] **Step 1: Add thumbnail generation**

Append to `this_week.py`:

```python
def photo_to_thumbnail_b64(photo_path: str, max_size: int = 400) -> str:
    """Convert a photo to a base64-encoded JPEG thumbnail."""
    try:
        img = Image.open(photo_path)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        # Handle HEIC orientation
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠ Thumbnail failed for {photo_path}: {e}")
        return ""
```

- [ ] **Step 2: Commit**

```bash
git add this_week.py
git commit -m "feat: add photo thumbnail generation"
```

---

### Task 4: Optional LLM event titles

**Files:**
- Modify: `this_week.py` (add LLM title generation)

- [ ] **Step 1: Add LLM title generation (optional, falls back to metadata)**

Append to `this_week.py`:

```python
def generate_event_title_llm(event: dict, ollama_host: str = "http://localhost:11434",
                              model: str = "gemma4:e4b") -> tuple[str, str]:
    """Try to generate an event title via LLM. Returns (title, description).
    Falls back to metadata title on any failure.
    """
    import requests

    # Pick first photo for visual context
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

        import re
        match = re.search(r"\{[\s\S]*?\}", raw)
        if match:
            data = json.loads(match.group(0))
            return data.get("title", event["fallback_title"]), data.get("description", "")
    except Exception as e:
        print(f"  ⚠ LLM failed for event {event['id']}: {e}")

    return event["fallback_title"], ""
```

- [ ] **Step 2: Commit**

```bash
git add this_week.py
git commit -m "feat: add optional LLM event title generation"
```

---

### Task 5: HTML rendering

**Files:**
- Modify: `this_week.py` (add HTML renderer + main)

- [ ] **Step 1: Add HTML rendering function**

Append to `this_week.py`:

```python
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

            # Generate thumbnails
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
  /* Lightbox */
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
// Feedback
const feedbackData = {{}};
function feedback(eventId, value) {{
  feedbackData[eventId] = value;
  document.getElementById('fb-' + eventId).textContent = value === 'useful' ? '已标记有用' : '已标记没用';
  // Style buttons
  const card = document.getElementById('event-' + eventId);
  card.querySelectorAll('button').forEach(b => b.classList.remove('selected'));
  const btn = value === 'useful' ? card.querySelector('.btn-useful') : card.querySelector('.btn-not-useful');
  btn.classList.add('selected');
  // Save to file via download trick
  saveFeedback();
}}

function saveFeedback() {{
  const blob = new Blob([JSON.stringify(feedbackData, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.style.display = 'none';
  // Don't auto-download, just store in localStorage
  localStorage.setItem('this_week_feedback', JSON.stringify(feedbackData));
}}

// Load previous feedback
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

// Lightbox
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

            # Pre-generate thumbnails progress
            print(f"    📷 Generating {len(event['selected_photos'])} thumbnails...")

        all_events[year] = events

    render_html(all_events)
    print(f"\n🎉 Done! Open this_week.html in your browser.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add to .gitignore**

Append to `/Users/macmini/photo-memory/.gitignore`:
```
this_week.html
this_week_feedback.json
```

- [ ] **Step 3: Commit**

```bash
git add this_week.py .gitignore
git commit -m "feat: add HTML renderer and main entry point for this-week prototype"
```

---

### Task 6: Run it and open

- [ ] **Step 1: Run without LLM first (fast, metadata-only titles)**

```bash
cd /Users/macmini/photo-memory
.venv/bin/python this_week.py
```

Expected: generates `this_week.html` with event cards using metadata titles.

- [ ] **Step 2: Open in browser**

```bash
open this_week.html
```

- [ ] **Step 3: Optional — run with LLM for better titles**

```bash
.venv/bin/python this_week.py --llm
```

- [ ] **Step 4: Human review — kill gate**

Open `this_week.html` and answer:
- Did you see a photo and think "我忘了这件事"?
- If YES → phototag direction validated, proceed to M2
- If NO → rethink the entire direction

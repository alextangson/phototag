#!/usr/bin/env python3
"""M1.5 Model Eval: sample 50 photos, run new prompt, save results."""

import base64
import json
import os
import random
import re
import sys
import time
from datetime import datetime

import osxphotos
import requests

from new_prompt import build_prompt

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "50"))
SEED = 42


def get_photo_context(photo) -> dict:
    """Extract Apple metadata context from an osxphotos PhotoInfo object."""
    location = "未知"
    if photo.place:
        parts = []
        if hasattr(photo.place, 'names') and photo.place.names:
            names = photo.place.names
            if names.city:
                parts.append(names.city[0])
            if names.state_province:
                parts.append(names.state_province[0])
            if names.country:
                parts.append(names.country[0])
        if parts:
            location = ", ".join(parts)
        elif photo.place.name:
            location = photo.place.name

    date_str = "未知"
    if photo.date:
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        d = photo.date
        date_str = f"{d.strftime('%Y-%m-%d')} {weekdays[d.weekday()]} {d.strftime('%H:%M')}"

    device = "未知"
    if photo.exif_info and photo.exif_info.camera_model:
        device = photo.exif_info.camera_model

    source_app = photo._info.get("imported_by_display_name", "相机") or "相机"
    apple_labels = photo.labels if photo.labels else []
    named = [pi.name for pi in photo.person_info if pi.name and not pi.name.startswith("_")]
    unnamed_count = sum(1 for pi in photo.person_info if not pi.name or pi.name.startswith("_"))
    albums = [a.title for a in photo.album_info if a.title]

    return {
        "date": date_str,
        "location": location,
        "device": device,
        "source_app": source_app,
        "apple_labels": str(apple_labels) if apple_labels else "无",
        "named_faces": str(named) if named else "无",
        "unnamed_face_count": unnamed_count,
        "is_selfie": photo.selfie,
        "is_screenshot": photo.screenshot,
        "albums": str(albums) if albums else "无",
    }


def sample_photos(photosdb) -> list:
    """Sample photos across categories for balanced eval."""
    photos = photosdb.photos(images=True, movies=False)
    accessible = [p for p in photos if p.path and os.path.isfile(p.path)]

    random.seed(SEED)

    categories = {
        "gps_and_faces": [p for p in accessible if p.latitude and p.person_info],
        "screenshots": [p for p in accessible if p.screenshot],
        "selfies": [p for p in accessible if p.selfie],
        "from_wechat": [p for p in accessible if p._info.get("imported_by_display_name") == "微信"],
        "landscape": [p for p in accessible if p.latitude and not p.person_info and not p.screenshot],
        "general": [p for p in accessible if not p.screenshot and not p.selfie],
    }

    targets = {
        "gps_and_faces": 12,
        "screenshots": min(4, len(categories["screenshots"])),
        "selfies": 6,
        "from_wechat": 10,
        "landscape": 8,
        "general": 10,
    }

    sampled = []
    seen_uuids = set()

    for cat_name, target in targets.items():
        pool = [p for p in categories[cat_name] if p.uuid not in seen_uuids]
        picked = random.sample(pool, min(target, len(pool)))
        for p in picked:
            sampled.append((cat_name, p))
            seen_uuids.add(p.uuid)

    return sampled[:SAMPLE_SIZE]


def call_ollama(image_path: str, prompt: str) -> dict:
    """Send image + prompt to Ollama, return parsed JSON or error."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return {"parsed": json.loads(match.group(0)), "raw": raw, "error": None}
        return {"parsed": None, "raw": raw, "error": "no JSON found"}
    except json.JSONDecodeError as e:
        return {"parsed": None, "raw": raw, "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"parsed": None, "raw": "", "error": str(e)}


def main():
    print("Opening Photos library...")
    photosdb = osxphotos.PhotosDB()

    print(f"Sampling {SAMPLE_SIZE} photos across categories...")
    sampled = sample_photos(photosdb)
    print(f"Sampled {len(sampled)} photos")

    results = []
    for i, (category, photo) in enumerate(sampled):
        print(f"\n[{i+1}/{len(sampled)}] {category}: {photo.original_filename}")

        context = get_photo_context(photo)
        prompt = build_prompt(context)
        print(f"  Context: {context['date']}, {context['location']}, faces={context['unnamed_face_count']}")

        start = time.time()
        result = call_ollama(photo.path, prompt)
        elapsed = time.time() - start

        entry = {
            "uuid": photo.uuid,
            "filename": photo.original_filename,
            "category": category,
            "context": context,
            "result": result["parsed"],
            "raw_response": result["raw"][:500],
            "error": result["error"],
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        if result["parsed"]:
            r = result["parsed"]
            print(f"  ✅ narrative: {r.get('narrative', '?')[:60]}...")
            print(f"     search_tags: {r.get('search_tags', [])}")
            print(f"     cleanup: {r.get('cleanup_class', '?')}")
        else:
            print(f"  ❌ Error: {result['error']}")

        print(f"  ⏱ {elapsed:.1f}s")

    os.makedirs("eval/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"eval/results/eval-{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Total: {len(results)} photos")
    print(f"Success: {sum(1 for r in results if r['result'])}")
    print(f"Failed: {sum(1 for r in results if r['error'])}")
    avg_time = sum(r['elapsed_seconds'] for r in results) / len(results) if results else 0
    print(f"Avg time per photo: {avg_time:.1f}s")


if __name__ == "__main__":
    main()

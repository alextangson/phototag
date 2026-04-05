#!/usr/bin/env python3
"""Score M1.5 eval results — prints a human-review scorecard."""

import glob
import json
import random
import sys
from collections import Counter


def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def score(results: list):
    total = len(results)
    success = [r for r in results if r["result"]]
    failed = [r for r in results if r["error"]]

    print(f"\n{'='*70}")
    print(f"  M1.5 MODEL EVAL SCORECARD")
    print(f"{'='*70}")
    print(f"  Model: gemma4:e4b | Photos: {total} | Success: {len(success)} | Failed: {len(failed)}")
    print(f"  Parse rate: {len(success)/total*100:.0f}%")

    if not success:
        print("  No successful results to score.")
        return

    avg_time = sum(r["elapsed_seconds"] for r in results) / total
    print(f"  Avg time: {avg_time:.1f}s/photo")
    print(f"  Est. total for 10k: {avg_time * 10000 / 3600:.0f} hours")

    print(f"\n  --- Field Completeness ---")
    fields = ["narrative", "event_hint", "search_tags", "has_text", "text_summary",
              "cleanup_class", "scene_category", "emotional_tone", "significance"]
    for field in fields:
        present = sum(1 for r in success if r["result"].get(field) not in (None, "", []))
        print(f"  {field:20s}: {present}/{len(success)} ({present/len(success)*100:.0f}%)")

    all_tags = []
    for r in success:
        all_tags.extend(r["result"].get("search_tags", []))
    print(f"\n  --- Search Tags ---")
    print(f"  Total unique tags: {len(set(all_tags))}")
    print(f"  Avg tags per photo: {len(all_tags)/len(success):.1f}")
    print(f"  Top 10: {[t for t, _ in Counter(all_tags).most_common(10)]}")

    cleanup_dist = Counter(r["result"].get("cleanup_class", "unknown") for r in success)
    print(f"\n  --- Cleanup Classification ---")
    for cls, cnt in cleanup_dist.most_common():
        print(f"  {cls:10s}: {cnt} ({cnt/len(success)*100:.0f}%)")

    print(f"\n  --- Results by Category ---")
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "success": 0, "time": []}
        by_cat[cat]["total"] += 1
        by_cat[cat]["time"].append(r["elapsed_seconds"])
        if r["result"]:
            by_cat[cat]["success"] += 1

    for cat, stats in sorted(by_cat.items()):
        rate = stats["success"] / stats["total"] * 100
        avg_t = sum(stats["time"]) / len(stats["time"])
        print(f"  {cat:20s}: {stats['success']}/{stats['total']} ({rate:.0f}%) avg {avg_t:.1f}s")

    print(f"\n  --- Sample Narratives (review for quality) ---")
    random.seed(42)
    samples = random.sample(success, min(10, len(success)))
    for i, r in enumerate(samples):
        ctx = r["context"]
        res = r["result"]
        print(f"\n  [{i+1}] {r['filename']} ({r['category']})")
        print(f"      Date: {ctx['date']} | Location: {ctx['location']}")
        print(f"      Narrative: {res.get('narrative', '?')}")
        print(f"      Event: {res.get('event_hint', '?')}")
        print(f"      Tags: {res.get('search_tags', [])}")
        print(f"      Cleanup: {res.get('cleanup_class', '?')}")
        print(f"      OCR: {res.get('text_summary', '-')}")

    print(f"\n{'='*70}")
    print(f"  KILL GATE CHECKLIST (human review required):")
    print(f"  [ ] Narrative accuracy >= 70% (check 10 samples above)")
    print(f"  [ ] Search tags useful >= 80% (would you find photos with these?)")
    print(f"  [ ] Screenshot OCR readable (check screenshot samples)")
    print(f"  [ ] Processing speed acceptable ({avg_time:.1f}s/photo)")
    print(f"{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        files = sorted(glob.glob("eval/results/eval-*.json"))
        if not files:
            print("No eval results found. Run eval/run_eval.py first.")
            sys.exit(1)
        path = files[-1]
    else:
        path = sys.argv[1]

    print(f"Loading: {path}")
    results = load_results(path)
    score(results)

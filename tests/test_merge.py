"""Tests for face cluster merge suggestions and operations."""
import json
from photo_memory.merge import suggest_merges, merge_clusters


def _person(fc_id, photo_count=10, first="2024-01-01T12:00:00", last="2024-06-01T12:00:00",
            co_appearances=None, user_name=None, apple_name=None):
    return {
        "face_cluster_id": fc_id,
        "apple_name": apple_name,
        "user_name": user_name,
        "photo_count": photo_count,
        "first_seen": first,
        "last_seen": last,
        "co_appearances": json.dumps(co_appearances or {}),
        "top_locations": "[]",
        "appearance_trend": "stable",
    }


def test_suggest_merges_returns_pairs_with_high_co_network_overlap():
    """Two clusters that share many co-appearances but never appear together → candidates."""
    people = [
        _person("fc_A", photo_count=20, co_appearances={"fc_X": 15, "fc_Y": 10, "fc_Z": 8}),
        _person("fc_B", photo_count=18, co_appearances={"fc_X": 12, "fc_Y": 9, "fc_Z": 6}),
        _person("fc_X", photo_count=100, co_appearances={"fc_A": 15, "fc_B": 12}),
        _person("fc_Y", photo_count=80),
        _person("fc_Z", photo_count=60),
    ]

    suggestions = suggest_merges(people, min_photos=5, min_shared_contacts=2)
    pair_ids = [(s["fc_a"], s["fc_b"]) for s in suggestions]
    assert ("fc_A", "fc_B") in pair_ids or ("fc_B", "fc_A") in pair_ids


def test_suggest_merges_skips_when_clusters_co_appear():
    """If fc_A and fc_B appear together in photos, they are different people."""
    people = [
        _person("fc_A", photo_count=20, co_appearances={"fc_B": 5, "fc_X": 10}),
        _person("fc_B", photo_count=18, co_appearances={"fc_A": 5, "fc_X": 8}),
        _person("fc_X", photo_count=50),
    ]
    suggestions = suggest_merges(people, min_photos=5, min_shared_contacts=1)
    pair_ids = [(s["fc_a"], s["fc_b"]) for s in suggestions]
    assert ("fc_A", "fc_B") not in pair_ids
    assert ("fc_B", "fc_A") not in pair_ids


def test_suggest_merges_skips_low_photo_count():
    people = [
        _person("fc_A", photo_count=3, co_appearances={"fc_X": 2}),
        _person("fc_B", photo_count=2, co_appearances={"fc_X": 1}),
        _person("fc_X", photo_count=50),
    ]
    suggestions = suggest_merges(people, min_photos=5)
    assert suggestions == []


def test_merge_clusters_sets_shared_user_name(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.upsert_person(face_cluster_id="fc_a", apple_name=None, user_name=None,
                     photo_count=10, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_b", apple_name=None, user_name=None,
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")

    merge_clusters(db, "fc_a", "fc_b", name="张三")

    p_a = db.execute("SELECT user_name FROM people WHERE face_cluster_id = 'fc_a'").fetchone()
    p_b = db.execute("SELECT user_name FROM people WHERE face_cluster_id = 'fc_b'").fetchone()
    assert p_a["user_name"] == "张三"
    assert p_b["user_name"] == "张三"
    db.close()

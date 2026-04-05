"""Tests for search and cleanup logic."""
from photo_memory.search import search_photos, list_cleanup_candidates


def test_search_photos_resolves_person_name(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.upsert_person(face_cluster_id="fc_001", apple_name="张三", user_name=None,
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]', description="test")
    db.update_photo_status("p1", "done")

    results = search_photos(db, person="张三")
    assert len(results) == 1
    assert results[0]["uuid"] == "p1"
    db.close()


def test_search_photos_unknown_person_returns_empty(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    results = search_photos(db, person="不存在")
    assert results == []
    db.close()


def test_list_cleanup_candidates_groups_by_class(tmp_path):
    from photo_memory.db import Database
    db = Database(str(tmp_path / "t.db"))
    db.upsert_photo("p1", importance="cleanup", status="done", description="模糊")
    db.upsert_photo("p2", importance="cleanup", status="done", description="重复")
    db.upsert_photo("p3", importance="review", status="done", description="不确定")

    report = list_cleanup_candidates(db)
    assert report["cleanup"]["count"] == 2
    assert report["review"]["count"] == 1
    assert len(report["cleanup"]["photos"]) == 2
    db.close()

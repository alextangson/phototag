import pytest
from photo_memory.dedup import compute_phash, find_duplicate_groups


def test_compute_phash_returns_hex_string(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    result = compute_phash(str(img_path))
    assert isinstance(result, str)
    assert len(result) == 16  # 64-bit hash as hex = 16 chars


def test_identical_images_have_same_hash(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="blue")
    path1 = tmp_path / "a.jpg"
    path2 = tmp_path / "b.jpg"
    img.save(path1)
    img.save(path2)
    assert compute_phash(str(path1)) == compute_phash(str(path2))


def test_find_duplicate_groups_clusters_similar():
    records = [
        {"uuid": "uuid-1", "phash": "aaaaaaaaaaaaaaaa"},
        {"uuid": "uuid-2", "phash": "aaaaaaaaaaaaaaaa"},
        {"uuid": "uuid-3", "phash": "ffffffffffffffff"},
    ]
    groups = find_duplicate_groups(records, threshold=5)
    assert len(groups) == 1
    assert set(groups[0]) == {"uuid-1", "uuid-2"}


def test_find_duplicate_groups_no_dupes():
    records = [
        {"uuid": "uuid-1", "phash": "0000000000000000"},
        {"uuid": "uuid-2", "phash": "ffffffffffffffff"},
    ]
    groups = find_duplicate_groups(records, threshold=5)
    assert len(groups) == 0

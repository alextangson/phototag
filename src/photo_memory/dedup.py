"""Perceptual hash based duplicate photo detection."""

import logging
from itertools import combinations

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


def compute_phash(image_path: str) -> str:
    """Compute perceptual hash of an image, returned as hex string."""
    img = Image.open(image_path)
    h = imagehash.phash(img)
    return str(h)


def _hamming_distance(hash1: str, hash2: str) -> int:
    """Compute hamming distance between two hex hash strings."""
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def find_duplicate_groups(phash_records: list[dict], threshold: int = 5) -> list[list[str]]:
    """Find groups of duplicate photos based on perceptual hash similarity.

    Args:
        phash_records: list of {"uuid": str, "phash": str}
        threshold: maximum hamming distance to consider as duplicate

    Returns:
        list of groups, where each group is a list of UUIDs
    """
    n = len(phash_records)
    adjacency: dict[str, set[str]] = {r["uuid"]: set() for r in phash_records}

    for i, j in combinations(range(n), 2):
        r1, r2 = phash_records[i], phash_records[j]
        dist = _hamming_distance(r1["phash"], r2["phash"])
        if dist <= threshold:
            adjacency[r1["uuid"]].add(r2["uuid"])
            adjacency[r2["uuid"]].add(r1["uuid"])

    visited = set()
    groups = []
    for uuid in adjacency:
        if uuid in visited or not adjacency[uuid]:
            continue
        group = []
        queue = [uuid]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            queue.extend(adjacency[current] - visited)
        if len(group) > 1:
            groups.append(group)

    logger.info(f"Found {len(groups)} duplicate groups from {n} photos")
    return groups

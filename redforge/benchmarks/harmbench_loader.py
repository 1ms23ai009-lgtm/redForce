"""HarmBench behavior dataset loader.

Downloads the official HarmBench behavior dataset from GitHub and
provides stratified sampling for cost-efficient evaluation.
"""

import csv
import hashlib
import io
import os
import random
import logging

import httpx

logger = logging.getLogger(__name__)

HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# HarmBench semantic categories
HARMBENCH_CATEGORIES = [
    "chemical_biological",
    "cybercrime",
    "harassment",
    "harmful",
    "illegal_activity",
    "malware",
    "misinformation",
]


def download_harmbench(cache_dir: str | None = None) -> str:
    """Download the HarmBench CSV from GitHub, caching locally.

    Returns:
        Path to the cached CSV file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, "harmbench_behaviors.csv")
    if os.path.exists(cache_path):
        logger.info(f"Using cached HarmBench data: {cache_path}")
        return cache_path

    logger.info("Downloading HarmBench behaviors from GitHub...")
    try:
        response = httpx.get(HARMBENCH_CSV_URL, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to download HarmBench dataset: {e}")
        raise RuntimeError(
            f"Could not download HarmBench dataset. Check your internet connection.\n"
            f"URL: {HARMBENCH_CSV_URL}\nError: {e}"
        )

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    logger.info(f"Downloaded {len(response.text):,} bytes -> {cache_path}")
    return cache_path


def load_behaviors(
    subset_size: int | None = None,
    categories: list[str] | None = None,
    functional_category: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load HarmBench behaviors, optionally sampling a subset.

    Args:
        subset_size: If set, return a stratified sample of this size.
        categories: Filter to these semantic categories only.
        functional_category: Filter to "standard" or "contextual".
        seed: Random seed for reproducible sampling.

    Returns:
        List of behavior dicts with keys:
          id, text, semantic_category, functional_category
    """
    csv_path = download_harmbench()

    behaviors = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behavior_text = row.get("Behavior", "").strip()
            if not behavior_text:
                continue

            sem_cat = row.get("SemanticCategory", "unknown").strip()
            func_cat = row.get("FunctionalCategory", "standard").strip()

            if categories and sem_cat not in categories:
                continue
            if functional_category and func_cat != functional_category:
                continue

            behaviors.append({
                "id": hashlib.md5(behavior_text.encode()).hexdigest()[:12],
                "text": behavior_text,
                "semantic_category": sem_cat,
                "functional_category": func_cat,
            })

    logger.info(f"Loaded {len(behaviors)} behaviors from HarmBench")

    if subset_size is None or subset_size >= len(behaviors):
        return behaviors

    # Stratified sampling: proportional from each category
    random.seed(seed)
    by_category: dict[str, list] = {}
    for b in behaviors:
        cat = b["semantic_category"]
        by_category.setdefault(cat, []).append(b)

    per_cat = max(1, subset_size // max(1, len(by_category)))
    sampled = []
    for cat, items in by_category.items():
        sampled.extend(random.sample(items, min(per_cat, len(items))))

    # Fill remaining if needed
    if len(sampled) < subset_size:
        sampled_ids = {b["id"] for b in sampled}
        extras = [b for b in behaviors if b["id"] not in sampled_ids]
        random.shuffle(extras)
        sampled.extend(extras[: subset_size - len(sampled)])

    sampled = sampled[:subset_size]
    cats = set(b["semantic_category"] for b in sampled)
    logger.info(
        f"Sampled {len(sampled)} behaviors across {len(cats)} categories"
    )
    return sampled


def get_dataset_info() -> dict:
    """Return metadata about the cached dataset without loading all behaviors."""
    csv_path = download_harmbench()

    total = 0
    categories: dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Behavior", "").strip():
                continue
            total += 1
            cat = row.get("SemanticCategory", "unknown").strip()
            categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_behaviors": total,
        "categories": categories,
        "source": HARMBENCH_CSV_URL,
        "cache_path": csv_path,
    }

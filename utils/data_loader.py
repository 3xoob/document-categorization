"""Dataset assembly for the document categorization/tagging project.

Source data: 20 Newsgroups (English), ~18.8k documents across 20 fine-grained
groups. These are collapsed into 5 parent categories. A stratified subsample
is machine-translated into Spanish (via Helsinki-NLP/opus-mt-en-es) to satisfy
the multi-language requirement, since a freely-downloadable, license-clean
multilingual topic-classification corpus of this size (e.g. MLDoc) is gated
behind an LDC/Reuters license.

Produces:
  data/raw_documents/{en,es}.jsonl        -- untouched source text + metadata
  data/processed_data/{train,val,test}.csv -- cleaned, deduped, split dataset
  data/processed_data/dataset_full.csv     -- full cleaned dataset (pre-split)
"""
import json
import os
import sys

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_preprocessing import normalize_text, dedupe_records

RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_documents")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data")

# Collapse the 20 fine-grained newsgroups into 5 parent categories.
CATEGORY_MAP = {
    "comp.graphics": "technology",
    "comp.os.ms-windows.misc": "technology",
    "comp.sys.ibm.pc.hardware": "technology",
    "comp.sys.mac.hardware": "technology",
    "comp.windows.x": "technology",
    "sci.electronics": "technology",
    "rec.autos": "recreation_sports",
    "rec.motorcycles": "recreation_sports",
    "rec.sport.baseball": "recreation_sports",
    "rec.sport.hockey": "recreation_sports",
    "sci.crypt": "science_health",
    "sci.space": "science_health",
    "sci.med": "science_health",
    "alt.atheism": "politics_society",
    "soc.religion.christian": "politics_society",
    "talk.religion.misc": "politics_society",
    "talk.politics.guns": "politics_society",
    "talk.politics.mideast": "politics_society",
    "talk.politics.misc": "politics_society",
    "misc.forsale": "marketplace",
}
CATEGORIES = sorted(set(CATEGORY_MAP.values()))

ES_DOCS_PER_CATEGORY = 300  # ~1500 translated docs -> solid per-language eval set
MIN_CHARS = 40  # drop near-empty posts after boilerplate stripping
MAX_TRANSLATE_CHARS = 500  # truncate before translation for speed/quality


def load_english_records() -> list[dict]:
    """Fetch the full 20 Newsgroups corpus and map each doc to a parent category."""
    print("Fetching 20 Newsgroups (subset='all')...")
    bunch = fetch_20newsgroups(subset="all", remove=(), random_state=RANDOM_SEED)
    records = []
    for doc_id, (raw_text, target_idx) in enumerate(zip(bunch.data, bunch.target)):
        subcategory = bunch.target_names[target_idx]
        category = CATEGORY_MAP[subcategory]
        records.append(
            {
                "doc_id": f"en_{doc_id}",
                "raw_text": raw_text,
                "category": category,
                "subcategory": subcategory,
                "language": "en",
            }
        )
    print(f"Loaded {len(records)} raw English documents.")
    return records


def save_raw_jsonl(records: list[dict], path: str, text_field: str = "raw_text"):
    """Write records as JSON Lines, copying `text_field` into a `text` key for a stable schema."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({**rec, "text": rec[text_field]}, ensure_ascii=False) + "\n")


def clean_english_records(records: list[dict]) -> list[dict]:
    """Normalize + strip newsgroup boilerplate from each record, dropping near-empty results."""
    cleaned = []
    for rec in records:
        text = normalize_text(rec["raw_text"], strip_boilerplate=True)
        if len(text) < MIN_CHARS:
            continue
        cleaned.append({**rec, "text": text})
    print(f"{len(cleaned)}/{len(records)} English documents survived cleaning.")
    return cleaned


def stratified_sample_for_translation(cleaned_en: list[dict], per_category: int) -> list[dict]:
    """Sample up to `per_category` English docs per category for Spanish translation."""
    df = pd.DataFrame(cleaned_en)
    parts = []
    for _, group in df.groupby("category"):
        parts.append(group.sample(n=min(per_category, len(group)), random_state=RANDOM_SEED))
    sampled = pd.concat(parts).to_dict("records")
    print(f"Selected {len(sampled)} English documents for Spanish translation.")
    return sampled


def load_raw_jsonl(path: str) -> list[dict]:
    """Read a JSON Lines file back into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_splits(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dedupe, then stratify-split (by category+language) into 80/10/10 train/val/test."""
    records = dedupe_records(records, text_key="text")
    df = pd.DataFrame(records)[["doc_id", "text", "category", "subcategory", "language"]]
    df["strata"] = df["category"] + "_" + df["language"]

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["strata"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["strata"]
    )
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(split)} docs")
    return train_df.drop(columns="strata"), val_df.drop(columns="strata"), test_df.drop(columns="strata")


def main():
    """Build the full dataset: fetch/reuse English docs, load pre-translated Spanish docs, split, save."""
    en_path = os.path.join(RAW_DIR, "en.jsonl")
    if os.path.exists(en_path):
        print(f"Reusing cached {en_path}")
        en_raw = load_raw_jsonl(en_path)
    else:
        en_raw = load_english_records()
        save_raw_jsonl(en_raw, en_path)

    en_clean = clean_english_records(en_raw)

    es_path = os.path.join(RAW_DIR, "es.jsonl")
    if not os.path.exists(es_path):
        raise FileNotFoundError(
            f"{es_path} not found. Run `translate_dataset.py` first (in a torch-enabled venv) "
            "to produce the Spanish translations -- TensorFlow's eager-mode generate() is too "
            "slow for this step. See README for details."
        )
    es_clean = [
        {**rec, "text": normalize_text(rec["text"], strip_boilerplate=False)} for rec in load_raw_jsonl(es_path)
    ]
    es_clean = [rec for rec in es_clean if len(rec["text"]) >= MIN_CHARS]
    print(f"Loaded {len(es_clean)} Spanish documents from {es_path}")

    all_records = en_clean + es_clean
    train_df, val_df, test_df = build_splits(all_records)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pd.DataFrame(all_records)[["doc_id", "text", "category", "subcategory", "language"]].to_csv(
        os.path.join(PROCESSED_DIR, "dataset_full.csv"), index=False
    )
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print("\nCategory distribution:")
    print(pd.DataFrame(all_records)["category"].value_counts())
    print("\nLanguage distribution:")
    print(pd.DataFrame(all_records)["language"].value_counts())
    print(f"\nTotal documents: {len(all_records)}")


if __name__ == "__main__":
    main()

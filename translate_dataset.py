"""One-off step: machine-translate a stratified sample of the English corpus
into Spanish using PyTorch (Helsinki-NLP/opus-mt-en-es).

Run this with the lightweight `.venv-mt` environment (torch + transformers
only) -- kept separate from the main `.venv` because torch and
tensorflow[and-cuda] pin conflicting CUDA library versions. TensorFlow's
eager-mode `generate()` was measured at ~0.45 docs/sec for this model on this
GPU; PyTorch's is dramatically faster thanks to proper KV-cache reuse.

Usage:
    source .venv-mt/bin/activate
    python translate_dataset.py
    deactivate
    source .venv/bin/activate   # back to the TF venv for the rest of the pipeline
"""
import json
import os
import time

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

from utils.data_loader import (
    ES_DOCS_PER_CATEGORY,
    MAX_TRANSLATE_CHARS,
    MIN_CHARS,
    RANDOM_SEED,
    RAW_DIR,
    clean_english_records,
    load_raw_jsonl,
    save_raw_jsonl,
    stratified_sample_for_translation,
)


def translate_to_spanish(records: list[dict], batch_size: int = 16) -> list[dict]:
    """Greedy-decode MarianMT translations for each record's text, batched."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Helsinki-NLP/opus-mt-en-es"
    print(f"Loading {model_name} (PyTorch, device={device})...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = [rec["text"][:MAX_TRANSLATE_CHARS] for rec in records]
    translated_texts = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            t0 = time.time()
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
            out = model.generate(**inputs, max_length=200, num_beams=1, do_sample=False)
            translated_texts.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            print(f"  translated {i + len(batch)}/{len(texts)} ({time.time() - t0:.1f}s for this batch)")

    es_records = []
    for rec, es_text in zip(records, translated_texts):
        if len(es_text) < MIN_CHARS:
            continue
        es_records.append(
            {
                "doc_id": rec["doc_id"].replace("en_", "es_"),
                "raw_text": es_text,
                "text": es_text,
                "category": rec["category"],
                "subcategory": rec["subcategory"],
                "language": "es",
                "source_doc_id": rec["doc_id"],
            }
        )
    print(f"Produced {len(es_records)} Spanish documents.")
    return es_records


def main():
    """Sample the cleaned English corpus and translate it to Spanish, saving raw_documents/es.jsonl."""
    en_path = os.path.join(RAW_DIR, "en.jsonl")
    if not os.path.exists(en_path):
        raise FileNotFoundError(f"{en_path} not found -- run `python -m utils.data_loader` first to fetch it.")
    en_raw = load_raw_jsonl(en_path)
    en_clean = clean_english_records(en_raw)
    es_source = stratified_sample_for_translation(en_clean, ES_DOCS_PER_CATEGORY)
    es_clean = translate_to_spanish(es_source)
    save_raw_jsonl(es_clean, os.path.join(RAW_DIR, "es.jsonl"))
    print(pd.DataFrame(es_clean)["category"].value_counts())


if __name__ == "__main__":
    main()

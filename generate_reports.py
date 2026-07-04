"""Generate reports/performance_metrics.json and reports/example_predictions.csv
from the trained checkpoint, per the project validation spec. Run after
utils/transfer_learning.py has produced models/checkpoints/*.
"""
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from utils.realtime_pipeline import DocumentPipeline

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data")

BENCHMARK_SAMPLE_SIZE = 500
EXAMPLE_SAMPLE_SIZE = 30


def evaluate_accuracy(pipeline: DocumentPipeline, test_df: pd.DataFrame) -> dict:
    """Run the given pipeline over the full test set and compute accuracy/F1/per-language."""
    predictions = pipeline.predict_batch(test_df["text"].tolist())
    y_true = test_df["category"].tolist()
    y_pred = [p["category"] for p in predictions]
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    per_lang = {}
    for lang in sorted(test_df["language"].unique()):
        idx = test_df["language"].values == lang
        per_lang[lang] = accuracy_score(
            [t for t, keep in zip(y_true, idx) if keep], [p for p, keep in zip(y_pred, idx) if keep]
        )
    return {"accuracy": accuracy, "f1_macro": f1_macro, "per_language_accuracy": per_lang}


def main():
    """Benchmark the trained pipeline and write performance_metrics.json + example_predictions.csv.

    Accuracy/F1/per-language are measured with the 3-model ensemble (higher
    accuracy, ~3x classification cost); throughput is measured with the
    single fastest checkpoint (the real-time serving default). These are two
    different, individually valid serving configurations -- reporting one
    pipeline's throughput next to a *different* pipeline's accuracy would be
    misleading, so both are labeled explicitly in the output.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
        config = json.load(f)

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    print("Loading fast (single-checkpoint) pipeline for throughput benchmark...")
    pipeline = DocumentPipeline(checkpoint_dir=CHECKPOINT_DIR, fast_tagging=True, ensemble=False)

    bench_texts = test_df["text"].sample(
        n=min(BENCHMARK_SAMPLE_SIZE, len(test_df)), random_state=42
    ).tolist()
    print(f"Benchmarking throughput on {len(bench_texts)} documents...")
    bench = pipeline.benchmark_throughput(bench_texts)
    print(f"Throughput: {bench['docs_per_second']:.1f} docs/sec")

    # Release the fast pipeline's ONNX Runtime CUDA arena before loading 3 more
    # models -- ONNX Runtime doesn't grow its GPU memory pool on demand the way
    # TF does with set_memory_growth, so it'll hold onto its allocation until
    # the session object is actually collected.
    del pipeline
    import gc

    gc.collect()

    print("Loading ensemble pipeline for accuracy evaluation (3 checkpoints)...")
    ensemble_pipeline = DocumentPipeline(checkpoint_dir=CHECKPOINT_DIR, fast_tagging=True, ensemble=True)
    print(f"Evaluating on the full {len(test_df)}-document test set...")
    ensemble_eval = evaluate_accuracy(ensemble_pipeline, test_df)
    print(f"Ensemble test accuracy={ensemble_eval['accuracy']:.4f}  macro-F1={ensemble_eval['f1_macro']:.4f}")

    metrics = {
        "classification_accuracy": round(ensemble_eval["accuracy"], 4),
        "f1_score_macro": round(ensemble_eval["f1_macro"], 4),
        "accuracy_measured_with": "3-model ensemble (models/checkpoints/ensemble/)",
        "processing_speed_docs_per_sec": round(bench["docs_per_second"], 1),
        "processing_speed_measured_with": "single checkpoint (models/checkpoints/text_classifier_best.h5)",
        "languages_supported": config["languages_supported"],
        "per_language_accuracy": {k: round(v, 4) for k, v in ensemble_eval["per_language_accuracy"].items()},
    }

    baseline_path = os.path.join(CHECKPOINT_DIR, "baseline_metrics.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        metrics["baseline_accuracy"] = round(baseline["accuracy"], 4)
        metrics["improvement_over_baseline"] = round(metrics["classification_accuracy"] - baseline["accuracy"], 4)

    with open(os.path.join(REPORTS_DIR, "performance_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {os.path.join(REPORTS_DIR, 'performance_metrics.json')}")
    print(json.dumps(metrics, indent=2))

    example_df = test_df.sample(n=min(EXAMPLE_SAMPLE_SIZE, len(test_df)), random_state=7).reset_index(drop=True)
    predictions = ensemble_pipeline.predict_batch(example_df["text"].tolist())
    rows = []
    for (_, row), pred in zip(example_df.iterrows(), predictions):
        rows.append(
            {
                "doc_id": row["doc_id"],
                "true_category": row["category"],
                "language": row["language"],
                "predicted_category": pred["category"],
                "confidence": pred["confidence"],
                "correct": pred["category"] == row["category"],
                "tags": "; ".join(pred["tags"]),
                "text_preview": row["text"][:150].replace("\n", " "),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(REPORTS_DIR, "example_predictions.csv"), index=False)
    print(f"Wrote {os.path.join(REPORTS_DIR, 'example_predictions.csv')}")


if __name__ == "__main__":
    main()

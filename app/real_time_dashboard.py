"""Streamlit dashboard for the document categorization/tagging system.

Run with: streamlit run app/real_time_dashboard.py
"""
import json
import os
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.realtime_pipeline import DocumentPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")

st.set_page_config(page_title="Document Categorization & Tagging", layout="wide")


@st.cache_resource
def load_pipeline():
    """Load the trained classifier + tagger once per session (expensive: loads DistilBERT + spaCy)."""
    return DocumentPipeline(checkpoint_dir=CHECKPOINT_DIR, fast_tagging=True)


@st.cache_data
def load_metrics():
    """Load reports/performance_metrics.json, or None if it hasn't been generated yet."""
    path = os.path.join(REPORTS_DIR, "performance_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_example_predictions():
    """Load reports/example_predictions.csv, or None if it hasn't been generated yet."""
    path = os.path.join(REPORTS_DIR, "example_predictions.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_test_sample(n=300):
    """A fixed random sample of the held-out test set, for the corpus-overview charts."""
    path = os.path.join(PROCESSED_DIR, "test.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.sample(n=min(n, len(df)), random_state=1)


@st.cache_data
def load_tag_frequency(_pipeline, n=150):
    """Top tags across a corpus sample -- `_pipeline` is unhashed (leading
    underscore) since Streamlit can't hash a loaded model/spaCy pipeline."""
    df = load_test_sample(n)
    if df is None:
        return None
    from collections import Counter

    counts = Counter()
    for result in _pipeline.predict_batch(df["text"].tolist()):
        counts.update(result["tags"])
    if not counts:
        return None
    return pd.DataFrame(counts.most_common(20), columns=["tag", "count"])


if "history" not in st.session_state:
    st.session_state.history = []

st.title("📄 Intelligent Document Categorization & Tagging")
st.caption("DistilBERT (multilingual) classifier + spaCy context-aware tagging — real-time dashboard")

metrics = load_metrics()

# ---- Top metrics row ----
st.subheader("Performance Metrics")
if metrics:
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['classification_accuracy'] * 100:.1f}%")
    cols[1].metric("Macro F1", f"{metrics['f1_score_macro']:.3f}")
    cols[2].metric("Speed", f"{metrics['processing_speed_docs_per_sec']:.0f} docs/s")
    if "improvement_over_baseline" in metrics:
        cols[3].metric("vs. Baseline", f"+{metrics['improvement_over_baseline'] * 100:.1f} pts")
    cols[4].metric("Languages", ", ".join(metrics["languages_supported"]))

    lang_cols = st.columns(len(metrics["per_language_accuracy"]))
    for col, (lang, acc) in zip(lang_cols, metrics["per_language_accuracy"].items()):
        col.metric(f"Accuracy ({lang})", f"{acc * 100:.1f}%")
else:
    st.warning("No reports/performance_metrics.json found yet. Run `python generate_reports.py` after training.")

st.divider()

# ---- Real-time categorization ----
left, right = st.columns([1, 1])

with left:
    st.subheader("Real-Time Categorization")
    default_text = (
        "NASA announced today that its upcoming mission will study radiation "
        "effects on new spacecraft materials. The team, led by Dr. Elena Ruiz, "
        "will present findings at next month's conference in Houston."
    )
    text_input = st.text_area("Paste a document (English or Spanish):", value=default_text, height=180)
    run_btn = st.button("Categorize & Tag", type="primary")

    if run_btn and text_input.strip():
        try:
            pipeline = load_pipeline()
        except (FileNotFoundError, RuntimeError, OSError) as e:
            st.error(f"Pipeline failed to load: {e}")
            pipeline = None

        if pipeline is not None:
            start = time.perf_counter()
            result = pipeline.predict(text_input)
            latency_ms = (time.perf_counter() - start) * 1000

            if result.get("error"):
                st.error(f"Could not process this document: {result['error']}")
            else:
                st.session_state.history.append(result)

                st.success(f"**Category:** {result['category']}  ·  **Confidence:** {result['confidence'] * 100:.1f}%")
                st.write(f"**Detected language:** `{result['language']}`  ·  **Latency:** {latency_ms:.0f} ms")
                st.write(
                    "**Tags:**", ", ".join(f"`{t}`" for t in result["tags"]) if result["tags"] else "_none found_"
                )
                if result["entities"]:
                    ent_df = pd.DataFrame(result["entities"]).drop_duplicates()
                    st.write("**Named entities:**")
                    st.dataframe(ent_df, hide_index=True, use_container_width=True)
                with st.expander("Why this category / these tags?"):
                    st.markdown(
                        "- Category is predicted by the fine-tuned DistilBERT model from the pooled document "
                        "representation.\n"
                        "- Tags combine spaCy Named Entity Recognition with frequency-ranked keywords, "
                        "boosted when they match vocabulary typical of the predicted category (context-aware "
                        "ranking)."
                    )

with right:
    st.subheader("Session History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            hist_df[["language", "category", "confidence", "tags"]],
            hide_index=True,
            use_container_width=True,
            height=200,
        )
        cat_counts = hist_df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        st.plotly_chart(px.bar(cat_counts, x="category", y="count", title="Session category counts"), use_container_width=True)
    else:
        st.info("Categorize a document to start building session history.")

st.divider()

# ---- Corpus-level visualizations ----
st.subheader("Corpus Overview (test set sample)")
test_sample = load_test_sample()
if test_sample is not None:
    v1, v2 = st.columns(2)
    with v1:
        cat_dist = test_sample["category"].value_counts().reset_index()
        cat_dist.columns = ["category", "count"]
        st.plotly_chart(px.pie(cat_dist, names="category", values="count", title="Category distribution"), use_container_width=True)
    with v2:
        lang_dist = test_sample["language"].value_counts().reset_index()
        lang_dist.columns = ["language", "count"]
        st.plotly_chart(px.pie(lang_dist, names="language", values="count", title="Language distribution"), use_container_width=True)

    tag_freq = load_tag_frequency(load_pipeline())
    if tag_freq is not None:
        st.plotly_chart(
            px.bar(tag_freq, x="tag", y="count", title="Most common tags (corpus sample)"),
            use_container_width=True,
        )

examples = load_example_predictions()
if examples is not None:
    st.subheader("Example Predictions (held-out test set)")
    accuracy_on_sample = examples["correct"].mean()
    st.write(f"Sample accuracy: **{accuracy_on_sample * 100:.1f}%** ({len(examples)} documents)")
    st.dataframe(examples, hide_index=True, use_container_width=True)

st.divider()
history_csv = os.path.join(CHECKPOINT_DIR, "training_history.csv")
if os.path.exists(history_csv):
    st.subheader("Training History")
    hist = pd.read_csv(history_csv)
    st.plotly_chart(
        px.line(hist, x="epoch", y=["accuracy", "val_accuracy"], title="Accuracy over epochs"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(hist, x="epoch", y=["loss", "val_loss"], title="Loss over epochs"),
        use_container_width=True,
    )

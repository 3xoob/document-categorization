# Document Categorization & Tagging

An intelligent document categorization and tagging system built with a
fine-tuned multilingual DistilBERT classifier (5 categories, English +
Spanish) paired with a spaCy-based context-aware tagger (Named Entity
Recognition + keyword extraction), served through a real-time Streamlit
dashboard.

## Highlights

- **Multilingual transformer classification** — `distilbert-base-multilingual-cased`
  fine-tuned via transfer learning, reaching **91.14% test accuracy** and a
  **macro F1 of 0.8991**, a **+5.07 point** improvement over a tuned
  TF-IDF + Logistic Regression baseline.
- **Strong per-language performance** — 90.87% (English) and 94.63%
  (Spanish) accuracy on held-out data.
- **Model ensembling** — three independently-trained checkpoints combined
  by a validation-set-selected weighted average, improving on any single
  model's accuracy.
- **Context-aware tagging** — Named Entity Recognition merged with
  frequency-ranked, category-boosted keywords for tags that reflect both
  what a document mentions and what it's about.
- **Optimized real-time inference** — a custom ONNX Runtime export path
  (PyTorch re-conversion + BERT-specific operator fusion) plus a
  producer/consumer GPU/CPU pipeline deliver up to **145 docs/sec**
  classification throughput and **~100 docs/sec** end-to-end
  (classification + tagging).
- **Production-grade error handling** — the pipeline isolates and reports
  per-document failures instead of crashing a whole batch, and the
  dashboard degrades gracefully on load/predict errors.
- **Interactive dashboard** — live categorization and tagging, performance
  metrics, corpus visualizations, and training curves.

## Dataset

- **Source:** [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) (~18.8k English documents, 20 fine-grained groups).
- **Categories:** the 20 groups are collapsed into 5 parent categories: `technology`, `recreation_sports`, `science_health`, `politics_society`, `marketplace`.
- **Languages:** English (native) + Spanish. A stratified subsample (~550/category) is machine-translated into Spanish with `Helsinki-NLP/opus-mt-en-es`, since freely-downloadable multilingual topic-classification corpora at this scale (e.g. MLDoc) are gated behind an LDC/Reuters license.
- Resulting corpus: 10,000+ documents, 5 categories, 2 languages — see `data/processed_data/dataset_full.csv` after running the data pipeline.

## Project layout

```text
data/                   raw + processed documents
models/                 classifier architecture, tagger, trained checkpoints
notebooks/              EDA + training walkthrough (executed, with outputs)
reports/                performance_metrics.json, example_predictions.csv
utils/                  data loading, preprocessing, transfer learning, real-time pipeline
app/                    Streamlit dashboard

generate_reports.py     generates the reports/ artifacts from a trained checkpoint
export_onnx.py          converts a trained checkpoint to the accelerated ONNX format
translate_dataset.py    builds the Spanish portion of the dataset
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

Requires an internet connection on first run (downloads 20 Newsgroups, the
DistilBERT/MarianMT checkpoints, and spaCy models). A CUDA GPU is used
automatically if available; CPU-only also works.

**Optional: accelerated real-time inference via ONNX Runtime.** This uses a
second, small venv, since `onnxruntime-gpu` needs cuDNN 9 while
`tensorflow[and-cuda]` pins cuDNN 8 — both coexist fine side by side on
`LD_LIBRARY_PATH` (different `.so` filenames), but pip can't resolve two
pinned versions of the same package into one venv:

```bash
python3 -m venv .venv-onnx
.venv-onnx/bin/pip install nvidia-cudnn-cu12 onnx onnxruntime-gpu==1.19.2 \
    torch --index-url https://download.pytorch.org/whl/cpu
.venv-onnx/bin/pip install --index-url https://pypi.org/simple \
    transformers==4.42.4 tf-keras tensorflow-cpu "numpy<2" "scipy<1.14"
```

`.venv/bin/activate` already points at `.venv-onnx`'s cuDNN 9 for you.
Skipping this is fine — `DocumentPipeline` falls back to the TF Keras
model automatically, and only uses ONNX Runtime if
`models/checkpoints/model.onnx` exists (see step 3 below).

## Running the pipeline end to end

```bash
# 1. Build the dataset (fetch, clean, translate, split)
python -m utils.data_loader

# 2. Train the TF-IDF baseline + fine-tune DistilBERT (>=5 epochs)
python -m utils.transfer_learning

# 3. Export a fused ONNX model for accelerated real-time inference (optional
#    but recommended -- see "Real-time inference backend" below)
python -c "
import json, numpy as np
from models.text_classifier import build_model
with open('models/checkpoints/config.json') as f:
    config = json.load(f)
model = build_model(num_labels=config['num_labels'], max_len=config['max_len'])
model.load_weights('models/checkpoints/text_classifier_best.h5')
model.get_layer('base_transformer').save_pretrained('models/checkpoints/_tf_base_export')
w1, b1 = model.get_layer('pre_classifier').get_weights()
w2, b2 = model.get_layer('category_output').get_weights()
np.savez('models/checkpoints/_head_weights.npz', w1=w1, b1=b1, w2=w2, b2=b2)
"
deactivate
source .venv-onnx/bin/activate
unset LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=""
python export_onnx.py
deactivate
source .venv/bin/activate

# 4. Generate reports/performance_metrics.json + example_predictions.csv
python generate_reports.py

# 5. Launch the dashboard
streamlit run app/real_time_dashboard.py
```

Or work through `notebooks/EDA_and_Training.ipynb`, which documents EDA and
runs the same training steps interactively with plots and commentary.

## Model

- **Base:** `distilbert-base-multilingual-cased` (Hugging Face `transformers`, TensorFlow backend).
- **Head:** masked mean-pooling over token embeddings → Dense(256, relu) → Dropout(0.4) → Dense(5, softmax), trained with the base unfrozen.
- **Training:** 6 epochs, class-balanced weighting, `max_len=256` (59% of documents exceed 128 tokens, so a shorter length was measurably truncating signal). `utils/transfer_learning.py` uses an `AdamW` + linear-warmup/decay learning-rate schedule (implemented in `models/text_classifier.py`) to control overfitting; best checkpoint selected by validation accuracy.
- **Checkpoints:** `models/checkpoints/text_classifier_best.h5` (single best checkpoint, 90.15% test accuracy — used for real-time serving), `text_classifier_epoch{N}.h5` (per-epoch), `config.json` (label maps + metrics), `training_history.csv` (per-epoch loss/accuracy).
- **Ensemble** (`models/checkpoints/ensemble/`): three independently-trained checkpoints combined by a weighted average of their softmax outputs. Weights are chosen by grid search on the *validation* set, then applied once to the held-out test set for the reported number — no test-set leakage. `DocumentPipeline(ensemble=True)` loads and uses it; `generate_reports.py` uses it for the reported accuracy/F1 numbers.

## Real-time inference backend

`DocumentPipeline(ensemble=False)` (the default) loads
`models/checkpoints/model.onnx` via ONNX Runtime's `CUDAExecutionProvider`
instead of calling the TF Keras model directly, when that file exists
(export step in Setup above), falling back to the TF model automatically
otherwise.

The shipped `model.onnx` is re-exported through PyTorch rather than
converted directly from the trained Keras model. The trained DistilBERT
encoder weights convert losslessly via `transformers`' `from_tf=True`
loader; the small custom head (2 Dense layers) is ported by hand
(`kernel.T` for TF→PyTorch's transposed `Linear` weight convention) and
verified to match the original TF model's output to 6 decimal places. This
path lets ONNX Runtime's BERT fusion optimizer
(`onnxruntime.transformers.optimizer`) fuse Gelu and LayerNorm operations
that a direct TensorFlow export leaves as separate kernels. Measured on
the full test set with predictions verified identical to the original TF
model: up to **~145 docs/sec** for classification alone.

`DocumentPipeline.__init__` also runs a short GPU warm-up pass at load
time (synthetic dummy input, ~2-3 seconds, nothing per request) so the
classifier's GPU kernels and boost clock are already at steady state
before the first real prediction — not just for benchmarking, but so real
user-facing requests are fast from the start.

## Tagging

`models/tagger.py` runs a language-specific spaCy pipeline per document:
Named Entity Recognition (PERSON/ORG/GPE/...) plus stopword-filtered
keyword frequency, merged into a single ranked tag list. Candidates whose
words match a small category-specific seed vocabulary are boosted, so tags
reflect the document's predicted topic, not just raw term frequency. The
real-time pipeline strips the tagger/parser/lemmatizer pipes, keeping only
tokenization + NER for speed.

For batches of 32+ documents, `DocumentPipeline._predict_batch_pipelined`
(in `utils/realtime_pipeline.py`) runs classification (GPU) for one chunk
concurrently with tagging (CPU) for the previous chunk, instead of
strictly sequentially — they're different hardware resources and both
release the GIL during their real work, so this is genuine overlap,
measured at ~1.6x over running them one after another end to end. Below
that batch size (e.g. the dashboard's single-document case), the plain
sequential path is used since thread setup costs more than it saves.

## Error handling

- `DocumentPipeline` raises a clear, actionable error if checkpoint files are
  missing or corrupted, instead of a raw stack trace.
- `DocumentPipeline.predict_batch` isolates failures: if a batch call raises
  (malformed input, unexpected encoding, etc.), it retries documents one at a
  time so the rest of the batch still succeeds, and returns an `"error"` field
  on the ones that don't.
- `DocumentTagger` raises a message pointing at the exact `spacy download`
  command if a language model isn't installed.
- `TerminateOnNaN` stops training early on a diverged loss instead of burning
  hours on a broken run.
- The dashboard catches pipeline load/predict failures and shows `st.error`
  instead of crashing the app.

## Reports

`reports/performance_metrics.json` follows:

```json
{
  "classification_accuracy": 0.9114,
  "f1_score_macro": 0.8991,
  "accuracy_measured_with": "3-model ensemble (models/checkpoints/ensemble/)",
  "processing_speed_docs_per_sec": 100.0,
  "processing_speed_measured_with": "single checkpoint (models/checkpoints/text_classifier_best.h5)",
  "languages_supported": ["en", "es"],
  "per_language_accuracy": {"en": 0.9087, "es": 0.9463}
}
```

Accuracy/F1/per-language are measured with the ensemble (higher accuracy);
throughput is measured with the single fastest checkpoint (the real-time
serving default) — two different, clearly-labeled serving configurations
rather than one pipeline measured twice.

`reports/example_predictions.csv` has a sample of held-out test documents
with true/predicted category, confidence, tags, and correctness.

## Dashboard

`streamlit run app/real_time_dashboard.py` shows: live categorization + tagging
for pasted text, performance metrics, category/language/tag-frequency
distribution charts, training history curves, and example predictions from
the held-out set.

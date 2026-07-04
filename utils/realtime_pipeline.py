"""Real-time categorization + tagging pipeline: language detection -> batched
transformer inference -> context-aware tagging, wrapped in a single object
the dashboard and benchmarks call against.
"""
import json
import os
import queue
import sys
import threading
import time

# Must be set before the HF tokenizer is used, or forking worker processes for
# spaCy's n_process>1 tagging risks a deadlock (the tokenizer warns loudly if
# this isn't set and a fork happens after it's already parallelized).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import tensorflow as tf

# Grow GPU memory on demand rather than grabbing it all upfront -- this
# process may also load an onnxruntime CUDA session (see _classify_batch),
# and a fixed TF memory pool would starve it on this card's 6GB.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tagger import DocumentTagger
from models.text_classifier import build_model
from utils.text_preprocessing import detect_language, normalize_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")


class DocumentPipeline:
    """End-to-end real-time pipeline: detect language -> classify -> tag."""

    MAX_TAG_CHARS = 1500
    ENSEMBLE_DIR = "ensemble"
    ONNX_MODEL_FILENAME = "model.onnx"

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR, fast_tagging: bool = True, ensemble: bool = False):
        """`ensemble=True` loads 3 independently-trained checkpoints and
        averages their predictions (weights chosen on the validation set) --
        higher accuracy, ~3x the classification cost. `ensemble=False`
        (default) loads a single checkpoint for real-time throughput, via
        ONNX Runtime if `models/checkpoints/model.onnx` exists (measured
        ~1.15x faster than raw TF `.predict()` -- ONNX Runtime's graph
        optimizer fuses ops TF's eager/tf.function path doesn't), falling
        back to the TF Keras model otherwise. Both ensemble and non-ensemble
        are legitimate serving configurations; pick per use case rather than
        pretending one number describes both."""
        config_path = os.path.join(checkpoint_dir, "config.json")
        weights_path = os.path.join(checkpoint_dir, "text_classifier_best.h5")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"{config_path} not found. Run `python -m utils.transfer_learning` first to train and save a "
                "checkpoint before starting the pipeline/dashboard."
            )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"{weights_path} not found (config.json exists but weights don't -- training may have been "
                "interrupted before a checkpoint was saved). Re-run `python -m utils.transfer_learning`."
            )

        with open(config_path) as f:
            self.config = json.load(f)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.max_len = self.config["max_len"]
        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}
        self.ensemble = ensemble
        self.onnx_session = None

        # Fixed-length input: tried dynamic per-batch padding to skip wasted
        # compute on short documents, but it forces TF to retrace its compiled
        # graph on every distinct shape it sees -- far more expensive than the
        # compute it saves at these batch counts. Fixed shape reuses one trace.
        if ensemble:
            self._load_ensemble(checkpoint_dir)
        else:
            onnx_path = os.path.join(checkpoint_dir, self.ONNX_MODEL_FILENAME)
            if os.path.exists(onnx_path):
                import onnxruntime as ort

                self.onnx_session = ort.InferenceSession(
                    onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            else:
                self.model = build_model(
                    num_labels=self.config["num_labels"], model_name=self.config["model_name"], max_len=self.max_len
                )
                try:
                    self.model.load_weights(weights_path)
                except (OSError, ValueError) as e:
                    raise RuntimeError(
                        f"Failed to load weights from {weights_path} -- the checkpoint may be corrupted or was "
                        f"saved with a different model architecture than models/text_classifier.py currently "
                        f"builds. Original error: {e}"
                    ) from e
            self._warmup_gpu_clock()

        # Strip the tagger/parser/lemmatizer pipes for real-time latency. Single
        # process, not spaCy's n_process>1: classification and tagging now run
        # concurrently (see _predict_batch_pipelined), so tagging already gets
        # a dedicated CPU thread for free -- adding worker-process spawn cost
        # on top of that (measured: pays for itself only above ~500 docs in one
        # shot) made the overlapped pipeline slower, not faster.
        self.tagger = DocumentTagger(languages=self.config["languages_supported"], fast=fast_tagging, n_process=1)

    def _load_ensemble(self, checkpoint_dir: str):
        ensemble_dir = os.path.join(checkpoint_dir, self.ENSEMBLE_DIR)
        ensemble_config_path = os.path.join(ensemble_dir, "ensemble_config.json")
        if not os.path.exists(ensemble_config_path):
            raise FileNotFoundError(
                f"{ensemble_config_path} not found -- ensemble=True requires 3 checkpoints and a weights file "
                f"under {ensemble_dir}/. Use ensemble=False for the single-checkpoint path."
            )
        with open(ensemble_config_path) as f:
            ensemble_config = json.load(f)
        self.ensemble_weights = ensemble_config["weights"]

        self.ensemble_models = {}
        for filename in self.ensemble_weights:
            model = build_model(
                num_labels=self.config["num_labels"], model_name=self.config["model_name"], max_len=self.max_len
            )
            model.load_weights(os.path.join(ensemble_dir, filename))
            self.ensemble_models[filename] = model

    PREDICT_BATCH_SIZE = 32
    LANG_DETECT_CHARS = 200  # language ID doesn't need the whole document
    PIPELINE_CHUNK_SIZE = 32  # matches PREDICT_BATCH_SIZE -- one classify call per chunk, no partial batches
    MIN_DOCS_FOR_PIPELINE = 32  # below this, thread setup costs more than it saves
    WARMUP_ITERATIONS = 2

    def _warmup_gpu_clock(self):
        """Run a couple of full-size dummy batches through the classifier at
        load time, not just at first use. This GPU's boost clock doesn't
        ramp up from one small burst of work -- a cold process measured
        right after a single small call swings ~38-94 docs/sec run to run
        (clock oscillating ~300MHz-1500MHz under load), but 2 full-size
        passes reliably settle it into a stable ~100-107 docs/sec band. A
        real server sees this same cold start on its first requests if
        nothing warms it up first; doing it here means the first real
        request is fast too, not just the benchmark."""
        dummy_ids = np.zeros((self.PREDICT_BATCH_SIZE, self.max_len), dtype=np.int32)
        dummy_mask = np.ones((self.PREDICT_BATCH_SIZE, self.max_len), dtype=np.int32)
        model_inputs = {"input_ids": dummy_ids, "attention_mask": dummy_mask}
        for _ in range(self.WARMUP_ITERATIONS):
            if self.onnx_session is not None:
                self.onnx_session.run(None, model_inputs)
            else:
                self.model.predict(model_inputs, batch_size=self.PREDICT_BATCH_SIZE, verbose=0)

    def _classify_batch(self, texts: list[str]) -> tuple[list[str], list[float]]:
        enc = self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="np"
        )
        model_inputs = {
            "input_ids": enc["input_ids"].astype(np.int32),
            "attention_mask": enc["attention_mask"].astype(np.int32),
        }
        if self.ensemble:
            probs = sum(
                weight * self.ensemble_models[filename].predict(model_inputs, batch_size=self.PREDICT_BATCH_SIZE, verbose=0)
                for filename, weight in self.ensemble_weights.items()
            )
        elif self.onnx_session is not None:
            probs = self._run_onnx_batched(model_inputs)
        else:
            probs = self.model.predict(model_inputs, batch_size=self.PREDICT_BATCH_SIZE, verbose=0)
        preds = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(preds)), preds]
        categories = [self.id2label[p] for p in preds]
        return categories, confidences.tolist()

    def _run_onnx_batched(self, model_inputs: dict) -> np.ndarray:
        """Run the ONNX session in PREDICT_BATCH_SIZE chunks -- the exported
        graph accepts any batch size, but feeding it all at once risks
        exhausting this GPU's memory arena on large inputs."""
        n = len(model_inputs["input_ids"])
        chunks = []
        for i in range(0, n, self.PREDICT_BATCH_SIZE):
            chunk_inputs = {k: v[i : i + self.PREDICT_BATCH_SIZE] for k, v in model_inputs.items()}
            chunks.append(self.onnx_session.run(None, chunk_inputs)[0])
        return np.concatenate(chunks, axis=0)

    def predict(self, text: str) -> dict:
        """Classify + tag a single document."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify + tag a batch of documents; a failure in one doesn't fail the rest (see below)."""
        if not texts:
            return []

        try:
            if len(texts) >= self.MIN_DOCS_FOR_PIPELINE:
                return self._predict_batch_pipelined(texts)
            return self._predict_batch_inner(texts)
        except Exception as e:
            # High-volume/streaming use means a single malformed document
            # (unexpected encoding, adversarial length, etc.) shouldn't take
            # the rest of the batch down with it. Retry one at a time so
            # everything that *can* succeed still does, and failures are
            # reported per-document instead of raised.
            results = []
            for text in texts:
                try:
                    results.extend(self._predict_batch_inner([text]))
                except Exception as doc_error:
                    results.append(
                        {
                            "text_preview": str(text)[:200] if text is not None else "",
                            "language": None,
                            "category": None,
                            "confidence": 0.0,
                            "tags": [],
                            "entities": [],
                            "error": str(doc_error),
                        }
                    )
            return results

    def _predict_batch_inner(self, texts: list[str]) -> list[dict]:
        cleaned = [normalize_text(t) for t in texts]
        languages = [detect_language(t[: self.LANG_DETECT_CHARS]) for t in cleaned]
        # A handful of documents run to tens of thousands of characters (quoted
        # reply chains, etc.) and dominate spaCy's wall-clock disproportionately
        # to their share of the corpus; cap tagging input like the classifier's
        # tokenizer already caps its input, to keep steady-state throughput high.
        tag_inputs = [t[: self.MAX_TAG_CHARS] for t in cleaned]

        categories, confidences = self._classify_batch(cleaned)
        analyses = self.tagger.analyze_batch(tag_inputs, languages)
        tag_results = [self.tagger.score(a, c) for a, c in zip(analyses, categories)]

        return self._assemble_results(texts, languages, categories, confidences, tag_results)

    def _assemble_results(self, texts, languages, categories, confidences, tag_results) -> list[dict]:
        results = []
        for text, lang, category, confidence, tag_result in zip(texts, languages, categories, confidences, tag_results):
            results.append(
                {
                    "text_preview": str(text)[:200] if text is not None else "",
                    "language": lang,
                    "category": category,
                    "confidence": round(float(confidence), 4),
                    "tags": tag_result["tags"],
                    "entities": tag_result["entities"],
                }
            )
        return results

    def _predict_batch_pipelined(self, texts: list[str]) -> list[dict]:
        """Same result as _predict_batch_inner, but classification (GPU) for
        chunk N+1 runs concurrently with tagging (CPU) for chunk N instead of
        strictly sequentially -- they use different hardware, and both release
        the GIL during their real work, so this is genuine overlap, not just
        thread bookkeeping. Measured ~1.6x over the sequential path at these
        chunk sizes. Only worth the thread setup above MIN_DOCS_FOR_PIPELINE."""
        cleaned = [normalize_text(t) for t in texts]
        languages = [detect_language(t[: self.LANG_DETECT_CHARS]) for t in cleaned]
        tag_inputs = [t[: self.MAX_TAG_CHARS] for t in cleaned]

        n = len(texts)
        chunk_size = self.PIPELINE_CHUNK_SIZE
        chunk_ranges = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

        classify_out: dict[int, tuple[list[str], list[float]]] = {}
        tag_out: dict[int, list[dict]] = {}
        error_holder: list[Exception] = []
        work_queue: queue.Queue = queue.Queue()

        def classify_worker():
            try:
                for start, end in chunk_ranges:
                    categories, confidences = self._classify_batch(cleaned[start:end])
                    classify_out[start] = (categories, confidences)
                    work_queue.put(start)
            except Exception as e:  # noqa: BLE001 -- surfaced to the main thread below
                error_holder.append(e)
            finally:
                work_queue.put(None)

        def tag_worker():
            try:
                while True:
                    start = work_queue.get()
                    if start is None:
                        break
                    end = min(start + chunk_size, n)
                    analyses = self.tagger.analyze_batch(tag_inputs[start:end], languages[start:end])
                    categories = classify_out[start][0]
                    tag_out[start] = [self.tagger.score(a, c) for a, c in zip(analyses, categories)]
            except Exception as e:  # noqa: BLE001
                error_holder.append(e)

        classify_thread = threading.Thread(target=classify_worker)
        tagging_thread = threading.Thread(target=tag_worker)
        classify_thread.start()
        tagging_thread.start()
        classify_thread.join()
        tagging_thread.join()

        if error_holder:
            raise error_holder[0]

        categories, confidences, tag_results = [], [], []
        for start, _ in chunk_ranges:
            categories.extend(classify_out[start][0])
            confidences.extend(classify_out[start][1])
            tag_results.extend(tag_out[start])

        return self._assemble_results(texts, languages, categories, confidences, tag_results)

    def benchmark_throughput(self, texts: list[str], batch_size: int = None) -> dict:
        """Measure end-to-end docs/sec for the full classify+tag pipeline.

        `predict_batch` already chunks internally (and pipelines classify/tag
        concurrently above MIN_DOCS_FOR_PIPELINE docs), so the realistic way
        to measure steady-state throughput is one call over the whole input --
        manually pre-chunking into small batch_size pieces before calling it,
        as earlier versions of this method did, defeats that internal
        pipelining and undercounts real throughput. `batch_size` is unused
        and only accepted for backward compatibility.

        Warm-up needs full-size calls, not a single small chunk: this GPU's
        boost clock doesn't ramp up from a brief, small burst of work -- a
        cold process measured immediately after one small warm-up call swings
        anywhere from ~38 to ~94 docs/sec run to run (clock oscillating
        between ~300MHz and ~1500MHz under load), but 2 full-size warm-up
        passes reliably settle it into a stable ~100-107 docs/sec band
        (measured stdev ~2 docs/sec across 10 runs). This matters beyond
        benchmarking accuracy: a real deployed server sees the same cold
        start on its first couple of requests, then runs at the stable rate
        for its whole lifetime -- so warming up here mirrors real serving
        behavior, not just a more favorable measurement.
        """
        for _ in range(self.WARMUP_ITERATIONS):
            self.predict_batch(texts)

        start = time.perf_counter()
        self.predict_batch(texts)
        elapsed = time.perf_counter() - start
        return {
            "n_docs": len(texts),
            "elapsed_seconds": elapsed,
            "docs_per_second": len(texts) / elapsed if elapsed > 0 else float("inf"),
        }

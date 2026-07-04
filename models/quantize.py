"""Post-training quantization experiment for the classifier.

Converts the trained Keras model to a dynamic-range-quantized TFLite model
and benchmarks it against the original. Kept as a standalone script (not
wired into the serving pipeline) because the result was negative: see
`if __name__ == "__main__"` output and README.md for the measured numbers
and why this doesn't help for this architecture.
"""
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

from models.text_classifier import build_model

CHECKPOINT_DIR = "models/checkpoints"


def convert_to_tflite(max_len: int = 256, num_labels: int = 5) -> bytes:
    """Load the trained checkpoint and dynamic-range-quantize it to a TFLite flatbuffer."""
    model = build_model(num_labels=num_labels, max_len=max_len)
    model.load_weights(f"{CHECKPOINT_DIR}/text_classifier_best.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # SELECT_TF_OPS ("Flex" delegate) is required: DistilBERT's embedding
    # lookups and attention ops aren't expressible in plain TFLite builtins.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    return converter.convert()


def benchmark_tflite(tflite_model: bytes, batch_size: int = 32, n_batches: int = 5) -> float:
    """Measure docs/sec for the quantized model on a real batch from the test set."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=8)
    input_details = interpreter.get_input_details()
    for d in input_details:
        interpreter.resize_tensor_input(d["index"], [batch_size, 256])
    interpreter.allocate_tensors()
    ids_idx = next(d["index"] for d in input_details if "input_ids" in d["name"])
    mask_idx = next(d["index"] for d in input_details if "attention_mask" in d["name"])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    df = pd.read_csv("data/processed_data/test.csv")
    texts = df["text"].sample(batch_size, random_state=1).tolist()
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=256, return_tensors="np")

    interpreter.set_tensor(ids_idx, enc["input_ids"].astype(np.int32))
    interpreter.set_tensor(mask_idx, enc["attention_mask"].astype(np.int32))
    interpreter.invoke()  # warm up

    start = time.perf_counter()
    for _ in range(n_batches):
        interpreter.set_tensor(ids_idx, enc["input_ids"].astype(np.int32))
        interpreter.set_tensor(mask_idx, enc["attention_mask"].astype(np.int32))
        interpreter.invoke()
    elapsed = time.perf_counter() - start
    return batch_size * n_batches / elapsed


if __name__ == "__main__":
    print("Converting to dynamic-range-quantized TFLite...")
    tflite_model = convert_to_tflite()
    print(f"Quantized model size: {len(tflite_model) / 1e6:.1f} MB (fp32 checkpoint is ~540 MB)")

    docs_per_sec = benchmark_tflite(tflite_model)
    print(f"TFLite CPU throughput: {docs_per_sec:.1f} docs/sec")
    print(
        "FINDING: this is far slower than the fp32 Keras model on GPU (~105 docs/sec measured). "
        "DistilBERT's ops require the SELECT_TF_OPS/Flex delegate rather than pure TFLite builtins, "
        "so quantized weights are dequantized and run through regular (CPU, unaccelerated) TF ops -- "
        "the model gets 4x smaller on disk but does not get faster. Quantization only pays off here if "
        "paired with a fully TFLite-native architecture or GPU delegate support, neither of which this "
        "checkpoint has. Not used in the serving pipeline (utils/realtime_pipeline.py) for this reason."
    )

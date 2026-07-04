"""Export the trained classifier to a fused ONNX model for fast real-time
inference (see README "Real-time inference backend").

Run this with the `.venv-onnx` environment (torch + transformers + onnx +
onnxruntime + tf2onnx + tensorflow-cpu) -- kept separate from the main
`.venv` for the same reason as `.venv-mt`: onnxruntime-gpu needs cuDNN 9,
tensorflow[and-cuda] needs cuDNN 8, and pip can't resolve two pinned
versions of the same package into one venv.

The trained model isn't converted directly (tf2onnx -> onnx): ONNX
Runtime's BERT fusion optimizer only recognizes graph patterns from known
export paths and rejects a tf2onnx-converted graph outright ("Model
producer not matched: Expected 'pytorch', Got 'tf2onnx'"). So this script
re-exports through PyTorch instead: the trained DistilBERT encoder weights
convert losslessly via `transformers`' `from_tf=True` loader; the small
custom head (2 Dense layers) is ported by hand (`kernel.T` for TF-to-
PyTorch's transposed `Linear` convention).

Usage (two venvs, because of the cuDNN conflict above):

    # 1. Under the main venv (has TF/Keras), extract the trained weights:
    source .venv/bin/activate
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

    # 2. Under .venv-onnx, convert + fuse + verify:
    source .venv-onnx/bin/activate
    unset LD_LIBRARY_PATH   # avoid a stray CUDA path making TF try (and fail) GPU JIT
    export CUDA_VISIBLE_DEVICES=""
    python export_onnx.py

Verify the result against the original TF model on real held-out documents
(not just the two hardcoded examples this script checks) before trusting a
change here -- see the correctness check in generate_reports.py's ensemble
evaluation for the pattern.
"""
import json
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel

CHECKPOINT_DIR = "models/checkpoints"
TF_BASE_EXPORT_DIR = "models/checkpoints/_tf_base_export"
HEAD_WEIGHTS_PATH = "models/checkpoints/_head_weights.npz"


class DocumentClassifierPT(nn.Module):
    """PyTorch mirror of models.text_classifier.build_model's architecture:
    DistilBERT encoder -> masked mean pool -> Dense(256, relu) -> Dropout -> Dense(num_labels, softmax)."""

    def __init__(self, tf_base_export_dir: str, head_weights: dict, num_labels: int, dropout: float = 0.4):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(tf_base_export_dir, from_tf=True)
        hidden_size = head_weights["w1"].shape[1]
        self.pre_classifier = nn.Linear(self.distilbert.config.dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.category_output = nn.Linear(hidden_size, num_labels)

        with torch.no_grad():
            # TF Dense kernel is (in, out); PyTorch Linear weight is (out, in).
            self.pre_classifier.weight.copy_(torch.from_numpy(head_weights["w1"].T))
            self.pre_classifier.bias.copy_(torch.from_numpy(head_weights["b1"]))
            self.category_output.weight.copy_(torch.from_numpy(head_weights["w2"].T))
            self.category_output.bias.copy_(torch.from_numpy(head_weights["b2"]))

    def forward(self, input_ids, attention_mask):
        sequence_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
        pooled = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        x = torch.relu(self.pre_classifier(pooled))
        x = self.dropout(x)
        return torch.softmax(self.category_output(x), dim=-1)


def main():
    if not os.path.exists(os.path.join(TF_BASE_EXPORT_DIR, "tf_model.h5")) or not os.path.exists(HEAD_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"{TF_BASE_EXPORT_DIR}/tf_model.h5 or {HEAD_WEIGHTS_PATH} not found -- run the extraction step "
            "under `.venv` first (see this file's module docstring for the exact command)."
        )

    with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
        config = json.load(f)
    max_len = config["max_len"]

    head_weights = dict(np.load(HEAD_WEIGHTS_PATH))

    print("Building PyTorch model from converted TF weights...")
    model = DocumentClassifierPT(TF_BASE_EXPORT_DIR, head_weights, config["num_labels"])
    model.eval()

    print("Sanity-checking a couple of examples (compare by eye against the TF model's output)...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    texts = [
        "NASA announced today that its upcoming mission will study radiation effects on new spacecraft materials.",
        "El gobierno anuncio nuevas leyes sobre el medio ambiente.",
    ]
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        print(model(enc["input_ids"], enc["attention_mask"]).numpy())

    print("Exporting to ONNX...")
    raw_onnx_path = os.path.join(CHECKPOINT_DIR, "_model_pt_raw.onnx")
    dummy_ids = torch.zeros((2, max_len), dtype=torch.long)
    dummy_mask = torch.ones((2, max_len), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        raw_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["probs"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    assert onnx.load(raw_onnx_path).producer_name == "pytorch"

    print("Running BERT fusion optimizer...")
    final_onnx_path = os.path.join(CHECKPOINT_DIR, "model.onnx")
    exit_code = os.system(
        f"python -m onnxruntime.transformers.optimizer --input {raw_onnx_path} --output {final_onnx_path} "
        f"--model_type bert --num_heads {model.distilbert.config.n_heads} "
        f"--hidden_size {model.distilbert.config.dim} --input_int32 --opt_level 99"
    )
    if exit_code != 0:
        raise RuntimeError(f"onnxruntime.transformers.optimizer exited with code {exit_code}")
    print(f"Wrote {final_onnx_path}")

    import shutil

    for path in [raw_onnx_path, raw_onnx_path + ".data", HEAD_WEIGHTS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(TF_BASE_EXPORT_DIR):
        shutil.rmtree(TF_BASE_EXPORT_DIR)


if __name__ == "__main__":
    main()

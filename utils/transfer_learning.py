"""Fine-tuning orchestration: tokenization, tf.data pipelines, training loop,
checkpointing, and evaluation (overall + per-language) for the transformer
classifier. Also includes the TF-IDF + Logistic Regression baseline used to
verify the transformer earns its complexity.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.text_classifier import DEFAULT_MAX_LEN, DEFAULT_MODEL_NAME, build_model

# Grow GPU memory on demand instead of grabbing a fixed pool -- this machine's
# 6GB GPU is sometimes shared with other training jobs, and a fixed pool
# reservation fails outright instead of coexisting with whatever is already
# allocated. Must run before any GPU op executes.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")

EPOCHS = 6
BATCH_SIZE = 8
LEARNING_RATE = 3e-5


def load_splits():
    """Load the train/val/test CSVs produced by `utils.data_loader`."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train_df, val_df, test_df


def build_label_maps(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """Build category<->integer-id maps from the sorted unique categories in `train_df`."""
    categories = sorted(train_df["category"].unique())
    label2id = {c: i for i, c in enumerate(categories)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label


def tokenize_texts(texts, tokenizer, max_len=DEFAULT_MAX_LEN):
    """Tokenize a batch of texts to fixed-length (input_ids, attention_mask) numpy arrays."""
    enc = tokenizer(
        list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="np"
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


def make_dataset(input_ids, attention_mask, labels, batch_size=BATCH_SIZE, shuffle=False):
    """Wrap tokenized arrays + labels into a batched, prefetching tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_mask}, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_baseline(train_df, test_df, label2id) -> dict:
    """TF-IDF + Logistic Regression baseline that the transformer must beat by >=5pts."""
    print("\n=== Training baseline: TF-IDF + Logistic Regression ===")
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train = train_df["category"].map(label2id).values
    y_test = test_df["category"].map(label2id).values

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    print(f"Baseline accuracy={acc:.4f}  macro-F1={f1_macro:.4f}")
    return {"accuracy": float(acc), "f1_macro": float(f1_macro)}


def per_language_accuracy(test_df, y_true, y_pred) -> dict:
    """Accuracy broken out per value of `test_df['language']`."""
    result = {}
    langs = test_df["language"].values
    for lang in sorted(set(langs)):
        idx = langs == lang
        result[lang] = float(accuracy_score(np.array(y_true)[idx], np.array(y_pred)[idx]))
    return result


def train_transformer(
    model_name: str = DEFAULT_MODEL_NAME,
    max_len: int = DEFAULT_MAX_LEN,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    """Fine-tune the transformer, checkpoint the best-by-val-accuracy weights, and evaluate on the test set."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_df, val_df, test_df = load_splits()
    label2id, id2label = build_label_maps(train_df)
    num_labels = len(label2id)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenizing splits...")
    train_ids, train_mask = tokenize_texts(train_df["text"], tokenizer, max_len)
    val_ids, val_mask = tokenize_texts(val_df["text"], tokenizer, max_len)
    test_ids, test_mask = tokenize_texts(test_df["text"], tokenizer, max_len)

    y_train = train_df["category"].map(label2id).values
    y_val = val_df["category"].map(label2id).values
    y_test = test_df["category"].map(label2id).values

    train_ds = make_dataset(train_ids, train_mask, y_train, batch_size, shuffle=True)
    val_ds = make_dataset(val_ids, val_mask, y_val, batch_size)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_labels), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    steps_per_epoch = -(-len(train_df) // batch_size)  # ceil
    total_steps = steps_per_epoch * epochs
    print(f"Building model ({model_name}, {num_labels} labels, {total_steps} total steps, warmup+decay LR)...")
    model = build_model(
        num_labels=num_labels,
        model_name=model_name,
        max_len=max_len,
        learning_rate=learning_rate,
        dropout=0.4,
        total_steps=total_steps,
    )

    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "text_classifier_best.h5")
    epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, "text_classifier_epoch{epoch:02d}.h5")
    history_csv_path = os.path.join(CHECKPOINT_DIR, "training_history.csv")

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),  # stop early on a diverged (NaN) loss instead of wasting hours
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(epoch_ckpt_path, save_weights_only=True, verbose=0),
        tf.keras.callbacks.CSVLogger(history_csv_path),
    ]

    print(f"Fine-tuning for {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    print("Loading best checkpoint (by val_accuracy) for evaluation...")
    model.load_weights(best_ckpt_path)

    test_probs = model.predict({"input_ids": test_ids, "attention_mask": test_mask}, batch_size=batch_size)
    test_preds = np.argmax(test_probs, axis=1)

    test_accuracy = float(accuracy_score(y_test, test_preds))
    test_f1_macro = float(f1_score(y_test, test_preds, average="macro"))
    report = classification_report(
        y_test, test_preds, target_names=[id2label[i] for i in range(num_labels)], output_dict=True
    )
    lang_accuracy = per_language_accuracy(test_df, y_test, test_preds)

    config = {
        "model_name": model_name,
        "max_len": max_len,
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "languages_supported": sorted(test_df["language"].unique().tolist()),
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "per_language_accuracy": lang_accuracy,
    }
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTest accuracy={test_accuracy:.4f}  macro-F1={test_f1_macro:.4f}")
    print(f"Per-language accuracy: {lang_accuracy}")

    return {
        "history_csv": history_csv_path,
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "per_language_accuracy": lang_accuracy,
        "classification_report": report,
        "label2id": label2id,
        "id2label": id2label,
        "test_df": test_df,
        "test_preds": test_preds,
        "test_probs": test_probs,
    }


if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()
    label2id, _ = build_label_maps(train_df)
    baseline_metrics = train_baseline(train_df, test_df, label2id)
    transformer_results = train_transformer()

    print("\n=== Summary ===")
    print(f"Baseline accuracy:    {baseline_metrics['accuracy']:.4f}")
    print(f"Transformer accuracy: {transformer_results['test_accuracy']:.4f}")
    improvement = transformer_results["test_accuracy"] - baseline_metrics["accuracy"]
    print(f"Improvement over baseline: {improvement * 100:.2f} pts")

    with open(os.path.join(CHECKPOINT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)

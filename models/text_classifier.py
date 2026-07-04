"""Transformer-based document classifier: DistilBERT (multilingual) body + a
small Keras classification head trained via transfer learning.

The base transformer is loaded from Hugging Face; only the pooling + dense
head is randomly initialized, so nearly all of the "knowledge" the model
starts with comes from pretraining and fine-tuning adapts it to our 5
categories across English and Spanish.
"""
import tensorflow as tf
from transformers import TFAutoModel

DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"
DEFAULT_MAX_LEN = 256  # ~59% of training docs exceed 128 tokens; TF-IDF baseline sees full docs


class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to peak_lr, then linear decay to 0 -- standard BERT-style
    schedule. A constant LR let training overfit fast (train acc ~98% by
    epoch 6 while val plateaued ~89-91%); warmup avoids destabilizing the
    pretrained weights early on, and decay lets later epochs make smaller,
    more stable steps instead of continuing to overfit at full LR."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.peak_lr * (step / warmup_steps)
        decay_lr = self.peak_lr * tf.maximum(0.0, (total_steps - step) / tf.maximum(total_steps - warmup_steps, 1.0))
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        """Keras serialization hook so the schedule can round-trip through model save/load."""
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


def build_model(
    num_labels: int,
    model_name: str = DEFAULT_MODEL_NAME,
    max_len: int | None = DEFAULT_MAX_LEN,
    learning_rate: float = 3e-5,
    dropout: float = 0.3,
    freeze_base: bool = False,
    total_steps: int | None = None,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> tf.keras.Model:
    """Build a functional Keras model: transformer body -> masked mean pool -> dense head.

    Plain tensors in/out (no HF ModelOutput dataclasses) so the model behaves
    like any other Keras model for .fit / .save_weights / ModelCheckpoint.

    max_len=None builds a dynamic-length input (weights are unaffected by
    sequence length -- pooling is masked-mean, not position-indexed) so
    inference can pad each batch to its own longest document instead of
    always paying for a fixed length, which matters when the training-time
    max_len is sized for worst-case documents but most real traffic is shorter.
    """
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    base_model = TFAutoModel.from_pretrained(model_name, name="base_transformer")
    base_model.trainable = not freeze_base

    transformer_output = base_model(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = transformer_output.last_hidden_state  # (batch, seq_len, hidden)

    # Masked mean pooling over real tokens (ignores padding) -- more stable
    # than CLS-token pooling for a from-scratch head on DistilBERT.
    mask = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
    summed = tf.reduce_sum(sequence_output * mask, axis=1)
    counts = tf.clip_by_value(tf.reduce_sum(mask, axis=1), 1e-9, tf.float32.max)
    pooled = summed / counts

    x = tf.keras.layers.Dense(256, activation="relu", name="pre_classifier")(pooled)
    x = tf.keras.layers.Dropout(dropout)(x)
    probs = tf.keras.layers.Dense(num_labels, activation="softmax", name="category_output")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=probs, name="document_classifier")

    if total_steps:
        lr_schedule = WarmupLinearDecay(learning_rate, int(total_steps * warmup_ratio), total_steps)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_model_for_inference(checkpoint_path: str, num_labels: int, model_name: str, max_len: int) -> tf.keras.Model:
    """Rebuild the architecture and load trained weights for inference."""
    model = build_model(num_labels=num_labels, model_name=model_name, max_len=max_len)
    model.load_weights(checkpoint_path)
    return model

"""Text normalization, cleaning, language detection and tokenization utilities.

Shared by the data pipeline (utils/data_loader.py), the training notebook,
and the real-time inference pipeline so that documents are treated
identically at train and serve time.
"""
import hashlib
import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect is non-deterministic by default; pin the seed for reproducible splits.
DetectorFactory.seed = 42

_HEADER_LINE_RE = re.compile(
    r"^(From|Subject|Organization|Lines|Nntp-Posting-Host|Reply-To|Distribution|"
    r"X-[\w-]+|In-Reply-To|Article-I\.D\.|Summary|Keywords|Sender):.*$",
    re.MULTILINE | re.IGNORECASE,
)
_QUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E -￿]")

SUPPORTED_LANGUAGES = {"en", "es"}


def strip_newsgroup_boilerplate(text: str) -> str:
    """Remove email-style headers, quoted reply lines, and signature blocks."""
    text = _HEADER_LINE_RE.sub(" ", text)
    text = _QUOTE_LINE_RE.sub(" ", text)
    # Drop everything after a standalone signature delimiter.
    sig_split = re.split(r"\n--\s*\n", text, maxsplit=1)
    text = sig_split[0]
    return text


def normalize_text(text: str, strip_boilerplate: bool = False) -> str:
    """Generic language-agnostic cleaning: unicode normalization, de-noising, whitespace."""
    if not isinstance(text, str):
        return ""
    if strip_boilerplate:
        text = strip_newsgroup_boilerplate(text)
    text = unicodedata.normalize("NFKC", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def detect_language(text: str, default: str = "en") -> str:
    """Best-effort language detection with a safe fallback for short/ambiguous text."""
    sample = text[:1000].strip()
    if len(sample) < 20:
        return default
    try:
        lang = detect(sample)
    except LangDetectException:
        return default
    return lang if lang in SUPPORTED_LANGUAGES else default


def content_hash(text: str) -> str:
    """Stable hash used for exact-duplicate detection."""
    normalized = _MULTI_WS_RE.sub(" ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def dedupe_records(records: list[dict], text_key: str = "text") -> list[dict]:
    """Drop exact-duplicate documents, keeping the first occurrence."""
    seen = set()
    deduped = []
    for rec in records:
        h = content_hash(rec[text_key])
        if h in seen:
            continue
        seen.add(h)
        deduped.append(rec)
    return deduped


_BLANK_TOKENIZERS = {}


def _get_blank_tokenizer(lang: str):
    """Lightweight language-specific tokenizer (no full spaCy pipeline)."""
    if lang not in _BLANK_TOKENIZERS:
        import spacy

        _BLANK_TOKENIZERS[lang] = spacy.blank(lang if lang in SUPPORTED_LANGUAGES else "en")
    return _BLANK_TOKENIZERS[lang]


def tokenize(text: str, lang: str = "en") -> list[str]:
    """Language-specific word tokenization (used for EDA / keyword stats, not the transformer)."""
    tokenizer = _get_blank_tokenizer(lang)
    return [t.text for t in tokenizer(text) if not t.is_space]


def clean_document(text: str, strip_boilerplate: bool = True) -> dict:
    """Full cleaning pass for one raw document: normalize + detect language."""
    cleaned = normalize_text(text, strip_boilerplate=strip_boilerplate)
    lang = detect_language(cleaned)
    return {"text": cleaned, "language": lang, "n_chars": len(cleaned), "n_tokens": len(tokenize(cleaned, lang))}

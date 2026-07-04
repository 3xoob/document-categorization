"""Context-aware, multi-language document tagging built on spaCy.

Combines a rule-based signal (Named Entity Recognition) with a statistical
signal (noun-chunk / keyword frequency) and boosts candidates that match the
document's predicted category, so tags reflect both *what's mentioned*
(entities) and *what it's about* (category context) -- not just raw term
frequency.
"""
import re
from collections import Counter

import spacy

SPACY_MODEL_MAP = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
}

# Entity types worth surfacing as tags, weighted by how identifying they are.
ENTITY_LABEL_WEIGHTS = {
    "PERSON": 3.0,
    "PER": 3.0,
    "ORG": 3.0,
    "GPE": 2.5,
    "LOC": 2.5,
    "PRODUCT": 2.5,
    "EVENT": 2.5,
    "NORP": 2.0,
    "FAC": 2.0,
    "WORK_OF_ART": 2.0,
    "DATE": 1.5,
    "LAW": 2.0,
    "LANGUAGE": 1.5,
    "MISC": 1.5,
}

# Seed vocabulary per category, used only to boost (not gate) candidate tags
# that are already present in the document -- this is the "context-aware"
# layer on top of plain frequency-based keyword extraction.
CATEGORY_SEED_KEYWORDS = {
    "technology": {
        "software", "hardware", "computer", "graphics", "windows", "driver", "processor",
        "disk", "memory", "system", "programa", "computadora", "software", "sistema", "disco",
    },
    "recreation_sports": {
        "team", "game", "season", "player", "car", "engine", "bike", "league", "score",
        "equipo", "juego", "temporada", "jugador", "coche", "motor", "liga",
    },
    "science_health": {
        "space", "research", "medicine", "disease", "encryption", "security", "nasa",
        "treatment", "doctor", "espacio", "investigacion", "medicina", "enfermedad", "seguridad",
    },
    "politics_society": {
        "government", "religion", "belief", "law", "rights", "church", "god", "election",
        "gobierno", "religion", "ley", "derechos", "iglesia", "dios", "eleccion",
    },
    "marketplace": {
        "sale", "price", "shipping", "offer", "condition", "buy", "sell",
        "venta", "precio", "envio", "oferta", "comprar", "vender",
    },
}

_STOP_EXTRA = {"'s", "n't"}
_WORD_RE = re.compile(r"^[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\-]{1,}$")


class DocumentTagger:
    """Multi-language entity + keyword tagger with category-aware ranking."""

    # Components other than tokenization + NER (tagger/morphologizer, parser,
    # attribute_ruler, lemmatizer -- names vary by language pipeline) aren't
    # used by tag()/tag_batch() and roughly double per-doc latency, so
    # fast=True keeps only what's needed for real-time throughput.
    ESSENTIAL_PIPES = ("tok2vec", "ner")

    # Below this many docs, spawning worker processes costs more than it saves
    # (measured: n_process=4 beats n_process=1 by ~2.4x at 500 docs, but a
    # single interactive dashboard submission would just pay startup cost).
    MIN_DOCS_FOR_MULTIPROCESS = 32

    def __init__(self, languages=("en", "es"), max_tags: int = 8, fast: bool = False, n_process: int = 1):
        self.max_tags = max_tags
        self.fast = fast
        self.n_process = n_process
        self._pipelines = {}
        for lang in languages:
            self._load_pipeline(lang)

    def _load_pipeline(self, lang: str):
        if lang not in self._pipelines:
            model_name = SPACY_MODEL_MAP.get(lang, SPACY_MODEL_MAP["en"])
            try:
                nlp = spacy.load(model_name)
            except OSError as e:
                raise OSError(
                    f"spaCy model '{model_name}' is not installed. Run "
                    f"`python -m spacy download {model_name}` and try again."
                ) from e
            if self.fast:
                for pipe_name in list(nlp.pipe_names):
                    if pipe_name not in self.ESSENTIAL_PIPES:
                        nlp.disable_pipe(pipe_name)
            self._pipelines[lang] = nlp
        return self._pipelines[lang]

    def _get_pipeline(self, lang: str):
        if lang not in self._pipelines:
            return self._load_pipeline(lang if lang in SPACY_MODEL_MAP else "en")
        return self._pipelines[lang]

    def extract_entities(self, doc) -> list[dict]:
        """Named entities from spaCy's NER pass, as plain {text, label} dicts."""
        return [
            {"text": ent.text.strip(), "label": ent.label_}
            for ent in doc.ents
            if len(ent.text.strip()) > 1
        ]

    def extract_keyword_candidates(self, doc) -> Counter:
        """Stopword-filtered token frequency. Deliberately POS/parser-free (uses
        only tokenization + the vocab's static is_stop/is_punct lookups) so it
        works identically whether the tagger/parser/lemmatizer pipes are loaded
        or stripped out for real-time speed."""
        counts = Counter()
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            word = token.text.lower()
            if _WORD_RE.match(word) and word not in _STOP_EXTRA:
                counts[word] += 1
        return counts

    def tag(self, text: str, language: str = "en", category: str | None = None) -> dict:
        """Tag a single document: entities + keyword-ranked tags, boosted by category."""
        nlp = self._get_pipeline(language)
        doc = nlp(text)
        analysis = {
            "language": language,
            "entities": self.extract_entities(doc),
            "keyword_counts": self.extract_keyword_candidates(doc),
        }
        return self.score(analysis, category)

    def analyze_batch(self, texts: list[str], languages: list[str]) -> list[dict]:
        """The expensive, category-independent part of tagging: NER + keyword
        counts per doc. Split out from scoring so callers (e.g. the real-time
        pipeline) can run this concurrently with classification -- neither
        needs the other's output, only the final scoring step does."""
        results = [None] * len(texts)
        by_lang: dict[str, list[int]] = {}
        for i, lang in enumerate(languages):
            by_lang.setdefault(lang if lang in SPACY_MODEL_MAP else "en", []).append(i)

        for lang, idxs in by_lang.items():
            nlp = self._get_pipeline(lang)
            n_process = self.n_process if len(idxs) >= self.MIN_DOCS_FOR_MULTIPROCESS else 1
            docs = nlp.pipe([texts[i] for i in idxs], n_process=n_process, batch_size=50)
            for i, doc in zip(idxs, docs):
                results[i] = {
                    "language": lang,
                    "entities": self.extract_entities(doc),
                    "keyword_counts": self.extract_keyword_candidates(doc),
                }
        return results

    def score(self, analysis: dict, category: str | None) -> dict:
        """Cheap step: rank entities + keywords into final tags, boosted by category."""
        entities, keyword_counts = analysis["entities"], analysis["keyword_counts"]
        seeds = CATEGORY_SEED_KEYWORDS.get(category, set())

        scores = Counter()
        for ent in entities:
            scores[ent["text"]] += ENTITY_LABEL_WEIGHTS.get(ent["label"], 1.5)
        for phrase, freq in keyword_counts.items():
            scores[phrase] += freq
        for tag_text in list(scores.keys()):
            if set(tag_text.lower().split()) & seeds:
                scores[tag_text] *= 1.5

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "language": analysis["language"],
            "category": category,
            "entities": entities,
            "tags": [t for t, _ in ranked[: self.max_tags]],
            "tag_scores": {t: round(s, 2) for t, s in ranked[: self.max_tags]},
        }

    def tag_batch(self, texts: list[str], languages: list[str], categories: list[str | None]) -> list[dict]:
        """Convenience wrapper: analyze_batch + score for each doc, in one call."""
        analyses = self.analyze_batch(texts, languages)
        return [self.score(a, c) for a, c in zip(analyses, categories)]

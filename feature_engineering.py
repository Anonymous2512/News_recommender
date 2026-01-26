"""
feature_engineering.py
Editorial Intelligence — Headline & Article Feature Extraction

EXPLORATORY ONLY. These features (sentiment, NER, readability, clickbait, keyword topics) are NOT used by
the trained models or the API (see ``newsrec/features/schema.py`` for the features that are). They are
computed per unique article, not per impression row. The former "semantic_alignment" feature compared
each row to the global mean embedding (not a user profile) and was removed; real history-similarity
features live in ``newsrec/features/profiles.py``.
"""

import re
import string
import logging
from typing import Optional

import numpy as np
import pandas as pd
import spacy
from textstat import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    gunning_fog,
    smog_index,
    coleman_liau_index,
)
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ─── Model Registry ──────────────────────────────────────────────────────────

_NLP: Optional[spacy.language.Language] = None
_SENTIMENT_PIPE = None
_CLICKBAIT_PIPE = None

CLICKBAIT_PATTERNS = re.compile(
    r"\b(you won'?t believe|this is why|here'?s why|what happens next|"
    r"will shock you|mind.?blow|the truth about|secret(s)? (of|to|behind)|"
    r"never knew|top \d+|number \d+ will|only \d+%|"
    r"before you|must.?see|goes viral|break(ing)? the internet|"
    r"this (one |simple )?(trick|hack|thing)|changed (my|their) life)\b",
    flags=re.IGNORECASE,
)

QUESTION_RE = re.compile(r"\?")
EXCLAIM_RE = re.compile(r"!")
ELLIPSIS_RE = re.compile(r"\.{2,}|…")
CAPS_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+\b")


def get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser"])
        logger.info("Loaded spaCy model: en_core_web_sm")
    return _NLP


def get_sentiment_pipe():
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        _SENTIMENT_PIPE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=128,
            batch_size=64,
        )
        logger.info("Loaded sentiment pipeline")
    return _SENTIMENT_PIPE


def get_clickbait_pipe():
    global _CLICKBAIT_PIPE
    if _CLICKBAIT_PIPE is None:
        _CLICKBAIT_PIPE = pipeline(
            "text-classification",
            model="elozano/bert-base-cased-clickbait-news",
            truncation=True,
            max_length=64,
            batch_size=64,
        )
        logger.info("Loaded clickbait pipeline")
    return _CLICKBAIT_PIPE


# ─── 1. Headline Length Features ────────────────────────────────────────────

def headline_length_features(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    Produces:
        hl_char_len       — raw character count
        hl_word_count     — token count
        hl_avg_word_len   — mean chars per token
        hl_has_number     — bool: contains digit sequence
        hl_starts_number  — bool: first token is numeric
    """
    titles = df[col].fillna("")
    tokens = titles.str.split()

    out = pd.DataFrame(index=df.index)
    out["hl_char_len"] = titles.str.len()
    out["hl_word_count"] = tokens.str.len().fillna(0).astype(int)
    out["hl_avg_word_len"] = tokens.map(
        lambda ws: np.mean([len(w) for w in ws]) if ws else 0.0
    )
    out["hl_has_number"] = titles.str.contains(NUMBER_RE).astype(int)
    out["hl_starts_number"] = tokens.map(
        lambda ws: int(bool(ws and re.match(r"^\d+", ws[0])))
    )
    return out


# ─── 2. Punctuation Intensity Features ──────────────────────────────────────

def punctuation_features(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    Produces:
        punc_question_count   — number of '?'
        punc_exclaim_count    — number of '!'
        punc_ellipsis_count   — '...' / '…' occurrences
        punc_caps_token_count — ALL-CAPS word count
        punc_total_ratio      — punctuation chars / total chars
        punc_intensity_score  — composite signal ∈ [0, 1]
    """
    titles = df[col].fillna("")
    char_len = titles.str.len().clip(lower=1)

    out = pd.DataFrame(index=df.index)
    out["punc_question_count"] = titles.str.count(QUESTION_RE)
    out["punc_exclaim_count"] = titles.str.count(EXCLAIM_RE)
    out["punc_ellipsis_count"] = titles.str.count(ELLIPSIS_RE)
    out["punc_caps_token_count"] = titles.str.count(CAPS_TOKEN_RE)
    out["punc_total_ratio"] = titles.map(
        lambda t: sum(1 for c in t if c in string.punctuation) / max(len(t), 1)
    )
    out["punc_intensity_score"] = (
        out["punc_question_count"].clip(upper=3) / 3 * 0.25
        + out["punc_exclaim_count"].clip(upper=3) / 3 * 0.35
        + out["punc_ellipsis_count"].clip(upper=2) / 2 * 0.15
        + out["punc_caps_token_count"].clip(upper=4) / 4 * 0.25
    ).clip(0, 1)
    return out


# ─── 3. Sentiment Features ──────────────────────────────────────────────────

def sentiment_features(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    Produces:
        sent_label    — 'POSITIVE' / 'NEGATIVE'
        sent_score    — model confidence ∈ [0, 1]
        sent_polarity — signed score: positive → +score, negative → −score
    """
    texts = df[col].fillna("").tolist()
    pipe = get_sentiment_pipe()
    results = pipe(texts)

    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    polarity = [s if l == "POSITIVE" else -s for l, s in zip(labels, scores)]

    return pd.DataFrame(
        {"sent_label": labels, "sent_score": scores, "sent_polarity": polarity},
        index=df.index,
    )


# ─── 4. Named Entity Features ────────────────────────────────────────────────

_ENTITY_TYPES = ["PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "NORP"]


def named_entity_features(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    Produces:
        ner_total_count      — total entity mentions
        ner_unique_count     — unique entity surface forms
        ner_type_diversity   — number of distinct entity types
        ner_has_person       — bool
        ner_has_org          — bool
        ner_has_location     — bool (GPE | LOC)
        ner_{type}_count     — per-type counts for _ENTITY_TYPES
    """
    nlp = get_nlp()
    texts = df[col].fillna("").tolist()
    docs = list(nlp.pipe(texts, batch_size=256))

    rows = []
    for doc in docs:
        ents = doc.ents
        row: dict = {
            "ner_total_count": len(ents),
            "ner_unique_count": len({e.text.lower() for e in ents}),
            "ner_type_diversity": len({e.label_ for e in ents}),
            "ner_has_person": int(any(e.label_ == "PERSON" for e in ents)),
            "ner_has_org": int(any(e.label_ == "ORG" for e in ents)),
            "ner_has_location": int(any(e.label_ in ("GPE", "LOC") for e in ents)),
        }
        for et in _ENTITY_TYPES:
            row[f"ner_{et.lower()}_count"] = sum(1 for e in ents if e.label_ == et)
        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


# ─── 5. Topic Encoding ───────────────────────────────────────────────────────

_TOPIC_TAXONOMY = {
    "politics": ["election", "president", "congress", "senate", "democrat", "republican",
                 "vote", "policy", "government", "white house", "biden", "trump"],
    "sports":   ["nfl", "nba", "mlb", "nhl", "game", "player", "team", "score",
                 "championship", "league", "coach", "season", "trade"],
    "tech":     ["ai", "tech", "apple", "google", "microsoft", "software", "startup",
                 "data", "cyber", "algorithm", "robot", "cloud", "app"],
    "health":   ["health", "covid", "cancer", "vaccine", "drug", "hospital", "study",
                 "disease", "mental", "fda", "treatment", "therapy"],
    "finance":  ["stock", "market", "economy", "inflation", "fed", "bank", "invest",
                 "recession", "earnings", "crypto", "bitcoin", "rate"],
    "entertainment": ["movie", "film", "actor", "music", "celebrity", "award",
                      "netflix", "show", "album", "concert", "oscar", "grammy"],
    "crime":    ["murder", "arrest", "police", "court", "trial", "suspect", "prison",
                 "shooting", "violence", "investigation", "charges"],
    "world":    ["russia", "china", "ukraine", "war", "un", "nato", "europe",
                 "middle east", "sanction", "diplomat", "treaty"],
}

_TOPIC_RE = {
    topic: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for topic, kws in _TOPIC_TAXONOMY.items()
}


def topic_encoding_features(
    df: pd.DataFrame,
    col: str = "title",
    use_category: bool = True,
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Produces:
        topic_{name}_score    — keyword match count per topic
        topic_primary         — argmax topic label
        topic_ambiguity       — number of topics with score > 0
        topic_category_id     — encoded category from metadata (if use_category)
    """
    texts = (df[col].fillna("") + " " + df.get(category_col, pd.Series("", index=df.index)).fillna("")).tolist()

    scores: dict[str, list] = {t: [] for t in _TOPIC_RE}
    for text in texts:
        for topic, pattern in _TOPIC_RE.items():
            scores[topic].append(len(pattern.findall(text)))

    out = pd.DataFrame({f"topic_{t}_score": v for t, v in scores.items()}, index=df.index)

    score_cols = [f"topic_{t}_score" for t in _TOPIC_RE]
    # Rows with no keyword match get "none" (idxmax alone would label them with the first topic).
    primary = out[score_cols].idxmax(axis=1).str.replace("topic_|_score", "", regex=True)
    primary = primary.where(out[score_cols].sum(axis=1) > 0, "none")
    # Stable, batch-independent encoding: fixed taxonomy order, "none" = -1.
    topic_index = {t: i for i, t in enumerate(_TOPIC_TAXONOMY)}
    out["topic_primary"] = primary.map(lambda t: topic_index.get(t, -1)).astype(int)
    out["topic_ambiguity"] = (out[score_cols] > 0).sum(axis=1)

    if use_category and category_col in df.columns:
        out["topic_category_id"] = df[category_col].astype("category").cat.codes

    return out


# ─── 6. Readability Features ─────────────────────────────────────────────────

def readability_features(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """
    Produces:
        read_flesch_ease       — Flesch Reading Ease (higher = easier)
        read_fk_grade          — Flesch-Kincaid Grade Level
        read_gunning_fog       — Gunning Fog Index
        read_smog              — SMOG Index
        read_coleman_liau      — Coleman-Liau Index
        read_avg_grade         — mean of grade-level metrics
        read_complexity_bucket — 0=easy, 1=medium, 2=hard
    """
    texts = df[col].fillna("").tolist()

    rows = []
    for text in texts:
        fk = flesch_kincaid_grade(text)
        fog = gunning_fog(text)
        smog = smog_index(text)
        cli = coleman_liau_index(text)
        avg = np.mean([fk, fog, smog, cli])
        rows.append({
            "read_flesch_ease": flesch_reading_ease(text),
            "read_fk_grade": fk,
            "read_gunning_fog": fog,
            "read_smog": smog,
            "read_coleman_liau": cli,
            "read_avg_grade": avg,
            "read_complexity_bucket": int(np.clip(avg // 7, 0, 2)),
        })

    return pd.DataFrame(rows, index=df.index)

# ─── 7. Clickbait Probability ────────────────────────────────────────────────

def clickbait_features(
    df: pd.DataFrame,
    col: str = "title",
    use_model: bool = True,
) -> pd.DataFrame:
    """
    Produces:
        cb_pattern_score     — regex pattern match count
        cb_pattern_flag      — bool: ≥1 pattern match
        cb_model_score       — transformer model probability (if use_model)
        cb_model_label       — 'clickbait' / 'not-clickbait' (if use_model)
        cb_composite_score   — weighted fusion ∈ [0, 1]
    """
    titles = df[col].fillna("")
    out = pd.DataFrame(index=df.index)

    out["cb_pattern_score"] = titles.map(lambda t: len(CLICKBAIT_PATTERNS.findall(t)))
    out["cb_pattern_flag"] = (out["cb_pattern_score"] > 0).astype(int)

    pattern_norm = out["cb_pattern_score"].clip(upper=5) / 5.0

    if use_model:
        pipe = get_clickbait_pipe()
        results = pipe(titles.tolist())
        model_scores = []
        model_labels = []
        for r in results:
            is_cb = r["label"].lower() in ("clickbait", "label_1", "1")
            score = r["score"] if is_cb else 1 - r["score"]
            model_scores.append(score)
            model_labels.append("clickbait" if is_cb else "not-clickbait")

        out["cb_model_score"] = model_scores
        out["cb_model_label"] = model_labels
        out["cb_composite_score"] = (0.4 * pattern_norm + 0.6 * out["cb_model_score"]).clip(0, 1)
    else:
        out["cb_composite_score"] = pattern_norm

    return out


# ─── Pipeline ────────────────────────────────────────────────────────────────

def build_editorial_features(
    df: pd.DataFrame,
    title_col: str = "title",
    category_col: str = "category",
    use_sentiment: bool = True,
    use_clickbait_model: bool = True,
) -> pd.DataFrame:
    logger.info("Building editorial features for %d rows", len(df))

    feature_frames = [
        headline_length_features(df, title_col),
        punctuation_features(df, title_col),
        named_entity_features(df, title_col),
        topic_encoding_features(df, title_col, category_col=category_col),
        readability_features(df, title_col),
        clickbait_features(df, title_col, use_model=use_clickbait_model),
    ]

    if use_sentiment:
        feature_frames.append(sentiment_features(df, title_col))

    result = pd.concat(feature_frames, axis=1)
    logger.info("Editorial features shape: %s", result.shape)
    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/news.parquet",
                        help="article table (news.parquet from `python -m newsrec prepare`)")
    parser.add_argument("--output", default="data/processed/editorial_features.parquet")
    parser.add_argument("--max-articles", type=int, default=50000)
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--no-clickbait-model", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input).drop_duplicates("news_id").reset_index(drop=True)
    if len(df) > args.max_articles:
        df = df.sample(args.max_articles, random_state=42).reset_index(drop=True)
    logger.info("Computing exploratory editorial features for %d unique articles", len(df))
    features = build_editorial_features(
        df,
        use_sentiment=not args.no_sentiment,
        use_clickbait_model=not args.no_clickbait_model,
    )
    out = pd.concat([df, features], axis=1)
    out.to_parquet(args.output, index=False)
    logger.info("Saved enriched dataset → %s", args.output)
    print(out.filter(regex="^(hl_|punc_|sent_|ner_|topic_|read_|cb_)").head())
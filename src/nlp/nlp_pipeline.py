"""
NLP Pipeline — Eco-Smart Classifier  (Personne 3)
Preprocessing, vectorization comparison, training, inference.

ZONE ROUGE: preprocess_text(), tokenize(), remove_stopwords() — no AI.
"""

import argparse
import json
import logging
import pickle
import re
import string
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Stopwords  (ZONE ROUGE — written manually)
# ─────────────────────────────────────────────────────────────────────────────
FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "au", "aux", "à", "par", "pour", "sur", "avec", "dans", "est",
    "sont", "ont", "a", "il", "elle", "ils", "elles", "nous", "vous",
    "je", "tu", "se", "si", "ou", "mais", "car", "que", "qui", "dont",
    "où", "ne", "pas", "plus", "très", "bien", "tout", "tous", "cette",
    "ce", "ces", "cet", "mon", "ton", "son", "ma", "ta", "sa", "même",
    "lors", "via", "non",
}

DOMAIN_STOPWORDS = {
    "lot", "collecte", "collecté", "récupéré", "provenance", "matériau",
    "état", "général", "poids", "volume", "type", "identifié", "renseigné",
    "site", "kg", "litre", "aspect", "bon", "moyen", "léger", "masse",
    "totale", "estimé", "mesuré", "total", "autres",
}

ALL_STOPWORDS = FRENCH_STOPWORDS | DOMAIN_STOPWORDS


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing  (ZONE ROUGE)
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+[.,]?\d*\s*(kg|l|ml|cm|m|litre|litres)?", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return [t for t in text.split() if len(t) >= 2]


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in ALL_STOPWORDS]


def stem_fr(word: str) -> str:
    """Minimal French suffix-stripping stemmer — no library needed."""
    suffixes = [
        "ations", "ation", "ements", "ement", "iques", "ique",
        "istes", "iste", "ables", "able", "ibles", "ible",
        "eurs", "eur", "euses", "euse", "eux", "aux",
        "ages", "age", "ures", "ure", "ises", "ise",
        "iers", "ier", "ières", "ière",
        "ers", "er", "irs", "ir", "es", "e", "s",
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess_text(text: str, use_stemming: bool = True) -> str:
    """Full pipeline: clean → tokenize → remove stopwords → stem."""
    tokens = remove_stopwords(tokenize(clean_text(text)))
    if use_stemming:
        tokens = [stem_fr(t) for t in tokens]
    return " ".join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    log.info("Preprocessing %d documents…", len(series))
    return series.fillna("").apply(preprocess_text)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorizers
# ─────────────────────────────────────────────────────────────────────────────
def make_bow(p: dict) -> CountVectorizer:
    return CountVectorizer(max_features=p["max_features"], min_df=p["min_df"])


def make_tfidf(p: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=p["max_features"],
        ngram_range=tuple(p["ngram_range"]),
        min_df=p["min_df"],
        sublinear_tf=p["sublinear_tf"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(params: dict) -> Pipeline:
    # Load data — train.csv has all columns including Rapport_Collecte
    train_df = pd.read_csv(params["data"]["train_path"])
    val_df   = pd.read_csv(params["data"]["val_path"])
    text_col = params["columns"]["text"]      # Rapport_Collecte
    target   = params["columns"]["target"]    # Categorie

    # Drop unlabelled rows (514 rows without Categorie)
    train_df = train_df.dropna(subset=[target])
    val_df   = val_df.dropna(subset=[target])

    X_tr = preprocess_series(train_df[text_col])
    X_va = preprocess_series(val_df[text_col])
    y_tr = train_df[target]
    y_va = val_df[target]

    nlp_p = params["nlp"]
    bow   = make_bow(nlp_p)
    tfidf = make_tfidf(nlp_p)

    # Fit vectorizers
    X_tr_bow   = bow.fit_transform(X_tr)
    X_va_bow   = bow.transform(X_va)
    X_tr_tfidf = tfidf.fit_transform(X_tr)
    X_va_tfidf = tfidf.transform(X_va)

    configs = [
        ("BoW",    X_tr_bow,   X_va_bow,   "naive_bayes",   MultinomialNB(alpha=0.1),                                              bow),
        ("TF-IDF", X_tr_tfidf, X_va_tfidf, "naive_bayes",   MultinomialNB(alpha=0.1),                                              tfidf),
        ("TF-IDF", X_tr_tfidf, X_va_tfidf, "logistic",      LogisticRegression(max_iter=1000, C=1.0, random_state=42),             tfidf),
        ("TF-IDF", X_tr_tfidf, X_va_tfidf, "svm",           LinearSVC(max_iter=2000, C=1.0, random_state=42),                     tfidf),
    ]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    best_acc, best_pipeline = 0.0, None
    all_results = {}

    for vec_name, X_tr_v, X_va_v, clf_name, clf, vec in configs:
        run_name = f"nlp_{vec_name}_{clf_name}"
        log.info("Training %s…", run_name)

        with mlflow.start_run(run_name=run_name):
            clf.fit(X_tr_v, y_tr)
            preds    = clf.predict(X_va_v)
            acc      = accuracy_score(y_va, preds)
            report   = classification_report(y_va, preds, output_dict=True)
            macro_f1 = report["macro avg"]["f1-score"]

            mlflow.log_params({"vectorizer": vec_name, "classifier": clf_name})
            mlflow.log_metrics({"val_accuracy": acc, "val_macro_f1": macro_f1})

            # Build full sklearn pipeline for self-contained inference
            pipe = Pipeline([("vectorizer", vec), ("classifier", clf)])
            mlflow.sklearn.log_model(pipe, artifact_path="nlp_model")

            all_results[run_name] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
            }
            log.info("  acc=%.4f  macro_f1=%.4f", acc, macro_f1)

            if acc > best_acc:
                best_acc = acc
                best_pipeline = pipe

    # Save best model
    Path("models").mkdir(exist_ok=True)
    with open("models/nlp_classifier.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/nlp_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("Best NLP val accuracy: %.4f — saved to models/nlp_classifier.pkl", best_acc)
    return best_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper (used by FastAPI)
# ─────────────────────────────────────────────────────────────────────────────
def predict_text(text: str, model_path: str = "models/nlp_classifier.pkl") -> str:
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    return pipe.predict([preprocess_text(text)])[0]

#
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--text", default="")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    if args.mode == "train":
        train(p)
    else:
        print(predict_text(args.text))

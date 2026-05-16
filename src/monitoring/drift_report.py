"""
Drift Report — Eco-Smart Classifier  (Personne 3)
Numeric drift via Evidently AI + text drift via Jensen-Shannon divergence.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def run_evidently_report(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          numeric_cols: list, output_path: str) -> None:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=train_df[numeric_cols],
                   current_data=test_df[numeric_cols])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)
        log.info("Evidently report → %s", output_path)

    except ImportError:
        log.warning("evidently not installed — using KS fallback")
        _ks_fallback(train_df, test_df, numeric_cols, output_path)


def _ks_fallback(train_df, test_df, numeric_cols, output_path):
    rows = []
    for col in numeric_cols:
        ref = train_df[col].dropna().values
        cur = test_df[col].dropna().values
        stat, pval = ks_2samp(ref, cur)
        rows.append({"feature": col, "ks_stat": round(stat, 4),
                      "p_value": round(pval, 4), "drift": pval < 0.05})
    html = pd.DataFrame(rows).to_html(index=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        f"<html><body><h2>Drift Report (KS test)</h2>{html}</body></html>")
    log.info("KS fallback report → %s", output_path)


def compute_js_text_drift(train_texts: pd.Series, test_texts: pd.Series,
                           max_vocab: int = 2000) -> float:
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(max_features=max_vocab, min_df=2)
    vec.fit(train_texts.fillna(""))
    ref = vec.transform(train_texts.fillna("")).toarray().sum(axis=0).astype(float)
    cur = vec.transform(test_texts.fillna("")).toarray().sum(axis=0).astype(float)
    eps = 1e-9
    ref = (ref + eps) / (ref + eps).sum()
    cur = (cur + eps) / (cur + eps).sum()
    return float(jensenshannon(ref, cur, base=2))


def main(params_path: str = "params.yaml") -> None:
    with open(params_path) as f:
        params = yaml.safe_load(f)

    train_df = pd.read_csv(params["data"]["train_path"])
    test_df  = pd.read_csv(params["data"]["test_path"])

    numeric_cols = params["columns"]["numeric"]
    text_col     = params["columns"]["text"]
    js_threshold = params["monitoring"]["js_divergence_threshold"]

    log.info("Running numeric drift report…")
    run_evidently_report(train_df, test_df, numeric_cols, "reports/drift_report.html")

    log.info("Computing text drift (Jensen-Shannon)…")
    js = compute_js_text_drift(train_df[text_col], test_df[text_col])
    alert = {
        "js_divergence": round(js, 4),
        "threshold": js_threshold,
        "text_drift_detected": js > js_threshold,
    }
    if alert["text_drift_detected"]:
        log.warning("⚠️  Text drift detected! JS=%.4f", js)
    else:
        log.info("✅  No text drift. JS=%.4f", js)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/drift_summary.json", "w") as f:
        json.dump(alert, f, indent=2)
    log.info("Drift summary → reports/drift_summary.json")


if __name__ == "__main__":
    main()

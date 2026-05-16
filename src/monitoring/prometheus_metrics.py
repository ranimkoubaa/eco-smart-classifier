"""Prometheus metrics registry — Eco-Smart Classifier (Personne 3)"""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status_code"])

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])

PREDICTION_COUNT = Counter(
    "predictions_total", "Predictions per category", ["category", "model"])

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence", "Confidence score distribution", ["model"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

MODEL_ACCURACY = Gauge(
    "model_accuracy", "Latest observed accuracy", ["model"])

DRIFT_JS_SCORE = Gauge(
    "drift_js_divergence", "Jensen-Shannon text drift score")

DRIFT_FEATURE = Gauge(
    "drift_feature_score", "KS statistic per feature", ["feature"])

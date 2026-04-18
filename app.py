"""
app.py  —  Flask Fraud Prediction API
GET  /health         liveness probe
GET  /metrics        running counters
POST /predict        score one transaction
POST /predict/batch  score up to 100 transactions
"""
import logging, os, time
from collections import defaultdict
from functools import wraps

import joblib, numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__)

# API key — read from environment (set in .env / docker-compose)
API_KEY = os.getenv("API_KEY", "secret-key-dev-only")

MODEL_PATHS = {
    "isolation_forest":  "models/isolation_forest.pkl",
    "gradient_boosting": "models/gradient_boosting.pkl",
    "scaler":            "models/scaler.pkl",
}
models = {}

_counters: dict = defaultdict(int)
_latencies: list = []

FEATURE_ORDER = [
    "amount", "hour_of_day", "is_weekend",
    "merchant_cat_atm", "merchant_cat_fuel",
    "merchant_cat_grocery", "merchant_cat_online", "merchant_cat_travel",
    "country_IN", "country_NG", "country_SG", "country_UK", "country_US",
]


def load_models():
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = joblib.load(path)
            log.info("Loaded %s", name)
        except FileNotFoundError:
            log.warning("Model not found: %s  (predictions will use fallback)", path)
            models[name] = None


def record(decision, latency_ms):
    _counters["total"]  += 1
    _counters[decision] += 1
    _latencies.append(latency_ms)
    if len(_latencies) > 10_000:
        _latencies.pop(0)


def build_feature_vector(txn):
    cat = {f"merchant_cat_{c}": 0 for c in ["atm","fuel","grocery","online","travel"]}
    ck  = f"merchant_cat_{txn.get('merchant_cat','other')}"
    if ck in cat: cat[ck] = 1

    cty = {f"country_{c}": 0 for c in ["IN","NG","SG","UK","US"]}
    ctk = f"country_{txn.get('country','IN')}"
    if ctk in cty: cty[ctk] = 1

    row = {"amount": float(txn.get("amount",0)),
           "hour_of_day": int(txn.get("hour_of_day",0)),
           "is_weekend":  int(txn.get("is_weekend",0)),
           **cat, **cty}
    return np.array([row[f] for f in FEATURE_ORDER], dtype=np.float64).reshape(1,-1)


def score_transaction(txn):
    vec    = build_feature_vector(txn)
    scaler = models.get("scaler")
    vec_sc = scaler.transform(vec) if scaler else vec

    iso    = models.get("isolation_forest")
    if_raw = float(-iso.score_samples(vec_sc)[0]) if iso else 0.5
    if_norm= float(np.clip((if_raw - 0.3) / 0.4, 0, 1))

    gb     = models.get("gradient_boosting")
    gb_prob= float(gb.predict_proba(vec_sc)[0, 1]) if gb else 0.5

    score  = round(0.4 * if_norm + 0.6 * gb_prob, 4)
    decision = "block" if score >= 0.70 else "flag" if score >= 0.45 else "allow"

    return {
        "transaction_id": txn.get("transaction_id", "unknown"),
        "card_id":        txn.get("card_id", "unknown"),
        "fraud_score":    score,
        "decision":       decision,
        "model_scores":   {"isolation_forest": round(if_norm,4),
                           "gradient_boosting": round(gb_prob,4)},
    }


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get("X-API-Key","") != API_KEY:
            return jsonify({"error": "Unauthorised"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/health")
def health():
    return jsonify({"status": "ok",
                    "models_loaded": {k: v is not None for k,v in models.items()}})


@app.route("/metrics")
def metrics():
    avg = round(sum(_latencies)/len(_latencies), 2) if _latencies else 0
    return jsonify({"total_scored": _counters["total"],
                    "blocked":      _counters["block"],
                    "flagged":      _counters["flag"],
                    "allowed":      _counters["allow"],
                    "avg_latency_ms": avg})


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    txn = request.get_json(silent=True)
    if not txn:
        return jsonify({"error": "Invalid JSON"}), 400
    missing = {"transaction_id","card_id","amount"} - txn.keys()
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 422
    t0  = time.perf_counter()
    res = score_transaction(txn)
    ms  = round((time.perf_counter()-t0)*1000, 2)
    res["latency_ms"] = ms
    record(res["decision"], ms)
    log.info("%s  score=%.3f  %s  %.1fms",
             res["transaction_id"][:12], res["fraud_score"], res["decision"], ms)
    return jsonify(res)


@app.route("/predict/batch", methods=["POST"])
@require_api_key
def predict_batch():
    body = request.get_json(silent=True) or {}
    txns = body.get("transactions", [])
    if not isinstance(txns, list) or not txns:
        return jsonify({"error": "Provide 'transactions' list"}), 422
    if len(txns) > 100:
        return jsonify({"error": "Max 100 per batch"}), 422
    t0      = time.perf_counter()
    results = [score_transaction(t) for t in txns]
    ms      = round((time.perf_counter()-t0)*1000, 2)
    for r in results: record(r["decision"], ms/len(txns))
    return jsonify({"results": results, "latency_ms": ms})


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)

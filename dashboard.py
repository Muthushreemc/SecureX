"""
dashboard.py  -  FraudSentinel v2.0
Three tabs: Live Monitor | Manual Detection | Batch CSV Upload

Run:  py -3.11 dashboard.py
Open: http://localhost:8050

Works WITHOUT trained models (rule-based fallback).
Run train_models.py first for full ML scoring.
"""
import os, random, uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# Load real ML models if they exist
# ---------------------------------------------------------------------------
try:
    import joblib
    _scaler = joblib.load(Path("models/scaler.pkl"))
    _iso    = joblib.load(Path("models/isolation_forest.pkl"))
    _gb     = joblib.load(Path("models/gradient_boosting.pkl"))
    MODELS_LOADED = True
except Exception:
    _scaler = _iso = _gb = None
    MODELS_LOADED = False

FEATURE_ORDER = [
    "amount", "hour_of_day", "is_weekend",
    "merchant_cat_atm", "merchant_cat_fuel", "merchant_cat_grocery",
    "merchant_cat_online", "merchant_cat_travel",
    "country_IN", "country_NG", "country_SG", "country_UK", "country_US",
]

# ---------------------------------------------------------------------------
# Flask app — serves static files from the same folder
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=".", static_url_path="")

# ---------------------------------------------------------------------------
# Simulated live transaction feed
# ---------------------------------------------------------------------------
TRANSACTIONS = deque(maxlen=200)
MERCHANTS = ["Amazon", "Flipkart", "Zomato", "HDFC ATM", "Shell Fuel",
             "BookMyShow", "Swiggy", "MakeMyTrip", "IRCTC", "Uber"]
CATEGORIES = ["online", "grocery", "fuel", "atm", "travel"]


def make_transaction():
    is_fraud = random.random() < 0.08
    amount   = round(random.uniform(800, 5000), 2) if is_fraud \
               else round(random.uniform(10, 800), 2)
    country  = random.choice(["NG", "CN", "AE", "US"]) if is_fraud \
               else random.choice(["IN", "IN", "IN", "UK"])
    score    = round(random.uniform(0.70, 0.97), 3) if is_fraud \
               else round(random.uniform(0.02, 0.38), 3)
    decision = "block" if (is_fraud and score > 0.85) else \
               "flag"  if (is_fraud or score > 0.45) else "allow"
    return {
        "id":       str(uuid.uuid4())[:8].upper(),
        "card":     f".... {random.randint(1000, 9999)}",
        "merchant": random.choice(MERCHANTS),
        "category": random.choice(CATEGORIES),
        "amount":   amount,
        "country":  country,
        "score":    score,
        "decision": decision,
        "is_fraud": is_fraud,
        "ts":       datetime.now().strftime("%H:%M:%S"),
    }


random.seed(42)
for _ in range(40):
    t  = make_transaction()
    dt = datetime.now() - timedelta(minutes=random.randint(0, 59))
    t["ts"] = dt.strftime("%H:%M:%S")
    TRANSACTIONS.appendleft(t)

# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
def build_feature_vector(txn):
    cat = {f"merchant_cat_{c}": 0
           for c in ["atm", "fuel", "grocery", "online", "travel"]}
    ck  = f"merchant_cat_{str(txn.get('merchant_cat', 'other')).lower()}"
    if ck in cat:
        cat[ck] = 1

    cty = {f"country_{c}": 0 for c in ["IN", "NG", "SG", "UK", "US"]}
    ctk = f"country_{txn.get('country', 'IN')}"
    if ctk in cty:
        cty[ctk] = 1

    row = {
        "amount":      float(txn.get("amount", 0)),
        "hour_of_day": int(txn.get("hour_of_day", 12)),
        "is_weekend":  int(txn.get("is_weekend", 0)),
        **cat, **cty,
    }
    return np.array([row[f] for f in FEATURE_ORDER],
                    dtype=np.float64).reshape(1, -1)


def score_transaction(txn):
    if MODELS_LOADED:
        vec     = build_feature_vector(txn)
        vec_sc  = _scaler.transform(vec)
        if_raw  = float(-_iso.score_samples(vec_sc)[0])
        if_norm = float(np.clip((if_raw - 0.3) / 0.4, 0, 1))
        gb_prob = float(_gb.predict_proba(vec_sc)[0, 1])
        score   = round(0.4 * if_norm + 0.6 * gb_prob, 4)
        source  = "ML Model (Isolation Forest + Gradient Boosting)"
    else:
        s = 0.0
        amt = float(txn.get("amount", 0))
        if amt > 3000:   s += 0.35
        elif amt > 1500: s += 0.15
        c = txn.get("country", "IN")
        if c in ["NG", "CN", "AE"]:       s += 0.30
        elif c in ["US", "UK", "SG"]:     s += 0.10
        if str(txn.get("merchant_cat", "")).lower() == "atm": s += 0.15
        h = int(txn.get("hour_of_day", 12))
        if h < 5 or h > 22: s += 0.15
        if int(txn.get("is_weekend", 0)) and amt > 1000: s += 0.10
        score   = round(min(s, 0.99), 4)
        if_norm = round(min(s * 1.1, 0.99), 4)
        gb_prob = round(min(s * 0.9, 0.99), 4)
        source  = "Rule-based fallback (run train_models.py to enable ML)"

    decision = ("block" if score >= 0.70 else
                "flag"  if score >= 0.45 else "allow")

    return {
        "transaction_id": txn.get("transaction_id",
                                  str(uuid.uuid4())[:8].upper()),
        "card_id":        txn.get("card_id", "manual"),
        "fraud_score":    score,
        "decision":       decision,
        "risk_level":     ("HIGH"   if score >= 0.70 else
                           "MEDIUM" if score >= 0.45 else "LOW"),
        "model_scores": {
            "isolation_forest":  round(if_norm, 4),
            "gradient_boosting": round(gb_prob, 4),
        },
        "scoring_source": source,
        "factors":        explain(txn),
    }


def explain(txn):
    f = []
    amt = float(txn.get("amount", 0))
    if amt > 3000:    f.append(f"High amount Rs.{amt:,.0f}")
    elif amt > 1500:  f.append(f"Elevated amount Rs.{amt:,.0f}")
    c = txn.get("country", "IN")
    if c in ["NG", "CN", "AE"]:
        f.append(f"High-risk country ({c})")
    elif c not in ["IN", "UK", "SG", "US"]:
        f.append(f"Unusual country ({c})")
    if str(txn.get("merchant_cat", "")).lower() == "atm":
        f.append("ATM withdrawal")
    h = int(txn.get("hour_of_day", 12))
    if h < 5:    f.append(f"Odd hour ({h}:00 AM)")
    elif h > 22: f.append(f"Late night ({h}:00)")
    if int(txn.get("is_weekend", 0)):
        f.append("Weekend transaction")
    if not f:
        f.append("No significant risk factors")
    return f


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/transactions")
def api_transactions():
    for _ in range(random.randint(1, 3)):
        TRANSACTIONS.appendleft(make_transaction())
    txns    = list(TRANSACTIONS)
    total   = len(txns)
    blocked = sum(1 for t in txns if t["decision"] == "block")
    flagged = sum(1 for t in txns if t["decision"] == "flag")
    allowed = sum(1 for t in txns if t["decision"] == "allow")
    avg_sc  = round(sum(t["score"] for t in txns) / total, 3) if total else 0
    return jsonify({
        "transactions":  txns,
        "models_loaded": MODELS_LOADED,
        "metrics": {
            "total":      total,
            "blocked":    blocked,
            "flagged":    flagged,
            "allowed":    allowed,
            "block_rate": round(blocked / total * 100, 1) if total else 0,
            "flag_rate":  round(flagged / total * 100, 1) if total else 0,
            "allow_rate": round(allowed / total * 100, 1) if total else 0,
            "avg_score":  avg_sc,
        },
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    txn = request.get_json(silent=True) or {}
    if not txn.get("amount"):
        return jsonify({"error": "amount is required"}), 422
    result = score_transaction(txn)
    result["merchant_cat"] = txn.get("merchant_cat", "")
    result["country"]      = txn.get("country", "IN")
    result["hour_of_day"]  = txn.get("hour_of_day", 12)
    result["amount"]       = txn.get("amount", 0)
    return jsonify(result)


@app.route("/api/detect/batch", methods=["POST"])
def api_detect_batch():
    body = request.get_json(silent=True) or {}
    txns = body.get("transactions", [])
    if not isinstance(txns, list) or not txns:
        return jsonify({"error": "Provide 'transactions' list"}), 422
    if len(txns) > 5000:
        return jsonify({"error": "Max 5000 rows"}), 422
    results = []
    for row in txns:
        try:
            txn = {
                "transaction_id": row.get("transaction_id",
                                          str(uuid.uuid4())[:8]),
                "card_id":  row.get("card_id", "unknown"),
                "amount":   float(row.get("amount", 0)),
                "merchant_cat": row.get("merchant_cat", "online"),
                "country":  row.get("country", "IN"),
                "hour_of_day": int(float(row.get("hour_of_day", 12))),
                "is_weekend":  int(float(row.get("is_weekend", 0))),
            }
            r = score_transaction(txn)
            r.update({
                "merchant_cat": txn["merchant_cat"],
                "country":      txn["country"],
                "hour_of_day":  txn["hour_of_day"],
                "amount":       txn["amount"],
            })
            results.append(r)
        except Exception as e:
            results.append({
                "transaction_id": row.get("transaction_id", "?"),
                "error":    str(e),
                "decision": "allow",
                "fraud_score": 0,
                "risk_level":  "LOW",
                "factors": [],
            })
    return jsonify({"results": results, "total": len(results)})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    status = ("ML models loaded" if MODELS_LOADED
              else "Rule-based mode  (run train_models.py to enable ML)")
    print(f"\n  FraudSentinel v2.0")
    print(f"  Open   : http://localhost:8050")
    print(f"  Models : {status}")
    print(f"  Stop   : Ctrl-C\n")
    app.run(host="0.0.0.0", port=8050, debug=False)

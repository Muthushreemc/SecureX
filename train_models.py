"""
train_models.py  —  Train and save ML models
Place reduced_creditcard_small.csv in the same folder, then run:
    py -3.11 train_models.py
Creates a models/ folder with .pkl files that Docker will use.
"""
import json, logging
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_PATH    = Path("reduced_creditcard_small.csv")
MODEL_DIR    = Path("models")
RANDOM_STATE = 42


def smote_resample(X, y, k=5, rs=42):
    rng = check_random_state(rs)
    minority = np.argmin(np.bincount(y))
    X_min = X[y == minority]
    n_synth = np.bincount(y).max() - np.bincount(y).min()
    k = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    idx = idx[:, 1:]
    src  = rng.randint(0, len(X_min), n_synth)
    pick = rng.randint(0, k, n_synth)
    lam  = rng.uniform(0, 1, (n_synth, 1))
    synth = X_min[src] + lam * (X_min[idx[src, pick]] - X_min[src])
    return (np.vstack([X, synth]),
            np.concatenate([y, np.ones(n_synth, dtype=y.dtype)]))


def best_threshold(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1s = np.where(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
    return float(thr[np.argmax(f1s[:-1])]) if len(thr) else 0.5


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        log.error("CSV not found: %s", DATA_PATH.resolve())
        log.error("Copy reduced_creditcard_small.csv into this folder first.")
        raise SystemExit(1)

    log.info("Loading %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    df = df.drop(columns=["Time"], errors="ignore")
    X  = df.drop(columns=["Class"]).values.astype(np.float64)
    y  = df["Class"].values.astype(np.int64)
    log.info("Rows: %d  Fraud: %d (%.2f%%)", len(X), y.sum(), 100*y.mean())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    X_res, y_res = smote_resample(X_tr_sc, y_tr, rs=RANDOM_STATE)

    log.info("Training Isolation Forest …")
    iso = IsolationForest(n_estimators=300,
                          contamination=max(float(y_tr.mean()), 1e-4),
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr_sc)
    if_scores = -iso.score_samples(X_te_sc)
    thr_if    = best_threshold(y_te, if_scores)
    log.info("IF  ROC-AUC=%.3f", roc_auc_score(y_te, if_scores))

    log.info("Training Gradient Boosting …")
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=4, subsample=0.8,
                                    min_samples_leaf=5, random_state=RANDOM_STATE)
    gb.fit(X_res, y_res)
    gb_proba = gb.predict_proba(X_te_sc)[:, 1]
    thr_gb   = best_threshold(y_te, gb_proba)
    log.info("GB  ROC-AUC=%.3f", roc_auc_score(y_te, gb_proba))

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl",            compress=3)
    joblib.dump(iso,    MODEL_DIR / "isolation_forest.pkl",  compress=3)
    joblib.dump(gb,     MODEL_DIR / "gradient_boosting.pkl", compress=3)
    (MODEL_DIR / "thresholds.json").write_text(json.dumps(
        {"isolation_forest": round(thr_if, 4),
         "gradient_boosting": round(thr_gb, 4)}, indent=2))

    log.info("Saved: models/scaler.pkl, isolation_forest.pkl, gradient_boosting.pkl")
    log.info("Done! Now run:  docker compose up -d")


if __name__ == "__main__":
    main()

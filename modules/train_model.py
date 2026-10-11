import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.preprocess import build_training_table, FEATURES, NUM, CAT
from source.config import HOLD_COST, CAPACITY, POST_WEIGHT, SEED, OUTPUT_DIR, EVIDENCE_DIR, MODEL_DIR

SPLITS = [  
    ("A", "train <=30 Apr 2026, test May-Jun 2026 (after the rule change)", "2026-04-30 23:59", "2026-05-01", "2026-06-30 23:59"),
    ("B", "train <=31 May 2026, test Jun 2026", "2026-05-31 23:59", "2026-06-01", "2026-06-30 23:59"),
    ("C", "train <=31 Dec 2025, test Jan-Apr 2026 (before the rule change)", "2025-12-31 23:59", "2026-01-01", "2026-04-30 23:59"),
]


def make_model(kind):
    if kind == "logreg":
        pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
        return Pipeline([("pre", pre), ("m", LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))])
    pre = ColumnTransformer([("n", "passthrough", NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20,
                                                                        l2_regularization=1.0, random_state=SEED))])


class Ensemble:
    """Average of the two models' probabilities. Both local - no API key."""
    def __init__(self):
        self.models = {k: make_model(k) for k in ("logreg", "hgb")}

    def fit(self, X, y, post):
        w = np.where(np.asarray(post) == 1, POST_WEIGHT, 1.0)
        for m in self.models.values():
            m.fit(X[FEATURES], y, m__sample_weight=w)
        return self

    def predict_proba1(self, X):
        return np.mean([m.predict_proba(X[FEATURES])[:, 1] for m in self.models.values()], axis=0)


def queue_size(val_start, val_end):
    """Desk capacity for a validation window: 40 claims per month, rounded."""
    return int(round(CAPACITY * ((pd.Timestamp(val_end) - pd.Timestamp(val_start)).days / 30.4)))


def top_k_metrics(df, score, k):
    """Review the k highest scores; compare with approving everything."""
    order = np.argsort(-np.asarray(score))[:k]
    flagged = np.zeros(len(df), dtype=bool); flagged[order] = True
    y = df.is_fraud.values == 1
    tp, fp, fn, tn = int((flagged & y).sum()), int((flagged & ~y).sum()), int((~flagged & y).sum()), int((~flagged & ~y).sum())
    stopped = float(df.claim_amount_inr.values[flagged & y].sum())
    return {"queue_size": k, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / k, 3), "false_positive_rate": round(fp / max(fp + tn, 1), 4),
            "accuracy_top_k": round((tp + tn) / len(df), 4), "accuracy_approve_all": round(1 - y.mean(), 4),
            "inr_stopped_per_claim_checked": round(stopped / k), "net_inr_per_claim_checked": round((stopped - HOLD_COST * fp) / k),
            "best_possible_inr_per_claim": round(float(df.claim_amount_inr[y].nlargest(k).sum()) / k)}


def rule_scores(v):
    rng = np.random.RandomState(SEED)
    return {"random order": rng.rand(len(v)),
            "rule: new partner": v.partner_new_cohort.values + 0.001 * rng.rand(len(v)),
            "rule: small uninspected claim": v.auto_approved.values + 0.001 * rng.rand(len(v)),
            "rule: biggest amount first": v.claim_amount_inr.values.astype(float)}


def bootstrap_net(va, score, k, n=500):
    """90% interval for net Rs per claim checked (only a few dozen frauds, so the number is uncertain)."""
    rng = np.random.RandomState(SEED); out = []
    for _ in range(n):
        ix = rng.randint(0, len(va), len(va))
        out.append(top_k_metrics(va.iloc[ix].reset_index(drop=True), score[ix], k)["net_inr_per_claim_checked"])
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def evaluate(L):
    rows, keep = [], {}
    for sid, label, tr_end, vs, ve in SPLITS:
        tr, va = L[L.t <= tr_end], L[(L.t >= vs) & (L.t <= ve)].reset_index(drop=True)
        k = queue_size(vs, ve)
        scores = {}
        for kind in ("logreg", "hgb"):
            w = np.where(tr.post_policy == 1, POST_WEIGHT, 1.0)
            scores[kind] = make_model(kind).fit(tr[FEATURES], tr.is_fraud, m__sample_weight=w).predict_proba(va[FEATURES])[:, 1]
        scores["ENSEMBLE (final)"] = np.mean([scores["logreg"], scores["hgb"]], axis=0)
        scores.update(rule_scores(va))
        keep[sid] = (va, scores["ENSEMBLE (final)"], k)
        for name, s in scores.items():
            r = {"split": sid, "description": label, "model": name, "train_claims": len(tr), "train_frauds": int(tr.is_fraud.sum()),
                 "test_claims": len(va), "test_frauds": int(va.is_fraud.sum()), **top_k_metrics(va, s, k),
                 "roc_auc": round(roc_auc_score(va.is_fraud, s), 3), "pr_auc": round(average_precision_score(va.is_fraud, s), 3)}
            if name.startswith("ENSEMBLE"):
                r["net_inr_ci90_low"], r["net_inr_ci90_high"] = [round(x) for x in bootstrap_net(va, s, k)]
            rows.append(r)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUTPUT_DIR / "validation_report.csv", index=False)
    return rep, keep


def confusion_figure(rep, keep):
    va, s, k = keep["B"]
    r = rep[(rep.split == "B") & rep.model.str.startswith("ENSEMBLE")].iloc[0]
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["not flagged", "flagged"]); ax.set_yticks([0, 1], ["genuine", "fraud"])
    ax.set_title(f"Jun 2026 hold-out, top {k} claims flagged"); fig.tight_layout()
    EVIDENCE_DIR.mkdir(exist_ok=True); fig.savefig(EVIDENCE_DIR / "confusion_matrix_split_B.png", dpi=140); plt.close(fig)


def train_final():
    """Fit on ALL decided claims and save the model the API uses."""
    L, _, _ = build_training_table()
    model = Ensemble().fit(L, L.is_fraud, L.post_policy)
    MODEL_DIR.mkdir(exist_ok=True); joblib.dump(model, MODEL_DIR / "model.joblib")
    return model, L


def print_report():
    rep = pd.read_csv(OUTPUT_DIR / "validation_report.csv")
    e = rep[rep.model.str.startswith("ENSEMBLE")]
    b = e[e.split == "B"].iloc[0]
    print("\nHOLD-OUT JUNE 2026 (claims the model never saw)")
    print(f"  confusion matrix  TN {b.tn}  FP {b.fp}  FN {b.fn}  TP {b.tp}   (top {b.queue_size} flagged)")
    print(f"  recall {b.recall:.1%} | precision {b.precision:.1%} | false-positive rate {b.false_positive_rate:.2%} | accuracy {b.accuracy_top_k:.2%}")
    print(f"  accuracy of approving everything: {b.accuracy_approve_all:.2%}  <- paying every claim scores about the same as the model, so accuracy cannot be the KPI")
    print(f"  Rs stopped per claim checked {b.inr_stopped_per_claim_checked}, net of goodwill {b.net_inr_per_claim_checked} (90% interval {b.net_inr_ci90_low:.0f} to {b.net_inr_ci90_high:.0f})")
    print("\nALL SPLITS - final model vs simple rules (top-40/month queue)")
    cols = ["split", "model", "test_frauds", "queue_size", "tp", "fp", "roc_auc", "pr_auc", "inr_stopped_per_claim_checked", "net_inr_per_claim_checked"]
    print(rep[cols].to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    L, _, stats = build_training_table()
    print("DATA:", stats)
    rep, keep = evaluate(L)
    confusion_figure(rep, keep)
    train_final()
    print_report()
    return rep


if __name__ == "__main__":
    main()
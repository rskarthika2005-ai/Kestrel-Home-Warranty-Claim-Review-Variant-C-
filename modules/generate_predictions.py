import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from modules.preprocess import build_test_table, _combined
from modules.train_model import train_final, Ensemble
from modules.analyze_claim import build_reasons, facts_from
from source.config import (DATA_DIR, MODEL_DIR, OUTPUT_DIR, PREDICTIONS_FILE, HOLD_COST, CAPACITY, PRIOR_RATE, PRIOR_STRENGTH)

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def write_predictions(te):
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    out = sub[["claim_id"]].merge(te[["claim_id", "score"]].drop_duplicates("claim_id"), on="claim_id", how="left")
    assert len(out) == len(sub) and out.score.notna().all(), "row/ID mismatch with sample_submission"
    out["score"] = out["score"].round(6)
    out.to_csv(PREDICTIONS_FILE, index=False)
    return out


def write_model_files(L, te):
    """Everything the API needs besides model.joblib."""
    allc, _ = _combined()
    end = allc.t.max(); recent = allc[allc.t > end - pd.Timedelta(days=30)]
    g = recent.groupby("partner_id").agg(partner_vol_30d=("claim_id", "size"), partner_small_share_30d=("small_claim", "mean"),
                                         partner_auto_share_30d=("auto_approved", "mean"))
    hist = L.groupby("partner_id").agg(f=("is_fraud", "sum"), n=("is_fraud", "size"))
    hist["partner_hist_rate"] = (hist.f + PRIOR_RATE * PRIOR_STRENGTH) / (hist.n + PRIOR_STRENGTH); hist["partner_hist_n"] = hist.n
    snap = hist[["partner_hist_rate", "partner_hist_n"]].join(g, how="outer").fillna(
        {"partner_vol_30d": 0, "partner_small_share_30d": 0, "partner_auto_share_30d": 0, "partner_hist_rate": PRIOR_RATE, "partner_hist_n": 0})
    snap.reset_index().to_csv(MODEL_DIR / "partner_snapshot.csv", index=False)
    json.dump(allc.serial_norm.value_counts().to_dict(), open(MODEL_DIR / "serial_counts.json", "w"))

    
    a, b = L[L.t <= "2026-05-31 23:59"], L[L.t >= "2026-06-01"]
    sb = Ensemble().fit(a, a.is_fraud, a.post_policy).predict_proba1(b)
    cal = LogisticRegression(C=100).fit(logit(sb).reshape(-1, 1), b.is_fraud)
    calib = {"a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])}
    te["p_cal"] = 1 / (1 + np.exp(-(calib["a"] * logit(te.score.values) + calib["b"])))

    
    months = (te.t.max() - te.t.min()).days / 30.4
    k_high = int(round(CAPACITY * months))
    s = np.sort(te.score.values)[::-1]
    thr_high, thr_med = float(s[k_high - 1]), float(s[min(len(s) - 1, 4 * k_high - 1)])
    cfg = {"version": "ens-v1", "calibration": calib, "thr_high": thr_high, "thr_medium": thr_med, "hold_cost": HOLD_COST,
           "prior_rate": PRIOR_RATE, "stats": facts_from(L)}
    json.dump(cfg, open(MODEL_DIR / "scorer_config.json", "w"), indent=2)
    return cfg, k_high


def write_scored_claims(te, cfg, k_high):
    te["risk_level"] = np.where(te.score >= cfg["thr_high"], "HIGH", np.where(te.score >= cfg["thr_medium"], "MEDIUM", "LOW"))
    te["rank"] = te.score.rank(ascending=False, method="first").astype(int)
    # the desk works per calendar month: its queue = the 40 highest scores of each month
    te["month"] = te.t.dt.to_period("M")
    te["in_review_queue"] = np.where(te.groupby("month").score.rank(ascending=False, method="first") <= CAPACITY, "YES", "no")
    reasons = [build_reasons(r, cfg["stats"])[:5] for _, r in te.iterrows()]
    te["top_reason"] = [x[0]["text"] for x in reasons]
    te["reasons"] = [" | ".join(y["text"] for y in x) for x in reasons]
    te["new_partner"] = te.partner_new_cohort.map({1: "Y", 0: "N"})
    te["auto_approved_flag"] = te.auto_approved.map({1: "Y", 0: "N"})
    te["week_start"] = (te.t - pd.to_timedelta(te.t.dt.dayofweek, unit="D")).dt.normalize().dt.strftime("%Y-%m-%d")
    cols = {"claim_id": "claim_id", "submitted_at": "submitted_at", "week_start": "week_start", "partner_id": "partner_id", "city": "city",
            "partner_type": "partner_type", "new_partner": "new_partner", "family": "product_family", "sku": "sku",
            "claim_amount_inr": "claim_amount_inr", "partner_inspected": "partner_inspected", "auto_approved_flag": "auto_approved",
            "customer_prior_claims": "customer_prior_claims", "rank": "rank", "score": "fraud_score", "p_cal": "est_fraud_probability",
            "risk_level": "risk_level", "in_review_queue": "in_review_queue", "top_reason": "top_reason", "reasons": "reasons"}
    scored = te.sort_values("rank")[list(cols)].rename(columns=cols)
    scored["est_fraud_probability"], scored["fraud_score"] = scored.est_fraud_probability.round(4), scored.fraud_score.round(4)
    scored.to_csv(OUTPUT_DIR / "scored_claims.csv", index=False)
    return scored


def main():
    MODEL_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)
    model, L = train_final()
    te = build_test_table()
    te["score"] = model.predict_proba1(te)
    out = write_predictions(te)
    cfg, k_high = write_model_files(L, te)
    scored = write_scored_claims(te, cfg, k_high)
    q = scored[scored.in_review_queue == "YES"]
    print(f"predictions.csv: {len(out)} rows | output/scored_claims.csv: {len(scored)} rows")
    print(scored.risk_level.value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).to_string())
    print(scored.head(10)[["rank", "claim_id", "partner_id", "claim_amount_inr", "fraud_score", "risk_level"]].to_string(index=False))
    print(f"thresholds HIGH>={cfg['thr_high']:.3f} MEDIUM>={cfg['thr_medium']:.3f} | queue {len(q)} claims from {q.partner_id.nunique()} outlets "
          f"| est. fraud Rs in queue {(q.est_fraud_probability * q.claim_amount_inr).sum():,.0f} (estimate)")


if __name__ == "__main__":
    main()

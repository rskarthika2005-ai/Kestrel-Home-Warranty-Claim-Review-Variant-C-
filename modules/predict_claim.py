import json
import re
import joblib
import numpy as np
import pandas as pd

from modules.preprocess import base_features, load_reference
from modules.analyze_claim import build_reasons
from source.config import MODEL_DIR

REQUIRED = ["partner_id", "sku", "days_since_purchase", "claim_amount_inr", "photo_attached", "partner_inspected"]
OPTIONAL_DEFAULTS = {"customer_prior_claims": 0, "product_serial": "", "claim_description": "", "inspector_note": None, "resubmissions": 0}


class ClaimError(ValueError):
    """Bad input - the message is written for an employee, not a developer."""


def parse_time(v):
    if v in (None, ""):
        return pd.Timestamp.now().floor("min")
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return pd.to_datetime(str(v), format=fmt)
        except ValueError:
            continue
    raise ClaimError("submitted_at must look like 25-09-2026 14:30 (dd-mm-yyyy hh:mm).")


class ClaimScorer:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(model_dir / "model.joblib")
        self.cfg = json.load(open(model_dir / "scorer_config.json"))
        self.snap = pd.read_csv(model_dir / "partner_snapshot.csv").set_index("partner_id")
        self.serials = json.load(open(model_dir / "serial_counts.json"))
        self.pa, self.pr = load_reference()

   
    def validate(self, rec):
        r = dict(rec)
        missing = [k for k in REQUIRED if r.get(k) in (None, "")]
        if missing:
            raise ClaimError("Missing required field(s): " + ", ".join(missing))
        for k, v in OPTIONAL_DEFAULTS.items():
            if r.get(k) in (None, ""):
                r[k] = v
        for k in ("days_since_purchase", "claim_amount_inr", "customer_prior_claims"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                raise ClaimError(f"{k} must be a number (got {r[k]!r}).")
            if r[k] < 0:
                raise ClaimError(f"{k} cannot be negative.")
        if r["claim_amount_inr"] == 0:
            raise ClaimError("claim_amount_inr must be greater than 0.")
        for k in ("photo_attached", "partner_inspected"):
            v = str(r[k]).strip().upper()
            if v not in ("Y", "N"):
                raise ClaimError(f"{k} must be Y or N.")
            r[k] = v
        if r["partner_id"] not in set(self.pa.partner_id):
            raise ClaimError(f"Unknown partner_id {r['partner_id']}. Add it to partners.csv first.")
        if r["sku"] not in set(self.pr.sku):
            raise ClaimError(f"Unknown sku {r['sku']}. Add it to products.csv first.")
        r["t"] = parse_time(r.get("submitted_at"))
        r["claim_id"] = r.get("claim_id") or "NEW"
        r["resubmissions"] = int(r["resubmissions"] or 0)
        if r["partner_inspected"] == "N":
            r["inspector_note"] = None            # a note without a sign-off is not evidence of inspection
        r["inspector_note"] = r["inspector_note"] if r["inspector_note"] is not None else np.nan
        return r

    def features(self, r):
        cols = ["claim_id", "t", "partner_id", "sku", "product_serial", "days_since_purchase", "claim_amount_inr", "photo_attached",
                "partner_inspected", "claim_description", "inspector_note", "customer_prior_claims", "resubmissions"]
        d = base_features(pd.DataFrame([{k: r[k] for k in cols}]), self.pa, self.pr)
        key = re.sub(r"[^A-Z0-9]", "", str(r["product_serial"]).upper())
        d["serial_prior_claims"] = self.serials.get(key, 0) if key else 0
        s = self.snap.loc[r["partner_id"]] if r["partner_id"] in self.snap.index else None
        d["partner_hist_rate"] = float(s.partner_hist_rate) if s is not None else self.cfg["prior_rate"]
        for c in ("partner_hist_n", "partner_vol_30d", "partner_small_share_30d", "partner_auto_share_30d"):
            d[c] = float(s[c]) if s is not None else 0.0
        return d

    def score(self, rec):
        r = self.validate(rec)
        d = self.features(r)
        p = float(self.model.predict_proba1(d)[0])
        row, c = d.iloc[0], self.cfg
        level = "HIGH" if p >= c["thr_high"] else "MEDIUM" if p >= c["thr_medium"] else "LOW"
        lg = float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
        p_cal = 1 / (1 + np.exp(-(c["calibration"]["a"] * lg + c["calibration"]["b"])))
        reasons = [{"code": x["code"], "text": x["text"]} for x in build_reasons(row, c["stats"])[:5]]
        return {"claim_id": r["claim_id"], "fraud_score": round(p, 4), "estimated_fraud_probability": round(float(p_cal), 4),
                "risk_level": level,
                "recommended_action": {"HIGH": "Send to the investigation desk before paying.",
                                       "MEDIUM": "Pay only after a quick manual check of photos/notes if desk capacity allows.",
                                       "LOW": "Normal approval flow."}[level],
                "expected_value_of_review_inr": round(p_cal * row.claim_amount_inr - (1 - p_cal) * c["hold_cost"]),
                "reasons": reasons,
                "note": "A high score means 'worth a human look', not that fraud is proven.",
                "model_version": c["version"]}

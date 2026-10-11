import re
import numpy as np
import pandas as pd
from source.config import (DATA_DIR, POLICY_CHANGE, SMALL_LIMIT, NEW_COHORT_START, LAG_DAYS, PRIOR_RATE, PRIOR_STRENGTH)

FAULTS = ["remote not working", "loud noise while running", "power button not working",
          "filter indicator stuck", "motor not running", "water leaking", "not charging",
          "blade jammed", "tripping mcb", "burning smell", "unit not heating", "display not working"]


def clean_fault(text: str) -> str:
    t = str(text).lower()
    for f in FAULTS:
        if t.startswith(f):
            return f
    return "other"


def norm_serial(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def serial_format(s) -> str:
    s = str(s)
    if s != s.strip():
        return "leading_space"
    if "-" in s:
        return "hyphen"
    if s != s.upper():
        return "lowercase"
    return "clean"


def load_reference():
    pa = pd.read_csv(DATA_DIR / "partners.csv")
    pa["onboarded"] = pd.to_datetime(pa["onboarded_date"], format="%d-%m-%Y")
    pr = pd.read_csv(DATA_DIR / "products.csv")
    return pa, pr


def load_claims(name: str):
    df = pd.read_csv(DATA_DIR / name)
    df["t"] = pd.to_datetime(df["submitted_at"], format="%d-%m-%Y %H:%M")
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["t", "claim_id"], kind="stable")
    n = df.groupby("claim_id")["t"].transform("size")
    out = df.drop_duplicates("claim_id", keep="first").copy()
    out["resubmissions"] = (n.loc[out.index] - 1).astype(int)
    return out.sort_values(["t", "claim_id"], kind="stable").reset_index(drop=True)


def base_features(df: pd.DataFrame, pa: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    """Row-level features that need only the claim itself + reference tables."""
    d = df.merge(pa[["partner_id", "city", "onboarded", "partner_type"]], on="partner_id", how="left") \
          .merge(pr, on="sku", how="left")
    d["fault"] = d["claim_description"].map(clean_fault)
    d["serial_norm"] = d["product_serial"].map(norm_serial)
    d["serial_fmt"] = d["product_serial"].map(serial_format)
    d["has_note"] = d["inspector_note"].notna().astype(int)
    d["photo"] = (d["photo_attached"] == "Y").astype(int)
    d["inspected"] = (d["partner_inspected"] == "Y").astype(int)
    d["log_amount"] = np.log1p(d["claim_amount_inr"])
    d["amount_ratio"] = d["claim_amount_inr"] / d["list_price_inr"]
    d["warranty_days"] = d["warranty_months"] * 30.4
    d["warranty_used"] = d["days_since_purchase"] / d["warranty_days"]
    d["out_of_warranty"] = (d["days_since_purchase"] > d["warranty_days"]).astype(int)
    d["partner_age_days"] = (d["t"] - d["onboarded"]).dt.days
    d["partner_new_180"] = (d["partner_age_days"] <= 180).astype(int)
    d["partner_new_cohort"] = (d["onboarded"] >= NEW_COHORT_START).astype(int)
    d["small_claim"] = (d["claim_amount_inr"] < SMALL_LIMIT).astype(int)
    d["post_policy"] = (d["t"] >= POLICY_CHANGE).astype(int)
    d["auto_approved"] = ((d["small_claim"] == 1) & (d["post_policy"] == 1) & (d["inspected"] == 0)).astype(int)
    d["hour"] = d["t"].dt.hour
    d["dow"] = d["t"].dt.dayofweek
    d["serial_fmt_odd"] = (d["serial_fmt"] != "clean").astype(int)
    d["new_x_auto"] = d["partner_new_cohort"] * d["auto_approved"]
    return d


def add_partner_behaviour(d: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Label-free behaviour of a partner in the trailing window (strictly earlier claims only).

    Works for brand-new partners because it needs no investigation outcomes."""
    d = d.sort_values(["t", "claim_id"], kind="stable").reset_index(drop=True)
    vol, small, auto, big = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for pid, g in d.groupby("partner_id"):
        t = g["t"].values.astype("datetime64[s]").astype(np.int64)
        sm, au = g["small_claim"].values, g["auto_approved"].values
        csm, cau = np.concatenate([[0], np.cumsum(sm)]), np.concatenate([[0], np.cumsum(au)])
        lo = np.searchsorted(t, t - window_days * 86400, side="left")
        hi = np.arange(len(g))                   
        n = hi - lo
        vol[g.index] = n
        small[g.index] = np.where(n > 0, (csm[hi] - csm[lo]) / np.maximum(n, 1), 0)
        auto[g.index] = np.where(n > 0, (cau[hi] - cau[lo]) / np.maximum(n, 1), 0)
    d["partner_vol_30d"], d["partner_small_share_30d"], d["partner_auto_share_30d"] = vol, small, auto
    return d


def add_serial_counts(d: pd.DataFrame, history_serials: pd.Series | None = None) -> pd.DataFrame:
    """How many OTHER claims (earlier in time) used the same normalised serial."""
    d = d.sort_values(["t", "claim_id"], kind="stable").copy()
    d["serial_prior_claims"] = d.groupby("serial_norm").cumcount()
    if history_serials is not None:
        d["serial_prior_claims"] += d["serial_norm"].map(history_serials.value_counts()).fillna(0).astype(int)
    return d


def add_partner_history(d: pd.DataFrame, labelled_history: pd.DataFrame, lag_days: int = 45,
                        prior_rate: float = PRIOR_RATE, strength: float = PRIOR_STRENGTH) -> pd.DataFrame:
    """Smoothed partner fraud rate using only outcomes that would have been KNOWN at claim time.

    An outcome is assumed known `lag_days` after submission (investigations are not instant).
    `labelled_history` needs columns partner_id, t, is_fraud (decided cases only).
    """
    h = labelled_history.sort_values("t", kind="stable")
    by_p = {p: g for p, g in h.groupby("partner_id")}
    rates, counts = [], []
    for pid, t in zip(d["partner_id"], d["t"]):
        g = by_p.get(pid)
        if g is None:
            rates.append(prior_rate); counts.append(0); continue
        k = g[g["t"] <= t - pd.Timedelta(days=lag_days)]
        n, f = len(k), k["is_fraud"].sum()
        rates.append((f + prior_rate * strength) / (n + strength)); counts.append(n)
    d = d.copy()
    d["partner_hist_rate"] = rates
    d["partner_hist_n"] = counts
    return d


NUM = ["log_amount", "amount_ratio", "days_since_purchase", "warranty_used", "customer_prior_claims",
       "photo", "inspected", "has_note", "partner_age_days", "partner_new_180", "small_claim",
          "post_policy", "auto_approved", "serial_prior_claims", "resubmissions",
       "partner_hist_rate", "partner_hist_n", "hour", "dow", "list_price_inr",
       "partner_new_cohort", "new_x_auto", "partner_vol_30d", "partner_small_share_30d", "partner_auto_share_30d"]
CAT = ["partner_type", "family"]
FEATURES = NUM + CAT


def _combined():
    """Train (deduped, incl. undecided) + test on ONE timeline so behaviour features see everything earlier."""
    pa, pr = load_reference()
    raw = load_claims("train.csv")
    tr = dedupe(raw)
    te = dedupe(load_claims("test_unlabelled.csv"))
    tr["is_test"], te["is_test"] = 0, 1
    allc = pd.concat([tr, te], ignore_index=True)
    allc = base_features(allc, pa, pr)
    allc = add_serial_counts(allc)
    allc = add_partner_behaviour(allc)
    stats = {"raw_rows": len(raw), "duplicate_claim_ids": int(raw.claim_id.duplicated().sum()),
             "rows_after_dedupe": len(tr), "undecided_dropped": int(tr.is_fraud.isna().sum())}
    return allc, stats


def build_training_table():
    allc, stats = _combined()
    d = allc[allc.is_test == 0]
    labelled = d[d.is_fraud.notna()].copy()
    labelled["is_fraud"] = labelled["is_fraud"].astype(int)
    labelled = add_partner_history(labelled, labelled[["partner_id", "t", "is_fraud"]], lag_days=LAG_DAYS)
    stats["labelled_rows"], stats["fraud_rows"] = len(labelled), int(labelled.is_fraud.sum())
    return labelled.reset_index(drop=True), d, stats


def build_test_table():
    allc, _ = _combined()
    train_all, _, _ = build_training_table()
    te = allc[allc.is_test == 1].copy()
    return add_partner_history(te, train_all[["partner_id", "t", "is_fraud"]], lag_days=0).reset_index(drop=True)

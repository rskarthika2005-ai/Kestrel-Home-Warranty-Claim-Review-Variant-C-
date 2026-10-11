import numpy as np
import pandas as pd
from modules.preprocess import load_claims, load_reference, base_features, dedupe
from source.config import DATA_DIR, OUTPUT_DIR, PREDICTIONS_FILE, POLICY_CHANGE, SMALL_LIMIT


def data_quality() -> pd.DataFrame:
    pa, pr = load_reference()
    raw, te_raw = load_claims("train.csv"), load_claims("test_unlabelled.csv")
    d = base_features(dedupe(raw), pa, pr)
    te = base_features(dedupe(te_raw), pa, pr)
    rows = []

    def add(check, value, ok, note=""):
        rows.append({"check": check, "value": value, "status": "OK" if ok else "ATTENTION", "note": note})

    n_dup = int(raw.claim_id.duplicated().sum())
    add("Train rows (raw)", len(raw), True)
    add("Unique claims after dedupe", len(d), True, f"{n_dup} resubmitted rows ({n_dup / len(raw):.1%}) removed, first submission kept")
    add("Test rows", len(te_raw), True, f"{te_raw.claim_id.duplicated().sum()} duplicate ids")
    add("Test ids also present in train", len(set(raw.claim_id) & set(te_raw.claim_id)), len(set(raw.claim_id) & set(te_raw.claim_id)) == 0, "must be 0 (leakage)")
    und = int(d.is_fraud.isna().sum())
    add("Undecided outcomes (blank)", und, True, "excluded from training")
    zero_legacy = int(((d.source == "legacy_zoho") & (d.is_fraud == 0)).sum())
    add("Legacy Zoho rows labelled 0", zero_legacy, True, "may include undecided cases stored as 0 (cannot be separated) - label-noise risk")
    lab = d[d.is_fraud.notna()]
    add("Fraud rate, decided claims", f"{lab.is_fraud.mean():.2%}", True, f"{int(lab.is_fraud.sum())} of {len(lab):,} - 'approve all' already scores {1 - lab.is_fraud.mean():.1%} accuracy")
    add("Partners missing from partners.csv (train/test)", f"{(~d.partner_id.isin(pa.partner_id)).sum()}/{(~te.partner_id.isin(pa.partner_id)).sum()}", True)
    add("SKUs missing from products.csv (train/test)", f"{(~d.sku.isin(pr.sku)).sum()}/{(~te.sku.isin(pr.sku)).sum()}", True)
    neg = int(((d.claim_amount_inr <= 0) | (d.days_since_purchase < 0)).sum() + ((te.claim_amount_inr <= 0) | (te.days_since_purchase < 0)).sum())
    add("Zero/negative amounts or days", neg, neg == 0)
    add("Claims before partner onboarding (train/test)", f"{(d.partner_age_days < 0).sum()}/{(te.partner_age_days < 0).sum()}", (d.partner_age_days < 0).sum() == 0)
    add("Claims beyond warranty period (train/test)", f"{d.out_of_warranty.sum()}/{te.out_of_warranty.sum()}", True, "none - warranty mismatch is NOT a fraud signal in this data")
    add("Serials typed with space/hyphen/lowercase", f"{d.serial_fmt_odd.mean():.0%}", True, "normalised before use; format itself is not a fraud signal")
    big_unins = d[(d.t >= POLICY_CHANGE) & (d.claim_amount_inr >= SMALL_LIMIT) & (d.inspected == 0)]
    add("Claims >= Rs 2,000 uninspected since 1 May", len(big_unins), len(big_unins) == 0, "policy says larger claims still need inspection")
    note_no_insp = int(((d.inspected == 0) & d.inspector_note.notna()).sum())
    add("Inspector note without inspection sign-off", note_no_insp, note_no_insp == 0)
    add("Train period", f"{d.t.min():%d %b %Y} - {d.t.max():%d %b %Y}", True)
    add("Test period", f"{te.t.min():%d %b %Y} - {te.t.max():%d %b %Y}", te.t.min() > d.t.max(), "test must be later than train (time-based validation)")
    add("Test claims auto-approved (small, uninspected)", f"{te.auto_approved.mean():.0%}", True, "test is almost entirely the post-policy regime")
    add("Test partners with no decided history", int((~te.partner_id.isin(d[d.is_fraud.notna()].partner_id)).sum() and te[~te.partner_id.isin(d[d.is_fraud.notna()].partner_id)].partner_id.nunique()), True, "scored on behaviour only")
    return pd.DataFrame(rows)


def prediction_check(path=PREDICTIONS_FILE) -> pd.DataFrame:
    p, s = pd.read_csv(path), pd.read_csv(DATA_DIR / "sample_submission.csv")
    checks = [("columns are claim_id,score", list(p.columns) == ["claim_id", "score"]),
              ("one row per input claim", len(p) == len(s)),
              ("same ids, same order as sample_submission", p.claim_id.tolist() == s.claim_id.tolist()),
              ("no duplicate ids", p.claim_id.is_unique),
              ("no blank scores", bool(p.score.notna().all())),
              ("scores between 0 and 1", bool(p.score.between(0, 1).all())),
              ("scores are not all identical", p.score.nunique() > 1)]
    return pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in checks])


def main():
    dq = data_quality(); dq.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
    pc = prediction_check()
    print("DATA QUALITY"); print(dq.to_string(index=False))
    print("\nPREDICTIONS.CSV CHECKS"); print(pc.to_string(index=False))
    assert (pc.status == "PASS").all(), "predictions.csv failed validation"


if __name__ == "__main__":
    main()

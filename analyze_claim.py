import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from modules.preprocess import build_training_table
from source.config import OUTPUT_DIR, EVIDENCE_DIR, POLICY_CHANGE, SMALL_LIMIT, CAPACITY

def facts_from(L):
    """Rates quoted inside reasons. Taken from the data, never typed in by hand."""
    post = L[L.post_policy == 1]
    return {"base_rate": float(L.is_fraud.mean()),
            "auto_fraud_rate": float(post[post.auto_approved == 1].is_fraud.mean()),
            "inspected_post_fraud_rate": float(post[post.inspected == 1].is_fraud.mean()),
            "high_ratio_fraud_rate": float(L[L.amount_ratio >= .5].is_fraud.mean()),
            "low_ratio_fraud_rate": float(L[L.amount_ratio < .5].is_fraud.mean()),
            "repeat2_fraud_rate": float(L[L.customer_prior_claims >= 2].is_fraud.mean())}


def build_reasons(d, st):
    """d = one claim's features (dict or Series). Returns [{'code','text','weight'}] strongest first.
    Every sentence quotes a value that is really in the claim or in the training data."""
    out = []

    def add(code, weight, text):
        out.append({"code": code, "weight": weight, "text": text})

    if d["auto_approved"]:
        add("AUTO_APPROVED_SMALL", 3,
            f"Small claim (Rs {d['claim_amount_inr']:,.0f}, under Rs {SMALL_LIMIT:,}) with no inspection - it would be auto-approved under the "
            f"{POLICY_CHANGE:%d %b %Y} rule. Since that rule, {st['auto_fraud_rate']:.1%} of such claims proved fraud versus "
            f"{st['inspected_post_fraud_rate']:.1%} of inspected claims.")
    if d["partner_hist_n"] >= 10 and d["partner_hist_rate"] >= 0.05:
        add("PARTNER_HISTORY", 3,
            f"Partner {d['partner_id']} has a high past fraud rate: about {d['partner_hist_rate']:.0%} of its decided claims "
            f"(smoothed, over {int(d['partner_hist_n'])} earlier decided claims) proved fraud, versus {st['base_rate']:.1%} overall.")
    if d["partner_vol_30d"] >= 15 and d["partner_auto_share_30d"] >= 0.6:
        add("PARTNER_PATTERN", 2,
            f"Partner filed {int(d['partner_vol_30d'])} claims in the last 30 days, {d['partner_auto_share_30d']:.0%} of them small and uninspected.")
    if d["amount_ratio"] >= 0.5:
        add("HIGH_AMOUNT_RATIO", 2,
            f"Claim is {d['amount_ratio']:.0%} of the product's list price (Rs {d['list_price_inr']:,.0f}). Claims at 50%+ of price proved fraud "
            f"{st['high_ratio_fraud_rate']:.1%} of the time versus {st['low_ratio_fraud_rate']:.1%} for the rest.")
    if d["customer_prior_claims"] >= 2:
        add("REPEAT_CUSTOMER", 2,
            f"Customer already has {int(d['customer_prior_claims'])} earlier warranty claims "
            f"({st['repeat2_fraud_rate']:.1%} of customers with 2+ earlier claims proved fraud).")
    if d["out_of_warranty"]:
        add("OUT_OF_WARRANTY", 2, "Claim date is beyond the product's warranty period.")
    if d["claim_amount_inr"] > d["list_price_inr"]:
        add("ABOVE_LIST_PRICE", 2, "Claim amount is higher than the product's list price.")
    if d["serial_prior_claims"] >= 4:
        add("SERIAL_REUSED", 1, f"The same serial number (after fixing spaces/hyphens/case) appears on {int(d['serial_prior_claims'])} earlier claims.")
    if d["partner_new_cohort"]:
        onb = d["onboarded"]
        when = f" {onb:%d %b %Y}" if pd.notna(onb) else ""
        add("NEW_PARTNER", 1, f"Partner onboarded{when} (new-partner cohort). Most new partners are fine - this only matters together with the other points.")
    if d["photo"] == 0:
        add("NO_PHOTO", 1, "No photo is attached.")
    # protective context, only shown when there is little else to say
    if len([o for o in out if o["weight"] >= 2]) == 0:
        if d["inspected"]:
            add("INSPECTED", 0, "Partner inspection sign-off is present, which lowers the risk.")
        else:
            add("NOT_INSPECTED", 0, "No partner inspection sign-off on this claim.")
        if not d["partner_new_cohort"]:
            add("ESTABLISHED_PARTNER", 0, "The outlet has been with Kestrel for more than a year.")
    if d["serial_fmt_odd"]:
        add("SERIAL_FORMAT", 0, "Serial was typed with extra spaces, a hyphen or lowercase letters; it was cleaned before checking (not a risk signal by itself).")
    if len(out) < 3:
        add("NOTHING_ELSE", 0, "Nothing else unusual was found in this claim.")
    return sorted(out, key=lambda x: -x["weight"])


def reasons_for(row, facts):
    """3 to 5 short reasons a Kestrel employee can read."""
    return [r["text"] for r in build_reasons(row, facts)[:5]]


def risk_level(score, thresholds):
    """thresholds = {'high': .., 'medium': ..}"""
    if score >= thresholds["high"]:
        return "High"
    if score >= thresholds["medium"]:
        return "Medium"
    return "Low"


# ----------------------------------------------------------------------------- statistics helpers
def wilson(frauds, n, z=1.96):
    """95% interval for a rate. With 36 frauds a bare percentage looks more certain than it is."""
    if n == 0:
        return (0.0, 0.0)
    p, d = frauds / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_table(df, by):
    base = df.is_fraud.mean()
    g = df.groupby(by, observed=True).is_fraud.agg(claims="size", frauds="sum").reset_index()
    g["fraud_rate_pct"] = (g.frauds / g.claims * 100).round(2)
    ci = [wilson(f, n) for f, n in zip(g.frauds, g.claims)]
    g["ci95_low_pct"] = [round(c[0] * 100, 2) for c in ci]
    g["ci95_high_pct"] = [round(c[1] * 100, 2) for c in ci]
    g["lift_vs_overall"] = (g.fraud_rate_pct / 100 / base).round(2)
    return g


def label_period(L):
    out = L.copy()
    out["period"] = np.where(out.t >= POLICY_CHANGE, "from_May_2026", "before_May_2026")
    out["claim_size"] = np.where(out.small_claim == 1, f"small_under_{SMALL_LIMIT}", f"large_{SMALL_LIMIT}_plus")
    out["partner_age"] = np.where(out.partner_new_cohort == 1, "new_partner", "older_partner")
    return out


# ----------------------------------------------------------------------------- pattern reports
def make_pattern_reports(L):
    """date_change_analysis.csv: fraud before/after the 1 May rule by claim size and partner age (with intervals).
    fraud_pattern_analysis.csv: the outlets behind the small-claim fraud since 1 May."""
    df = label_period(L)
    change = rate_table(df, ["period", "claim_size", "partner_age"])
    change.to_csv(OUTPUT_DIR / "date_change_analysis.csv", index=False)

    small = df[(df.period == "from_May_2026") & (df.small_claim == 1)]
    outlets = (small.groupby("partner_id").agg(small_claims=("claim_id", "size"), frauds=("is_fraud", "sum"), city=("city", "first"),
                                               partner_type=("partner_type", "first"), onboarded=("onboarded", "first"),
                                               new_partner=("partner_new_cohort", "first")).reset_index())
    amt = small[small.is_fraud == 1].groupby("partner_id").claim_amount_inr.sum()
    outlets["fraud_amount_inr"] = outlets.partner_id.map(amt).fillna(0).astype(int)
    outlets["fraud_rate_pct"] = (outlets.frauds / outlets.small_claims * 100).round(1)
    outlets["onboarded"] = outlets.onboarded.dt.strftime("%Y-%m-%d")
    outlets = outlets[outlets.frauds > 0].sort_values(["frauds", "fraud_rate_pct"], ascending=False)
    outlets.to_csv(OUTPUT_DIR / "fraud_pattern_analysis.csv", index=False)
    return change, outlets


def test_new_partner_idea(L):
    """Ritu: 'it is the newer partners'. Tested overall, before and since 1 May, with Fisher's exact test."""
    res = {}
    for name, sub in {"all_period": L, "since_1_May_2026": L[L.post_policy == 1], "before_1_May_2026": L[L.post_policy == 0]}.items():
        new, old = sub[sub.partner_new_cohort == 1], sub[sub.partner_new_cohort == 0]
        fn, fo, nn, no = int(new.is_fraud.sum()), int(old.is_fraud.sum()), len(new), len(old)
        _, p = fisher_exact([[fn, nn - fn], [fo, no - fo]])
        rn, ro = fn / max(nn, 1), fo / max(no, 1)
        res[name] = {"new_claims": nn, "new_frauds": fn, "new_rate_pct": round(rn * 100, 2), "older_claims": no, "older_frauds": fo,
                     "older_rate_pct": round(ro * 100, 2), "rate_ratio": round(rn / ro, 1) if ro > 0 else None, "fisher_p": float(f"{p:.2g}")}
    new = L[(L.partner_new_cohort == 1) & (L.post_policy == 1)]
    per = new.groupby("partner_id").is_fraud.agg(["sum", "size"])
    active = per[per["size"] >= 3]
    bad = active[active["sum"] >= 2]
    res["inside_new_cohort_since_May"] = {
        "active_new_outlets": len(active), "outlets_with_2plus_frauds": len(bad),
        "share_of_new_partner_fraud_from_those": round(float(bad["sum"].sum() / max(per["sum"].sum(), 1)), 2),
        "verdict": "Concentrated in a few outlets - not the whole cohort" if len(bad) < 0.25 * len(active) else "Broad across the cohort"}
    return res


def risk_factor_table(L):
    """Every factor, every band: fraud rate with interval and lift."""
    d = label_period(L)
    d["amount_vs_price"] = pd.cut(d.amount_ratio, [0, .1, .25, .5, .75, 1.5], labels=["<10%", "10-25%", "25-50%", "50-75%", "75%+"])
    d["customer_prior_claims_band"] = d.customer_prior_claims.clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
    d["days_since_purchase_band"] = pd.cut(d.days_since_purchase, [-1, 30, 90, 180, 365, 800], labels=["0-30", "31-90", "91-180", "181-365", "366+"])
    d["photo_attached"] = d.photo.map({1: "Y", 0: "N"})
    d["inspected_by_partner"] = d.inspected.map({1: "Y", 0: "N"})
    d["partner_cohort"] = d.partner_new_cohort.map({1: "new (since Oct 2025)", 0: "older"})
    parts = []
    for f in ["amount_vs_price", "customer_prior_claims_band", "days_since_purchase_band", "photo_attached", "inspected_by_partner",
              "partner_cohort", "partner_type", "family", "source", "serial_fmt", "period"]:
        t = rate_table(d, f).rename(columns={f: "band"}); t.insert(0, "factor", f); t["band"] = t.band.astype(str); parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(EVIDENCE_DIR / "risk_factor_table.csv", index=False)
    return out


def outlet_concentration(L):
    res = {}
    for name, sub in {"before_1_May_2026": L[L.post_policy == 0], "since_1_May_2026": L[L.post_policy == 1]}.items():
        c = sub.groupby("partner_id").is_fraud.sum().sort_values(ascending=False)
        total = int(c.sum())
        res[name] = {"frauds": total, "outlets_with_fraud": int((c > 0).sum()),
                     "top5_outlets_share": round(float(c.head(5).sum() / max(total, 1)), 2), "top5": {k: int(v) for k, v in c.head(5).items()}}
    return res


def failure_cases(L):
    """June 2026 hold-out: what the model missed and what it wrongly held."""
    from modules.train_model import Ensemble, queue_size
    a, b = L[L.t <= "2026-05-31 23:59"], L[L.t >= "2026-06-01"].reset_index(drop=True)
    b["score"] = Ensemble().fit(a, a.is_fraud, a.post_policy).predict_proba1(b)
    k = queue_size("2026-06-01", "2026-06-30 23:59")
    thr = np.sort(b.score.values)[::-1][k - 1]
    cols = ["claim_id", "partner_id", "partner_new_cohort", "claim_amount_inr", "auto_approved", "partner_hist_n", "score", "is_fraud"]
    missed = b[(b.score < thr) & (b.is_fraud == 1)][cols].sort_values("score", ascending=False)
    alarms = b[(b.score >= thr) & (b.is_fraud == 0)][cols].sort_values("score", ascending=False)
    missed.to_csv(EVIDENCE_DIR / "failure_missed_fraud_june.csv", index=False)
    alarms.to_csv(EVIDENCE_DIR / "failure_false_alarms_june.csv", index=False)
    return {"queue_size": k, "missed": len(missed), "missed_amount_inr": int(missed.claim_amount_inr.sum()),
            "missed_with_no_partner_history": int((missed.partner_hist_n < 5).sum()),
            "false_alarms": len(alarms), "false_alarm_partners": {k_: int(v) for k_, v in alarms.partner_id.value_counts().head(5).items()}}


def make_charts(change, outlets):
    EVIDENCE_DIR.mkdir(exist_ok=True)
    c = change.copy(); c["group"] = c.claim_size.str.replace("_", " ") + "\n" + c.partner_age.str.replace("_", " ")
    piv = c.pivot(index="group", columns="period", values="fraud_rate_pct").fillna(0)[["before_May_2026", "from_May_2026"]]
    ax = piv.plot.bar(figsize=(8, 4.2), color=["#9aa5b1", "#d64545"], rot=0)
    ax.set_ylabel("fraud rate, % of claims"); ax.set_title("Fraud before vs from 1 May 2026 (small claims from new partners jumped)")
    ax.legend(["before 1 May", "from 1 May"]); plt.tight_layout(); plt.savefig(EVIDENCE_DIR / "chart_before_after_may.png", dpi=130); plt.close()
    o = outlets.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(o.partner_id + " (" + o.city + ")", o.frauds, color=np.where(o.new_partner == 1, "#d64545", "#9aa5b1"))
    ax.set_xlabel("confirmed frauds in small claims since 1 May (red = new partner)"); ax.set_title("Outlets behind the new fraud")
    plt.tight_layout(); plt.savefig(EVIDENCE_DIR / "chart_outlets_since_may.png", dpi=130); plt.close()

def main():
    OUTPUT_DIR.mkdir(exist_ok=True); EVIDENCE_DIR.mkdir(exist_ok=True)
    L, _, stats = build_training_table()
    change, outlets = make_pattern_reports(L)
    risk_factor_table(L)
    make_charts(change, outlets)
    report = {"data_quality": stats,
              "fraud_rate_pct": {"all": round(float(L.is_fraud.mean()) * 100, 2), "before_May": round(float(L[L.post_policy == 0].is_fraud.mean()) * 100, 2),
                                 "since_May": round(float(L[L.post_policy == 1].is_fraud.mean()) * 100, 2)},
              "new_partner_idea": test_new_partner_idea(L), "outlet_concentration": outlet_concentration(L), "failure_cases_june": failure_cases(L)}
    json.dump(report, open(EVIDENCE_DIR / "fraud_patterns.json", "w"), indent=2, default=str)

    print("FRAUD RATE %:", report["fraud_rate_pct"])
    print("\nBEFORE vs FROM 1 MAY (rate with 95% interval):"); print(change.to_string(index=False))
    n = report["new_partner_idea"]
    print("\nIS IT THE NEW PARTNERS?")
    for k in ("all_period", "before_1_May_2026", "since_1_May_2026"):
        print(f"  {k:18s} new {n[k]['new_rate_pct']}% vs older {n[k]['older_rate_pct']}%  ratio {n[k]['rate_ratio']}  p={n[k]['fisher_p']}")
    print("  inside the cohort:", n["inside_new_cohort_since_May"])
    print("\nOUTLET CONCENTRATION:", report["outlet_concentration"])
    print("\nTOP OUTLETS (small claims since May):"); print(outlets.head(8).to_string(index=False))
    print("\nFAILURES (June hold-out):", report["failure_cases_june"])


if __name__ == "__main__":
    main()

import sys
sys.dont_write_bytecode = True   

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUT = BASE / "output"
EVIDENCE = BASE / "evidence"

st.set_page_config(page_title="Kestrel warranty claims", layout="wide")



# PAGE 1: score one claim

def read_reference(name):
    path = DATA / name
    return pd.read_csv(path) if path.exists() else None


def score_page():
    st.title("Kestrel warranty claim check")
    st.write("Enter one claim to see how risky it looks and why. "
             "A high score means 'worth a human look', not that fraud is proven.")

    partners = read_reference("partners.csv")
    products = read_reference("products.csv")

    # ---- Section 1: claim form ----
    st.header("1. Claim details")
    with st.form("claim_form"):
        if partners is not None:
            city_of = dict(zip(partners["partner_id"], partners["city"]))
            partner_id = st.selectbox("Partner outlet", sorted(city_of),
                                      format_func=lambda p: p + " - " + str(city_of[p]))
        else:
            partner_id = st.text_input("Partner outlet (for example SP3160)")

        if products is not None:
            family_of = dict(zip(products["sku"], products["family"]))
            sku = st.selectbox("Product", sorted(family_of),
                               format_func=lambda s: s + " - " + str(family_of[s]))
        else:
            sku = st.text_input("Product SKU (for example KH-RV-03)")

        amount = st.number_input("Claim amount (Rs)", min_value=1, value=1500, step=50)
        days = st.number_input("Days since purchase", min_value=0, value=120, step=1)
        prior = st.number_input("Customer's earlier warranty claims", min_value=0, value=0, step=1)
        photo = st.radio("Photo attached?", ["Y", "N"], horizontal=True)
        inspected = st.radio("Partner inspected the product?", ["Y", "N"], index=1, horizontal=True)
        serial = st.text_input("Product serial (optional)")
        description = st.text_input("Fault description (optional)")
        use_ai = st.checkbox("Also write a short GPT explanation (needs an API key, costs a little)")
        submitted = st.form_submit_button("Score claim")

    # ---- Section 2: result ----
    st.header("2. Result")
    if not submitted:
        st.caption("Fill in the form and press 'Score claim'.")
        return

    claim = {"partner_id": partner_id, "sku": sku, "claim_amount_inr": amount,
             "days_since_purchase": days, "customer_prior_claims": prior,
             "photo_attached": photo, "partner_inspected": inspected}
    if serial:
        claim["product_serial"] = serial
    if description:
        claim["claim_description"] = description

    try:
        reply = requests.post(API_URL + "/score", json=claim,
                              params={"ai": "true" if use_ai else "false"}, timeout=60)
    except requests.RequestException:
        st.error("The scoring service is not running. Start everything with: python app.py")
        return

    if reply.status_code != 200:
        # The API sends a message written for an employee
        try:
            st.warning(reply.json().get("detail", "This claim could not be scored."))
        except ValueError:
            st.warning("This claim could not be scored.")
        return

    result = reply.json()
    level = result["risk_level"]
    if level == "HIGH":
        st.error("Risk level: HIGH")
    elif level == "MEDIUM":
        st.warning("Risk level: MEDIUM")
    else:
        st.success("Risk level: LOW")
    st.write("**What to do:** " + result["recommended_action"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud score (0 to 1)", result["fraud_score"])
    col2.metric("Estimated chance of fraud", str(round(result["estimated_fraud_probability"] * 100, 1)) + "%")
    col3.metric("Expected value of review (Rs)", result["expected_value_of_review_inr"])

    st.subheader("Why")
    for reason in result["reasons"]:
        st.write("- " + reason["text"])

    ai = result.get("ai_explanation")
    if ai is not None:
        st.subheader("GPT explanation")
        if ai["status"] == "ok":
            st.info(ai["text"])
        else:
            st.caption(ai["text"])

    st.caption(result["note"] + "  Model version: " + str(result["model_version"]))


# PAGE 2: what the data says

def read_output(name):
    path = OUTPUT / name
    return pd.read_csv(path) if path.exists() else None


def data_page():
    st.title("What the data says")

    validation = read_output("validation_report.csv")
    scored = read_output("scored_claims.csv")
    change = read_output("date_change_analysis.csv")
    outlets = read_output("fraud_pattern_analysis.csv")

    if validation is None or scored is None or change is None or outlets is None:
        st.warning("The result files are missing. Run 'python run_pipeline.py' first, then refresh this page.")
        return

    
    st.header("1. Money and the review queue")

   
    june = validation[(validation["split"] == "B") & validation["model"].str.startswith("ENSEMBLE")].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims checked (June)", int(june["queue_size"]))
    c2.metric("Frauds caught", str(int(june["tp"])) + " of " + str(int(june["test_frauds"])))
    c3.metric("Fraud stopped per claim checked", "Rs " + str(int(june["inr_stopped_per_claim_checked"])))
    c4.metric("After goodwill cost (Rs 380 per wrong hold)", "Rs " + str(int(june["net_inr_per_claim_checked"])))
    st.caption("90% range for the last number: Rs " + str(int(june["net_inr_ci90_low"])) + " to Rs "
               + str(int(june["net_inr_ci90_high"])) + ". Only 22 frauds in June, so treat it as rough.")

    st.write("**Why accuracy is the wrong KPI:** approving every claim scores "
             + str(round(june["accuracy_approve_all"] * 100, 1)) + "% accuracy and catches nothing. "
             + "Checking the top claims scores " + str(round(june["accuracy_top_k"] * 100, 1))
             + "% accuracy but catches " + str(round(june["recall"] * 100)) + "% of the fraud.")

    st.subheader("Review queue: the 40 highest scores each month")
    scored["month"] = pd.to_datetime(scored["submitted_at"], format="%d-%m-%Y %H:%M").dt.strftime("%Y-%m")
    month = st.selectbox("Month", sorted(scored["month"].unique()))
    queue = scored[(scored["month"] == month) & (scored["in_review_queue"] == "YES")]
    queue_view = queue[["rank", "claim_id", "partner_id", "city", "claim_amount_inr",
                        "fraud_score", "risk_level", "top_reason"]]
    estimate = (queue["est_fraud_probability"] * queue["claim_amount_inr"]).sum()
    st.write(str(len(queue)) + " claims from " + str(queue["partner_id"].nunique())
             + " outlets. Estimated fraud in this queue: about Rs " + format(int(estimate), ",") + " (an estimate).")
    st.dataframe(queue_view)
    st.download_button("Download this queue (CSV)", queue_view.to_csv(index=False),
                       file_name="review_queue_" + month + ".csv")

    
    st.header("2. Patterns and proof")

    st.subheader("Before and from 1 May 2026")
    st.dataframe(change)
    chart1 = EVIDENCE / "chart_before_after_may.png"
    if chart1.exists():
        st.image(str(chart1))
    st.write("Since 1 May, small claims from new partners are the problem. "
             "Before that date, the fraud sat in larger claims from older partners.")

    st.subheader("Outlets behind the small-claim fraud since 1 May")
    st.dataframe(outlets)
    chart2 = EVIDENCE / "chart_outlets_since_may.png"
    if chart2.exists():
        st.image(str(chart2))
    st.write("A few outlets carry most of it. Most new partners are fine, so do not judge them as one group.")

    st.subheader("Is the model better than simple rules? (3 time-based tests)")
    st.caption("Split A trains before 1 May and tests May to June: the model fails there because the fraud pattern changed. "
               "The final model learns from all claims, with claims since 1 May counted more.")
    table = validation[["split", "model", "test_frauds", "queue_size", "tp", "fp", "roc_auc",
                        "inr_stopped_per_claim_checked", "net_inr_per_claim_checked"]]
    st.dataframe(table)
    cm = EVIDENCE / "confusion_matrix_split_B.png"
    if cm.exists():
        st.image(str(cm))



# Choose the page

page = st.sidebar.radio("Page", ["Score a claim", "What the data says"])
if page == "Score a claim":
    score_page()
else:
    data_page()

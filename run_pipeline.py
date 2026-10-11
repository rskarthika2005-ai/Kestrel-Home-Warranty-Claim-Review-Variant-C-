import sys
sys.dont_write_bytecode = True  

import pandas as pd
from modules import validation, analyze_claim, train_model, generate_predictions
from modules.predict_claim import ClaimScorer
from source.config import OUTPUT_DIR, EVIDENCE_DIR, MODEL_DIR, PREDICTIONS_FILE


EXPECTED_FILES = [
    PREDICTIONS_FILE,
    OUTPUT_DIR / "data_quality_report.csv",
    OUTPUT_DIR / "fraud_pattern_analysis.csv",
    OUTPUT_DIR / "date_change_analysis.csv",
    OUTPUT_DIR / "validation_report.csv",
    OUTPUT_DIR / "scored_claims.csv",
    EVIDENCE_DIR / "confusion_matrix_split_B.png",
    EVIDENCE_DIR / "chart_before_after_may.png",
    EVIDENCE_DIR / "chart_outlets_since_may.png",
    MODEL_DIR / "model.joblib",
    MODEL_DIR / "scorer_config.json",
    MODEL_DIR / "partner_snapshot.csv",
    MODEL_DIR / "serial_counts.json",
]


def check_outputs():
    # Simple checks that every module really produced its output
    problems = []

    # 1. every file exists and is not empty
    for path in EXPECTED_FILES:
        if not path.exists() or path.stat().st_size == 0:
            problems.append("missing or empty: " + path.name)

    # 2. scored_claims.csv has the same claims as predictions.csv
    if not problems:
        preds = pd.read_csv(PREDICTIONS_FILE)
        scored = pd.read_csv(OUTPUT_DIR / "scored_claims.csv")
        if set(preds.claim_id) != set(scored.claim_id):
            problems.append("scored_claims.csv and predictions.csv have different claims")
        if scored.risk_level.isna().any() or scored.reasons.isna().any():
            problems.append("scored_claims.csv has blank risk levels or reasons")

    # 3. the one-claim scorer (used by the API) works
    if not problems:
        claim = {"partner_id": "SP3160", "sku": "KH-RV-03", "days_since_purchase": 120,
                 "claim_amount_inr": 1400, "photo_attached": "N", "partner_inspected": "N"}
        result = ClaimScorer().score(claim)
        if not (0 <= result["fraud_score"] <= 1) or len(result["reasons"]) < 3:
            problems.append("predict_claim.py gave a strange result")

    return problems


def main():
    print("STEP 1 of 6: data quality checks")
    dq = validation.data_quality()
    dq.to_csv(validation.OUTPUT_DIR / "data_quality_report.csv", index=False)
    print(dq[dq.status == "ATTENTION"].to_string(index=False))

    print("\nSTEP 2 of 6: where the fraud is (patterns and charts)")
    analyze_claim.main()

    print("\nSTEP 3 of 6: train and validate the model")
    train_model.main()

    print("\nSTEP 4 of 6: write predictions.csv and scored_claims.csv")
    generate_predictions.main()

    print("\nSTEP 5 of 6: check predictions.csv against sample_submission.csv")
    checks = validation.prediction_check()
    print(checks.to_string(index=False))
    assert (checks.status == "PASS").all(), "predictions.csv failed a check"

    print("\nSTEP 6 of 6: check every module made its output")
    problems = check_outputs()
    if problems:
        for p in problems:
            print("  PROBLEM:", p)
        raise SystemExit("Pipeline finished with problems")
    print("  All", len(EXPECTED_FILES), "files found, scored_claims matches predictions, one-claim scorer works")

    print("\nDone. Start the app next.")


if __name__ == "__main__":
    main()
import pandas as pd
from source.config import DATA_DIR, PREDICTIONS_FILE


def test_predictions_match_sample_submission():
    preds = pd.read_csv(PREDICTIONS_FILE)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert list(preds.columns) == ["claim_id", "score"]
    assert preds["claim_id"].tolist() == sample["claim_id"].tolist()


def test_scores_are_valid():
    preds = pd.read_csv(PREDICTIONS_FILE)
    assert preds["score"].notna().all()
    assert preds["score"].between(0, 1).all()
    assert preds["score"].nunique() > 1

import pandas as pd
from modules.preprocess import norm_serial, dedupe


def test_norm_serial():
    assert norm_serial(" kh-123456789") == "KH123456789"
    assert norm_serial("KH123456789") == "KH123456789"


def test_dedupe_keeps_first_submission():
    df = pd.DataFrame({
        "claim_id": ["A", "A", "B"],
        "t": pd.to_datetime(["2025-04-01", "2025-04-03", "2025-04-02"]),
    })
    out = dedupe(df)
    assert len(out) == 2
    first = out[out["claim_id"] == "A"].iloc[0]
    assert first["t"] == pd.Timestamp("2025-04-01")
    assert first["resubmissions"] == 1
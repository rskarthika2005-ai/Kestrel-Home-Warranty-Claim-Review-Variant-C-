import os
import sys
import types

from source import llm_client

RESULT = {"risk_level": "HIGH", "fraud_score": 0.8,
          "reasons": [{"code": "X", "text": "Small claim with no inspection."}]}
CLAIM = {"claim_amount_inr": 1500, "days_since_purchase": 100,
         "photo_attached": "N", "partner_inspected": "N"}


def test_no_key_gives_polite_message():
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        out = llm_client.explain(CLAIM, RESULT)
        assert out["status"] == "off"
        assert "no API key" in out["text"]
    finally:
        if old:
            os.environ["OPENAI_API_KEY"] = old


def test_failed_call_does_not_crash():
    # Pretend the openai library exists but the call fails
    class BrokenClient:
        def __init__(self, **kwargs):
            raise RuntimeError("network down")

    fake = types.ModuleType("openai")
    fake.OpenAI = BrokenClient
    old_module = sys.modules.get("openai")
    old_key = os.environ.get("OPENAI_API_KEY")
    sys.modules["openai"] = fake
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        out = llm_client.explain(CLAIM, RESULT)
        assert out["status"] == "failed"
    finally:
        if old_module is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = old_module
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key
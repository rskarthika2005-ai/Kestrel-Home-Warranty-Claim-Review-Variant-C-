import sys
sys.dont_write_bytecode = True  

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from source.prompts import SYSTEM_PROMPT, build_user_prompt


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PRICE_IN_PER_M = float(os.getenv("OPENAI_PRICE_IN_PER_M", "0.15"))
PRICE_OUT_PER_M = float(os.getenv("OPENAI_PRICE_OUT_PER_M", "0.60"))

NO_KEY_TEXT = "AI explanation is off because no API key is set. The reasons above still apply."
FAILED_TEXT = "AI explanation is not available right now. The reasons above still apply."

usage = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}


saved = {}


def total_cost_usd():
    cost_in = usage["input_tokens"] / 1_000_000 * PRICE_IN_PER_M
    cost_out = usage["output_tokens"] / 1_000_000 * PRICE_OUT_PER_M
    return cost_in + cost_out


def usage_summary():
    out = dict(usage)
    out["model"] = MODEL
    out["cost_usd"] = round(total_cost_usd(), 6)
    return out


def explain(claim, result):
    """claim = the JSON the user sent, result = what predict_claim.py returned."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return {"status": "off", "text": NO_KEY_TEXT}

    facts = {
        "risk_level": result["risk_level"],
        "score": result["fraud_score"],
        "amount": claim.get("claim_amount_inr"),
        "days": claim.get("days_since_purchase"),
        "photo": claim.get("photo_attached"),
        "inspected": claim.get("partner_inspected"),
        "reasons": [r["text"] for r in result["reasons"]],
    }
    prompt = build_user_prompt(facts)

  
    if prompt in saved:
        usage["cached"] += 1
        return {"status": "ok", "text": saved[prompt]}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=15)
        reply = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=150,
        )
        text = reply.choices[0].message.content.strip()
        usage["calls"] += 1
        usage["input_tokens"] += reply.usage.prompt_tokens
        usage["output_tokens"] += reply.usage.completion_tokens
        saved[prompt] = text
        return {"status": "ok", "text": text}
    except Exception as error:
        print("AI explanation failed:", type(error).__name__)
        return {"status": "failed", "text": FAILED_TEXT}
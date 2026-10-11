import sys
sys.dont_write_bytecode = True   

from fastapi import Body, FastAPI, HTTPException

from modules.predict_claim import ClaimScorer, ClaimError
from source import llm_client

app = FastAPI(title="Kestrel warranty claim scoring")

state = {"scorer": None, "error": None}


def get_scorer():
    if state["scorer"] is None:
        try:
            state["scorer"] = ClaimScorer()
            state["error"] = None
        except Exception as error:
            state["error"] = type(error).__name__
            print("Could not load the model:", error)
    return state["scorer"]


@app.get("/health")
def health():
    ready = get_scorer() is not None
    return {"status": "ok" if ready else "model_not_ready",
            "ai_explanations": "on" if llm_client.os.getenv("OPENAI_API_KEY") else "off"}


@app.post("/score")
def score(claim: dict = Body(...), ai: bool = False):
    """Send one claim as JSON. Add ?ai=true for an optional GPT explanation."""
    scorer = get_scorer()
    if scorer is None:
        raise HTTPException(status_code=503,
                            detail="The model is not ready. Run 'python run_pipeline.py' first.")
    try:
        result = scorer.score(claim)
    except ClaimError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        print("Scoring failed:", type(error).__name__, error)
        raise HTTPException(status_code=500,
                            detail="Sorry, this claim could not be scored. Please check the values.")

    if ai:
        result["ai_explanation"] = llm_client.explain(claim, result)
    return result


@app.get("/ai_usage")
def ai_usage():
    return llm_client.usage_summary()
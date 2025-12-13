import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from FlagEmbedding import FlagReranker
import argparse

# --- Configuration ---
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True

# Global variable to hold the model
reranker_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # --- Startup Logic ---
    global reranker_model
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    reranker_model = FlagReranker(MODEL_NAME, use_fp16=FP16, device=DEVICE)
    print("Model loaded successfully. Ready to serve.")

    yield  # Control is yielded to the application here

    # --- Shutdown Logic (Optional) ---
    print("Shutting down model...")
    del reranker_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialize FastAPI with the lifespan handler
app = FastAPI(title="Custom Reranker Server", lifespan=lifespan)


class ScoreRequest(BaseModel):
    text_1: List[str]  # Queries
    text_2: List[str]  # Documents
    model: Optional[str] = None


@app.post("/v1/score")
async def score(request: ScoreRequest):
    """
    Mimics the vLLM /score endpoint structure.
    """
    if len(request.text_1) != len(request.text_2):
        raise HTTPException(status_code=400, detail="text_1 and text_2 lists must have the same length.")

    # Pair them up: [(q1, d1), (q2, d2), ...]
    pairs = list(zip(request.text_1, request.text_2))

    if not pairs:
        return {"data": []}

    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized yet.")

    try:
        # FlagReranker.compute_score returns a list of floats (logits or scores)
        scores = reranker_model.compute_score(pairs, normalize=True)

        # If single pair, it returns a float, ensure it's a list
        if isinstance(scores, float):
            scores = [scores]

        # Format response to match vLLM style
        data = [{"score": float(s), "index": i} for i, s in enumerate(scores)]
        return {"data": data}

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Reranker Server")

    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    args = parser.parse_args()

    print(f"Starting Reranker Server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

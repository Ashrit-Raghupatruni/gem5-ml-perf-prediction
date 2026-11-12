"""FastAPI main application serving REST endpoints and static UI dashboard."""
import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import SingleHardwareConfig, SinglePredictionResponse, BatchPredictionResponse
from backend.services import predict_single, predict_batch_df, get_pareto_reference, REPO_ROOT

app = FastAPI(
    title="gem5 ML Performance Prediction Engine",
    description="Microarchitectural performance prediction, uncertainty estimation, and design space exploration.",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")

@app.post("/api/predict", response_model=SinglePredictionResponse)
def api_predict_single(config: SingleHardwareConfig):
    try:
        config_dict = config.model_dump(exclude_none=True)
        result = predict_single(config_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def api_predict_batch(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        contents = await file.read()
        
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .json files are supported.")
            
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file contains 0 rows.")
            
        result = predict_batch_df(df)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/api/pareto")
def api_get_pareto():
    return {
        "status": "success",
        "top5_candidates": get_pareto_reference()
    }

@app.get("/api/defaults/{workload}")
def api_get_workload_defaults(workload: str):
    valid_workloads = ["matmul", "stream", "quicksort", "hash"]
    if workload not in valid_workloads:
        workload = "matmul"
    return {
        "workload": workload,
        "fetch_width": 4,
        "decode_width": 4,
        "issue_width": 4,
        "commit_width": 4,
        "rob_entries": 128,
        "branch_predictor": "TournamentBP",
        "l1d_size_kb": 64,
        "l1d_assoc": 2,
        "l1i_size_kb": 32,
        "l1i_assoc": 2,
        "l2_size_kb": 1024,
        "l2_assoc": 4,
        "clk_freq_ghz": 2.4
    }

# Mount static frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "gem5 ML Prediction API active. Frontend index.html not yet initialized."}

"""FastAPI main application serving REST endpoints and static UI dashboard."""
import os
import io
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import SingleHardwareConfig, SinglePredictionResponse, BatchPredictionResponse
from backend.services import predict_single, predict_batch_df, get_pareto_reference, REPO_ROOT

app = FastAPI(
    title="gem5 ML Performance Prediction Engine",
    description="Microarchitectural performance prediction, dataset inspection, and active learning.",
    version="2.1.0"
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
VISUALIZATIONS_DIR = os.path.join(REPO_ROOT, "visualizations")
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
MODELS_DIR = os.path.join(REPO_ROOT, "ml", "saved_models")

# Mount visualizations as static route
if os.path.exists(VISUALIZATIONS_DIR):
    app.mount("/visualizations", StaticFiles(directory=VISUALIZATIONS_DIR), name="visualizations")

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

@app.get("/api/dataset/view")
def api_view_dataset(view_type: str = "features", limit: int = 50):
    """View-only inspection for Input Features or Output Targets without download."""
    if view_type == "outputs":
        csv_file = os.path.join(DATASET_DIR, "dataset_outputs.csv")
    else:
        csv_file = os.path.join(DATASET_DIR, "dataset_features.csv")
        
    if not os.path.exists(csv_file):
        # Fallback to dataset_cleaned.csv if split files aren't found
        csv_file = os.path.join(DATASET_DIR, "dataset_cleaned.csv")
        
    if not os.path.exists(csv_file):
        raise HTTPException(status_code=404, detail="Dataset file not found")
        
    df = pd.read_csv(csv_file, nrows=limit)
    # Total count without loading full memory
    with open(csv_file, "r") as f:
        total_rows = sum(1 for _ in f) - 1
        
    return {
        "view_type": view_type,
        "total_rows": total_rows,
        "columns": list(df.columns),
        "data": df.fillna("").to_dict(orient="records")
    }

@app.get("/api/validation/top5")
def api_get_validation_top5():
    """Ground-truth validation comparison for Top-5 candidates."""
    val_csv = os.path.join(DATASET_DIR, "validation_results.csv")
    val_json = os.path.join(DATASET_DIR, "validation_summary.json")
    
    table_data = []
    if os.path.exists(val_csv):
        df = pd.read_csv(val_csv)
        table_data = df.to_dict(orient="records")
        
    summary = {}
    if os.path.exists(val_json):
        with open(val_json, "r") as f:
            summary = json.load(f)
            
    return {
        "status": "success",
        "table": table_data,
        "summary": summary
    }

@app.get("/api/models/metrics")
def api_get_model_metrics():
    """Metrics across the 4 evaluated models."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

@app.get("/api/active-learning/queries")
def api_get_active_queries():
    """Queries selected by active learning uncertainty sampling."""
    queries_path = os.path.join(CONFIGS_DIR, "active_queries.json")
    if os.path.exists(queries_path):
        with open(queries_path, "r") as f:
            data = json.load(f)
            return {"status": "success", "queries": data}
    return {"status": "error", "queries": []}

@app.get("/api/pareto")
def api_get_pareto():
    return {
        "status": "success",
        "top5_candidates": get_pareto_reference()
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


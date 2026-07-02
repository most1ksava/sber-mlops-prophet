import os
import pickle

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from model import StockForecastModel

app = FastAPI(title="Stock Prediction API", docs_url="/docs")

MODEL_PATH = os.getenv("MODEL_PATH", "mlflow_artifacts/model.pkl")
LAST_DATE_PATH = os.getenv("LAST_DATE_PATH", "mlflow_artifacts/last_date.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sber_stock_forecast")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

predict_fn = None
_model_info = {"source": "none", "run_id": ""}
_last_training_date = ""


def load_model_from_mlflow():
    global predict_fn, _model_info
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    client = mlflow.tracking.MlflowClient()
    try:
        latest_version = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if not latest_version:
            latest_version = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if latest_version:
            version = latest_version[-1].version
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{version}"
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            predict_fn = pyfunc_model.predict
            _model_info = {"source": "mlflow", "run_id": model_uri}
            logger.success(f"Loaded model from MLFlow: {model_uri}")
            return True
    except Exception as e:
        logger.warning(f"Could not load from MLFlow: {e}")
    return False


def load_model_local():
    global predict_fn, _model_info
    wrapper = StockForecastModel()
    artifacts = {"model": MODEL_PATH, "last_date": LAST_DATE_PATH}

    class LocalContext:
        def __init__(self, artifacts):
            self.artifacts = artifacts

    ctx = LocalContext(artifacts)
    wrapper.load_context(ctx)
    predict_fn = lambda data: wrapper.predict(ctx, data)
    _model_info = {"source": "local", "run_id": ""}
    logger.info(f"Loaded model from local: {MODEL_PATH}")


def load_model():
    global _last_training_date
    if MLFLOW_TRACKING_URI:
        if not load_model_from_mlflow():
            load_model_local()
    else:
        load_model_local()
    try:
        with open(LAST_DATE_PATH, "rb") as f:
            _last_training_date = str(pickle.load(f).date())
    except Exception:
        _last_training_date = "unknown"


class PredictRequest(BaseModel):
    ticker: str = "SBER"
    days: int = 7


class PredictResponse(BaseModel):
    ticker: str
    predictions: list[dict]


@app.on_event("startup")
def startup():
    logger.add("app.log", rotation="10 MB")
    logger.info("Starting Stock Prediction API")
    load_model()


@app.post("/_reload", include_in_schema=False)
def reload_model():
    logger.info("Reload triggered by retrainer")
    load_model()
    logger.success(f"Model reloaded: {_model_info}")
    return {"status": "reloaded", "model": _model_info, "last_training_date": _last_training_date}


@app.get("/")
def root():
    return {
        "message": "Stock Prediction API",
        "endpoints": {
            "GET /health": "Health check",
            "GET /docs": "Swagger documentation",
            "POST /predict": "Make a prediction",
        },
        "model": _model_info,
        "last_training_date": _last_training_date,
    }


@app.get("/health")
def health():
    return {"status": "OK", "model": _model_info, "last_training_date": _last_training_date}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info(f"Predict request: ticker={req.ticker}, days={req.days}")
    try:
        input_df = pd.DataFrame({"days": [req.days]})
        pred = predict_fn(input_df)

        if isinstance(pred, pd.DataFrame):
            records = pred.to_dict(orient="records")
        else:
            records = pred

        logger.success(f"Prediction successful: {len(records)} days")
        return PredictResponse(ticker=req.ticker, predictions=records)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

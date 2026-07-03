import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.config import get_settings
from app.model_service import ModelService
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Stock Prediction API", docs_url="/docs")
service = ModelService()


@app.on_event("startup")
async def startup():
    logger.add("app.log", rotation="10 MB")
    logger.info("Starting Stock Prediction API")
    await service.load_model()


@app.post("/_reload", include_in_schema=False)
async def reload_model():
    logger.info("Reload triggered by retrainer")
    await service.load_model()
    logger.success(f"Model reloaded: {service.model_info}")
    return {
        "status": "reloaded",
        "model": service.model_info,
        "last_training_date": service.last_training_date,
    }


@app.get("/")
async def root():
    return {
        "message": "Stock Prediction API",
        "endpoints": {
            "GET /health": "Health check",
            "GET /docs": "Swagger documentation",
            "POST /predict": "Make a prediction",
        },
        "model": service.model_info,
        "last_training_date": service.last_training_date,
    }


@app.get("/health")
async def health():
    return {"status": "OK", "model": service.model_info, "last_training_date": service.last_training_date}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    logger.info(f"Predict request: ticker={req.ticker}, days={req.days}")
    try:
        pred = await service.predict(req.days)
        if isinstance(pred, pd.DataFrame):
            records = pred.to_dict(orient="records")
        else:
            records = pred
        logger.success(f"Prediction successful: {len(records)} days")
        return PredictResponse(ticker=req.ticker, predictions=records)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

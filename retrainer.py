import asyncio
import json
import os
import pickle
from datetime import datetime, timedelta

import httpx
import mlflow
import mlflow.pyfunc
import pandas as pd
from loguru import logger
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from model import StockForecastModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sber_stock_forecast")
TICKER = os.getenv("TICKER", "SBER")
TZ = os.getenv("TZ", "Europe/Moscow")
TRAINING_YEARS = int(os.getenv("TRAINING_YEARS", "3"))

ARTIFACTS_DIR = "/app/mlflow_artifacts"


async def fetch_data(ticker, years=3):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
    all_data = []
    current_start = start_date

    async with httpx.AsyncClient(timeout=30) as client:
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=180), end_date)
            params = {
                "from": current_start.strftime("%Y-%m-%d"),
                "till": current_end.strftime("%Y-%m-%d"),
                "interval": 24,
                "iss.meta": "off",
            }
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                rows = data["candles"]["data"]
                if rows:
                    df_chunk = pd.DataFrame(rows, columns=data["candles"]["columns"])
                    all_data.append(df_chunk)
            except Exception as e:
                logger.warning(f"Fetch error {current_start.date()}: {e}")
            current_start = current_end + timedelta(days=1)

    if not all_data:
        raise RuntimeError("No data fetched from MOEX")
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    df_prophet = pd.DataFrame()
    df_prophet["ds"] = pd.to_datetime(df["begin"]).dt.tz_localize(None)
    df_prophet["y"] = df["close"]
    df_prophet = df_prophet.dropna().reset_index(drop=True)
    return df_prophet


def train_model(df):
    best_params = {
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 0.01,
        "holidays_prior_scale": 0.1,
    }
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        **best_params,
    )
    model.fit(df)
    last_date = df["ds"].max()
    metrics = {}
    split = int(len(df) * 0.8)
    if split > 30:
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        test_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **best_params,
        )
        test_model.fit(train_df)
        future = test_model.make_future_dataframe(periods=len(test_df), freq="D")
        forecast = test_model.predict(future)
        forecast_test = forecast[forecast["ds"] >= test_df["ds"].min()]
        y_true = test_df["y"].values[: len(forecast_test)]
        y_pred = forecast_test["yhat"].values[: len(y_true)]
        metrics = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        }
    return model, last_date, best_params, metrics


def save_artifacts(model, last_date, best_params, metrics, df):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(f"{ARTIFACTS_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{ARTIFACTS_DIR}/last_date.pkl", "wb") as f:
        pickle.dump(last_date, f)
    metadata = {
        "model_type": "Prophet",
        "ticker": TICKER,
        "training_data_range": {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end": df["ds"].max().strftime("%Y-%m-%d"),
        },
        "training_samples": len(df),
        "best_params": best_params,
        "metrics": metrics,
        "last_training_date": last_date.strftime("%Y-%m-%d"),
        "trained_at": datetime.now().isoformat(),
    }
    with open(f"{ARTIFACTS_DIR}/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def log_to_mlflow(metadata):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    with mlflow.start_run(run_name=f"prophet_{TICKER}_{datetime.now():%Y%m%d_%H%M}") as run:
        mlflow.log_params(metadata.get("best_params", {}))
        mlflow.log_metrics(metadata.get("metrics", {}))
        mlflow.log_param("training_samples", metadata.get("training_samples"))
        mlflow.log_param("ticker", TICKER)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=StockForecastModel(),
            artifacts={
                "model": f"{ARTIFACTS_DIR}/model.pkl",
                "last_date": f"{ARTIFACTS_DIR}/last_date.pkl",
            },
        )
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        logger.success(f"Model v{result.version} logged to MLFlow (run: {run_id})")
        return run_id, result.version


async def retrain():
    logger.info(f"=== Retrain started at {datetime.now()} ===")
    try:
        df = await fetch_data(TICKER, TRAINING_YEARS)
        logger.info(f"Fetched {len(df)} rows from MOEX ({df['ds'].min().date()} - {df['ds'].max().date()})")
        model, last_date, best_params, metrics = train_model(df)
        metadata = save_artifacts(model, last_date, best_params, metrics, df)
        logger.info(f"Model saved. Metrics: {metrics}")
        run_id, version = log_to_mlflow(metadata)
        logger.success(f"Retrain complete: v{version}, run={run_id}")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post("http://fastapi:8000/_reload")
            logger.info("FastAPI reload triggered")
        except Exception as e:
            logger.warning(f"FastAPI reload failed: {e}")
    except Exception as e:
        logger.error(f"Retrain failed: {e}")


async def scheduler():
    last_run_day = -1
    while True:
        now = datetime.now()
        if now.weekday() < 5 and now.hour == 19 and now.minute == 0 and now.day != last_run_day:
            await retrain()
            last_run_day = now.day
        await asyncio.sleep(30)


async def main():
    logger.add("retrainer.log", rotation="10 MB")
    logger.info("Retrainer started")

    await retrain()

    logger.info("Scheduled retrain: weekdays at 19:00 MSK")
    await scheduler()


if __name__ == "__main__":
    asyncio.run(main())

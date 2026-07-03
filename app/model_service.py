import asyncio
import os
import pickle

import mlflow
import mlflow.pyfunc
import pandas as pd
from loguru import logger

from app.config import get_settings
from model import StockForecastModel


class ModelService:
    def __init__(self):
        self._predict_fn = None
        self.model_info = {"source": "none", "run_id": ""}
        self.last_training_date = ""

    async def load_model(self):
        settings = get_settings()
        loop = asyncio.get_running_loop()

        if settings.mlflow_tracking_uri:
            loaded = await loop.run_in_executor(None, self._load_from_mlflow, settings)
            if not loaded:
                await loop.run_in_executor(None, self._load_local, settings)
        else:
            await loop.run_in_executor(None, self._load_local, settings)

        await loop.run_in_executor(None, self._load_last_date, settings)

    def _load_from_mlflow(self, settings) -> bool:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key

        client = mlflow.tracking.MlflowClient()
        try:
            latest_version = client.get_latest_versions(
                settings.mlflow_model_name, stages=["None"]
            )
            if not latest_version:
                latest_version = client.get_latest_versions(
                    settings.mlflow_model_name, stages=["Production"]
                )
            if latest_version:
                version = latest_version[-1].version
                model_uri = f"models:/{settings.mlflow_model_name}/{version}"
                pyfunc_model = mlflow.pyfunc.load_model(model_uri)
                self._predict_fn = pyfunc_model.predict
                self.model_info = {"source": "mlflow", "run_id": model_uri}
                logger.success(f"Loaded model from MLFlow: {model_uri}")
                return True
        except Exception as e:
            logger.warning(f"Could not load from MLFlow: {e}")
        return False

    def _load_local(self, settings):
        wrapper = StockForecastModel()
        artifacts = {"model": settings.model_path, "last_date": settings.last_date_path}

        class LocalContext:
            def __init__(self, artifacts):
                self.artifacts = artifacts

        ctx = LocalContext(artifacts)
        wrapper.load_context(ctx)
        self._predict_fn = lambda data: wrapper.predict(ctx, data)
        self.model_info = {"source": "local", "run_id": ""}
        logger.info(f"Loaded model from local: {settings.model_path}")

    def _load_last_date(self, settings):
        try:
            with open(settings.last_date_path, "rb") as f:
                self.last_training_date = str(pickle.load(f).date())
        except Exception:
            self.last_training_date = "unknown"

    async def predict(self, days: int) -> pd.DataFrame:
        input_df = pd.DataFrame({"days": [days]})
        loop = asyncio.get_running_loop()
        pred = await loop.run_in_executor(None, self._predict_fn, input_df)
        return pred

import os
from functools import lru_cache


class Settings:
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "mlflow_artifacts/model.pkl")
        self.last_date_path = os.getenv("LAST_DATE_PATH", "mlflow_artifacts/last_date.pkl")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        self.mlflow_model_name = os.getenv("MLFLOW_MODEL_NAME", "sber_stock_forecast")
        self.mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")


@lru_cache()
def get_settings() -> Settings:
    return Settings()

import pickle
import pandas as pd
from datetime import timedelta
from prophet import Prophet
import mlflow.pyfunc


class StockForecastModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["last_date"], "rb") as f:
            self.last_date = pickle.load(f)

    def predict(self, context, model_input):
        if "days" in model_input.columns:
            days = int(model_input["days"].iloc[0])
            start_date = self.last_date + timedelta(days=1)
        elif "start_date" in model_input.columns:
            start_date = pd.to_datetime(model_input["start_date"].iloc[0])
            if "days" in model_input.columns:
                days = int(model_input["days"].iloc[0])
            else:
                days = 30
        else:
            days = 30
            start_date = self.last_date + timedelta(days=1)

        future = self.model.make_future_dataframe(periods=days, freq="D")
        forecast = self.model.predict(future)
        forecast_future = forecast[forecast["ds"] >= start_date].copy()

        result = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.rename(columns={
            "yhat": "prediction",
            "yhat_lower": "prediction_lower",
            "yhat_upper": "prediction_upper",
        })
        result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")
        return result

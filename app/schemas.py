from pydantic import BaseModel


class PredictRequest(BaseModel):
    ticker: str = "SBER"
    days: int = 7


class PredictResponse(BaseModel):
    ticker: str
    predictions: list[dict]

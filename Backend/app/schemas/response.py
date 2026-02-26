from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float

class ResultRecord(BaseModel):
    id: Optional[str]
    filename: str
    prediction: str
    confidence: float
    timestamp: datetime
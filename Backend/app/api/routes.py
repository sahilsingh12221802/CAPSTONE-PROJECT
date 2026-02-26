from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.model_service import predict_animal_type
from app.db.mongo import results_collection
from app.core.config import settings
from app.schemas.response import PredictionResponse, ResultRecord
from datetime import datetime
from typing import List

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")

    prediction, confidence = predict_animal_type(contents)

    record = {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    }

    results_collection.insert_one(record)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence
    }


@router.get("/results", response_model=List[ResultRecord])
def get_results(
    limit: int = Query(10, ge=1, le=100),
    prediction: str | None = None
):
    query = {}
    if prediction:
        query["prediction"] = prediction

    results = list(results_collection.find(query).sort("timestamp", -1).limit(limit))

    formatted = []
    for r in results:
        formatted.append({
            "id": str(r["_id"]),
            "filename": r["filename"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "timestamp": r["timestamp"]
        })

    return formatted
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model_service import predict_animal_type
from app.db.mongo import results_collection
from datetime import datetime

router = APIRouter()

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_MB = 5


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
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
@router.get("/results")
def get_results(limit: int = 10):
    results = list(results_collection.find().sort("timestamp", -1).limit(limit))

    for r in results:
        r["_id"] = str(r["_id"])

    return results

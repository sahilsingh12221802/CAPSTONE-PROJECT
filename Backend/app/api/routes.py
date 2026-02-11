from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model_service import predict_animal_type

router = APIRouter()

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_MB = 5


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG and PNG images are allowed."
        )

    # Read file bytes
    contents = await file.read()

    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 5MB limit."
        )

    # Placeholder AI prediction
    prediction, confidence = predict_animal_type(contents)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence
    }
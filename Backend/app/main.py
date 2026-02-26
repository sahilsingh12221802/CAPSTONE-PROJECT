from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for Image-based Cattle and Buffalo Classification",
    version=settings.APP_VERSION
)

@app.get("/")
def root():
    return {"message": "Animal Classification Backend is running"}

@app.get("/api/health")
def health_check():
    return {"status": "OK", "message": "Backend server is healthy"}

app.include_router(api_router, prefix="/api")
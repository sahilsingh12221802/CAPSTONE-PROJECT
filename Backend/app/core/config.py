import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "Animal Type Classification Backend"
    APP_VERSION = "1.0.0"

    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "animal_classifier")

    MAX_FILE_SIZE_MB = 5
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

settings = Settings()
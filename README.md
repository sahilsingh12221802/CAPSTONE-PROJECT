# Cattle and Buffalo Classification Capstone

End-to-end project for image-based classification of cattle and buffalo, with explicit rejection for non-target images.

## Quick Navigation

- [Backend Guide](backend/README.md)
- [Frontend Guide](frontend/README.md)
- [ML Guide](ml/README.md)
- [Model Artifacts Guide](models/README.md)

## Overview

This project combines four layers:

- A React web dashboard for users.
- A FastAPI service for inference and result APIs.
- A TensorFlow model pipeline for train/evaluate/predict.
- A MongoDB store for prediction records.

### Core Features

- 3-class model training: `Cattle`, `Buffalo`, `Other`.
- Runtime rejection support for non-bovine images (returns `Unknown`).
- Prediction metadata and history tracking in MongoDB.
- Dockerized full stack with one command.

## Architecture

```text
Browser (http://localhost)
  |
  v
Frontend (React + Nginx, port 80)
  |
  v
Backend API (FastAPI, port 8000)
  |
  +--> ML inference (models/*.h5 + ml/predict.py)
  |
  +--> MongoDB (port 27017)
```

## Project Layout

```text
.
├── backend/                 # FastAPI API service
├── frontend/                # React app and Nginx container config
├── ml/                      # Training, prediction, evaluation scripts
├── models/                  # Trained model and labels
├── Dockerfile               # Backend image
├── docker-compose.yml       # Full stack orchestration
└── requirements.txt         # Backend/ML Python dependencies
```

## Run with Docker (Recommended)

### Prerequisites

- Docker Desktop is running.
- These ports are available on your machine: `80`, `8000`, `27017`.

### Start entire stack

```bash
docker compose up --build
```

Open:

- Website: http://localhost
- API docs: http://localhost:8000/docs
- MongoDB: mongodb://localhost:27017

### Stop stack

```bash
docker compose down
```

Remove containers and database volume:

```bash
docker compose down -v
```

## API Smoke Test

Classify an image:

```bash
curl -X POST http://localhost:8000/classify \
  -F "image=@Cattle-Buffalo-breeds.folder/test/Gir/Gir_282_jpeg.rf.69627fadb90729dcd11f4b7d849f5398.jpg"
```

Get recent records:

```bash
curl "http://localhost:8000/records?limit=5"
```

## Local Development (Without Docker)

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Website (dev mode): http://localhost:3000

## ML Commands

Train model:

```bash
python -m ml.train
```

Evaluate model:

```bash
python -m ml.evaluate
```

Single-image prediction:

```bash
python -m ml.predict "path/to/image.jpg"
```

## Important Notes

- Backend expects model files inside `models/`.
- Classification logs are saved in database `animal_classification`, collection `results`.
- Docker frontend is a production build served by Nginx.

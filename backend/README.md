# Backend Service (FastAPI)

Backend API for model inference, validation, and prediction history storage.

## Purpose

The backend does three main jobs:

- Accept image uploads and classify them.
- Return structured prediction output with confidence and reasoning fields.
- Save prediction records to MongoDB for dashboard/history views.

## Entry Point

- `backend/app.py`

## API Endpoints

### `POST /classify`

Input:

- Multipart form-data field: `image`

Behavior:

- Validates file type.
- Sends image through model pipeline (`ml/predict.py`).
- Persists result in MongoDB.

Response shape:

- `success`
- `data.id`
- `data.filename`
- `data.label`
- `data.confidence`
- `data.class_probabilities`
- `data.is_target_animal`
- `data.message`
- `data.gate_reason`
- `data.timestamp`

### `GET /records?limit=50`

Returns latest saved predictions with key metadata.

## Data Store

- MongoDB database: `animal_classification`
- Collection: `results`

## Configuration

Environment variables:

- `MONGO_URI` (default: `mongodb://127.0.0.1:27017`)

## Run Locally

From project root:

```bash
source .venv/bin/activate
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open docs at:

- http://localhost:8000/docs

## Docker Run

Managed by Compose service `backend`.

- Image built from root [Dockerfile](../Dockerfile)
- Port mapped: `8000:8000`

## Troubleshooting

- If startup fails with Mongo error, check `MONGO_URI` and database container status.
- If classify fails with model error, ensure files exist in `models/` and are readable.
- If uploads fail, check that request uses `multipart/form-data` and field name `image`.

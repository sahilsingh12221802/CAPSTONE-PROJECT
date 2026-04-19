import os
import shutil
import uuid
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient

app = FastAPI(title='Cattle/Buffalo Classifier API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://127.0.0.1:27017')
MONGO_TIMEOUT_MS = int(os.getenv('MONGO_TIMEOUT_MS', '1000'))
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
db = client['animal_classification']
collection = db['results']
mongo_available = True

MODEL_PATH = os.path.abspath('models/cattle_buffalo_mobile.h5')


@app.on_event('startup')
def startup_event():
    global mongo_available
    try:
        client.server_info()
    except Exception as e:
        mongo_available = False
        print(f'Warning: Could not connect to MongoDB: {e}')

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Trained model not found at: {MODEL_PATH}')


@app.post('/classify')
async def classify_animal(image: UploadFile = File(...)):
    try:
        from ml.predict import classify

        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail='Only image files are allowed')

        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        extension = os.path.splitext(image.filename)[1] or '.jpg'
        unique_name = f'{uuid.uuid4().hex}{extension}'
        temp_path = os.path.join(temp_dir, unique_name)

        with open(temp_path, 'wb') as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = classify(temp_path)
        record = {
            'filename': image.filename,
            'predicted_label': result['label'],
            'predicted_species': result.get('species'),
            'predicted_breed': result.get('breed'),
            'confidence': result['confidence'],
            'raw_scores': result.get('raw_scores', []),
            'class_probabilities': result.get('class_probabilities', {}),
            'top_predictions': result.get('top_predictions', []),
            'is_target_animal': result.get('is_target_animal', False),
            'message': result.get('message', ''),
            'gate_reason': result.get('gate_reason', ''),
            'timestamp': datetime.utcnow().isoformat()
        }

        inserted_id = None
        if mongo_available:
            try:
                insert_result = collection.insert_one(record)
                inserted_id = str(insert_result.inserted_id)
            except Exception as e:
                # Continue serving classification even if DB write fails at runtime.
                print(f'Warning: could not persist classification record: {e}')

        # pymongo mutates inserted dictionaries by adding _id (ObjectId), so return a clean JSON-safe payload.
        record_response = {
            'id': inserted_id,
            'filename': record['filename'],
            'label': record['predicted_label'],
            'species': record['predicted_species'],
            'breed': record['predicted_breed'],
            'confidence': record['confidence'],
            'raw_scores': record['raw_scores'],
            'class_probabilities': record['class_probabilities'],
            'top_predictions': record['top_predictions'],
            'is_target_animal': record['is_target_animal'],
            'message': record['message'],
            'gate_reason': record['gate_reason'],
            'timestamp': record['timestamp']
        }

        return JSONResponse(status_code=200, content={'success': True, 'data': record_response})

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/records')
def get_records(limit: int = 50):
    if not mongo_available:
        return {'success': True, 'count': 0, 'data': []}

    docs = []
    try:
        for doc in collection.find().sort([('timestamp', -1)]).limit(limit):
            docs.append({
                'id': str(doc.get('_id')),
                'filename': doc.get('filename', ''),
                'label': doc.get('predicted_label', ''),
                'species': doc.get('predicted_species'),
                'breed': doc.get('predicted_breed'),
                'confidence': doc.get('confidence', 0.0),
                'class_probabilities': doc.get('class_probabilities', {}),
                'top_predictions': doc.get('top_predictions', []),
                'is_target_animal': doc.get('is_target_animal', False),
                'message': doc.get('message', ''),
                'timestamp': doc.get('timestamp', '')
            })
    except Exception as e:
        print(f'Warning: failed to fetch records: {e}')
        return {'success': True, 'count': 0, 'data': []}

    return {'success': True, 'count': len(docs), 'data': docs}

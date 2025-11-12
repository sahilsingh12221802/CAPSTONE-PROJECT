# app.py
from flask import Flask, request, jsonify
from datetime import datetime
import random

app = Flask(__name__)

# Simulated in-memory database
database = []

@app.route('/')
def home():
    return jsonify({"message": "AI-based Animal Type Classification API is running!"})

@app.route('/classify', methods=['POST'])
def classify_animal():
    # Simulate receiving an image file
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    filename = image.filename

    # Simulate model prediction (for demonstration)
    animal_type = random.choice(["Cattle", "Buffalo"])
    confidence = round(random.uniform(85.0, 99.9), 2)

    # Simulated record for MongoDB
    result_record = {
        "filename": filename,
        "classification": animal_type,
        "confidence": f"{confidence}%",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Append to in-memory database
    database.append(result_record)

    # Return structured JSON response
    return jsonify({
        "message": "Classification successful!",
        "result": result_record
    })

@app.route('/records', methods=['GET'])
def get_records():
    # Return all saved classification results
    return jsonify({
        "total_records": len(database),
        "data": database
    })

if __name__ == '__main__':
    app.run(debug=True)

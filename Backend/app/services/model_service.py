import random

def predict_animal_type(image_bytes: bytes):
    """
    Placeholder prediction service.
    This will be replaced with actual ML model inference later.
    """
    possible_classes = ["cattle", "buffalo"]
    prediction = random.choice(possible_classes)
    confidence = round(random.uniform(0.70, 0.95), 2)

    return prediction, confidence
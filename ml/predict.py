import json
import os

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2,
                                                        decode_predictions,
                                                        preprocess_input)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

MODEL_PATH = os.path.abspath('models/cattle_buffalo_mobile.h5')
LABELS_PATH = os.path.abspath('models/class_labels.json')
IMG_SIZE = (224, 224)
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70
CLASSIFIER_MARGIN_THRESHOLD = 0.12
OTHER_CLASS_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_CLASS_LABELS = ['Cattle', 'Buffalo']
TARGET_IMAGENET_LABELS = {'ox', 'water_buffalo', 'bison'}
GATE_MATCH_MIN_SCORE = 0.03
GATE_OVERRIDE_CONFIDENCE = 0.98
GATE_OVERRIDE_MARGIN = 0.75
GATE_OVERRIDE_OTHER_MAX = 0.08

_CLASSIFIER_MODEL = None
_GATE_MODEL = None
_CLASS_LABELS_CACHE = None


def load_and_preprocess(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = arr.astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_and_preprocess_for_gate(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def get_classifier_model():
    global _CLASSIFIER_MODEL
    if _CLASSIFIER_MODEL is None:
        _CLASSIFIER_MODEL = load_model(MODEL_PATH)
    return _CLASSIFIER_MODEL


def get_class_labels(model_output_size: int):
    global _CLASS_LABELS_CACHE

    if _CLASS_LABELS_CACHE is None:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                _CLASS_LABELS_CACHE = json.load(f)
        else:
            _CLASS_LABELS_CACHE = DEFAULT_CLASS_LABELS

    labels = list(_CLASS_LABELS_CACHE)
    if len(labels) != model_output_size:
        labels = DEFAULT_CLASS_LABELS[:]
        if model_output_size == 3:
            labels.append('Other')

    return labels


def get_gate_model():
    global _GATE_MODEL
    if _GATE_MODEL is None:
        _GATE_MODEL = MobileNetV2(weights='imagenet', include_top=True)
    return _GATE_MODEL


def is_bovine_image(image_path):
    gate_model = get_gate_model()
    x_gate = load_and_preprocess_for_gate(image_path)
    gate_probs = gate_model.predict(x_gate, verbose=0)
    top_preds = decode_predictions(gate_probs, top=5)[0]

    for _, label, score in top_preds:
        if label in TARGET_IMAGENET_LABELS and score >= GATE_MATCH_MIN_SCORE:
            return True, top_preds, f'Gate matched: {label} ({score:.2f})'

    top_label = top_preds[0][1]
    top_score = float(top_preds[0][2])
    return False, top_preds, f'Gate top class was {top_label} ({top_score:.2f}), not a bovine class.'


def classify(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    is_bovine, gate_top_preds, gate_reason = is_bovine_image(image_path)
    gate_predictions = [
        {'label': label, 'score': float(score)}
        for _, label, score in gate_top_preds
    ]

    model = get_classifier_model()
    x = load_and_preprocess(image_path)
    preds = model.predict(x, verbose=0)[0]
    class_labels = get_class_labels(len(preds))
    probs_by_class = {
        class_labels[i]: float(preds[i])
        for i in range(len(preds))
    }

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    sorted_probs = np.sort(preds)
    second_best = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
    margin = float(confidence - second_best)
    predicted_label = class_labels[idx]

    if not is_bovine:
        # External gate can miss some real breeds, so allow only very strong model evidence to override it.
        if (
            predicted_label in {'Cattle', 'Buffalo'}
            and confidence >= GATE_OVERRIDE_CONFIDENCE
            and margin >= GATE_OVERRIDE_MARGIN
            and probs_by_class.get('Other', 1.0) <= GATE_OVERRIDE_OTHER_MAX
        ):
            return {
                'label': predicted_label,
                'confidence': confidence,
                'raw_scores': preds.tolist(),
                'class_probabilities': probs_by_class,
                'is_target_animal': True,
                'message': 'External gate was uncertain, but classifier confidence is very high. Returning cattle/buffalo prediction.',
                'gate_reason': gate_reason,
                'gate_predictions': gate_predictions,
            }

        return {
            'label': 'Unknown',
            'confidence': confidence,
            'raw_scores': preds.tolist(),
            'class_probabilities': probs_by_class,
            'is_target_animal': False,
            'message': 'This image does not appear to be cattle or buffalo.',
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    if 'Other' in probs_by_class and probs_by_class['Other'] >= OTHER_CLASS_CONFIDENCE_THRESHOLD:
        return {
            'label': 'Unknown',
            'confidence': probs_by_class['Other'],
            'raw_scores': preds.tolist(),
            'class_probabilities': probs_by_class,
            'is_target_animal': False,
            'message': 'This image was classified as non-target (Other), not cattle or buffalo.',
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    if confidence < CLASSIFIER_CONFIDENCE_THRESHOLD or margin < CLASSIFIER_MARGIN_THRESHOLD:
        return {
            'label': 'Unknown',
            'confidence': confidence,
            'raw_scores': preds.tolist(),
            'class_probabilities': probs_by_class,
            'is_target_animal': True,
            'message': 'Image may be animal-like, but model confidence is too low for reliable cattle/buffalo classification.',
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    if predicted_label not in {'Cattle', 'Buffalo'}:
        return {
            'label': 'Unknown',
            'confidence': confidence,
            'raw_scores': preds.tolist(),
            'class_probabilities': probs_by_class,
            'is_target_animal': False,
            'message': 'Prediction is outside cattle/buffalo target classes.',
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    return {
        'label': predicted_label,
        'confidence': confidence,
        'raw_scores': preds.tolist(),
        'class_probabilities': probs_by_class,
        'is_target_animal': True,
        'message': 'Classification completed successfully.',
        'gate_reason': gate_reason,
        'gate_predictions': gate_predictions,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Classify cattle vs buffalo image')
    parser.add_argument('image', type=str, help='Path to image')
    args = parser.parse_args()
    result = classify(args.image)
    print(result)

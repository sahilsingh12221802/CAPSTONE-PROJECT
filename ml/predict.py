import json
import os

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2,
                                                        decode_predictions,
                                                        preprocess_input)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

MODEL_PATH = os.path.abspath('models/cattle_buffalo_mobile.h5')
LABELS_PATH = os.path.abspath('models/class_labels.json')
IMG_SIZE = (224, 224)
TTA_ENABLED = os.getenv('TTA_ENABLED', 'true').lower() == 'true'
TTA_BRIGHTNESS_DELTA = float(os.getenv('TTA_BRIGHTNESS_DELTA', '0.08'))
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70
CLASSIFIER_MARGIN_THRESHOLD = 0.12
OTHER_CLASS_CONFIDENCE_THRESHOLD = 0.45
BREED_TO_SPECIES = {
    'Gir': 'Cattle',
    'Holstein_Friesian': 'Cattle',
    'Jersey': 'Cattle',
    'Sahiwal': 'Cattle',
    'Jaffrabadi': 'Buffalo',
    'Murrah': 'Buffalo',
}
BREED_CLASSES = list(BREED_TO_SPECIES.keys())
DEFAULT_CLASS_LABELS = BREED_CLASSES + ['Other']
TARGET_IMAGENET_LABELS = {'ox', 'water_buffalo', 'bison'}
GATE_MATCH_MIN_SCORE = 0.03
GATE_OVERRIDE_CONFIDENCE = 0.98
GATE_OVERRIDE_MARGIN = 0.75
GATE_OVERRIDE_OTHER_MAX = 0.08

_CLASSIFIER_MODEL = None
_GATE_MODEL = None
_CLASS_LABELS_CACHE = None
_FACE_CASCADE = None
_HOG_PEOPLE = None


def load_and_preprocess(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = arr.astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_tta_batch(image_path):
    """Build a small deterministic TTA batch for more stable breed predictions."""
    img = load_img(image_path, target_size=IMG_SIZE)
    base = img_to_array(img).astype('float32') / 255.0

    variants = [
        base,
        np.fliplr(base),
        np.clip(base + TTA_BRIGHTNESS_DELTA, 0.0, 1.0),
        np.clip(base - TTA_BRIGHTNESS_DELTA, 0.0, 1.0),
    ]
    return np.stack(variants, axis=0)


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
        if model_output_size == 3:
            labels = ['Cattle', 'Buffalo', 'Other']
        elif model_output_size == len(DEFAULT_CLASS_LABELS):
            labels = DEFAULT_CLASS_LABELS[:]
        else:
            labels = [f'class_{i}' for i in range(model_output_size)]

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


def make_top_predictions(primary_label, primary_score, class_labels, preds, fallback_scores=None):
    max_model_score = float(np.max(preds)) if len(preds) else 0.0
    adjusted_primary = float(max(primary_score, max_model_score + 1e-6))
    result = [{'label': primary_label, 'score': adjusted_primary}]
    used = {primary_label}

    if fallback_scores:
        for label, score in sorted(fallback_scores.items(), key=lambda x: x[1], reverse=True):
            if label in used:
                continue
            result.append({'label': label, 'score': float(score)})
            used.add(label)
            if len(result) >= 3:
                return result

    for i in np.argsort(preds)[::-1]:
        label = class_labels[i]
        if label in used:
            continue
        result.append({'label': label, 'score': float(preds[i])})
        used.add(label)
        if len(result) >= 3:
            break

    return result


def merge_non_target_probs(class_labels, probs_by_class):
    merged = {}
    other_score = 0.0

    for label in class_labels:
        score = float(probs_by_class.get(label, 0.0))
        if label in {'Other', 'Human'}:
            other_score += score
        else:
            merged[label] = score

    merged['Other'] = merged.get('Other', 0.0) + other_score
    return merged


def get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def get_hog_people_detector():
    global _HOG_PEOPLE
    if _HOG_PEOPLE is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _HOG_PEOPLE = hog
    return _HOG_PEOPLE


def detect_human_cue_score(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 0.0

    score = 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        img_area = float(max(image.shape[0] * image.shape[1], 1))
        mean_face_area_ratio = float(np.mean([(w * h) / img_area for (_, _, w, h) in faces]))
        face_score = 0.45 + 0.10 * min(len(faces), 3) + 2.20 * mean_face_area_ratio
        score = max(score, min(0.98, face_score))

    # Body cue fallback for images without a clearly visible face.
    hog = get_hog_people_detector()
    h, w = image.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    resized = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
    boxes, _ = hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(boxes) > 0:
        resized_area = float(max(resized.shape[0] * resized.shape[1], 1))
        mean_box_area_ratio = float(np.mean([(w * h) / resized_area for (_, _, w, h) in boxes]))
        body_score = 0.38 + 0.09 * min(len(boxes), 3) + 1.50 * mean_box_area_ratio
        score = max(score, min(0.90, body_score))

    return float(min(score, 0.95))


def make_non_target_display_predictions(other_confidence, merged_labels, merged_preds):
    other_confidence = float(min(max(other_confidence, 0.0), 1.0))
    remainder = max(0.0, 1.0 - other_confidence)

    other_idx = merged_labels.index('Other') if 'Other' in merged_labels else -1
    alt_indices = [i for i in np.argsort(merged_preds)[::-1] if i != other_idx]
    alt_total = float(sum(float(merged_preds[i]) for i in alt_indices))

    display = [{'label': 'Other', 'score': other_confidence}]
    for i in alt_indices[:2]:
        raw = float(merged_preds[i])
        scaled = (raw / alt_total) * remainder if alt_total > 0 else 0.0
        display.append({'label': merged_labels[i], 'score': float(scaled)})

    return display


def classify(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    is_bovine, gate_top_preds, gate_reason = is_bovine_image(image_path)
    gate_top_score = float(gate_top_preds[0][2])
    gate_predictions = [
        {'label': label, 'score': float(score)}
        for _, label, score in gate_top_preds
    ]

    model = get_classifier_model()
    if TTA_ENABLED:
        x_batch = load_tta_batch(image_path)
        preds = np.mean(model.predict(x_batch, verbose=0), axis=0)
    else:
        x = load_and_preprocess(image_path)
        preds = model.predict(x, verbose=0)[0]
    class_labels = get_class_labels(len(preds))
    supports_breed = any(label in BREED_CLASSES for label in class_labels)
    probs_by_class = {
        class_labels[i]: float(preds[i])
        for i in range(len(preds))
    }

    merged_probs = merge_non_target_probs(class_labels, probs_by_class)
    merged_labels = list(merged_probs.keys())
    merged_preds = np.array([merged_probs[label] for label in merged_labels], dtype=float)

    idx = int(np.argmax(merged_preds))
    confidence = float(merged_preds[idx])
    sorted_probs = np.sort(merged_preds)
    second_best = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
    margin = float(confidence - second_best)
    predicted_label = merged_labels[idx]

    top_indices = np.argsort(merged_preds)[::-1][:3]
    top_predictions = [
        {
            'label': merged_labels[i],
            'score': float(merged_preds[i]),
        }
        for i in top_indices
    ]

    predicted_species = BREED_TO_SPECIES.get(predicted_label)
    predicted_breed = predicted_label if predicted_species else None
    if predicted_label in {'Cattle', 'Buffalo'}:
        predicted_species = predicted_label
        predicted_breed = None

    bovine_labels = {label for label in merged_labels if label in BREED_TO_SPECIES or label in {'Cattle', 'Buffalo'}}
    bovine_score = float(sum(merged_probs.get(label, 0.0) for label in bovine_labels))
    max_bovine_class_prob = float(max((merged_probs.get(label, 0.0) for label in bovine_labels), default=0.0))
    non_target_score = float(max(0.0, 1.0 - bovine_score))
    human_cue_score = detect_human_cue_score(image_path)

    # Calibrate non-target confidence to avoid low/confusing percentages on clearly non-bovine images.
    calibrated_other_confidence = float(
        max(
            merged_probs.get('Other', 0.0),
            non_target_score,
            1.0 - max_bovine_class_prob,
            gate_top_score if not is_bovine else 0.0,
        )
    )
    if not is_bovine:
        # Adaptive boost avoids a fixed confidence for all non-bovine images.
        non_target_signal = max(
            merged_probs.get('Other', 0.0),
            non_target_score,
            1.0 - max_bovine_class_prob,
        )
        adaptive_boost = min(0.95, 0.45 + 0.50 * non_target_signal)
        if human_cue_score > 0.0:
            # Smooth blend so confidence varies continuously across different human images.
            human_boost = 0.74 + 0.16 * human_cue_score + 0.10 * non_target_signal
            adaptive_boost = max(adaptive_boost, min(0.96, human_boost))
        calibrated_other_confidence = max(calibrated_other_confidence, adaptive_boost)

    if not is_bovine:
        # External gate can miss some real breeds, so allow only very strong model evidence to override it.
        if (
            predicted_species in {'Cattle', 'Buffalo'}
            and confidence >= GATE_OVERRIDE_CONFIDENCE
            and margin >= GATE_OVERRIDE_MARGIN
            and merged_probs.get('Other', 1.0) <= GATE_OVERRIDE_OTHER_MAX
        ):
            return {
                'label': predicted_species,
                'species': predicted_species,
                'breed': predicted_breed,
                'confidence': confidence,
                'raw_scores': preds.tolist(),
                'class_probabilities': merged_probs,
                'top_predictions': top_predictions,
                'is_target_animal': True,
                'message': (
                    'External gate was uncertain, but classifier confidence is '
                    'very high. Returning bovine prediction.'
                ),
                'gate_reason': gate_reason,
                'gate_predictions': gate_predictions,
            }

        return {
            'label': 'Other',
            'species': None,
            'breed': None,
            'confidence': calibrated_other_confidence,
            'raw_scores': preds.tolist(),
            'class_probabilities': merged_probs,
            'top_predictions': make_non_target_display_predictions(
                calibrated_other_confidence,
                merged_labels,
                merged_preds,
            ),
            'is_target_animal': False,
            'message': 'Detected as non-target image (not cattle or buffalo).',
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    if predicted_species is None:
        return {
            'label': 'Other',
            'species': None,
            'breed': None,
            'confidence': calibrated_other_confidence,
            'raw_scores': preds.tolist(),
            'class_probabilities': merged_probs,
            'top_predictions': make_non_target_display_predictions(
                calibrated_other_confidence,
                merged_labels,
                merged_preds,
            ),
            'is_target_animal': False,
            'message': 'Prediction is outside cattle/buffalo target classes.',
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    if confidence < CLASSIFIER_CONFIDENCE_THRESHOLD or margin < CLASSIFIER_MARGIN_THRESHOLD:
        return {
            'label': 'Unknown',
            'species': None,
            'breed': None,
            'confidence': confidence,
            'raw_scores': preds.tolist(),
            'class_probabilities': merged_probs,
            'top_predictions': top_predictions,
            'is_target_animal': True,
            'message': (
                'Image may be animal-like, but model confidence is too low for '
                'reliable cattle/buffalo classification.'
            ),
            'gate_reason': gate_reason,
            'gate_predictions': gate_predictions,
        }

    return {
        'label': predicted_species,
        'species': predicted_species,
        'breed': predicted_breed,
        'confidence': confidence,
        'raw_scores': preds.tolist(),
        'class_probabilities': merged_probs,
        'top_predictions': top_predictions,
        'is_target_animal': True,
        'message': (
            'Classification completed successfully.'
            if supports_breed or predicted_breed
            else 'Legacy model detected: retrain to enable breed-level prediction.'
        ),
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

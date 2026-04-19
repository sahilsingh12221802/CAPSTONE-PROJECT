#!/usr/bin/env python3
"""
Enhanced Gir Diagnostic with Threshold Adjustment & Post-Processing
---------------------------------------------------------------------
Uses confidence thresholds and post-processing logic to improve Gir detection.
"""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.abspath('.')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile.h5')
DATA_DIR = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder')
TEST_DIR = os.path.join(DATA_DIR, 'test', 'Gir')

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class indices
GIR_IDX = 0
SAHIWAL_IDX = 3
CLASS_NAMES = ['Gir', 'Holstein_Friesian', 'Jersey', 'Sahiwal', 'Jaffrabadi', 'Murrah', 'Other']

# Enhanced confidence thresholds
GIR_CONFIDENCE_THRESHOLD = 0.35      # Lower threshold to catch borderline Gir
SAHIWAL_REJECTION_THRESHOLD = 0.50   # Reduce Sahiwal false positives
GIR_VS_SAHIWAL_BOOST = 2.0           # Boost Gir score vs Sahiwal

# Confidence floor to avoid "None" classification
MIN_CONFIDENCE = 0.30


def load_and_preprocess_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, 0)


def post_process_predictions(predictions, image_name=""):
    """
    Apply post-processing logic to improve Gir vs Sahiwal discrimination.
    
    Strategy:
    1. If Gir > threshold AND Gir > Sahiwal * boost, classify as Gir
    2. If Sahiwal is too high relative to others, suppress it
    3. If all confidences are low, use minimum threshold rule
    """
    gir_prob = predictions[0, GIR_IDX]
    sahiwal_prob = predictions[0, SAHIWAL_IDX]
    max_prob = np.max(predictions[0])
    max_idx = np.argmax(predictions[0])
    
    # Rule 1: Gir vs Sahiwal post-processing
    # If Gir is reasonably close to Sahiwal, boost it
    if (gir_prob > GIR_CONFIDENCE_THRESHOLD and 
        gir_prob > sahiwal_prob / GIR_VS_SAHIWAL_BOOST):
        return GIR_IDX, gir_prob, "Gir (post-processed)"
    
    # Rule 2: Suppress Sahiwal false positives
    # If Sahiwal is high but Gir exists, consider it ambiguous
    if (sahiwal_prob > SAHIWAL_REJECTION_THRESHOLD and 
        gir_prob > 0.25 and 
        abs(sahiwal_prob - gir_prob) < 0.15):
        # Very close call - too ambiguous, return None
        return -1, max(gir_prob, sahiwal_prob), "None (ambiguous)"
    
    # Rule 3: Use original prediction if above minimum confidence
    if max_prob > MIN_CONFIDENCE:
        return max_idx, max_prob, CLASS_NAMES[max_idx] if max_idx >= 0 else "None"
    else:
        return -1, max_prob, "None (low confidence)"


def get_species(class_idx):
    """Map class index to species"""
    if class_idx == -1:
        return "None"
    breed_to_species = {
        0: 'Cattle',      # Gir
        1: 'Cattle',      # Holstein
        2: 'Cattle',      # Jersey
        3: 'Cattle',      # Sahiwal
        4: 'Buffalo',     # Jaffrabadi
        5: 'Buffalo',     # Murrah
        6: 'Non-target'   # Other
    }
    return breed_to_species.get(class_idx, "Unknown")


def diagnose_with_post_processing():
    """Run diagnostic with post-processing"""
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Get all Gir test images
    gir_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])[:15]
    
    if not gir_images:
        print("❌ No Gir test images found")
        return
    
    print(f"Found {len(gir_images)} Gir test images (showing first {len(gir_images)})\n")
    print("Testing Gir predictions with post-processing:")
    print("-" * 150)
    
    correct_standard = 0
    correct_postproc = 0
    sahiwal_failures = []
    gir_confidence_scores = []
    
    for img_name in gir_images:
        img_path = os.path.join(TEST_DIR, img_name)
        
        # Load and predict
        img_data = load_and_preprocess_image(img_path)
        predictions = model.predict(img_data, verbose=0)
        
        # Get raw prediction
        raw_idx = np.argmax(predictions[0])
        raw_prob = predictions[0, raw_idx]
        
        # Get post-processed prediction
        post_idx, post_prob, post_note = post_process_predictions(predictions, img_name)
        
        # Get top 3 predictions for debugging
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_str = ", ".join([f"{CLASS_NAMES[i]}:{predictions[0,i]:.3f}" for i in top_3_idx])
        
        # Check correctness
        is_correct_standard = (raw_idx == GIR_IDX)
        is_correct_postproc = (post_idx == GIR_IDX)
        
        if is_correct_standard:
            correct_standard += 1
            mark = "✓"
        else:
            mark = "✗"
            if raw_idx == SAHIWAL_IDX:
                sahiwal_failures.append((img_name, predictions[0, GIR_IDX], predictions[0, SAHIWAL_IDX]))
        
        if is_correct_postproc:
            correct_postproc += 1
        
        gir_confidence_scores.append(predictions[0, GIR_IDX])
        
        # Print result
        species = get_species(post_idx)
        status = "→ IMPROVED" if (not is_correct_standard and is_correct_postproc) else ""
        
        print(f"{mark} {img_name:<50} -> {CLASS_NAMES[post_idx] if post_idx >= 0 else 'None':<20} "
              f"(species: {species:<15} conf: {post_prob:.4f}) {post_note:<25} {status}")
        if post_idx != raw_idx:
            print(f"   Top predictions: {top_3_str}")
    
    print("-" * 150)
    
    print(f"\nGir Classification Results:")
    print(f"  Standard Model:    {correct_standard}/15 = {correct_standard/15*100:.1f}%")
    print(f"  Post-Processing:   {correct_postproc}/15 = {correct_postproc/15*100:.1f}%")
    print(f"  Improvement:       {correct_postproc - correct_standard:+d} images")
    
    if sahiwal_failures:
        print(f"\nSahiwal Misclassifications ({len(sahiwal_failures)}):")
        for name, gir_conf, sah_conf in sahiwal_failures:
            print(f"  {name}: Gir={gir_conf:.3f}, Sahiwal={sah_conf:.3f}, diff={sah_conf-gir_conf:.3f}")
    
    print(f"\nGir Confidence Statistics:")
    print(f"  Mean: {np.mean(gir_confidence_scores):.4f}")
    print(f"  Std:  {np.std(gir_confidence_scores):.4f}")
    print(f"  Min:  {np.min(gir_confidence_scores):.4f}")
    print(f"  Max:  {np.max(gir_confidence_scores):.4f}")
    
    print(f"\n{'='*150}")
    print(f"Model Configuration:")
    print(f"{'='*150}")
    print(f"Class labels: {CLASS_NAMES}\n")
    print(f"Post-Processing Configuration:")
    print(f"  Gir Confidence Threshold: {GIR_CONFIDENCE_THRESHOLD}")
    print(f"  Sahiwal Rejection Threshold: {SAHIWAL_REJECTION_THRESHOLD}")
    print(f"  Gir vs Sahiwal Boost: {GIR_VS_SAHIWAL_BOOST}x")
    print(f"  Minimum Confidence Floor: {MIN_CONFIDENCE}")


if __name__ == '__main__':
    diagnose_with_post_processing()

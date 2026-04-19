#!/usr/bin/env python3
"""
Diagnostic script to test Gir breed predictions
"""
import json
import os
from pathlib import Path

from ml.predict import classify

# Test on Gir images
gir_test_dir = Path('Cattle-Buffalo-breeds.folder/test/Gir')
gir_images = sorted(
    list(gir_test_dir.glob('*.jpg')) + list(gir_test_dir.glob('*.png'))
)[:15]

print(f"\nFound {len(gir_images)} Gir test images (showing first 15)")
print("\nTesting Gir predictions:")
print("-" * 120)

gir_correct = 0
gir_to_sahiwal = 0
gir_to_unknown = 0
gir_to_other = 0
gir_total = 0

for img_path in gir_images:
    try:
        result = classify(str(img_path))
        breed = result.get('breed', 'ERROR')
        species = result.get('species', 'ERROR')
        confidence = result.get('confidence', 0)

        is_correct = (breed and breed.lower() == 'gir')
        if is_correct:
            gir_correct += 1
        elif breed and 'sahiwal' in breed.lower():
            gir_to_sahiwal += 1
        elif breed and 'unknown' in breed.lower():
            gir_to_unknown += 1
        elif breed and 'other' in breed.lower():
            gir_to_other += 1

        gir_total += 1

        status = "✓" if is_correct else "✗"
        print(
            f"{status} {img_path.name:45} -> {str(breed):20} "
            f"(species: {str(species):15} conf: {confidence:.4f})"
        )
        if not is_correct:
            # Show top predictions for errors
            top = result.get('top_predictions', [])
            if top:
                top_str = ', '.join(
                    [f"{p['label']}:{p['score']:.3f}" for p in top[:2]]
                )
                print(f"  {'':45}    Top predictions: {top_str}")
    except Exception as e:
        print(f"✗ {img_path.name:45} -> ERROR: {str(e)[:40]}")

print("-" * 120)
if gir_total > 0:
    print("\nGir Classification Results:")
    print(
        f"  Correct (Gir):        {gir_correct}/{gir_total} "
        f"= {100 * gir_correct / gir_total:.1f}%"
    )
    print(f"  Misclassified as Sahiwal: {gir_to_sahiwal}")
    print(f"  Misclassified as Unknown: {gir_to_unknown}")
    print(f"  Misclassified as Other:   {gir_to_other}")
    print(
        "  Other misclassifications: "
        f"{gir_total - gir_correct - gir_to_sahiwal - gir_to_unknown - gir_to_other}"
    )

# Check the trained model
print("\n" + "="*120)
print("Model Configuration:")
print("="*120)

with open('models/class_labels.json', 'r') as f:
    class_labels = json.load(f)
print(f"Class labels: {class_labels}")

# Check training history
if os.path.exists('models/training_history.json'):
    with open('models/training_history.json', 'r') as f:
        history = json.load(f)
    print(f"\nTraining stages: {history.get('stages', 'Not recorded')}")
    if 'final_metrics' in history:
        print(f"Final test accuracy: {history['final_metrics'].get('accuracy', 'N/A')}")

#!/usr/bin/env python3
"""
Deep diagnostic to understand Gir vs training issues
"""
import json
import os
from collections import Counter
from pathlib import Path

# Check training data distribution
print("="*100)
print("TRAINING DATA DISTRIBUTION")
print("="*100)

breed_counts = {}
for breed_dir in Path('Cattle-Buffalo-breeds.folder/train').iterdir():
    if breed_dir.is_dir():
        images = list(breed_dir.glob('*.jpg')) + list(breed_dir.glob('*.png'))
        breed_counts[breed_dir.name] = len(images)

print("\nBreed counts in train folder:")
for breed, count in sorted(breed_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {breed:25} {count:4} images")

# Check test data 
print("\n" + "="*100)
print("TEST DATA DISTRIBUTION")
print("="*100)

test_counts = {}
for breed_dir in Path('Cattle-Buffalo-breeds.folder/test').iterdir():
    if breed_dir.is_dir():
        images = list(breed_dir.glob('*.jpg')) + list(breed_dir.glob('*.png'))
        test_counts[breed_dir.name] = len(images)

print("\nBreed counts in test folder:")
for breed, count in sorted(test_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {breed:25} {count:4} images")

# Compare ratios
print("\n" + "="*100)
print("ANALYSIS")
print("="*100)

if 'Gir' in breed_counts and 'Sahiwal' in breed_counts:
    gir_train = breed_counts.get('Gir', 0)
    sahiwal_train = breed_counts.get('Sahiwal', 0)
    print(f"\nTraining data ratio:")
    print(f"  Gir:    {gir_train} images")
    print(f"  Sahiwal: {sahiwal_train} images")
    print(f"  Ratio (Gir:Sahiwal): 1:{sahiwal_train/gir_train:.2f}")
    
if 'Gir' in test_counts and 'Sahiwal' in test_counts:
    gir_test = test_counts.get('Gir', 0)
    sahiwal_test = test_counts.get('Sahiwal', 0)
    print(f"\nTest data ratio:")
    print(f"  Gir:    {gir_test} images")
    print(f"  Sahiwal: {sahiwal_test} images")
    print(f"  Ratio (Gir:Sahiwal): 1:{sahiwal_test/gir_test:.2f}")

# Check if there's a training_history.json to see what happened
print("\n" + "="*100)
print("TRAINING CONFIGURATION")
print("="*100)

if os.path.exists('models/training_history.json'):
    with open('models/training_history.json', 'r') as f:
        history = json.load(f)
    if 'config' in history:
        config = history['config']
        print(f"\nHard-breed weight boost: {config.get('hard_breed_weight_boost', 'not set')}")
        print(f"Hard-breed oversample factor: {config.get('hard_breed_oversample_factor', 'not set')}")
    if 'stages' in history:
        print(f"\nTraining stages:")
        for stage_name, stage_info in history['stages'].items():
            print(f"  {stage_name}: {stage_info}")

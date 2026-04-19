#!/usr/bin/env python3
"""
Retrain model with aggressive Gir balancing to fix misclassification
"""
import os

# Set strong hard-breed balancing BEFORE importing ml.train
os.environ['HARD_BREED_WEIGHT_BOOST'] = '3.0'
os.environ['HARD_BREED_OVERSAMPLE_FACTOR'] = '3.0'
os.environ['INITIAL_EPOCHS'] = '20'
os.environ['FINETUNE_EPOCHS'] = '12'
os.environ['REFINE_EPOCHS'] = '8'

# Now import and run training
from ml.train import main as train_main  # noqa: E402

if __name__ == '__main__':
    print("="*100)
    print("RETRAINING MODEL WITH AGGRESSIVE GIR BALANCING")
    print("="*100)
    print("\nParameters:")
    print(f"  HARD_BREED_WEIGHT_BOOST: {os.environ['HARD_BREED_WEIGHT_BOOST']}")
    print(f"  HARD_BREED_OVERSAMPLE_FACTOR: {os.environ['HARD_BREED_OVERSAMPLE_FACTOR']}")
    print(f"  INITIAL_EPOCHS: {os.environ['INITIAL_EPOCHS']}")
    print(f"  FINETUNE_EPOCHS: {os.environ['FINETUNE_EPOCHS']}")
    print(f"  REFINE_EPOCHS: {os.environ['REFINE_EPOCHS']}")
    print("\nThis will retrain for ~40 total epochs across 3 stages. Typical runtime: 2-3 hours\n")

    train_main()

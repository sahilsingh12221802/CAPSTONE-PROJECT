#!/usr/bin/env python3
"""
Gir-vs-Sahiwal Focused Refinement Training (Simplified)
-------------------------------------------------------
Fine-tunes the existing model specifically on Gir and Sahiwal discrimination.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile.h5')
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile_backup.h5')

# Refinement parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
REFINEMENT_EPOCHS = 15
BASE_LEARNING_RATE = 5e-6  # Very small LR for fine-tuning

RNG_SEED = 42
np.random.seed(RNG_SEED)
tf.keras.utils.set_random_seed(RNG_SEED)


def build_gir_sahiwal_dataframe(split_dir):
    """Build dataframe with only Gir and Sahiwal images"""
    rows = []
    split_path = Path(split_dir)
    
    # All 7 classes as defined in the model
    all_classes = ['Gir', 'Holstein_Friesian', 'Jersey', 'Sahiwal', 'Jaffrabadi', 'Murrah', 'Other']
    
    for breed in ['Gir', 'Sahiwal']:
        breed_dir = split_path / breed
        if breed_dir.exists():
            for img_path in sorted(breed_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    rows.append({'filepath': str(img_path), 'class': breed})
    
    return pd.DataFrame(rows)


def main():
    print("="*100)
    print("GIR-VS-SAHIWAL FOCUSED REFINEMENT TRAINING")
    print("="*100)
    print(f"\nRefinement Parameters:")
    print(f"  Refinement Epochs: {REFINEMENT_EPOCHS}")
    print(f"  Learning Rate: {BASE_LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        return
    
    print(f"\nLoading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Create backup
    print(f"Creating backup at {BACKUP_MODEL_PATH}...")
    model.save(BACKUP_MODEL_PATH)
    
    # Build dataframes for Gir and Sahiwal only
    print("\nBuilding Gir-Sahiwal datasets...")
    train_df = build_gir_sahiwal_dataframe(TRAIN_DIR)
    val_df = build_gir_sahiwal_dataframe(VAL_DIR)
    test_df = build_gir_sahiwal_dataframe(TEST_DIR)
    
    print(f"\nData summary:")
    print(f"  Train: {len(train_df)} images")
    print(f"    - Gir: {(train_df['class']=='Gir').sum()}")
    print(f"    - Sahiwal: {(train_df['class']=='Sahiwal').sum()}")
    print(f"  Val: {len(val_df)} images")
    print(f"    - Gir: {(val_df['class']=='Gir').sum()}")
    print(f"    - Sahiwal: {(val_df['class']=='Sahiwal').sum()}")
    print(f"  Test: {len(test_df)} images")
    print(f"    - Gir: {(test_df['class']=='Gir').sum()}")
    print(f"    - Sahiwal: {(test_df['class']=='Sahiwal').sum()}")
    
    # Freeze all but last 30 layers for refinement
    print("\nFreezing early layers, tuning last 30 layers...")
    for layer in model.layers[:-30]:
        layer.trainable = False
    
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    # Compile with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.06),
        metrics=['accuracy']
    )
    
    # Create data generators
    datagen_train = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    datagen_val = ImageDataGenerator()
    
    # For the 7-class model, map Gir->0, Sahiwal->3, and weight others as negative
    # We'll create a custom mapping
    class_mapping = {
        'Gir': 0,
        'Sahiwal': 3
    }
    
    # Create generators with 7 classes - pad missing classes with zeros
    print("\nCreating data generators...")
    train_gen = datagen_train.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RNG_SEED,
        classes=class_mapping  # Only use Gir and Sahiwal
    )
    
    val_gen = datagen_val.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=RNG_SEED,
        classes=class_mapping
    )
    
    test_gen = datagen_val.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=RNG_SEED,
        classes=class_mapping
    )
    
    # Class weights: heavily boost Gir detection
    # Since we only have Gir and Sahiwal, we'll focus weight on Gir
    class_weights = {
        0: 3.0,      # Gir
        3: 1.0       # Sahiwal
    }
    
    print(f"\nClass weights: {class_weights}")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
    
    # Train
    print("\n" + "="*100)
    print("STARTING REFINEMENT TRAINING")
    print("="*100 + "\n")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=REFINEMENT_EPOCHS,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    # Evaluate on test set
    print("\n" + "="*100)
    print("EVALUATING ON TEST SET")
    print("="*100 + "\n")
    
    test_loss, test_accuracy = model.evaluate(test_gen, steps=len(test_gen), verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Save refined model
    print(f"\n\nSaving refined model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    # Save training history
    history_dict = {
        'refinement_epochs': len(history.history['loss']),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_time_minutes': training_time,
        'timestamp': datetime.now().isoformat()
    }
    
    history_path = os.path.join(MODEL_DIR, 'refinement_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Refinement history saved to {history_path}")
    
    print("\n" + "="*100)
    print(f"REFINEMENT COMPLETED in {training_time:.1f} minutes")
    print(f"Test Accuracy on Gir/Sahiwal: {test_accuracy*100:.2f}%")
    print("="*100)
    print("\nNext: Run 'python diagnose_gir.py' to check if Gir/Sahiwal discrimination improved")


if __name__ == '__main__':
    main()

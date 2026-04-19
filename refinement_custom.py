#!/usr/bin/env python3
"""
Gir-vs-Sahiwal Focused Refinement - Custom Pipeline
---------------------------------------------------
Fine-tunes using a custom TensorFlow pipeline to handle class mapping properly.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuration
BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile.h5')
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile_backup.h5')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
REFINEMENT_EPOCHS = 12
BASE_LEARNING_RATE = 3e-6

# Class mapping: Gir->0, Sahiwal->3 in the 7-class output
NUM_CLASSES = 7
CLASS_NAMES = ['Gir', 'Holstein_Friesian', 'Jersey', 'Sahiwal', 'Jaffrabadi', 'Murrah', 'Other']
GIR_IDX = 0
SAHIWAL_IDX = 3

RNG_SEED = 42
np.random.seed(RNG_SEED)
tf.keras.utils.set_random_seed(RNG_SEED)


def load_images_with_labels(breed_dirs):
    """Load images and their class labels (only Gir and Sahiwal)"""
    images = []
    labels = []
    
    for breed_dir in breed_dirs:
        breed_name = breed_dir.name
        class_idx = GIR_IDX if breed_name == 'Gir' else SAHIWAL_IDX
        
        if breed_dir.exists():
            for img_path in sorted(breed_dir.glob('*')):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images.append(str(img_path))
                    labels.append(class_idx)
    
    return np.array(images), np.array(labels)


def load_and_preprocess(img_path, label):
    """Load and preprocess single image"""
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    
    # Convert label to 7-class one-hot
    one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
    one_hot[label] = 1.0
    
    return img_array, one_hot


def create_tf_dataset(images, labels, batch_size=BATCH_SIZE, shuffle=True):
    """Create TensorFlow dataset with augmentation for training"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(len(images), reshuffle_each_iteration=True)
    
    def augment_and_load(img_path, label):
        img, one_hot = tf.py_function(
            lambda p, l: load_and_preprocess(p.numpy().decode('utf-8'), l.numpy()),
            [img_path, label],
            [tf.float32, tf.float32]
        )
        img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        one_hot.set_shape((NUM_CLASSES,))
        
        if shuffle:
            # Random augmentations
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
        
        return img, one_hot
    
    dataset = dataset.map(augment_and_load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def main():
    print("="*100)
    print("GIR-VS-SAHIWAL FOCUSED REFINEMENT (CUSTOM PIPELINE)")
    print("="*100)
    print(f"\nRefinement Parameters:")
    print(f"  Epochs: {REFINEMENT_EPOCHS}")
    print(f"  Learning Rate: {BASE_LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Class mapping: Gir->0, Sahiwal->{SAHIWAL_IDX} (7-class output)")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        return
    
    print(f"\nLoading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Backup
    print(f"Creating backup...")
    model.save(BACKUP_MODEL_PATH)
    
    # Load training data
    print("\nLoading Gir and Sahiwal images...")
    
    train_imgs, train_labels = load_images_with_labels([
        Path(TRAIN_DIR) / 'Gir',
        Path(TRAIN_DIR) / 'Sahiwal'
    ])
    
    val_imgs, val_labels = load_images_with_labels([
        Path(VAL_DIR) / 'Gir',
        Path(VAL_DIR) / 'Sahiwal'
    ])
    
    test_imgs, test_labels = load_images_with_labels([
        Path(TEST_DIR) / 'Gir',
        Path(TEST_DIR) / 'Sahiwal'
    ])
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_imgs)} ({np.sum(train_labels==GIR_IDX)} Gir, {np.sum(train_labels==SAHIWAL_IDX)} Sahiwal)")
    print(f"  Val:   {len(val_imgs)} ({np.sum(val_labels==GIR_IDX)} Gir, {np.sum(val_labels==SAHIWAL_IDX)} Sahiwal)")
    print(f"  Test:  {len(test_imgs)} ({np.sum(test_labels==GIR_IDX)} Gir, {np.sum(test_labels==SAHIWAL_IDX)} Sahiwal)")
    
    # Create datasets
    print("\nCreating TensorFlow datasets...")
    train_dataset = create_tf_dataset(train_imgs, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = create_tf_dataset(val_imgs, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = create_tf_dataset(test_imgs, test_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    # Freeze early layers
    print("\nFreezing early layers, tuning last 35 layers...")
    for layer in model.layers[:-35]:
        layer.trainable = False
    
    for layer in model.layers[-35:]:
        layer.trainable = True
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
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
    
    # Custom class weights focusing on Gir
    class_weight = {GIR_IDX: 4.0, SAHIWAL_IDX: 1.0}
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=REFINEMENT_EPOCHS,
        class_weight=class_weight,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    # Evaluate
    print("\n" + "="*100)
    print("EVALUATING ON TEST SET")
    print("="*100 + "\n")
    
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Save model
    print(f"\nSaving refined model...")
    model.save(MODEL_PATH)
    
    # Save history
    history_dict = {
        'refinement_type': 'gir_vs_sahiwal_focused',
        'refinement_epochs': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_time_minutes': training_time,
        'timestamp': datetime.now().isoformat()
    }
    
    history_path = os.path.join(MODEL_DIR, 'refinement_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Refinement history saved")
    
    print("\n" + "="*100)
    print(f"REFINEMENT COMPLETED in {training_time:.1f} minutes")
    print(f"Binary Classification Accuracy (Gir vs Sahiwal): {test_accuracy*100:.2f}%")
    print("="*100)
    print("\nNext: Run 'python diagnose_gir.py' to check if Gir/Sahiwal discrimination improved")


if __name__ == '__main__':
    main()

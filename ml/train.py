import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
GENERATED_OTHER_DIR = os.path.join(DATA_DIR, 'generated_other')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.json')

CLASS_NAMES = ['Cattle', 'Buffalo', 'Other']

# Map breed folders to the target classes.
BREED_TO_CLASS = {
    'Gir': 'Cattle',
    'Holstein_Friesian': 'Cattle',
    'Jersey': 'Cattle',
    'Sahiwal': 'Cattle',
    'Jaffrabadi': 'Buffalo',
    'Murrah': 'Buffalo'
}

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
INITIAL_EPOCHS = int(os.getenv('INITIAL_EPOCHS', '16'))
FINETUNE_EPOCHS = int(os.getenv('FINETUNE_EPOCHS', '8'))
BASE_LEARNING_RATE = float(os.getenv('BASE_LR', '1e-4'))
FINETUNE_LEARNING_RATE = float(os.getenv('FINETUNE_LR', '1e-5'))
FINE_TUNE_LAYERS = int(os.getenv('FINE_TUNE_LAYERS', '40'))

OTHER_TRAIN_COUNT = int(os.getenv('OTHER_TRAIN_COUNT', '2800'))
OTHER_VALID_COUNT = int(os.getenv('OTHER_VALID_COUNT', '600'))
OTHER_TEST_COUNT = int(os.getenv('OTHER_TEST_COUNT', '600'))

RNG_SEED = int(os.getenv('SEED', '42'))
rng = np.random.default_rng(RNG_SEED)


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old_file in path.glob('*.jpg'):
        old_file.unlink()


def _save_other_image(gray_array: np.ndarray, output_path: Path) -> None:
    img = Image.fromarray(gray_array.astype(np.uint8), mode='L').convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img.save(output_path, quality=90)


def _build_other_pool() -> np.ndarray:
    (mnist_train, _), (mnist_test, _) = tf.keras.datasets.mnist.load_data()
    (fashion_train, _), (fashion_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    return np.concatenate([mnist_train, mnist_test, fashion_train, fashion_test], axis=0)


def _create_other_split(split_name: str, sample_count: int, pool: np.ndarray) -> None:
    split_other_dir = Path(GENERATED_OTHER_DIR) / split_name / 'Other'
    _ensure_clean_dir(split_other_dir)

    chosen_indices = rng.choice(len(pool), size=sample_count, replace=False)
    for i, idx in enumerate(chosen_indices):
        _save_other_image(pool[idx], split_other_dir / f'other_{i:05d}.jpg')


def ensure_generated_other_dataset() -> None:
    print('Preparing generated non-target (Other) dataset...')
    pool = _build_other_pool()
    _create_other_split('train', OTHER_TRAIN_COUNT, pool)
    _create_other_split('valid', OTHER_VALID_COUNT, pool)
    _create_other_split('test', OTHER_TEST_COUNT, pool)
    print('Generated Other dataset under Cattle-Buffalo-breeds.folder/generated_other')


def build_dataframe(split_dir: str, other_split_dir: str) -> pd.DataFrame:
    rows = []
    split_path = Path(split_dir)

    for breed_name, class_name in BREED_TO_CLASS.items():
        breed_dir = split_path / breed_name
        if not breed_dir.exists():
            continue

        for image_file in breed_dir.iterdir():
            if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                rows.append({'filepath': str(image_file), 'label': class_name})

    other_dir = Path(other_split_dir) / 'Other'
    if other_dir.exists():
        for image_file in other_dir.iterdir():
            if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                rows.append({'filepath': str(image_file), 'label': 'Other'})

    return pd.DataFrame(rows)


def compute_class_weights(train_df: pd.DataFrame, class_indices: dict) -> dict:
    counts = train_df['label'].value_counts()
    total = float(len(train_df))
    num_classes = float(len(class_indices))

    class_weight = {}
    for label, idx in class_indices.items():
        class_count = float(counts.get(label, 1.0))
        class_weight[idx] = total / (num_classes * class_count)

    return class_weight


def make_generators(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        classes=CLASS_NAMES,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RNG_SEED
    )

    validation_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='label',
        classes=CLASS_NAMES,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='label',
        classes=CLASS_NAMES,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def build_model(num_classes: int):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base


def make_callbacks(model_path: str, patience: int = 5):
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]


def main():
    ensure_generated_other_dataset()

    train_df = build_dataframe(TRAIN_DIR, os.path.join(GENERATED_OTHER_DIR, 'train'))
    val_df = build_dataframe(VAL_DIR, os.path.join(GENERATED_OTHER_DIR, 'valid'))
    test_df = build_dataframe(TEST_DIR, os.path.join(GENERATED_OTHER_DIR, 'test'))

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError('Dataset split is empty. Verify cattle/buffalo folders and generated_other dataset.')

    print(f'Train images: {len(train_df)}')
    print(f'Valid images: {len(val_df)}')
    print(f'Test images: {len(test_df)}')
    print('Train label distribution:')
    print(train_df['label'].value_counts())

    train_generator, validation_generator, test_generator = make_generators(train_df, val_df, test_df)
    class_weight = compute_class_weights(train_df, train_generator.class_indices)
    print('Class indices:', train_generator.class_indices)
    print('Class weights:', class_weight)

    model, base = build_model(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'Stage 1 training for {INITIAL_EPOCHS} epochs (frozen backbone).')
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS,
        callbacks=make_callbacks(MODEL_PATH),
        class_weight=class_weight
    )

    # Fine-tune the top MobileNetV2 layers while keeping BatchNorm layers frozen for stability.
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'Stage 2 fine-tuning for {FINETUNE_EPOCHS} epochs (top {FINE_TUNE_LAYERS} layers).')
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=make_callbacks(MODEL_PATH, patience=4),
        class_weight=class_weight
    )

    best_model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = best_model.evaluate(test_generator, verbose=1)
    print(f'Test loss: {loss:.4f}, Test accuracy: {acc:.4f}')

    best_model.save(MODEL_PATH)
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f'Saved model to {MODEL_PATH}')
    print(f'Saved class labels to {LABELS_PATH}')


if __name__ == '__main__':
    main()

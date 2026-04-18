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
GENERATED_HUMAN_DIR = os.path.join(DATA_DIR, 'generated_human')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.json')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')

BREED_TO_SPECIES = {
    'Gir': 'Cattle',
    'Holstein_Friesian': 'Cattle',
    'Jersey': 'Cattle',
    'Sahiwal': 'Cattle',
    'Jaffrabadi': 'Buffalo',
    'Murrah': 'Buffalo'
}
BREED_CLASSES = list(BREED_TO_SPECIES.keys())
CLASS_NAMES = BREED_CLASSES + ['Other']

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
INITIAL_EPOCHS = int(os.getenv('INITIAL_EPOCHS', '16'))
FINETUNE_EPOCHS = int(os.getenv('FINETUNE_EPOCHS', '8'))
REFINE_EPOCHS = int(os.getenv('REFINE_EPOCHS', '4'))
BASE_LEARNING_RATE = float(os.getenv('BASE_LR', '1e-4'))
FINETUNE_LEARNING_RATE = float(os.getenv('FINETUNE_LR', '1e-5'))
REFINE_LEARNING_RATE = float(os.getenv('REFINE_LR', '5e-6'))
FINE_TUNE_LAYERS = int(os.getenv('FINE_TUNE_LAYERS', '40'))
LABEL_SMOOTHING = float(os.getenv('LABEL_SMOOTHING', '0.06'))
ADAM_CLIPNORM = float(os.getenv('ADAM_CLIPNORM', '1.0'))
ADAM_WEIGHT_DECAY = float(os.getenv('ADAM_WEIGHT_DECAY', '1e-5'))
ONLY_PREPARE_DATA = os.getenv('ONLY_PREPARE_DATA', 'false').lower() == 'true'

OTHER_TRAIN_COUNT = int(os.getenv('OTHER_TRAIN_COUNT', '1800'))
OTHER_VALID_COUNT = int(os.getenv('OTHER_VALID_COUNT', '450'))
OTHER_TEST_COUNT = int(os.getenv('OTHER_TEST_COUNT', '600'))
HUMAN_TRAIN_COUNT = int(os.getenv('HUMAN_TRAIN_COUNT', '1200'))
HUMAN_VALID_COUNT = int(os.getenv('HUMAN_VALID_COUNT', '350'))
HUMAN_TEST_COUNT = int(os.getenv('HUMAN_TEST_COUNT', '500'))

RNG_SEED = int(os.getenv('SEED', '42'))
rng = np.random.default_rng(RNG_SEED)
tf.keras.utils.set_random_seed(RNG_SEED)


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old_file in path.glob('*.jpg'):
        old_file.unlink()


def _save_gray_image(gray_array: np.ndarray, output_path: Path) -> None:
    img = Image.fromarray(gray_array.astype(np.uint8), mode='L').convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img.save(output_path, quality=90)


def _save_rgb_image(rgb_array: np.ndarray, output_path: Path) -> None:
    img = Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img.save(output_path, quality=90)


def _build_other_pool() -> np.ndarray:
    # Use natural-image negatives for stronger real-world non-target behavior (logos, people, objects).
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    return np.concatenate([x_train, x_test], axis=0)


def _build_human_pool() -> np.ndarray:
    # CIFAR-100 fine label IDs: baby(2), boy(11), girl(35), man(46), woman(98)
    human_label_ids = {2, 11, 35, 46, 98}
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train.reshape(-1), y_test.reshape(-1)], axis=0)
    return x_all[np.isin(y_all, list(human_label_ids))]


def _create_generated_split(
    root_dir: str,
    split_name: str,
    class_name: str,
    sample_count: int,
    pool: np.ndarray,
    grayscale: bool,
) -> None:
    split_dir = Path(root_dir) / split_name / class_name
    _ensure_clean_dir(split_dir)

    replace = sample_count > len(pool)
    chosen_indices = rng.choice(len(pool), size=sample_count, replace=replace)
    file_prefix = class_name.lower()

    for i, idx in enumerate(chosen_indices):
        output_path = split_dir / f'{file_prefix}_{i:05d}.jpg'
        if grayscale:
            _save_gray_image(pool[idx], output_path)
        else:
            _save_rgb_image(pool[idx], output_path)


def ensure_generated_non_target_datasets() -> None:
    print('Preparing generated non-target datasets (Other + Human)...')
    pool = _build_other_pool()
    _create_generated_split(GENERATED_OTHER_DIR, 'train', 'Other', OTHER_TRAIN_COUNT, pool, grayscale=False)
    _create_generated_split(GENERATED_OTHER_DIR, 'valid', 'Other', OTHER_VALID_COUNT, pool, grayscale=False)
    _create_generated_split(GENERATED_OTHER_DIR, 'test', 'Other', OTHER_TEST_COUNT, pool, grayscale=False)

    human_pool = _build_human_pool()
    _create_generated_split(GENERATED_HUMAN_DIR, 'train', 'Human', HUMAN_TRAIN_COUNT, human_pool, grayscale=False)
    _create_generated_split(GENERATED_HUMAN_DIR, 'valid', 'Human', HUMAN_VALID_COUNT, human_pool, grayscale=False)
    _create_generated_split(GENERATED_HUMAN_DIR, 'test', 'Human', HUMAN_TEST_COUNT, human_pool, grayscale=False)

    print('Generated datasets under Cattle-Buffalo-breeds.folder/generated_other and generated_human')


def build_dataframe(split_dir: str, other_split_dir: str, human_split_dir: str) -> pd.DataFrame:
    rows = []
    split_path = Path(split_dir)

    for breed_name in BREED_TO_SPECIES:
        breed_dir = split_path / breed_name
        if not breed_dir.exists():
            continue

        for image_file in breed_dir.iterdir():
            if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                rows.append({'filepath': str(image_file), 'label': breed_name})

    other_dir = Path(other_split_dir) / 'Other'
    if other_dir.exists():
        for image_file in other_dir.iterdir():
            if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                rows.append({'filepath': str(image_file), 'label': 'Other'})

    human_dir = Path(human_split_dir) / 'Human'
    if human_dir.exists():
        for image_file in human_dir.iterdir():
            if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                # Human samples are intentionally merged into Other as one unified non-target class.
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
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        channel_shift_range=20.0,
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


def make_refinement_generators(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Build a bovine-focused but still non-target-aware refinement stream."""
    train_bovine = train_df[train_df['label'] != 'Other'].copy()
    train_other = train_df[train_df['label'] == 'Other'].copy()

    # Keep enough non-target examples to preserve rejection, but avoid overwhelming breed learning.
    target_other_n = max(200, int(len(train_bovine) * 0.6))
    if len(train_other) > target_other_n:
        train_other = train_other.sample(n=target_other_n, random_state=RNG_SEED)

    refine_train_df = pd.concat([train_bovine, train_other], ignore_index=True)

    val_bovine = val_df[val_df['label'] != 'Other'].copy()
    val_other = val_df[val_df['label'] == 'Other'].copy()
    target_val_other_n = max(80, int(len(val_bovine) * 0.6))
    if len(val_other) > target_val_other_n:
        val_other = val_other.sample(n=target_val_other_n, random_state=RNG_SEED)
    refine_val_df = pd.concat([val_bovine, val_other], ignore_index=True)

    refine_train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=18,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='nearest',
    )
    refine_val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    refine_train_gen = refine_train_datagen.flow_from_dataframe(
        refine_train_df,
        x_col='filepath',
        y_col='label',
        classes=CLASS_NAMES,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RNG_SEED,
    )
    refine_val_gen = refine_val_datagen.flow_from_dataframe(
        refine_val_df,
        x_col='filepath',
        y_col='label',
        classes=CLASS_NAMES,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    return refine_train_df, refine_train_gen, refine_val_gen


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


def make_optimizer(learning_rate: float):
    # Clip gradients to reduce unstable updates and use light decay for better generalization.
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=ADAM_CLIPNORM,
        weight_decay=ADAM_WEIGHT_DECAY,
    )


def make_loss():
    # Label smoothing improves robustness on visually similar breeds (e.g., Gir vs Sahiwal).
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)


def make_callbacks(model_path: str, patience: int = 5):
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]


def main():
    ensure_generated_non_target_datasets()

    if ONLY_PREPARE_DATA:
        print('ONLY_PREPARE_DATA=true -> generated datasets prepared; skipping model training.')
        return

    train_df = build_dataframe(
        TRAIN_DIR,
        os.path.join(GENERATED_OTHER_DIR, 'train'),
        os.path.join(GENERATED_HUMAN_DIR, 'train'),
    )
    val_df = build_dataframe(
        VAL_DIR,
        os.path.join(GENERATED_OTHER_DIR, 'valid'),
        os.path.join(GENERATED_HUMAN_DIR, 'valid'),
    )
    test_df = build_dataframe(
        TEST_DIR,
        os.path.join(GENERATED_OTHER_DIR, 'test'),
        os.path.join(GENERATED_HUMAN_DIR, 'test'),
    )

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            'Dataset split is empty. Verify breed folders and generated non-target datasets.'
        )

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
        optimizer=make_optimizer(BASE_LEARNING_RATE),
        loss=make_loss(),
        metrics=['accuracy']
    )

    print(f'Stage 1 training for {INITIAL_EPOCHS} epochs (frozen backbone).')
    stage1_history = model.fit(
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
        optimizer=make_optimizer(FINETUNE_LEARNING_RATE),
        loss=make_loss(),
        metrics=['accuracy']
    )

    print(f'Stage 2 fine-tuning for {FINETUNE_EPOCHS} epochs (top {FINE_TUNE_LAYERS} layers).')
    stage2_history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=make_callbacks(MODEL_PATH, patience=4),
        class_weight=class_weight
    )

    refine_history_payload = {}
    if REFINE_EPOCHS > 0:
        print(f'Stage 3 breed refinement for {REFINE_EPOCHS} epochs (bovine-focused mix).')
        refine_train_df, refine_train_gen, refine_val_gen = make_refinement_generators(train_df, val_df)
        refine_class_weight = compute_class_weights(refine_train_df, refine_train_gen.class_indices)

        model.compile(
            optimizer=make_optimizer(REFINE_LEARNING_RATE),
            loss=make_loss(),
            metrics=['accuracy'],
        )
        refine_history = model.fit(
            refine_train_gen,
            validation_data=refine_val_gen,
            epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS + REFINE_EPOCHS,
            initial_epoch=INITIAL_EPOCHS + FINETUNE_EPOCHS,
            callbacks=make_callbacks(MODEL_PATH, patience=3),
            class_weight=refine_class_weight,
        )
        refine_history_payload = {k: [float(v) for v in values] for k, values in refine_history.history.items()}

    best_model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = best_model.evaluate(test_generator, verbose=1)
    print(f'Test loss: {loss:.4f}, Test accuracy: {acc:.4f}')

    best_model.save(MODEL_PATH)
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(CLASS_NAMES, f, indent=2)

    history_payload = {
        'stage1': {k: [float(v) for v in values] for k, values in stage1_history.history.items()},
        'stage2': {k: [float(v) for v in values] for k, values in stage2_history.history.items()},
        'config': {
            'seed': RNG_SEED,
            'batch_size': BATCH_SIZE,
            'initial_epochs': INITIAL_EPOCHS,
            'finetune_epochs': FINETUNE_EPOCHS,
            'base_lr': BASE_LEARNING_RATE,
            'finetune_lr': FINETUNE_LEARNING_RATE,
            'label_smoothing': LABEL_SMOOTHING,
            'adam_clipnorm': ADAM_CLIPNORM,
            'adam_weight_decay': ADAM_WEIGHT_DECAY,
            'refine_epochs': REFINE_EPOCHS,
            'refine_lr': REFINE_LEARNING_RATE,
        },
    }
    if refine_history_payload:
        history_payload['stage3_refine'] = refine_history_payload
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history_payload, f, indent=2)

    print(f'Saved model to {MODEL_PATH}')
    print(f'Saved class labels to {LABELS_PATH}')
    print(f'Saved training history to {HISTORY_PATH}')


if __name__ == '__main__':
    main()

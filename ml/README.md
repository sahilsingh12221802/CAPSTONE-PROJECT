# ML Module

This module contains all machine learning workflows: training, prediction, and evaluation.

## Goals

- Train a robust classifier for `Cattle`, `Buffalo`, and `Other`.
- Reduce false positives on unrelated images.
- Provide reproducible evaluation metrics and confusion output.

## Files

- `train.py`: complete training pipeline and model export.
- `predict.py`: single-image inference with rejection safeguards.
- `evaluate.py`: benchmark script for bovine accuracy and non-target rejection.

## Training Pipeline (High Level)

- Uses transfer learning backbone (MobileNetV2).
- Uses dataset label mapping from breed folders to target classes.
- Builds/generated `Other` examples for non-target awareness.
- Trains in stages (frozen backbone then fine-tuning).
- Saves best model and class labels into `models/`.

## Runtime Inference Behavior

- Loads model and labels.
- Applies preprocessing + prediction.
- Uses confidence/margin checks and gate signals.
- Returns one of:
	- `Cattle`
	- `Buffalo`
	- `Unknown` (for non-target or low-confidence cases)

## Evaluation Outputs

The evaluation script reports:

- Overall accuracy
- Bovine-only accuracy (`Cattle` vs `Buffalo`)
- Bovine rejected-as-unknown rate
- Non-target rejection rate
- Confusion matrix and sample error lines

## Run Commands

From project root:

```bash
source .venv/bin/activate
python -m ml.train
python -m ml.evaluate
python -m ml.predict "path/to/image.jpg"
```

## Model Artifacts

Saved in [models/](../models):

- `cattle_buffalo_mobile.h5`
- `class_labels.json`

## Dataset Requirements

- Main dataset root: `Cattle-Buffalo-breeds.folder/`
- Expected splits: `train/`, `valid/`, `test/`
- Breed folders are mapped to binary target species classes.
- `train.py` generates additional non-target examples under `generated_other/`.

## Generated `Other` Dataset (Important)

The `Other` class is auto-generated during training to teach the model what is **not** cattle/buffalo.

### Why it exists

- A 2-class model (`Cattle`/`Buffalo`) tends to force every input into one of those classes.
- The generated `Other` class improves rejection behavior for unrelated images.
- This is a key reason the API can return `Unknown` more reliably.

### Data source used for `Other`

`train.py` builds `Other` samples from grayscale datasets and converts them to RGB image files:

- MNIST
- Fashion-MNIST

These samples are resized to model input size and saved as JPEG files.

### Folder layout

Generated files are created under:

```text
Cattle-Buffalo-breeds.folder/generated_other/
├── train/Other/
├── valid/Other/
└── test/Other/
```

### Default generated sample counts

- Train: `2800`
- Validation: `600`
- Test: `600`

These defaults are controlled by environment variables:

- `OTHER_TRAIN_COUNT`
- `OTHER_VALID_COUNT`
- `OTHER_TEST_COUNT`

### Reproducibility

- Generation is deterministic based on the `SEED` value.
- Re-running training can regenerate this dataset and overwrite old generated files.

## Tunable Training Controls

Main controls are exposed via environment variables in `train.py`:

- batch size and epoch counts
- learning rates
- fine-tune depth
- generated `Other` sample counts

Use these to trade off speed vs quality.

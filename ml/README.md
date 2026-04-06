# ML Module

This module contains all machine learning workflows: training, prediction, and evaluation.

## Goals

- Train a robust classifier for breed-level bovine classes plus a unified non-target class.
- Reduce false positives on unrelated images.
- Improve human/logo/object rejection confidence using stronger non-target data.
- Provide reproducible evaluation metrics and confusion output.

## Files

- `train.py`: complete training pipeline and model export.
- `predict.py`: single-image inference with rejection safeguards.
- `evaluate.py`: benchmark script for bovine accuracy and non-target rejection.

## Training Pipeline (High Level)

- Uses transfer learning backbone (MobileNetV2).
- Trains on breed folders directly (`Gir`, `Murrah`, etc.) for breed-level prediction.
- Generates `Other` examples from CIFAR-10 natural images.
- Generates `Human` examples from CIFAR-100 human fine-label classes.
- Merges generated `Human` samples into `Other` during training (single non-target output class).
- Trains in stages (frozen backbone then fine-tuning).
- Saves best model and class labels into `models/`.

## Runtime Inference Behavior

- Loads model and labels.
- Applies preprocessing + prediction.
- Uses confidence/margin checks and gate signals.
- Returns species + breed metadata for bovine predictions.
- Returns `Other` for non-target images (human, logo, unrelated object).
- Uses adaptive non-target confidence calibration to avoid static confidence percentages.
- Returns `Unknown` only for low-confidence/ambiguous bovine decisions.

## Evaluation Outputs

The evaluation script reports:

- Overall accuracy including `Other`
- Bovine species accuracy (`Cattle` vs `Buffalo`)
- Bovine breed accuracy (`Gir`, `Murrah`, etc.)
- Non-target (`Other`) rejection rate
- Confusion matrix and sample error lines

## Run Commands

From project root:

```bash
source .venv311/bin/activate
ONLY_PREPARE_DATA=true python -m ml.train
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
- Breed folders are used as direct model classes.
- `train.py` generates additional non-target examples under `generated_other/` and `generated_human/`.

## Generated `Other` Dataset (Important)

The `Other` class is auto-generated during training to teach the model what is **not** cattle/buffalo.

### Why it exists

- A 2-class model (`Cattle`/`Buffalo`) tends to force every input into one of those classes.
- The generated `Other` class improves rejection behavior for unrelated images.
- This is a key reason the API can return `Unknown` more reliably.

### Data source used for `Other`

`train.py` builds `Other` samples from CIFAR-10 natural images.

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

## Generated `Human` Dataset (Important)

Generated human samples are not a separate final label. They are merged into `Other` during training.

### Data source used for `Human`

`train.py` builds `Human` samples from CIFAR-100 images filtered by fine labels:

- `baby` (2)
- `boy` (11)
- `girl` (35)
- `man` (46)
- `woman` (98)

### Folder layout

Generated files are created under:

```text
Cattle-Buffalo-breeds.folder/generated_human/
├── train/Human/
├── valid/Human/
└── test/Human/
```

### Default generated sample counts

- Train: `2400`
- Validation: `500`
- Test: `500`

These defaults are controlled by environment variables:

- `HUMAN_TRAIN_COUNT`
- `HUMAN_VALID_COUNT`
- `HUMAN_TEST_COUNT`

### How `Human` is used

- Human images are generated and stored under `generated_human/`.
- During dataframe construction, those files are labeled as `Other`.
- Final model output classes therefore include bovine breeds + `Other` only.

## Tunable Training Controls

Main controls are exposed via environment variables in `train.py`:

- batch size and epoch counts
- learning rates
- fine-tune depth
- generated `Other` sample counts

Use these to trade off speed vs quality.

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ml.predict import classify

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OTHER_TEST_DIR = os.path.join(DATA_DIR, 'generated_other', 'test', 'Other')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'class_labels.json')

BREED_TO_CLASS = {
    'Gir': 'Cattle',
    'Holstein_Friesian': 'Cattle',
    'Jersey': 'Cattle',
    'Sahiwal': 'Cattle',
    'Jaffrabadi': 'Buffalo',
    'Murrah': 'Buffalo'
}

VALID_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_class_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return ['Cattle', 'Buffalo']


def collect_test_rows():
    rows = []

    for breed, target in BREED_TO_CLASS.items():
        breed_dir = Path(TEST_DIR) / breed
        if not breed_dir.exists():
            continue

        for image_file in breed_dir.iterdir():
            if image_file.suffix.lower() in VALID_SUFFIXES:
                rows.append({'filepath': str(image_file), 'actual': target})

    other_dir = Path(OTHER_TEST_DIR)
    if other_dir.exists():
        for image_file in other_dir.iterdir():
            if image_file.suffix.lower() in VALID_SUFFIXES:
                rows.append({'filepath': str(image_file), 'actual': 'Unknown'})

    return pd.DataFrame(rows)


def summarize_metrics(df):
    total = len(df)
    correct = (df['actual'] == df['predicted']).sum()
    accuracy = correct / total if total else 0.0

    bovine_df = df[df['actual'].isin(['Cattle', 'Buffalo'])]
    non_target_df = df[df['actual'] == 'Unknown']

    bovine_total = len(bovine_df)
    bovine_correct = (bovine_df['actual'] == bovine_df['predicted']).sum()
    bovine_unknown = (bovine_df['predicted'] == 'Unknown').sum()

    non_target_total = len(non_target_df)
    non_target_rejected = (non_target_df['predicted'] == 'Unknown').sum()

    print('==== Evaluation Summary ====')
    print(f'Total samples: {total}')
    print(f'Overall accuracy (3-way incl. Unknown): {accuracy:.4f}')
    print('')
    print('Bovine-only performance:')
    print(f'  Total bovine samples: {bovine_total}')
    print(f'  Correct cattle/buffalo labels: {bovine_correct}')
    if bovine_total:
        print(f'  Bovine classification accuracy: {bovine_correct / bovine_total:.4f}')
        print(f'  Bovine rejected as Unknown: {bovine_unknown} ({bovine_unknown / bovine_total:.4f})')
    print('')
    print('Non-target rejection performance:')
    print(f'  Total non-target samples: {non_target_total}')
    print(f'  Correctly rejected (Unknown): {non_target_rejected}')
    if non_target_total:
        print(f'  Non-target rejection rate: {non_target_rejected / non_target_total:.4f}')


def print_confusion(df):
    labels = ['Cattle', 'Buffalo', 'Unknown']
    matrix = pd.crosstab(
        pd.Categorical(df['actual'], categories=labels),
        pd.Categorical(df['predicted'], categories=labels),
        dropna=False
    )
    print('')
    print('==== Confusion Matrix (actual x predicted) ====')
    print(matrix)


def main():
    class_labels = load_class_labels()
    print(f'Model class labels: {class_labels}')

    df = collect_test_rows()
    if df.empty:
        raise RuntimeError('No test samples found for evaluation.')

    predictions = []
    for row in df.itertuples(index=False):
        result = classify(row.filepath)
        predicted = result.get('label', 'Unknown')
        predictions.append(
            {
                'filepath': row.filepath,
                'actual': row.actual,
                'predicted': predicted,
                'confidence': float(result.get('confidence', 0.0)),
                'message': result.get('message', ''),
            }
        )

    pred_df = pd.DataFrame(predictions)
    summarize_metrics(pred_df)
    print_confusion(pred_df)

    mistakes = pred_df[pred_df['actual'] != pred_df['predicted']].head(20)
    if not mistakes.empty:
        print('')
        print('==== Sample Errors (first 20) ====')
        for row in mistakes.itertuples(index=False):
            print(f"{row.actual} -> {row.predicted} | conf={row.confidence:.3f} | {row.filepath}")


if __name__ == '__main__':
    main()

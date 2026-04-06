import json
import os
from pathlib import Path

import pandas as pd

from ml.predict import classify

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OTHER_TEST_DIR = os.path.join(DATA_DIR, 'generated_other', 'test', 'Other')
HUMAN_TEST_DIR = os.path.join(DATA_DIR, 'generated_human', 'test', 'Human')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'class_labels.json')

BREED_TO_SPECIES = {
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

    for breed, target in BREED_TO_SPECIES.items():
        breed_dir = Path(TEST_DIR) / breed
        if not breed_dir.exists():
            continue

        for image_file in breed_dir.iterdir():
            if image_file.suffix.lower() in VALID_SUFFIXES:
                rows.append({'filepath': str(image_file), 'actual_species': target, 'actual_breed': breed})

    other_dir = Path(OTHER_TEST_DIR)
    if other_dir.exists():
        for image_file in other_dir.iterdir():
            if image_file.suffix.lower() in VALID_SUFFIXES:
                rows.append({'filepath': str(image_file), 'actual_species': 'Unknown', 'actual_breed': None})

    human_dir = Path(HUMAN_TEST_DIR)
    if human_dir.exists():
        for image_file in human_dir.iterdir():
            if image_file.suffix.lower() in VALID_SUFFIXES:
                rows.append({'filepath': str(image_file), 'actual_species': 'Human', 'actual_breed': None})

    return pd.DataFrame(rows)


def summarize_metrics(df):
    total = len(df)
    correct = (df['actual_species'] == df['predicted_species']).sum()
    accuracy = correct / total if total else 0.0

    bovine_df = df[df['actual_species'].isin(['Cattle', 'Buffalo'])]
    bovine_species_correct = (bovine_df['actual_species'] == bovine_df['predicted_species']).sum()
    bovine_breed_correct = (bovine_df['actual_breed'] == bovine_df['predicted_breed']).sum()

    human_df = df[df['actual_species'] == 'Human']
    unknown_df = df[df['actual_species'] == 'Unknown']

    bovine_total = len(bovine_df)
    human_total = len(human_df)
    unknown_total = len(unknown_df)
    human_detected = (human_df['predicted_species'] == 'Human').sum()
    unknown_rejected = (unknown_df['predicted_species'] == 'Unknown').sum()

    print('==== Evaluation Summary ====')
    print(f'Total samples: {total}')
    print(f'Overall species accuracy: {accuracy:.4f}')
    print('')
    print('Bovine performance:')
    print(f'  Total bovine samples: {bovine_total}')
    print(f'  Correct species labels: {bovine_species_correct}')
    print(f'  Correct breed labels: {bovine_breed_correct}')
    if bovine_total:
        print(f'  Bovine species accuracy: {bovine_species_correct / bovine_total:.4f}')
        print(f'  Bovine breed accuracy: {bovine_breed_correct / bovine_total:.4f}')
    print('')
    print('Human detection performance:')
    print(f'  Total human samples: {human_total}')
    print(f'  Correctly detected as Human: {human_detected}')
    if human_total:
        print(f'  Human detection rate: {human_detected / human_total:.4f}')
    print('')
    print('Unknown rejection performance:')
    print(f'  Total unknown samples: {unknown_total}')
    print(f'  Correctly rejected as Unknown: {unknown_rejected}')
    if unknown_total:
        print(f'  Unknown rejection rate: {unknown_rejected / unknown_total:.4f}')


def print_confusion(df):
    labels = ['Cattle', 'Buffalo', 'Human', 'Unknown']
    matrix = pd.crosstab(
        pd.Categorical(df['actual_species'], categories=labels),
        pd.Categorical(df['predicted_species'], categories=labels),
        dropna=False
    )
    print('')
    print('==== Species Confusion Matrix (actual x predicted) ====')
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
        predicted_species = result.get('species') or result.get('label', 'Unknown')
        predicted_breed = result.get('breed')
        if predicted_species == 'Unknown' and result.get('label') == 'Human':
            predicted_species = 'Human'
        predictions.append(
            {
                'filepath': row.filepath,
                'actual_species': row.actual_species,
                'actual_breed': row.actual_breed,
                'predicted_species': predicted_species,
                'predicted_breed': predicted_breed,
                'confidence': float(result.get('confidence', 0.0)),
                'message': result.get('message', ''),
            }
        )

    pred_df = pd.DataFrame(predictions)
    summarize_metrics(pred_df)
    print_confusion(pred_df)

    mistakes = pred_df[pred_df['actual_species'] != pred_df['predicted_species']].head(20)
    if not mistakes.empty:
        print('')
        print('==== Sample Errors (first 20) ====')
        for row in mistakes.itertuples(index=False):
            print(
                f"{row.actual_species}/{row.actual_breed} -> "
                f"{row.predicted_species}/{row.predicted_breed} "
                f"| conf={row.confidence:.3f} | {row.filepath}"
            )


if __name__ == '__main__':
    main()

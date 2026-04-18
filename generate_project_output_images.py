import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow.keras.preprocessing.image import img_to_array, load_img

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'Cattle-Buffalo-breeds.folder'
MODEL_PATH = ROOT / 'models' / 'cattle_buffalo_mobile.h5'
LABELS_PATH = ROOT / 'models' / 'class_labels.json'
HISTORY_PATH = ROOT / 'models' / 'training_history.json'
OUT_DIR = ROOT / 'output_images'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BREEDS = ['Gir', 'Holstein_Friesian', 'Jersey', 'Sahiwal', 'Jaffrabadi', 'Murrah']
VALID_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BREED_TO_SPECIES = {
    'Gir': 'Cattle',
    'Holstein_Friesian': 'Cattle',
    'Jersey': 'Cattle',
    'Sahiwal': 'Cattle',
    'Jaffrabadi': 'Buffalo',
    'Murrah': 'Buffalo',
    'Other': 'Other',
}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_SUFFIXES])


def collect_split_rows(split: str):
    rows = []
    split_dir = DATA_DIR / split
    for breed in BREEDS:
        for p in list_images(split_dir / breed):
            rows.append({'path': p, 'label': breed, 'split': split})

    for p in list_images(DATA_DIR / 'generated_other' / split / 'Other'):
        rows.append({'path': p, 'label': 'Other', 'split': split})

    for p in list_images(DATA_DIR / 'generated_human' / split / 'Human'):
        rows.append({'path': p, 'label': 'Other', 'split': split})

    return rows


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=250)
    plt.close()


def sample_grid(rows, title, out_name, n=12):
    if not rows:
        return
    pick = rows[:n]
    cols = 4
    r = int(np.ceil(len(pick) / cols))
    plt.figure(figsize=(12, 3 * r))
    for i, row in enumerate(pick, 1):
        img = Image.open(row['path']).convert('RGB').resize((224, 224))
        ax = plt.subplot(r, cols, i)
        ax.imshow(img)
        ax.set_title(row['label'], fontsize=9)
        ax.axis('off')
    plt.suptitle(title)
    save_fig(out_name)


def augmentation_preview(image_path: Path, title: str, out_name: str):
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img).astype('float32') / 255.0

    variants = [
        arr,
        np.fliplr(arr),
        np.clip(arr + 0.12, 0.0, 1.0),
        np.clip(arr - 0.12, 0.0, 1.0),
        np.clip(np.rot90(arr, 1), 0.0, 1.0),
        np.clip(np.rot90(arr, 3), 0.0, 1.0),
    ]

    plt.figure(figsize=(10, 6))
    for i, v in enumerate(variants, 1):
        ax = plt.subplot(2, 3, i)
        ax.imshow(v)
        ax.set_title(f'Aug {i}', fontsize=9)
        ax.axis('off')
    plt.suptitle(title)
    save_fig(out_name)


def load_model_and_labels():
    labels = json.loads(LABELS_PATH.read_text(encoding='utf-8'))
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, labels


def load_batch(rows):
    x = []
    y = []
    paths = []
    for row in rows:
        arr = img_to_array(load_img(row['path'], target_size=(224, 224))).astype('float32') / 255.0
        x.append(arr)
        y.append(row['label'])
        paths.append(str(row['path']))
    return np.array(x), y, paths


def reliability_plot(y_true, y_prob_max, y_pred, out_name):
    correct = np.array([int(t == p) for t, p in zip(y_true, y_pred)])
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(y_prob_max, bins) - 1

    conf = []
    acc = []
    for i in range(10):
        idx = np.where(bin_ids == i)[0]
        if len(idx) == 0:
            conf.append((bins[i] + bins[i + 1]) / 2)
            acc.append(0)
            continue
        conf.append(float(np.mean(y_prob_max[idx])))
        acc.append(float(np.mean(correct[idx])))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.plot(conf, acc, marker='o', color='#1f77b4', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    save_fig(out_name)


def make_outputs():
    train_rows = collect_split_rows('train')
    val_rows = collect_split_rows('valid')
    test_rows = collect_split_rows('test')

    # 1) total class distribution
    df_all = pd.DataFrame(train_rows + val_rows + test_rows)
    plt.figure(figsize=(10, 5))
    df_all['label'].value_counts().reindex(BREEDS + ['Other']).plot(kind='bar', color='#1f77b4')
    plt.title('Total Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Image Count')
    save_fig('01_total_class_distribution.png')

    # 2) split/class heatmap
    split_table = pd.crosstab(df_all['split'], df_all['label']).reindex(index=['train', 'valid', 'test'])
    split_table = split_table.reindex(columns=BREEDS + ['Other'], fill_value=0)
    plt.figure(figsize=(11, 4))
    sns.heatmap(split_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Split vs Class Distribution')
    save_fig('02_split_class_heatmap.png')

    # 3-5) sample grids
    sample_grid(train_rows, 'Training Sample Grid', '03_train_samples_grid.png')
    sample_grid(val_rows, 'Validation Sample Grid', '04_valid_samples_grid.png')
    sample_grid(test_rows, 'Test Sample Grid', '05_test_samples_grid.png')

    # 6-7) augmentation previews
    if train_rows:
        augmentation_preview(
            train_rows[0]['path'],
            'Augmentation Preview (Generic)',
            '06_augmentation_preview_generic.png'
        )
        gir_rows = [r for r in train_rows if r['label'] == 'Gir']
        src = gir_rows[0]['path'] if gir_rows else train_rows[min(1, len(train_rows) - 1)]['path']
        augmentation_preview(src, 'Augmentation Preview (Breed Focus)', '07_augmentation_preview_breed_focus.png')

    # 8-9) training curves from history
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding='utf-8'))
        acc = history.get('stage1', {}).get('accuracy', []) + history.get('stage2', {}).get('accuracy', [])
        val_acc = history.get('stage1', {}).get('val_accuracy', []) + history.get('stage2', {}).get('val_accuracy', [])
        loss = history.get('stage1', {}).get('loss', []) + history.get('stage2', {}).get('loss', [])
        val_loss = history.get('stage1', {}).get('val_loss', []) + history.get('stage2', {}).get('val_loss', [])

        if acc and val_acc:
            plt.figure(figsize=(8, 5))
            plt.plot(acc, label='Train Acc')
            plt.plot(val_acc, label='Val Acc')
            plt.title('Training vs Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            save_fig('08_training_accuracy_curve.png')

        if loss and val_loss:
            plt.figure(figsize=(8, 5))
            plt.plot(loss, label='Train Loss')
            plt.plot(val_loss, label='Val Loss')
            plt.title('Training vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            save_fig('09_training_loss_curve.png')

    # Predict on test rows
    model, class_labels = load_model_and_labels()
    x_test, y_true, paths = load_batch(test_rows)
    probs = model.predict(x_test, verbose=0)
    pred_idx = np.argmax(probs, axis=1)
    y_pred = [class_labels[i] for i in pred_idx]
    y_conf = np.max(probs, axis=1)

    # 10) breed confusion matrix
    labels_order = [label for label in class_labels if label in BREEDS + ['Other']]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
    plt.title('Confusion Matrix (All Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_fig('10_confusion_matrix_all_classes.png')

    # 11) normalized confusion matrix
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma', xticklabels=labels_order, yticklabels=labels_order)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_fig('11_confusion_matrix_normalized.png')

    # 12) species confusion matrix
    y_true_species = [BREED_TO_SPECIES.get(y, 'Other') for y in y_true]
    y_pred_species = [BREED_TO_SPECIES.get(y, 'Other') for y in y_pred]
    sp_labels = ['Cattle', 'Buffalo', 'Other']
    cm_sp = confusion_matrix(y_true_species, y_pred_species, labels=sp_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_sp, annot=True, fmt='d', cmap='Greens', xticklabels=sp_labels, yticklabels=sp_labels)
    plt.title('Species-Level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_fig('12_confusion_matrix_species.png')

    # 13) per-class precision/recall/f1
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels_order, zero_division=0)
    x = np.arange(len(labels_order))
    w = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - w, p, width=w, label='Precision')
    plt.bar(x, r, width=w, label='Recall')
    plt.bar(x + w, f, width=w, label='F1')
    plt.xticks(x, labels_order, rotation=20, ha='right')
    plt.ylim(0, 1)
    plt.title('Per-Class Precision/Recall/F1')
    plt.legend()
    save_fig('13_per_class_precision_recall_f1.png')

    # 14) class support bar
    plt.figure(figsize=(9, 4))
    plt.bar(labels_order, s, color='#ff7f0e')
    plt.title('Test Set Class Support')
    plt.xlabel('Class')
    plt.ylabel('Support')
    plt.xticks(rotation=20, ha='right')
    save_fig('14_test_class_support.png')

    # 15) confidence histogram correct vs wrong
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])
    plt.figure(figsize=(8, 5))
    plt.hist(y_conf[correct], bins=15, alpha=0.7, label='Correct')
    plt.hist(y_conf[~correct], bins=15, alpha=0.7, label='Wrong')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    save_fig('15_confidence_histogram_correct_vs_wrong.png')

    # 16) reliability diagram
    reliability_plot(y_true, y_conf, y_pred, '16_reliability_diagram.png')

    # 17) top-k accuracy
    idx_map = {lbl: i for i, lbl in enumerate(class_labels)}
    y_idx_model = np.array([idx_map.get(y, -1) for y in y_true])
    topk = {}
    for k in [1, 2, 3]:
        top_preds = np.argsort(probs, axis=1)[:, -k:]
        hit = [int(yi in top_preds[i]) for i, yi in enumerate(y_idx_model)]
        topk[k] = float(np.mean(hit)) if hit else 0.0
    plt.figure(figsize=(6, 4))
    plt.bar(['Top-1', 'Top-2', 'Top-3'], [topk[1], topk[2], topk[3]], color='#9467bd')
    plt.ylim(0, 1)
    plt.title('Top-k Accuracy')
    save_fig('17_topk_accuracy.png')

    # 18) top confusion pairs
    conf_pairs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            key = f'{t} -> {p}'
            conf_pairs[key] = conf_pairs.get(key, 0) + 1
    top_pairs = sorted(conf_pairs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    plt.figure(figsize=(10, 4))
    if top_pairs:
        plt.bar([k for k, _ in top_pairs], [v for _, v in top_pairs], color='#d62728')
        plt.xticks(rotation=35, ha='right')
    plt.title('Top Misclassification Pairs')
    plt.ylabel('Count')
    save_fig('18_top_misclassification_pairs.png')

    # 19-20) misclassified sample grids
    mis_rows = [
        {'path': Path(paths[i]), 'actual': y_true[i], 'pred': y_pred[i], 'conf': y_conf[i]}
        for i in range(len(y_true)) if y_true[i] != y_pred[i]
    ]
    mis_rows = sorted(mis_rows, key=lambda r: r['conf'], reverse=True)

    for idx, out_name in enumerate(['19_misclassified_samples_grid_1.png', '20_misclassified_samples_grid_2.png']):
        start = idx * 12
        chunk = mis_rows[start:start + 12]
        if not chunk:
            continue
        plt.figure(figsize=(12, 9))
        for i, row in enumerate(chunk, 1):
            ax = plt.subplot(3, 4, i)
            img = Image.open(row['path']).convert('RGB').resize((224, 224))
            ax.imshow(img)
            ax.set_title(f"A:{row['actual']}\nP:{row['pred']} ({row['conf']:.2f})", fontsize=8)
            ax.axis('off')
        plt.suptitle(f'Misclassified Samples Grid {idx + 1}')
        save_fig(out_name)

    # Save text report too (useful for paper tables)
    report = classification_report(y_true, y_pred, labels=labels_order, zero_division=0)
    (OUT_DIR / 'classification_report.txt').write_text(report, encoding='utf-8')

    print(f'Generated output images in: {OUT_DIR}')


if __name__ == '__main__':
    make_outputs()

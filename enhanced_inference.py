#!/usr/bin/env python3
"""
Enhanced Inference Pipeline with Post-Processing
-------------------------------------------------
Wrapper for the cattle/buffalo classification model with Gir/Sahiwal refinement.
"""
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class CattleBuffaloClassifier:
    """Enhanced inference with post-processing for better Gir detection"""

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
            'Gir',
            'Holstein_Friesian',
            'Jersey',
            'Sahiwal',
            'Jaffrabadi',
            'Murrah',
            'Other',
        ]
        self.gir_idx = 0
        self.sahiwal_idx = 3

        # Post-processing thresholds (tuned based on diagnostic results)
        self.gir_confidence_threshold = 0.35
        self.sahiwal_rejection_threshold = 0.50
        self.gir_vs_sahiwal_boost = 2.0
        self.min_confidence = 0.30

        self.img_height = 224
        self.img_width = 224

    def preprocess_image(self, image_path_or_array):
        """Load and preprocess image"""
        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array).convert('RGB')
        else:
            img = Image.fromarray(np.uint8(image_path_or_array)).convert('RGB')

        img = img.resize(
            (self.img_width, self.img_height), Image.Resampling.BILINEAR
        )
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, 0)

    def predict_with_postprocessing(self, image_path_or_array, return_raw=False):
        """
        Predict breed with post-processing for improved Gir detection.

        Returns:
            dict: {
                'class': predicted class name,
                'confidence': prediction confidence,
                'species': 'Cattle'/'Buffalo'/'Non-target',
                'all_predictions': list of (class_name, prob) for all classes,
                'post_processed': bool indicating if post-processing was applied
            }
        """
        img_data = self.preprocess_image(image_path_or_array)
        predictions = self.model.predict(img_data, verbose=0)[0]

        # Raw prediction
        raw_idx = np.argmax(predictions)
        raw_prob = predictions[raw_idx]

        # Apply post-processing
        final_idx, final_prob, applied = self._apply_postprocessing(predictions)

        # Map to species
        breed_to_species = {
            0: 'Cattle', 1: 'Cattle', 2: 'Cattle', 3: 'Cattle',
            4: 'Buffalo', 5: 'Buffalo', 6: 'Non-target'
        }
        species = (
            breed_to_species.get(final_idx, 'Unknown')
            if final_idx >= 0
            else 'Unknown'
        )

        # All predictions sorted by confidence
        all_preds = [
            (self.class_names[i], float(predictions[i]))
            for i in range(len(self.class_names))
        ]
        all_preds.sort(key=lambda x: x[1], reverse=True)

        result = {
            'class': self.class_names[final_idx] if final_idx >= 0 else 'Unknown',
            'confidence': float(final_prob),
            'species': species,
            'all_predictions': all_preds,
            'post_processed': applied
        }

        if return_raw:
            result['raw_class'] = self.class_names[raw_idx]
            result['raw_confidence'] = float(raw_prob)

        return result

    def _apply_postprocessing(self, predictions):
        """Apply post-processing rules to improve Gir detection"""
        gir_prob = predictions[self.gir_idx]
        sahiwal_prob = predictions[self.sahiwal_idx]
        max_prob = np.max(predictions)
        max_idx = np.argmax(predictions)

        # Rule 1: Gir vs Sahiwal post-processing
        # If Gir is reasonably close to Sahiwal, boost it
        if (
            gir_prob > self.gir_confidence_threshold
            and gir_prob > sahiwal_prob / self.gir_vs_sahiwal_boost
        ):
            return self.gir_idx, gir_prob, True

        # Rule 2: Suppress Sahiwal false positives
        # If Sahiwal is high but Gir exists, consider it ambiguous
        if (
            sahiwal_prob > self.sahiwal_rejection_threshold
            and gir_prob > 0.25
            and abs(sahiwal_prob - gir_prob) < 0.15
        ):
            # Very close call - return None/Unknown
            return -1, max(gir_prob, sahiwal_prob), True

        # Rule 3: Use original prediction if above minimum confidence
        if max_prob > self.min_confidence:
            return (
                max_idx,
                max_prob,
                max_idx == self.gir_idx
                and max_prob <= self.gir_confidence_threshold,
            )
        else:
            return -1, max_prob, True

    def set_postprocessing_params(
        self,
        gir_threshold=None,
        sahiwal_threshold=None,
        gir_boost=None,
        min_conf=None,
    ):
        """Adjust post-processing parameters"""
        if gir_threshold is not None:
            self.gir_confidence_threshold = gir_threshold
        if sahiwal_threshold is not None:
            self.sahiwal_rejection_threshold = sahiwal_threshold
        if gir_boost is not None:
            self.gir_vs_sahiwal_boost = gir_boost
        if min_conf is not None:
            self.min_confidence = min_conf


if __name__ == '__main__':
    # Example usage
    BASE_DIR = os.path.abspath('.')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'cattle_buffalo_mobile.h5')

    print("Initializing Enhanced Classifier...")
    classifier = CattleBuffaloClassifier(MODEL_PATH)

    # Test on a sample image if available
    test_dir = os.path.join(BASE_DIR, 'Cattle-Buffalo-breeds.folder', 'test', 'Gir')
    test_images = list(Path(test_dir).glob('*'))[:3]

    if test_images:
        print("\nTest predictions:")
        for img_path in test_images:
            result = classifier.predict_with_postprocessing(
                str(img_path),
                return_raw=True,
            )
            print(f"\n{img_path.name}:")
            print(f"  Class: {result['class']} ({result['confidence']:.4f})")
            print(f"  Species: {result['species']}")
            print(f"  Post-processed: {result['post_processed']}")
            if result['post_processed']:
                print(
                    f"  Raw class: {result['raw_class']} "
                    f"({result['raw_confidence']:.4f})"
                )
            top3 = ', '.join(
                [f'{c}:{p:.3f}' for c, p in result['all_predictions'][:3]]
            )
            print(f"  Top-3: {top3}")

    print("\n✅ Enhanced Classifier ready for deployment")

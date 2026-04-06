# Model Artifacts

This folder stores trained model files used by backend inference.

## Files

- `cattle_buffalo_mobile.h5`
  - Main trained TensorFlow/Keras model.
- `class_labels.json`
  - Class label ordering expected by inference code.

Typical current labels:

- `Gir`
- `Holstein_Friesian`
- `Jersey`
- `Sahiwal`
- `Jaffrabadi`
- `Murrah`
- `Other`

## How These Files Are Used

- Backend calls `ml/predict.py`.
- Inference loads both model and labels.
- Class probabilities are mapped using `class_labels.json` order.

## Regenerating Artifacts

From project root:

```bash
source .venv311/bin/activate
python -m ml.train
```

This command updates:

- `models/cattle_buffalo_mobile.h5`
- `models/class_labels.json`

## Important Note

Do not rename these files unless you also update paths in inference and backend code.

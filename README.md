# MNIST MLP Classifier

A single-file scikit-learn project that trains a multilayer perceptron on the
MNIST handwritten digit dataset and evaluates it on the standard test split.

## Results

| Metric | Value |
| --- | --- |
| Test accuracy | **0.9645** (96.45%) |
| Training images | 60,000 |
| Test images | 10,000 |

Recorded in [`metrics.json`](metrics.json) after every training run.

## Model

- `MLPClassifier` with a single hidden layer of **64** units
- `relu` activation, `max_iter=30`, `random_state=42`
- Input pixels flattened to 784 features and scaled to `[0, 1]`
- Default scikit-learn Adam optimizer (`alpha=1e-4`, `batch_size=auto`)

## Project layout

```
train_mlp.py     # training + evaluation script
mnist.npz        # MNIST arrays (x_train/y_train, x_test/y_test)
model.joblib     # trained model, written by train_mlp.py
metrics.json     # test accuracy from the last run
requirements.txt # pinned dependencies
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

```bash
python train_mlp.py
```

The script prints the dataset sizes, then scikit-learn's per-iteration training
log (`verbose=True`), then the final test accuracy. On completion it writes
`model.joblib` and `metrics.json`.

Training takes roughly 1-2 minutes on a single CPU core. There is no GPU
requirement — scikit-learn runs the MLP on the CPU.

## Loading the trained model

```python
import joblib

model = joblib.load("model.joblib")
predictions = model.predict(images.reshape(len(images), -1) / 255.0)
```

Input images must be flattened to 784 features and scaled to `[0, 1]` to match
training, since scikit-learn applies no preprocessing pipeline of its own.

## Next steps

Accuracy is limited by the tiny hidden layer and the 30-iteration cap. Try
`hidden_layer_sizes=(128, 64)`, `max_iter=100`, and an explicit
`solver="adam"` learning rate schedule to push past 98%.

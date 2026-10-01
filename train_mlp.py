import json

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

DATA_FILE = "mnist.npz"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"


def load_data():
    with np.load(DATA_FILE) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_data()
    print("train images:", len(x_train))
    print("test images :", len(x_test))
    print()

    model = MLPClassifier(
        hidden_layer_sizes=64,
        activation="relu",
        max_iter=30,
        random_state=42,
        verbose=True,
    )
    model.fit(x_train, y_train)

    test_accuracy = model.score(x_test, y_test)
    print()
    print("test accuracy: %.4f" % test_accuracy)

    joblib.dump(model, MODEL_FILE)
    with open(METRICS_FILE, "w") as f:
        json.dump({"test_accuracy": test_accuracy}, f, indent=2)
    print("saved %s and %s" % (MODEL_FILE, METRICS_FILE))


if __name__ == "__main__":
    main()

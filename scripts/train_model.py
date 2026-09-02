"""Train a small Keras MLP to classify asana from joint angles.

Free & fast: generates a synthetic dataset by sampling around each asana's
ideal angles (from app/agent/pose_rules.py) with gaussian noise, so there
is no dependency on a labeled video dataset to get a working model.
Swap `generate_synthetic_dataset` for real extracted-landmark data later.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.agent.pose_rules import ASANA_RULES
from app.cv.pose_classifier import FEATURE_JOINTS, ASANAS, MODEL_PATH


def generate_synthetic_dataset(samples_per_class: int = 500):
    X, y = [], []
    for idx, asana in enumerate(ASANAS):
        rules = ASANA_RULES[asana]
        for _ in range(samples_per_class):
            row = []
            for joint in FEATURE_JOINTS:
                ideal, tol = rules.get(joint, (90, 40))
                noisy = np.random.normal(ideal, tol * 0.6)
                row.append(np.clip(noisy, 0, 180) / 180.0)
            X.append(row)
            y.append(idx)
    return np.array(X, dtype="float32"), np.array(y, dtype="int64")


def main():
    import tensorflow as tf

    X, y = generate_synthetic_dataset()
    n = len(X)
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]
    split = int(n * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_JOINTS),)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(ASANAS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32, verbose=2)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

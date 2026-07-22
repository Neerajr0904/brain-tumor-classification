"""
Training script for the Brain Tumor Classification model.

Expects a dataset laid out like the popular Kaggle "Brain Tumor MRI Dataset"
(masoudnickparvar), i.e.:

    data/Training/glioma/...
    data/Training/meningioma/...
    data/Training/notumor/...
    data/Training/pituitary/...
    data/Testing/glioma/...
    data/Testing/meningioma/...
    data/Testing/notumor/...
    data/Testing/pituitary/...

Usage:
    python -m src.train
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    TRAIN_DIR,
    VAL_DIR,
)
from src.model import build_model


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen


def plot_training_history(history, out_path="model/training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def evaluate_model(model, val_gen):
    val_gen.reset()
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    return cm


def main():
    train_gen, val_gen = build_data_generators()

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_training_history(history)
    evaluate_model(model, val_gen)

    print(f"\nBest model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

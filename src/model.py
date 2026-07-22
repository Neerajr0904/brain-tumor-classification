"""
Model definition: EfficientNetB0 backbone with a custom classification head,
used for 4-class brain tumor classification (glioma, meningioma, pituitary,
no tumor).
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from src.config import IMG_SIZE, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, fine_tune_at: int | None = None):
    """
    Build an EfficientNetB0-based classifier via transfer learning.

    Args:
        num_classes: number of output classes.
        fine_tune_at: if given, unfreezes backbone layers from this index
            onward for fine-tuning. If None, the backbone stays fully frozen
            (feature-extraction mode).
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="brain_tumor_efficientnet")
    return model

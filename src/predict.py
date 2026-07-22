"""
Inference helper: loads the trained model once and exposes a simple
`predict(image)` function used by the Streamlit app.
"""

import tensorflow as tf

from src.config import CLASS_NAMES, MODEL_SAVE_PATH
from src.preprocess import load_and_preprocess_image, postprocess_prediction

_model = None


def get_model():
    """Load the trained model once and cache it."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    return _model


def predict(image_path_or_array):
    """
    Run inference on a single image.

    Returns:
        predicted_class (str), confidence (float 0-1),
        class_probabilities (dict[str, float])
    """
    model = get_model()
    processed = load_and_preprocess_image(image_path_or_array)
    raw_output = model.predict(processed)
    return postprocess_prediction(raw_output, CLASS_NAMES)

"""
Preprocessing utilities for MRI images.

Used by both the training pipeline and the inference / Streamlit app so that
images are handled identically in both places.
"""

import numpy as np
import cv2

from src.config import IMG_SIZE


def load_and_preprocess_image(image_path_or_array):
    """
    Load an MRI image (from a file path, bytes, or an already-loaded numpy
    array) and prepare it for the model: resize, convert to RGB, normalize,
    and add the batch dimension.

    Returns a numpy array of shape (1, H, W, 3) with values in [0, 1].
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path_or_array}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_path_or_array)
        if image.ndim == 2:  # grayscale -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:  # RGBA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # batch dimension
    return image


def postprocess_prediction(pred_probs, class_names):
    """
    Convert raw model output (softmax probabilities) into a
    (predicted_class, confidence, all_class_probabilities) tuple.
    """
    pred_probs = pred_probs[0]
    predicted_index = int(np.argmax(pred_probs))
    predicted_class = class_names[predicted_index]
    confidence = float(pred_probs[predicted_index])
    class_probabilities = {
        class_names[i]: float(pred_probs[i]) for i in range(len(class_names))
    }
    return predicted_class, confidence, class_probabilities

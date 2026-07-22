"""
Shared configuration for the Brain Tumor Classification project.
"""

IMG_SIZE = (224, 224)          # EfficientNetB0 default input size
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Training hyperparameters
EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5

# Paths (adjust to match your local dataset layout)
TRAIN_DIR = "data/Training"
VAL_DIR = "data/Testing"
MODEL_SAVE_PATH = "model/brain_tumor_effnet.keras"

import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/recynexa_mobilenetv2_finetuned.keras"

DATASET_DIR = "dataset"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

SEED = 42


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading RecyNexa model...")

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(
    "models/class_names.json",
    "r"
) as file:

    class_names = json.load(file)


print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ==========================================
# LOAD SAME VALIDATION SET
# ==========================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    validation_split=VALIDATION_SPLIT,

    subset="validation",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=True
)


# ==========================================
# GET TRUE LABELS
# ==========================================

true_labels = []

predicted_labels = []


print("\nRunning evaluation...")


for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted
    )


true_labels = np.array(true_labels)

predicted_labels = np.array(
    predicted_labels
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n")
print("=" * 60)
print("RECY NEXA CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    true_labels,
    predicted_labels
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(matrix)


# ==========================================
# OVERALL ACCURACY
# ==========================================

accuracy = np.mean(
    true_labels == predicted_labels
)

print("\n")
print("=" * 60)
print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)
print("=" * 60)
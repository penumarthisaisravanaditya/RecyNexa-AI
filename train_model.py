import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.20
SEED = 42

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)


# ============================================================
# 3. GET CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nNumber of classes: {len(class_names)}")


# Save class names
with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "w"
) as file:
    json.dump(class_names, file, indent=4)


# ============================================================
# 4. CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = np.zeros(len(class_names), dtype=int)

for _, labels in train_dataset:
    for label in labels.numpy():
        class_counts[label] += 1

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.repeat(
        np.arange(len(class_names)),
        class_counts
    )
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass distribution:")

for i, name in enumerate(class_names):
    print(
        f"{name:12} : "
        f"{class_counts[i]} images | "
        f"weight = {class_weights[i]:.3f}"
    )


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")


# ============================================================
# 6. PREPARE DATA
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# 8. BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 9. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. SHOW MODEL
# ============================================================

print("\nModel architecture:")

model.summary()


# ============================================================
# 11. TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "recynexa_mobilenetv2.keras"
)

model.save(model_path)

print("\nModel saved successfully:")
print(model_path)


# ============================================================
# 13. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("RecyNexa Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "accuracy.png"
    )
)

# plt.show()


# ============================================================
# 14. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("RecyNexa Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "loss.png"
    )
)

# plt.show()


print("\n====================================")
print("RECY NEXA TRAINING COMPLETED")
print("====================================")
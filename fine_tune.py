import tensorflow as tf
import numpy as np
import json
import os

from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATASET_DIR = "dataset"
MODEL_PATH = "models/recynexa_mobilenetv2.keras"
OUTPUT_MODEL = "models/recynexa_mobilenetv2_finetuned.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

print("Classes:")
print(class_names)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False
)

# --------------------------------------------------
# PERFORMANCE
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# CALCULATE CLASS WEIGHTS
# --------------------------------------------------

labels = []

for images, batch_labels in train_ds:
    labels.extend(batch_labels.numpy())

labels = np.array(labels)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=labels
)

class_weights = {
    i: weight
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
for i, name in enumerate(class_names):
    print(name, ":", round(class_weights[i], 3))

# --------------------------------------------------
# LOAD EXISTING MODEL
# --------------------------------------------------

print("\nLoading existing model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Existing model loaded successfully.")

# --------------------------------------------------
# FIND MOBILENETV2 BASE MODEL
# --------------------------------------------------

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise RuntimeError("MobileNetV2 base model could not be found.")

print("\nBase model found:", base_model.name)

# --------------------------------------------------
# FINE-TUNING
# --------------------------------------------------

base_model.trainable = True

# Freeze most of MobileNetV2
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("\nFine-tuning last 30 layers of MobileNetV2...")

# --------------------------------------------------
# COMPILE
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7
    ),

    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        monitor="val_accuracy",
        save_best_only=True
    )
]

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nStarting fine-tuning...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

model.save(OUTPUT_MODEL)

print("\n======================================")
print("Fine-tuning completed!")
print("Model saved to:")
print(OUTPUT_MODEL)
print("======================================")

# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------

loss, accuracy = model.evaluate(val_ds, verbose=1)

print("\nFinal Validation Accuracy:")
print(f"{accuracy * 100:.2f}%")
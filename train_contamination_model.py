
from pathlib import Path
import json
import tensorflow as tf

# Dataset and output paths
DATA_DIR = Path("contamination_data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
SEED = 42

# Load training and validation data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.25,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["clean", "contaminated"],
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.25,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["clean", "contaminated"],
)

print("Class order:", ["clean", "contaminated"])

# MobileNetV2 expects its input pixels scaled to [-1, 1].
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
)

model_path = MODEL_DIR / "contamination_mobilenetv2.keras"
model.save(model_path)

with open(MODEL_DIR / "contamination_class_names.json", "w") as f:
    json.dump(["clean", "contaminated"], f)

print(f"\nModel saved to: {model_path}")
print("Training complete.")
print(
    "\nWARNING: This is only a pipeline test using 40 AI-generated images. "
    "Its accuracy does not demonstrate real-world contamination detection."
)


from pathlib import Path
import tensorflow as tf

model_path = Path("models/contamination_mobilenetv2.keras")
image_root = Path("contamination_data")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

for folder_name in ["clean", "contaminated"]:
    folder = image_root / folder_name
    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not images:
        print(f"No images found in {folder}")
        continue

    image_path = images[0]

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array, verbose=0)[0][0])

    if probability >= 0.5:
        prediction = "contaminated"
        confidence = probability * 100
    else:
        prediction = "clean"
        confidence = (1 - probability) * 100

    print(f"\nImage: {image_path.name}")
    print(f"Predicted class: {prediction}")
    print(f"Model score: {confidence:.2f}%")

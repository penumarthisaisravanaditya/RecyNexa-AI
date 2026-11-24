
from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("contamination_data")
MODEL_PATH = Path("models/contamination_mobilenetv2.keras")
IMAGE_SIZE = (224, 224)
EXTENSIONS = {".jpg", ".jpeg", ".png"}

model = tf.keras.models.load_model(MODEL_PATH)

correct = 0
total = 0

print("EXPERIMENTAL CONTAMINATION MODEL")
print("=" * 45)

for folder_name in ["clean", "contaminated"]:
    folder = DATA_DIR / folder_name
    image_paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    folder_correct = 0

    for image_path in image_paths:
        image = tf.keras.utils.load_img(
            image_path, target_size=IMAGE_SIZE
        )
        array = tf.keras.utils.img_to_array(image)
        array = tf.expand_dims(array, axis=0)

        score = float(model.predict(array, verbose=0)[0][0])
        predicted = "contaminated" if score >= 0.5 else "clean"

        is_correct = predicted == folder_name
        correct += int(is_correct)
        total += 1
        folder_correct += int(is_correct)

        print(
            f"{image_path.name}: predicted={predicted}, "
            f"score={score:.3f}, expected={folder_name}, "
            f"correct={is_correct}"
        )

    print(
        f"\n{folder_name}: {folder_correct}/{len(image_paths)} "
        "classified correctly"
    )

print("\n" + "=" * 45)
print(f"Starter-set accuracy: {correct}/{total}")

if total:
    print(f"Accuracy: {100 * correct / total:.1f}%")

print(
    "\nLIMITATION: These are AI-generated starter images, "
    "not an independent real-world test set. The score does "
    "not establish real-world contamination detection accuracy."
)

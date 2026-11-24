import json
import numpy as np
import tensorflow as tf
from PIL import Image


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/recynexa_mobilenetv2_finetuned.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMAGE_SIZE = (224, 224)

# If confidence is below 70%, return UNKNOWN
CONFIDENCE_THRESHOLD = 0.70


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading RecyNexa AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("Classes:", class_names)


# ==========================================
# RECYCLING RECOMMENDATIONS
# ==========================================

recommendations = {

    "plastic":
        "Place in the plastic/recyclable waste stream. "
        "Clean the item before recycling.",

    "paper":
        "Place in the paper recycling stream. "
        "Keep paper dry and free from food contamination.",

    "cardboard":
        "Place in the cardboard recycling stream. "
        "Flatten the cardboard and keep it dry.",

    "glass":
        "Place in the glass recycling stream. "
        "Handle broken glass carefully.",

    "metal":
        "Place in the metal recycling stream. "
        "Clean the container before recycling.",

    "organic":
        "Place in the organic/compostable waste stream.",

    "e_waste":
        "Do not place in regular waste. "
        "Take electronic waste to an authorized "
        "e-waste collection point."
}


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_waste(image_path):

    # --------------------------------------
    # Load image
    # --------------------------------------

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------
    # Resize image
    # --------------------------------------

    image = image.resize(IMAGE_SIZE)

    # --------------------------------------
    # Convert image to NumPy array
    # --------------------------------------

    image_array = np.array(image, dtype=np.float32)

    # --------------------------------------
    # Add batch dimension
    # Shape becomes:
    # (1, 224, 224, 3)
    # --------------------------------------

    image_array = np.expand_dims(image_array, axis=0)

    # --------------------------------------
    # IMPORTANT
    #
    # DO NOT use:
    #
    # tf.keras.applications.mobilenet_v2.preprocess_input()
    #
    # here.
    #
    # The trained model already contains the
    # MobileNetV2 preprocessing layer.
    # --------------------------------------

    # --------------------------------------
    # Make prediction
    # --------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------
    # Get highest probability
    # --------------------------------------

    predicted_index = np.argmax(predictions[0])

    confidence = float(
        predictions[0][predicted_index]
    )

    predicted_class = class_names[predicted_index]

    # --------------------------------------
    # Get Top 3 Predictions
    # --------------------------------------

    top_indices = np.argsort(
        predictions[0]
    )[::-1][:3]

    print("\nTop 3 predictions:")

    for index in top_indices:

        print(
            f"{class_names[index]:12} : "
            f"{predictions[0][index] * 100:.2f}%"
        )

    # --------------------------------------
    # Confidence-aware decision
    # --------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return {
            "class": "unknown",

            "confidence": confidence,

            "message":
                "The model is not confident enough "
                "to classify this image."
        }

    # --------------------------------------
    # Return valid prediction
    # --------------------------------------

    return {

        "class": predicted_class,

        "confidence": confidence,

        "message":
            recommendations.get(
                predicted_class,
                "No recycling recommendation available."
            )
    }


# ==========================================
# COMMAND LINE TEST
# ==========================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter the path of a waste image: "
    )

    try:

        result = predict_waste(image_path)

        print("\n================================")
        print("       RECY NEXA AI")
        print("================================")

        print(
            f"Prediction : {result['class']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            f"Recommendation : "
            f"{result['message']}"
        )

        print("================================")

    except FileNotFoundError:

        print("\nError: Image file not found.")

    except Exception as e:

        print("\nError:", e)
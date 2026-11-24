import pandas as pd
from database import save_scan, get_all_scans, clear_scan_history
import hashlib
import os

import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image



# ============================================================
# RECY NEXA AI
# Intelligent Waste Segregation & Recycling Assistant
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/recynexa_mobilenetv2_finetuned.keras"
CLASS_NAMES_PATH = "models/class_names.json"
CONTAMINATION_MODEL_PATH = "models/contamination_mobilenetv2.keras"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RecyNexa AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --rx-bg: #071411;
        --rx-panel: #10231d;
        --rx-panel-2: #142b23;
        --rx-line: rgba(148, 190, 165, .18);
        --rx-text: #edf8f0;
        --rx-muted: #a7c2b1;
        --rx-green: #67e6a5;
        --rx-mint: #b7f7d2;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        color: var(--rx-text);
        background:
          radial-gradient(ellipse at 8% 0%, rgba(34,197,94,.16), transparent 34%),
          radial-gradient(ellipse at 95% 10%, rgba(20,184,166,.12), transparent 30%),
          linear-gradient(145deg, #071411 0%, #0a1915 48%, #07120f 100%);
    }
    .block-container { max-width: 1420px; padding-top: 1.6rem; padding-bottom: 2.8rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; color: #f1fff5; }
    p, label, .stMarkdown { color: #d7e9dd; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #10251d 0%, #091711 100%); border-right: 1px solid var(--rx-line); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #c4ddcc; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(23,55,42,.94), rgba(13,34,26,.96));
        border: 1px solid rgba(103,230,165,.22); border-radius: 18px;
        padding: 17px 18px; box-shadow: 0 10px 30px rgba(0,0,0,.16);
    }
    [data-testid="stMetricLabel"] { color: #a9c9b5 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #effff3 !important; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] {
        background: rgba(17,43,32,.72); border: 1px dashed rgba(103,230,165,.42);
        border-radius: 18px; padding: 12px;
    }
    [data-testid="stFileUploader"] section { background: transparent; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px; border: 1px solid rgba(103,230,165,.38);
        background: linear-gradient(135deg, #1c8b5a, #126b50); color: white;
        font-weight: 700; padding: .55rem 1rem; transition: all .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #9bf2bd; color: white; transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(26,170,103,.2);
    }
    [data-testid="stRadio"] > div { gap: .55rem; }
    [data-testid="stRadio"] label {
        background: rgba(17,43,32,.62); border: 1px solid var(--rx-line);
        padding: .45rem .8rem; border-radius: 12px;
    }
    [data-testid="stChatMessage"] {
        background: rgba(18,43,33,.78); border: 1px solid var(--rx-line);
        border-radius: 16px; padding: 12px 15px; margin-bottom: 10px;
    }
    [data-testid="stChatInput"] textarea {
        background: #10251c; color: #effff3; border: 1px solid rgba(103,230,165,.35); border-radius: 14px;
    }
    [data-testid="stExpander"] { background: rgba(15,39,29,.68); border: 1px solid var(--rx-line); border-radius: 15px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--rx-line); border-radius: 14px; overflow: hidden; }
    hr { border-color: rgba(148,190,165,.18); }
    [data-testid="stAlert"] { border-radius: 14px; }
    .rx-hero {
        position: relative; overflow: hidden; padding: 32px 34px; border-radius: 26px;
        background: linear-gradient(115deg, rgba(18,74,49,.96), rgba(13,48,42,.94) 58%, rgba(13,57,54,.92));
        border: 1px solid rgba(133,244,180,.30); box-shadow: 0 20px 55px rgba(0,0,0,.24);
        margin: 4px 0 22px 0;
    }
    .rx-hero:after { content: '♻'; position: absolute; right: 4%; top: -45px; font-size: 190px; color: rgba(198,255,218,.07); transform: rotate(-15deg); }
    .rx-kicker { color: #9df4bd; text-transform: uppercase; letter-spacing: .18em; font-size: .76rem; font-weight: 800; }
    .rx-title { font: 700 clamp(2rem, 4vw, 3.25rem)/1.05 'Space Grotesk', sans-serif; color: #f4fff7; margin: 12px 0 10px; }
    .rx-subtitle { color: #c5e8d1; font-size: 1.05rem; max-width: 760px; line-height: 1.65; }
    .rx-pill { display: inline-block; border: 1px solid rgba(171,255,202,.32); background: rgba(7,28,19,.32); color: #d9ffe5; padding: 7px 12px; border-radius: 999px; font-size: .78rem; margin: 18px 7px 0 0; }
    .rx-section-tag { color: #8cebb0; text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 800; margin-bottom: 4px; }
    .rx-footer { text-align:center; padding: 25px 12px 8px; color:#89aa96; }
    .rx-footer strong { color:#c7f8d7; font-family:'Space Grotesk',sans-serif; font-size:1.15rem; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model



@st.cache_resource
def load_contamination_model():
    """Load the separate experimental clean/contaminated classifier."""
    return tf.keras.models.load_model(CONTAMINATION_MODEL_PATH)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        class_names = json.load(file)

    return class_names


# ============================================================
# LOAD MODEL AND CLASSES
# ============================================================

try:

    model = load_model()
    class_names = load_class_names()

except Exception as e:

    st.error(
        "Unable to load the RecyNexa AI model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# RECYCLING RECOMMENDATIONS
# ============================================================

recommendations = {

    "plastic": {

        "Clean":
            "Place the item in the plastic recycling stream. "
            "Empty and clean the item before recycling.",

        "Contaminated":
            "Clean the plastic item and remove food or liquid "
            "residue before recycling."
    },


    "paper": {

        "Clean":
            "Place the item in the paper recycling stream. "
            "Keep paper clean and dry.",

        "Contaminated":
            "Separate wet or food-contaminated paper. "
            "Only clean and dry paper should enter the "
            "paper recycling stream."
    },


    "cardboard": {

        "Clean":
            "Flatten the cardboard and place it in the "
            "cardboard recycling stream.",

        "Contaminated":
            "Remove food contamination and flatten the "
            "cardboard before recycling."
    },


    "glass": {

        "Clean":
            "Place the glass item in the glass recycling stream."
            "Handle broken glass carefully.",

        "Contaminated":
            "Empty and rinse the glass container before recycling. "
            "Handle broken glass carefully."
    },


    "metal": {

        "Clean":
            "Place the metal item in the metal recycling stream.",

        "Contaminated":
            "Empty and clean the metal container before recycling."
    },


    "organic": {

        "Clean":
            "Place the material in the appropriate "
            "organic or compostable waste stream.",

        "Contaminated":
            "Remove non-organic materials before composting "
            "or organic disposal."
    },


    "e_waste": {

        "Clean":
            "Do not place electronic waste in regular household "
            "waste. Take it to an authorized e-waste collection point.",

        "Contaminated":
            "Do not place electronic waste in regular household "
            "waste. Use an authorized e-waste collection facility."
    }
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# ♻️ RecyNexa AI"
    )

    st.divider()

    st.markdown(
        "## About"
    )

    st.write(
        """
        RecyNexa AI is an intelligent waste
        segregation and recycling assistant
        based on deep learning and computer vision.
        """
    )

    st.divider()

    st.markdown(
        "## 🤖 AI Model"
    )

    st.write(
        "**Architecture:** MobileNetV2"
    )

    st.write(
        "**Learning:** Transfer Learning + Fine-Tuning"
    )

    st.write(
        "**Input:** 224 × 224 RGB image"
    )

    st.write(
        "**Classes:** 7"
    )

    st.divider()

    st.markdown(
        "## ♻️ Waste Categories"
    )

    category_display_names = {

        "cardboard": "Cardboard",
        "e_waste": "E-Waste",
        "glass": "Glass",
        "metal": "Metal",
        "organic": "Organic",
        "paper": "Paper",
        "plastic": "Plastic"
    }

    for category in class_names:

        display_name = category_display_names.get(
            category,
            category.replace("_", " ").title()
        )

        st.write(
            f"• {display_name}"
        )

    st.divider()

    st.markdown(
        "## 📊 Current Prototype"
    )

    st.write(
        "✓ Waste classification"
    )

    st.write(
        "✓ Confidence analysis"
    )

    st.write(
        "✓ Top-3 predictions"
    )

    st.write(
        "✓ Contamination-aware guidance"
    )

    st.write(
        "✓ Recycling decision support"
    )

    st.divider()

    st.caption(
        "RecyNexa AI"
    )

    st.caption(
        "Intelligent Waste Segregation & Recycling Assistant"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <section class="rx-hero">
      <div class="rx-kicker">Smart • Sustainable • AI-powered</div>
      <div class="rx-title">RecyNexa AI</div>
      <div class="rx-subtitle">Intelligent Waste Segregation &amp; Recycling Assistant.
      Turn a waste photo into a clear classification, confidence insight and practical next step.</div>
      <span class="rx-pill">♻ 7 waste categories</span>
      <span class="rx-pill">◈ MobileNetV2</span>
      <span class="rx-pill">✓ Confidence-aware results</span>
    </section>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="rx-section-tag">Platform overview</div>', unsafe_allow_html=True)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        "AI Architecture",
        "MobileNetV2"
    )


with info2:

    st.metric(
        "Waste Classes",
        "7"
    )


with info3:

    st.metric(
        "Input Size",
        "224 × 224"
    )


with info4:

    st.metric(
        "Confidence Threshold",
        "70%"
    )


st.divider()


page = st.radio(
    "Navigation",
    ["Waste Scanner", "Analytics Dashboard"],
    horizontal=True
)

if page == "Analytics Dashboard":
    st.markdown('<div class="rx-section-tag">Insights & activity</div>', unsafe_allow_html=True)
    st.header("📊 Scan History & Analytics")
    columns = ["ID", "Scanned At", "Filename", "Predicted Class", "Confidence", "Status", "Contamination", "Recommendation"]
    history = pd.DataFrame(get_all_scans(), columns=columns)
    if history.empty:
        st.info("No scans saved yet. Scan an image to begin.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total Scans", len(history))
        b.metric("Identified", int((history["Status"] == "Identified").sum()))
        c.metric("Low Confidence", int((history["Status"] == "Low Confidence").sum()))
        st.subheader("Waste Category Distribution")
        identified = history.loc[history["Status"] == "Identified"]
        if not identified.empty:
            st.bar_chart(identified["Predicted Class"].value_counts())
        st.subheader("Scan Records")
        display_history = history.copy()
        display_history["Confidence"] = display_history["Confidence"].map(lambda value: f"{value * 100:.2f}%")
        st.dataframe(display_history, use_container_width=True, hide_index=True)
        st.download_button("Export CSV", history.to_csv(index=False).encode("utf-8"), "recynexa_history.csv", "text/csv")
        with st.expander("Delete history"):
            confirm = st.checkbox("I understand this deletes allrecords")
            if st.button("Delete All Records", disabled=not confirm):
                clear_scan_history()
                st.rerun()

if page == "Waste Scanner":
    # ============================================================
    # IMAGE INPUT
    # ============================================================

    st.markdown('<div class="rx-section-tag">01 / Image intelligence</div>', unsafe_allow_html=True)
    st.header("📷 Waste Image Analysis")
    st.caption("Upload a clear photo or use your camera. RecyNexa will analyze the item and guide the next step.")

    input_method = st.radio(
        "Select image source",
        [
            "Upload Image",
            "Use Camera"
        ],
        horizontal=True
    )


    image = None


    # ============================================================
    # UPLOAD IMAGE
    # ============================================================

    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Choose a waste image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help="Upload a clear image of a waste item."
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            )


    # ============================================================
    # CAMERA
    # ============================================================

    else:

        camera_image = st.camera_input(
            "Take a picture of the waste item"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            )


    # ============================================================
    # IMAGE PROCESSING AND PREDICTION
    # ============================================================

    if image is not None:

        image_name = uploaded_file.name if input_method == "Upload Image" else "camera_capture.jpg"

        st.divider()

        # Convert image to RGB
        image_rgb = image.convert(
            "RGB"
        )


        # --------------------------------------------------------
        # PROCESS IMAGE
        # --------------------------------------------------------

        with st.spinner(
            "Analyzing waste image..."
        ):

            resized_image = image_rgb.resize(
                IMAGE_SIZE
            )

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Model prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # Highest probability
            predicted_index = np.argmax(
                predictions
            )

            confidence = float(
                predictions[predicted_index]
            )

            predicted_class = class_names[
                predicted_index
            ]

            # Top 3
            top_indices = np.argsort(
                predictions
            )[::-1][:3]


        # ========================================================
        # IMAGE + AI RESULT
        # ========================================================

        image_column, result_column = st.columns(
            [1.1, 1],
            gap="large"
        )


        # --------------------------------------------------------
        # INPUT IMAGE
        # --------------------------------------------------------

        with image_column:

            st.subheader(
                "📸 Input Image"
            )

            st.image(
                image_rgb,
                width="stretch"
            )


        # --------------------------------------------------------
        # AI ANALYSIS
        # --------------------------------------------------------

        with result_column:

            st.subheader(
                "🤖 AI Analysis"
            )

            # High confidence
            if confidence >= CONFIDENCE_THRESHOLD:

                st.success(
                    "✓ Waste identified successfully"
                )

                metric1, metric2 = st.columns(2)

                with metric1:

                    st.metric(
                        "Predicted Category",
                        predicted_class.upper()
                    )

                with metric2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                st.progress(
                    confidence
                )

                st.caption(
                    f"Confidence threshold: "
                    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
                )

            # Low confidence
            else:

                st.warning(
                    "⚠️ Low Confidence / Unknown Item"
                )

                st.metric(
                    "Highest Prediction",
                    predicted_class.upper()
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.info(
                    "The model is not confident enough to "
                    "reliably assign this image to a known "
                    "waste category."
                )

                scan_key = (
                    hashlib.sha256(image_rgb.tobytes()).hexdigest(),
                    "Low Confidence"
                )
                if st.session_state.get("last_saved_scan_key") != scan_key:
                    save_scan(
                        image_name,
                        predicted_class,
                        confidence,
                        "Low Confidence"
                    )
                    st.session_state["last_saved_scan_key"] = scan_key

                # Continue to show the experimental contamination result,
                # but do not present recycling guidance for an uncertain class.


        # ========================================================
        # TOP 3 PREDICTIONS
        # ========================================================

        st.divider()

        st.header(
            "📊 Prediction Analysis"
        )

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            class_name = class_names[
                index
            ]

            probability = float(
                predictions[index]
            )

            display_name = category_display_names.get(
                class_name,
                class_name.replace(
                    "_",
                    " "
                ).title()
            )

            prediction_col, progress_col = st.columns(
                [2, 5]
            )

            with prediction_col:

                st.write(
                    f"**#{rank} {display_name}**"
                )

            with progress_col:

                st.progress(
                    probability
                )

                st.caption(
                    f"{probability * 100:.2f}%"
                )


        # ========================================================
        # EXPERIMENTAL CONTAMINATION ASSESSMENT
        # ========================================================

        st.divider()
        st.header("🧪 Contamination Assessment")

        st.warning(
            "Experimental prototype: this model was trained using only "
            "40 AI-generated starter images. Its predictions have not been "
            "validated for real-world food residue, dirt, or liquid. "
            "Please visually inspect the item before recycling."
        )

        contamination_model = None
        contamination = "Unknown"
        contamination_score = None

        try:
            contamination_model = load_contamination_model()

            contamination_image = image_rgb.resize(IMAGE_SIZE)
            contamination_array = np.array(
                contamination_image,
                dtype=np.float32
            )
            contamination_array = np.expand_dims(
                contamination_array,
                axis=0
            )

            # The contamination model contains MobileNetV2 preprocessing
            # internally, so do not preprocess the image a second time.
            raw_score = float(
                contamination_model.predict(
                    contamination_array,
                    verbose=0
                )[0][0]
            )

            contamination = (
                "Contaminated" if raw_score >= 0.5 else "Clean"
            )
            contamination_score = (
                raw_score if raw_score >= 0.5 else 1.0 - raw_score
            )

            st.metric(
                "Experimental Model Prediction",
                contamination
            )
            st.metric(
                "Model Score",
                f"{contamination_score * 100:.2f}%"
            )
            st.progress(max(0.0, min(1.0, contamination_score)))
            st.caption(
                "This score is the model's output for its predicted class, "
                "not a calibrated probability of real-world correctness."
            )
            st.info(
                "If the prediction disagrees with the visible condition, "
                "trust visual inspection. This experimental result should "
                "not be used as a safety or recycling guarantee."
            )

        except Exception as e:
            st.error(
                "The experimental contamination model could not run. "
                "Check that models/contamination_mobilenetv2.keras exists "
                "and can be loaded."
            )
            st.code(str(e))
            st.stop()

        # ========================================================
        # RECYCLING DECISION
        # ========================================================

        st.divider()

        st.header(
            "♻️ Recycling Decision"
        )

        if predicted_class in recommendations and contamination in recommendations[predicted_class]:
            decision = recommendations[predicted_class][contamination]
        else:
            decision = (
                "Inspect the item manually. The waste category or "
                "contamination result is uncertain."
            )

        scan_key = (hashlib.sha256(image_rgb.tobytes()).hexdigest(), contamination)
        if st.session_state.get("last_saved_scan_key") != scan_key:
            save_scan(image_name, predicted_class, confidence, "Identified", contamination, decision)
            st.session_state["last_saved_scan_key"] = scan_key


        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "### Manual review recommended\n\n"
                "The waste classifier is below its confidence threshold. "
                "Do not rely on the recycling recommendation until the "
                "waste category is confirmed."
            )
            st.write(decision)

        elif contamination == "Clean":
            st.success(
                f"### ✓ Suggested Recycling Guidance\n\n{decision}"
            )
        else:
            st.warning(
                f"### ⚠ Preparation May Be Required\n\n{decision}"
            )


        # ========================================================
        # FINAL GUIDANCE
        # ========================================================

        st.markdown('<div class="rx-section-tag">02 / Recommended action</div>', unsafe_allow_html=True)
        st.header("🎯 Final Guidance")


        if predicted_class == "e_waste":

            st.info(
                "🔌 **E-WASTE:** Do not place this item in "
                "regular household waste. Use an authorized "
                "e-waste collection facility."
            )

        elif predicted_class == "organic":

            st.success(
                "🌱 **ORGANIC:** Use the appropriate organic "
                "or compost waste stream."
            )

        elif contamination == "Contaminated":

            st.warning(
                "⚠️ **PREPARE BEFORE RECYCLING:** Clean or "
                "separate the contaminated material before disposal."
            )

        else:

            st.success(
                "♻️ **RECYCLABLE:** Send the material to the "
                "appropriate recycling stream."
            )


        # ========================================================
        # GROQ-POWERED AI RECYCLING ASSISTANT
        # Uses the current scan as context for follow-up answers.
        # ========================================================

        st.divider()
        st.markdown('<div class="rx-section-tag">03 / Conversational guidance</div>', unsafe_allow_html=True)
        st.header("🤖 RecyNexa AI Recycling Assistant")
        st.write(
            "Ask follow-up questions about this scanned item, "
            "recycling, waste segregation, or safe disposal."
        )

        # Never describe an uncertain classification as confirmed.
        scan_is_confident = confidence >= CONFIDENCE_THRESHOLD
        category_for_chat = (
            predicted_class.replace("_", " ").title()
            if scan_is_confident else "Uncertain / not identified"
        )
        contamination_for_chat = contamination
        scan_fingerprint = hashlib.sha256(
            image_rgb.tobytes()
        ).hexdigest()
        chat_context_key = (
            scan_fingerprint, category_for_chat,
            round(confidence, 5), contamination_for_chat
        )

        if st.session_state.get("recy_chat_context_key") != chat_context_key:
            st.session_state["recy_chat_context_key"] = chat_context_key
            st.session_state["recy_chat_messages"] = []

        with st.expander("Current scan context", expanded=True):
            context_col1, context_col2, context_col3 = st.columns(3)
            context_col1.metric("Detected category", category_for_chat)
            context_col2.metric("Classifier confidence", f"{confidence * 100:.2f}%")
            context_col3.metric("Contamination model", contamination_for_chat)
            if not scan_is_confident:
                st.warning(
                    "The waste category is below the 70% confidence threshold. "
                    "The assistant will treat the category as uncertain and "
                    "recommend manual verification."
                )
            st.caption(
                "Contamination assessment is experimental and was trained "
                "on a small AI-generated starter dataset. It is not a reliable "
                "real-world contamination or safety test. Visually inspect the item."
            )

        chat_messages = st.session_state.get("recy_chat_messages", [])
        for chat_message in chat_messages:
            with st.chat_message(chat_message["role"]):
                st.markdown(chat_message["content"])

        groq_api_key = ""
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            groq_api_key = ""
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        user_question = st.chat_input(
            "Ask about this scanned item...",
            key="recy_nexa_chat_input"
        )

        if user_question:
            st.session_state["recy_chat_messages"].append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            if not groq_api_key:
                assistant_reply = (
                    "The AI chat service is not configured yet. Add your Groq API key "
                    "to `.streamlit/secrets.toml` as `GROQ_API_KEY = \"your-key\"`, "
                    "or set the `GROQ_API_KEY` environment variable, then restart Streamlit."
                )
                with st.chat_message("assistant"):
                    st.warning(assistant_reply)
            else:
                try:
                    from groq import Groq

                    system_prompt = f"""You are RecyNexa AI, a concise and practical waste segregation and recycling assistant.
Use the following scan context when answering follow-up questions:
- Predicted waste category: {category_for_chat}
- Classifier confidence: {confidence * 100:.2f}%
- Classification status: {'Identified' if scan_is_confident else 'Low confidence / uncertain'}
- Experimental contamination model output: {contamination_for_chat}

Rules:
1. Clearly distinguish the model's prediction from confirmed facts. If classification confidence is below 70%, say the category is uncertain and ask the user to verify the item visually.
2. The contamination model is experimental and not validated for real-world contamination. Never claim it can reliably detect dirt, food residue, chemicals, or hazards. Treat its output as a weak experimental signal only.
3. Give short, actionable steps. Mention that local recycling rules vary.
4. For batteries, e-waste, chemicals, medicines, sharps, or unknown hazardous contents, advise safe handling and an authorized local collection point; do not recommend putting them in ordinary recycling.
5. Do not invent local collection locations or claim an item is accepted by a local facility without evidence.
6. If the question is unrelated to waste or recycling, politely guide the user back to the assistant's purpose.
"""

                    recent_messages = st.session_state["recy_chat_messages"][-12:]
                    request_messages = [{"role": "system", "content": system_prompt}]
                    request_messages.extend(recent_messages)

                    with st.chat_message("assistant"):
                        with st.spinner("RecyNexa AI is preparing guidance..."):
                            client = Groq(api_key=groq_api_key)
                            completion = client.chat.completions.create(
                                model=groq_model,
                                messages=request_messages,
                                temperature=0.4,
                                max_completion_tokens=600
                            )
                            assistant_reply = completion.choices[0].message.content
                            st.markdown(assistant_reply)

                except ImportError:
                    assistant_reply = "Groq package is missing. Install it with `pip install groq` and restart the app."
                    with st.chat_message("assistant"):
                        st.error(assistant_reply)
                except Exception as error:
                    assistant_reply = (
                        "I could not contact the AI service. Check your Groq API key, "
                        "model availability, internet connection, and account limits. "
                        f"Technical detail: {error}"
                    )
                    with st.chat_message("assistant"):
                        st.error(assistant_reply)

            st.session_state["recy_chat_messages"].append(
                {"role": "assistant", "content": assistant_reply}
            )

        if st.session_state.get("recy_chat_messages"):
            if st.button("Clear conversation", key="clear_recy_chat"):
                st.session_state["recy_chat_messages"] = []
                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="rx-footer">
      <strong>♻ RecyNexa AI</strong><br>
      Intelligent Waste Segregation &amp; Recycling Assistant<br>
      <span style="font-size:.8rem">MobileNetV2 • 7 waste categories • Confidence-aware classification</span>
    </div>
    """,
    unsafe_allow_html=True
)

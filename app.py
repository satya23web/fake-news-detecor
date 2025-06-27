import streamlit as st
from PIL import Image
import io
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoImageProcessor, ViTForImageClassification

# --- IMPORTANT: Secure your Hugging Face Token for deployment! ---
# For Colab, you can still use st.secrets.get() but you'd need to set up Colab Secrets.
# A simpler temporary way in Colab is to directly use the token or environment variables.
# If you're sharing the notebook, it's best to put this in Colab secrets.
# Go to "Runtime" -> "Manage Colab secrets" and add a secret named "HF_TOKEN"
# then uncomment the line below:
# HF_TOKEN = st.secrets.get("HF_TOKEN", "YOUR_ACTUAL_HF_TOKEN_HERE_IF_NOT_IN_SECRETS")

# For quick testing in Colab, you can hardcode, but be aware if sharing publicly:
HF_TOKEN = "hf_TjYAFDIJqfiUbmdduCmXpkhPNkffWDtim" # REPLACE WITH YOUR ACTUAL TOKEN

# --- Cache models to avoid reloading on every rerun ---
@st.cache_resource
def get_text_detector():
    """Load the fake text detection model (Hugging Face pipeline)."""
    try:
        text_pipeline = pipeline(
            "text-classification",
            model="openai-community/roberta-base-openai-detector",
            tokenizer="openai-community/roberta-base-openai-detector",
            use_auth_token=HF_TOKEN
        )
        st.success("Text detection model loaded successfully!")
        return text_pipeline
    except Exception as e:
        st.error(f"Error loading text detection model: {e}. Please ensure your HF_TOKEN is valid and you have internet access.")
        return None

@st.cache_resource
def get_image_detector():
    """Load the fake image detection model (Hugging Face pipeline)."""
    try:
        model_name = "prithivMLmods/deepfake-detector-model-v1"
        image_pipeline = pipeline(
            "image-classification",
            model=model_name,
            use_auth_token=HF_TOKEN
        )
        st.success(f"Image detection model '{model_name}' loaded successfully!")
        return image_pipeline
    except Exception as e:
        st.error(f"Error loading image detection model: {e}. Please ensure your HF_TOKEN is valid, model exists, and you have internet access.")
        return None

# Load models once when the app starts
text_detector = get_text_detector()
image_detector = get_image_detector()


# --- Prediction Functions ---
def detect_fake_text(text_input, model):
    """
    Detects fake text using the loaded Hugging Face model.
    Assumes the 'openai-community/roberta-base-openai-detector' model output format.
    """
    if model is None:
        return "Text detection model not loaded", 0.0

    try:
        result = model(text_input)
        label = result[0]['label']
        score = result[0]['score']

        if label == "Fake":
            return "FAKE", score
        else:
            return "REAL", score
    except Exception as e:
        return f"Error during text detection: {e}", 0.0


def detect_fake_image(image_bytes, model):
    """
    Detects fake images using the loaded Hugging Face model.
    Assumes the 'prithivMLmods/deepfake-detector-model-v1' model output format.
    """
    if model is None:
        return "Image detection model not loaded", 0.0

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = model(img)

        predicted_label = result[0]['label']
        predicted_score = result[0]['score']

        if predicted_label.lower() == "fake":
            return "FAKE", predicted_score
        elif predicted_label.lower() == "real":
            return "REAL", predicted_score
        else:
            return f"Unrecognized label: {predicted_label}", predicted_score

    except Exception as e:
        st.error(f"Debug: Error details - {e}")
        return f"Error during image detection: {e}", 0.0


# --- Streamlit UI ---
st.set_page_config(page_title="Fake Content Detector", layout="centered")
st.title("Fake Content Detector")
st.markdown("Identify potentially AI-generated or manipulated text and images.")

# --- Fake Text Detector Section ---
st.header("✍️ Fake Text Detector")
text_input = st.text_area("Enter text to analyze:", height=200, placeholder="Type or paste text here...")

if st.button("Analyze Text", key="analyze_text_btn"):
    if text_input:
        if text_detector:
            with st.spinner("Analyzing text..."):
                text_label, text_score = detect_fake_text(text_input, text_detector)
                if "Error" in text_label:
                    st.error(text_label)
                else:
                    st.subheader("Text Analysis Result:")
                    if text_label == "FAKE":
                        st.error(f"**Likely FAKE!** (Confidence: {text_score:.2f})")
                        st.markdown("The model suggests this text might be AI-generated or heavily manipulated.")
                    else:
                        st.success(f"**Likely REAL.** (Confidence: {text_score:.2f})")
                        st.markdown("The model suggests this text is likely human-written.")
                    st.progress(text_score)
        else:
            st.warning("Text detection model could not be loaded. Please check the console for errors.")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---") # Separator

# --- Fake Image Detector Section ---
st.header("🖼️ Fake Image Detector")
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"], help="Max file size 200MB.")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image", key="analyze_image_btn"):
        if image_detector:
            with st.spinner("Analyzing image..."):
                image_bytes = uploaded_file.getvalue()
                image_label, image_score = detect_fake_image(image_bytes, image_detector)

                if "Error" in image_label:
                    st.error(image_label)
                else:
                    st.subheader("Image Analysis Result:")
                    if image_label == "FAKE":
                        st.error(f"**Likely FAKE!** (Confidence: {image_score:.2f})")
                        st.markdown("The model suggests this image might be a deepfake or manipulated.")
                    elif image_label == "REAL":
                        st.success(f"**Likely REAL.** (Confidence: {image_score:.2f})")
                        st.markdown("The model suggests this image is likely authentic.")
                    else:
                        st.info(f"Analysis result: **{image_label}** (Confidence: {image_score:.2f})")
                        st.markdown("The image model returned an unexpected label. Interpretation might vary.")
                    st.progress(image_score)
        else:
            st.warning("Image detection model could not be loaded. Please check the console for errors.")
else:
    st.info("Upload an image to analyze its authenticity.")

st.markdown("---") # Separator

st.sidebar.markdown("## About this App")
st.sidebar.markdown(
    """
    This application leverages machine learning models to identify
    potentially fake or AI-generated text and images.

    **Disclaimer:** AI detection models are continuously evolving and
    are not 100% accurate. Results should be interpreted as an indication
    and not as definitive proof.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using [Streamlit](https://streamlit.io/)")

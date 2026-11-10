from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from fusion_model import FacialEmotionFusion


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
)


# --------------------------------------------------
# LOAD EXTERNAL CSS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
CSS_PATH = PROJECT_DIR / "assets" / "style.css"


def load_css(css_path: Path) -> None:
    """Load the application's stylesheet."""
    if not css_path.is_file():
        st.warning(f"CSS file not found: {css_path}")
        return

    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css(CSS_PATH)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return FacialEmotionFusion()


def get_emotion(probabilities: dict) -> str:
    return max(probabilities, key=probabilities.get)


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("🎭 Facial Emotion Recognition")
st.subheader(
    "Comparing Facial Appearance and Landmark Features "
    "for Emotion Recognition"
)

st.markdown(
    """
    This application compares three approaches to facial emotion recognition:

    - **Appearance-based CNN:** Learns visual patterns from facial images.
    - **Geometry-based Model:** Uses facial landmarks and engineered features.
    - **Decision-Level Fusion:** Combines the two models' probability predictions.
    """
)

st.divider()


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

try:
    fusion_system = load_model()
    st.success("Models loaded successfully.")
except Exception as exc:
    st.error(f"Unable to load the models: {exc}")
    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.header("1. Upload a facial image")

uploaded_file = st.file_uploader(
    "Choose an image containing a face",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read this image. Please upload another image.")
    else:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded image",
            width=350,
        )

        if st.button("Analyze emotion", type="primary"):
            try:
                with st.spinner("Analyzing facial expression..."):
                    result = fusion_system.predict(image)

                st.success("Analysis completed.")

                cnn_prob = result["cnn_probabilities"]
                geometry_prob = result["geometry_probabilities"]
                fusion_prob = result["probabilities"]

                classes = list(fusion_system.cnn_classes)

                # ------------------------------------------
                # MODEL COMPARISON
                # ------------------------------------------

                st.divider()
                st.header("2. Model comparison")

                col_cnn, col_geometry, col_fusion = st.columns(3)

                with col_cnn:
                    st.subheader("Appearance CNN")
                    cnn_emotion = get_emotion(cnn_prob)
                    st.metric("Prediction", cnn_emotion.upper())
                    st.metric(
                        "Confidence",
                        f"{cnn_prob[cnn_emotion] * 100:.2f}%",
                    )

                with col_geometry:
                    st.subheader("Geometry model")
                    if result["geometry_available"] and geometry_prob:
                        geometry_emotion = get_emotion(geometry_prob)
                        st.metric("Prediction", geometry_emotion.upper())
                        st.metric(
                            "Confidence",
                            f"{geometry_prob[geometry_emotion] * 100:.2f}%",
                        )
                    else:
                        st.warning(
                            "No usable facial landmarks were detected. "
                            "The system used CNN-only fallback for this image."
                        )

                with col_fusion:
                    st.subheader("Decision-level fusion")
                    fusion_emotion = get_emotion(fusion_prob)
                    st.metric("Prediction", fusion_emotion.upper())
                    st.metric(
                        "Confidence",
                        f"{fusion_prob[fusion_emotion] * 100:.2f}%",
                    )

                # ------------------------------------------
                # FINAL PREDICTION
                # ------------------------------------------

                st.divider()
                st.header("3. Final prediction")
                st.title(f"🎯 {result['emotion'].upper()}")

                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.metric(
                        "Final confidence",
                        f"{result['confidence'] * 100:.2f}%",
                    )
                with detail_col2:
                    st.metric(
                        "Geometry available",
                        "Yes" if result["geometry_available"] else "No",
                    )

                st.caption(f"Prediction method: `{result['method']}`")

                # ------------------------------------------
                # PROBABILITY COMPARISON
                # ------------------------------------------

                st.divider()
                st.header("4. Emotion probability comparison")

                probability_data = {
                    "Appearance CNN": [
                        cnn_prob[emotion] * 100 for emotion in classes
                    ],
                    "Decision-Level Fusion": [
                        fusion_prob[emotion] * 100 for emotion in classes
                    ],
                }

                if result["geometry_available"] and geometry_prob:
                    probability_data["Geometry Model"] = [
                        geometry_prob[emotion] * 100 for emotion in classes
                    ]

                comparison_df = pd.DataFrame(
                    probability_data,
                    index=classes,
                )

                st.write(
                    "Predicted probability for each of the seven emotion classes."
                )
                st.bar_chart(comparison_df)
                st.caption("All values are percentages (%).")

                # ------------------------------------------
                # FUSION CONFIGURATION
                # ------------------------------------------

                st.divider()
                st.header("5. Fusion configuration")

                # New fusion_model.py returns explicit weights.
                # Fallback keeps compatibility with the earlier version.
                cnn_weight = result.get(
                    "cnn_weight",
                    getattr(fusion_system, "cnn_weight", result["alpha"]),
                )
                geometry_weight = result.get(
                    "geometry_weight",
                    getattr(
                        fusion_system,
                        "geometry_weight",
                        1.0 - cnn_weight,
                    ),
                )

                weight_col1, weight_col2 = st.columns(2)

                with weight_col1:
                    st.metric("CNN weight", f"{cnn_weight * 100:.0f}%")
                with weight_col2:
                    st.metric(
                        "Geometry weight",
                        f"{geometry_weight * 100:.0f}%",
                    )

                weight_df = pd.DataFrame(
                    {
                        "Weight (%)": [
                            cnn_weight * 100,
                            geometry_weight * 100,
                        ]
                    },
                    index=["Appearance CNN", "Geometry Model"],
                )

                st.subheader("Model contribution")
                st.bar_chart(weight_df, horizontal=True)

                if not result["geometry_available"]:
                    st.info(
                        "The configured fusion weights are 60:40, but this "
                        "image used CNN-only fallback because geometry "
                        "features were unavailable."
                    )

            except Exception as exc:
                st.error(f"An error occurred during prediction: {exc}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Facial Emotion Recognition Mini-Project | "
    "Appearance CNN + Geometry Model + Decision-Level Fusion"
)

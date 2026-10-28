
from pathlib import Path
import json
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.feature_extraction import extract_features


# ============================================================
# Configuration
# ============================================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_DIR = ROOT / "results" / "metrics"
PLOTS_DIR = ROOT / "results" / "plots"

st.set_page_config(
    page_title="RiceCare AI | Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7faf7;
    }
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .hero {
        background: linear-gradient(120deg, #123d2a, #23744a);
        color: white;
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.35rem;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #e5f4e9;
        font-size: 1.05rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #dce8df;
        border-radius: 14px;
        padding: 1.1rem;
        min-height: 110px;
    }
    .metric-label {
        color: #52675a;
        font-size: 0.9rem;
    }
    .metric-value {
        color: #185b37;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .small-note {
        color: #53645a;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model and data loading
# ============================================================
@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(filename):
    path = METRICS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json(filename):
    path = METRICS_DIR / filename
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return None


bundle = load_model_bundle()


# ============================================================
# Disease advisory information
# ============================================================
ADVISORY = {
    "Bacterialblight": {
        "title": "Bacterial blight",
        "overview": (
            "A bacterial disease that can cause water-soaked lesions, "
            "yellowing and drying along rice leaves."
        ),
        "actions": [
            "Inspect affected plants and monitor whether symptoms spread.",
            "Use locally recommended resistant varieties when planning future crops.",
            "Maintain balanced fertilizer and water management.",
            "Consult a local agricultural extension officer for confirmation "
            "and locally appropriate management recommendations.",
        ],
    },
    "Blast": {
        "title": "Rice blast",
        "overview": (
            "A fungal disease that may produce spindle-shaped lesions "
            "on leaves. Symptoms vary with plant stage and conditions."
        ),
        "actions": [
            "Inspect leaves for characteristic lesions and monitor their spread.",
            "Avoid excessive nitrogen fertilizer; follow local soil and crop guidance.",
            "Use locally recommended resistant varieties where available.",
            "Seek local agricultural advice if the disease is suspected to be spreading.",
        ],
    },
    "Brownspot": {
        "title": "Brown spot",
        "overview": (
            "A rice disease commonly associated with brown leaf spots. "
            "Appearance can overlap with other leaf problems."
        ),
        "actions": [
            "Check whether spots occur on multiple leaves and plants.",
            "Maintain balanced crop nutrition based on soil testing or local advice.",
            "Use healthy seed and recommended crop-management practices.",
            "Consult an agricultural specialist to confirm the cause and appropriate action.",
        ],
    },
    "Tungro": {
        "title": "Rice tungro",
        "overview": (
            "A viral disease complex that may cause yellow-orange "
            "discoloration and stunted plant growth."
        ),
        "actions": [
            "Inspect the crop for yellowing and stunted growth across plants.",
            "Monitor the crop and follow local integrated pest-management guidance.",
            "Ask an agricultural extension officer to confirm suspected tungro.",
            "Use only locally approved management practices and certified guidance.",
        ],
    },
}


def get_advisory(label):
    if label in ADVISORY:
        return ADVISORY[label]

    # Fallback if the saved model uses different class spelling.
    normalized = label.lower().replace("_", "").replace(" ", "")
    for key, value in ADVISORY.items():
        if normalized == key.lower().replace("_", "").replace(" ", ""):
            return value

    return {
        "title": label,
        "overview": (
            "No verified advisory is configured for this class yet."
        ),
        "actions": [
            "Do not rely on the model alone for treatment decisions.",
            "Seek confirmation from a local agricultural specialist.",
        ],
    }


def make_prediction(image):
    """Run the saved model using the training-time feature extractor."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp_path = Path(temp.name)
        image.convert("RGB").save(temp_path, format="PNG")

    try:
        features = extract_features(temp_path).reshape(1, -1)
        model = bundle["model"]
        prediction = str(model.predict(features)[0])

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]

        return prediction, probabilities
    finally:
        temp_path.unlink(missing_ok=True)


# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("🌾 RiceCare AI")
st.sidebar.caption("Rice leaf disease screening and crop guidance")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home Dashboard",
        "Disease Detection",
        "Model Analytics",
        "Crop Advisory",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Project model**")
if bundle:
    st.sidebar.success(
        f"Loaded: {bundle.get('selected_model', 'Saved classifier')}"
    )
else:
    st.sidebar.error("Trained model not found")

st.sidebar.caption(
    "Educational prototype. Predictions require expert confirmation."
)


# ============================================================
# Shared header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🌾 RiceCare AI</h1>
        <p>Rice Leaf Disease Detection & Crop Advisory System</p>
        <p>Upload a leaf image, explore model performance, and view
        general management guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Home Dashboard
# ============================================================
if page == "Home Dashboard":
    st.subheader("Project Dashboard")
    st.write(
        "A traditional machine-learning application that classifies "
        "rice leaf images into four supported categories."
    )

    summary = load_csv("dataset_summary.csv")
    comparison = load_csv("validation_model_comparison.csv")
    final_metrics = load_json("final_test_metrics.json")

    total_images = int(summary["images"].sum()) if (
        summary is not None and "images" in summary.columns
    ) else 0
    total_classes = len(summary) if summary is not None else 4

    test_accuracy = None
    test_f1 = None
    if final_metrics:
        metrics = final_metrics.get("test_metrics", {})
        test_accuracy = metrics.get("accuracy")
        test_f1 = metrics.get("f1_score")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Cleaned images</div>
            <div class="metric-value">{total_images:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Disease classes</div>
            <div class="metric-value">{total_classes}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        value = f"{test_accuracy:.2%}" if test_accuracy is not None else "N/A"
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Test accuracy</div>
            <div class="metric-value">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        value = f"{test_f1:.2%}" if test_f1 is not None else "N/A"
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Test score F1</div>
            <div class="metric-value">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Dataset class distribution")
        plot_path = PLOTS_DIR / "class_distribution.png"
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)
        elif summary is not None:
            st.bar_chart(summary.set_index("class")["images"])
        else:
            st.info("Run data preprocessing to generate dataset statistics.")

    with right:
        st.markdown("### Project workflow")
        st.markdown(
            """
            1. Inspect and clean leaf images.
            2. Split data into training, validation and test sets.
            3. Extract HOG and colour histogram features.
            4. Compare four traditional ML classifiers.
            5. Predict a class from an uploaded leaf image.
            6. Present general crop-management information.
            """
        )

    st.info(
        "The reported test metrics describe performance on the held-out "
        "dataset. They do not guarantee equivalent accuracy on field photographs."
    )


# ============================================================
# Disease Detection
# ============================================================
elif page == "Disease Detection":
    st.subheader("Upload a Rice Leaf Image")
    st.write(
        "Use a clear, well-lit photograph of a single rice leaf. "
        "The model predicts one of the classes it was trained on."
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported image formats: JPG, JPEG, PNG and BMP.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            col_image, col_result = st.columns([1, 1.1], gap="large")

            with col_image:
                st.markdown("### Uploaded image")
                st.image(image, use_container_width=True)
                st.caption(f"Image dimensions: {image.width} × {image.height}")

            with col_result:
                st.markdown("### Prediction")

                if not bundle:
                    st.error(
                        "The trained model was not found. "
                        "Run `python src/train_models.py` first."
                    )
                elif st.button(
                    "🔍 Predict Disease",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing leaf image..."):
                        prediction, probabilities = make_prediction(image)

                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_image"] = image

                    advisory = get_advisory(prediction)

                    st.success(f"Predicted class: {advisory['title']}")

                    if probabilities is not None:
                        classes = list(bundle["model"].classes_)
                        index = classes.index(prediction)
                        score = float(probabilities[index])

                        st.metric(
                            "Model probability for predicted class",
                            f"{score:.1%}",
                        )
                        st.caption(
                            "This is the classifier's estimated probability, "
                            "not a guarantee of diagnostic certainty."
                        )

                    st.markdown("#### About this class")
                    st.write(advisory["overview"])

                    st.markdown("#### General guidance")
                    for action in advisory["actions"]:
                        st.markdown(f"- {action}")

                    st.warning(
                        "This is an educational screening result, not a "
                        "confirmed diagnosis. Similar symptoms can have "
                        "different causes. Confirm important decisions with "
                        "a local agricultural expert."
                    )

        except (UnidentifiedImageError, OSError, ValueError):
            st.error(
                "The uploaded file could not be read as a valid image. "
                "Please try another JPG or PNG file."
            )

    else:
        st.info("Upload an image to begin disease screening.")


# ============================================================
# Model Analytics
# ============================================================
elif page == "Model Analytics":
    st.subheader("Model Performance Analytics")
    st.write(
        "Four classifiers were compared using the same image features "
        "and stratified dataset split."
    )

    comparison = load_csv("validation_model_comparison.csv")
    final_metrics = load_json("final_test_metrics.json")

    if comparison is not None:
        display_df = comparison.copy()

        for column in [
            "accuracy", "precision", "recall", "f1_score"
        ]:
            if column in display_df.columns:
                display_df[column] = display_df[column].map(
                    lambda value: f"{value:.2%}"
                )

        if "training_seconds" in display_df.columns:
            display_df["training_seconds"] = display_df[
                "training_seconds"
            ].map(lambda value: f"{value:.2f} s")

        st.markdown("### Validation comparison")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        plot_path = PLOTS_DIR / "model_comparison_validation.png"
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)

        st.caption(
            "Validation metrics were used to select the model. "
            "Test metrics are reported separately below."
        )
    else:
        st.warning(
            "Model comparison results are missing. "
            "Run `python src/train_models.py` first."
        )

    st.markdown("### Final test evaluation")
    if final_metrics:
        test_metrics = final_metrics.get("test_metrics", {})
        metric_cols = st.columns(4)

        metric_names = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1-score"),
        ]

        for column, (key, label) in zip(metric_cols, metric_names):
            with column:
                value = test_metrics.get(key)
                st.metric(label, f"{value:.2%}" if value is not None else "N/A")

        st.write(
            "Selected model:",
            final_metrics.get("selected_model", "Not recorded"),
        )
    else:
        st.info("Final test metrics are not available yet.")

    cm_path = PLOTS_DIR / "best_model_confusion_matrix.png"
    if cm_path.exists():
        st.markdown("### Selected model confusion matrix")
        st.image(str(cm_path), use_container_width=True)
        st.caption(
            "Rows represent true classes and columns represent predicted classes."
        )


# ============================================================
# Crop Advisory
# ============================================================
elif page == "Crop Advisory":
    st.subheader("Rice Crop Advisory")
    st.write(
        "Browse general information for the supported disease classes. "
        "Guidance is educational and should be confirmed locally."
    )

    selected_class = st.selectbox(
        "Select a disease class",
        options=list(ADVISORY.keys()),
        format_func=lambda key: ADVISORY[key]["title"],
    )

    advisory = ADVISORY[selected_class]

    st.markdown(f"### {advisory['title']}")
    st.write(advisory["overview"])

    st.markdown("### General management considerations")
    for action in advisory["actions"]:
        st.markdown(f"- {action}")

    st.warning(
        "This application does not prescribe pesticides or determine "
        "chemical dosages. Follow local agricultural recommendations "
        "and applicable product labels."
    )


# ============================================================
# About Project
# ============================================================
elif page == "About Project":
    st.subheader("About RiceCare AI")

    st.markdown(
        """
        **Project objective**

        Develop an image-based rice leaf classification system and present
        general crop-management guidance for four supported classes.

        **Machine-learning methodology**

        - Image resizing to 128 × 128 pixels
        - RGB colour histogram features
        - Histogram of Oriented Gradients (HOG)
        - K-Nearest Neighbors
        - Support Vector Machine
        - Decision Tree
        - Logistic Regression

        **Data preparation**

        The project uses cleaned image manifests with stratified training,
        validation and test subsets. The test set is held out from model
        selection.

        **Limitations**

        - Only the classes represented by the training dataset are supported.
        - Predictions may be less reliable for field images with different
          lighting, backgrounds or image quality.
        - The model cannot establish a definitive agricultural diagnosis.
        - Crop guidance is general information and is not a substitute for
          local expert advice.
        """
    )

    st.markdown("### Technology stack")
    st.write(
        "Python · scikit-learn · scikit-image · Pillow · Pandas · "
        "Matplotlib · Streamlit"
    )

st.divider()
st.caption(
    "RiceCare AI | Academic mini-project | "
    "Image classification and educational crop guidance"
)

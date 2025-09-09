import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import pickle
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered",
)

# ---- Pastel UI (soft, clean) ----
PASTEL_CSS = """
<style>
/* Backgrounds */
.stApp {background: linear-gradient(180deg, #f9fbff 0%, #fdf7ff 100%);} 
header, .st-emotion-cache-18ni7ap, .st-emotion-cache-1dp5vir {background: transparent;}

/* Cards */
.block-container {padding-top: 3rem !important;}
div[data-testid="stVerticalBlock"] > div {background: #ffffffaa; border-radius: 1rem; padding: 1rem;}

/* Typography */
h1, h2, h3, h4, h5 {color: #3b3b58;}
p, span, label {color: #4c4c6d;}

/* Inputs */
textarea, input, .stTextArea, .stTextInput {background: #fff; border: 1px solid #e7e9f5 !important; border-radius: 12px !important;}
.stButton > button {background: #cfe8ff; border: none; color: #1f3b57; padding: 0.6rem 1.2rem; border-radius: 999px;}
.stButton > button:hover {background: #e4f1ff;}

/* Pills */
.badge-pill {display:inline-block; padding: 0.35rem 0.75rem; border-radius: 999px; background:#fce4ec; color:#6d2e46; font-weight:600; margin-right:0.4rem;}
.proba-bar {height: 12px; border-radius: 999px; background: #f1f3fb; overflow:hidden;}
.proba-fill {height: 100%; border-radius: 999px;}
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

st.title("📰 BBC News Category Classifier")
st.caption("Simple, clean, pastel UI • Single-text prediction • BBC dataset categories")

# ---- Sidebar ----
with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies a news article into one of the **BBC categories**: "
        "**business**, **entertainment**, **politics**, **sport**, **tech**."
    )
    st.write("Upload not needed — just paste or type your text below.")
    st.write("Model & vectorizer are loaded from local files.")
    st.markdown("---")
    st.write("Tips:")
    st.write("- Write a few sentences for best accuracy.")
    st.write("- Avoid headlines only; add some context.")
    st.write("- English text works best for this model/dataset.")

# ---- Utils to load model/vectorizer ----
@st.cache_resource(show_spinner=True)
def load_artifacts():
    # Try multiple common filenames/paths
    model_paths = [
        "model.pkl",
        os.path.join(os.getcwd(), "model.pkl"),
        "/mnt/data/model.pkl"
    ]
    vec_paths = [
        "vectorizer",
        "vectorizer.pkl",
        os.path.join(os.getcwd(), "vectorizer"),
        os.path.join(os.getcwd(), "vectorizer.pkl"),
        "/mnt/data/vectorizer",
        "/mnt/data/vectorizer.pkl"
    ]

    model = None
    vectorizer = None

    for p in model_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
            except Exception:
                with open(p, "rb") as f:
                    model = pickle.load(f)
            break

    for p in vec_paths:
        if os.path.exists(p):
            try:
                vectorizer = joblib.load(p)
            except Exception:
                with open(p, "rb") as f:
                    vectorizer = pickle.load(f)
            break

    if model is None or vectorizer is None:
        raise FileNotFoundError(
            "Could not find model and/or vectorizer. "
            "Ensure 'model.pkl' and 'vectorizer' are present."
        )
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    artifacts_status = "✅ Model & vectorizer loaded."
except Exception as e:
    artifacts_status = f"❌ Problem loading artifacts: {e}"

st.markdown(f"<div class='badge-pill'>{artifacts_status}</div>", unsafe_allow_html=True)

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

# ---- Text input ----
st.subheader("Enter your news text")
example = st.selectbox(
    "Optional: load an example",
    ["(none)",
     "The central bank announced changes to interest rates affecting the stock market.",
     "The movie premiere drew huge crowds and critics praised the lead actor's performance.",
     "The prime minister addressed parliament to outline the new education policy.",
     "The team secured a dramatic last-minute victory in the championship final.",
     "The company unveiled a new smartphone featuring advanced AI capabilities."],
    index=0
)
text = st.text_area(
    label="Paste or type a BBC-style article paragraph:",
    height=200,
    value="" if example == "(none)" else example,
    placeholder="Type your article text here..."
)

# ---- Predict button ----
if st.button("Classify Article"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
            else:
                # Fall back: create a pseudo-proba vector with 1 at predicted class
                proba = np.zeros(len(CATEGORIES))
                # Try to map label if model returns str label
                if isinstance(pred, str) and pred in CATEGORIES:
                    proba[CATEGORIES.index(pred)] = 1.0
                else:
                    # If pred is numeric label
                    try:
                        proba[int(pred)] = 1.0
                    except Exception:
                        pass

            # Normalize if needed
            if proba.sum() > 0:
                proba = proba / proba.sum()

            # If labels come from model classes_, align by order if possible
            display_labels = CATEGORIES
            if hasattr(model, "classes_"):
                # Attempt to reorder probabilities to CATEGORIES order if they match
                try:
                    cls = list(model.classes_)
                    if set(cls) == set(CATEGORIES):
                        reorder = [cls.index(c) for c in CATEGORIES]
                        proba = proba[reorder]
                except Exception:
                    pass

            st.success(f"**Predicted Category:** :violet[{pred}]")

            # Probability table
            df = pd.DataFrame({"Category": display_labels, "Probability": proba})
            df = df.sort_values("Probability", ascending=False).reset_index(drop=True)

            st.write("Confidence by category:")
            st.dataframe(df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

            # Probability bar chart (matplotlib, single plot, no explicit colors)
            fig, ax = plt.subplots()
            ax.bar(df["Category"], df["Probability"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_xlabel("Category")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---- Footer ----
st.markdown("---")
st.caption("Built with Streamlit • BBC News Classification (business, entertainment, politics, sport, tech) • Pastel UI")

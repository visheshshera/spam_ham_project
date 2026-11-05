import nltk
import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam / Ham Classifier")


@st.cache_resource
def load_pipeline():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return PredictionPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()

message = st.text_area("Enter a message", placeholder="Type your message here...")

if st.button("Classify"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        label = pipeline.initiate_prediction_pipeline(message)
        if label == "spam":
            st.error("🚨 Spam")
        else:
            st.success("✅ Ham (not spam)")

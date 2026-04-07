import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>📩 Spam Message Classifier</h1>",
    unsafe_allow_html=True
)

st.write("Detect whether a message is **Spam** or **Not Spam** using Machine Learning.")

st.divider()

# Text input box
input_text = st.text_area(
    "✉️ Enter your message",
    placeholder="Type or paste your message here...",
    height=120
)

col1, col2 = st.columns(2)

# Train Model Button
with col1:
    if st.button("⚙️ Train Model", use_container_width=True):
        with st.spinner("Training model... please wait"):
            from src.pipeline.training_pipeline import TrainingPipeline
            obj = TrainingPipeline()
            obj.initiate_training_pipeline()

        st.success("✅ Model Trained Successfully!")

# Predict Button
with col2:
    if st.button("🔍 Predict", use_container_width=True):

        if input_text.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Analyzing message..."):
                from src.pipeline.prediction_pipeline import PredictionPipeline
                obj = PredictionPipeline()
                prediction = obj.initiate_prediction_pipeline(input_text)

            st.divider()

            if prediction == 1:
                st.error("🚨 This message is **SPAM**")
            else:
                st.success("✅ This message is **NOT SPAM**")

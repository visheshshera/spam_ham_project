import os,sys
import streamlit as st
import joblib

st.title("Spam Classifier")

if st.button("Train Model"):
    from src.pipeline.training_pipeline import TrainingPipeline
    obj=TrainingPipeline()
    obj.initiate_training_pipeline()
    st.success("Model Trained Successfully")

input_text=st.text_input("Enter the text")

if st.button("Predict"):
    from src.pipeline.prediction_pipeline import PredictionPipeline

    obj=PredictionPipeline()
    prediction=obj.initiate_prediction_pipeline(input_text)

    if prediction==1:
        st.success("Spam")
    else:
        st.success("Not Spam")
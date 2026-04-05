import os,sys
import streamlit as st
import joblib
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

st.title("Spam Classifier")

# LABEL_ENCODER_FILE_PATH=os.path.join('artifacts','label_encoder.pkl')

# if not os.path.exists(LABEL_ENCODER_FILE_PATH):
#     raise Exception('label encoder not found')
# else:
#     label_encoder=joblib.load(LABEL_ENCODER_FILE_PATH)




if st.button("Train Model"):
    obj=TrainingPipeline()
    obj.initiate_training_pipeline()
    st.success("Model Trained Successfully")

input_text=st.text_input("Enter the text")

if st.button("Predict"):

    # LABEL_ENCODER_FILE_PATH=os.path.join('artifacts','target_encoder.pkl')

    # if not os.path.exists(LABEL_ENCODER_FILE_PATH):
    #     raise Exception('label encoder not found')
    # else:
    #     label_encoder=joblib.load(LABEL_ENCODER_FILE_PATH)

    obj=PredictionPipeline()
    prediction=obj.initiate_prediction_pipeline(input_text)

    if prediction==1:
        st.success("Spam")
    else:
        st.success("Not Spam")
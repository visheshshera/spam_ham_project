import os,sys
from src.constants import *
from src.exception import CustomException
from src.logger import logging
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.text_preprocessing import preprocess_text

class PredictionPipeline:
    MODEL_FILE_PATH=os.path.join('artifacts','model.pkl')
    VECTORIZER_FILE_PATH=os.path.join('artifacts/tfidf_vectorizer.pkl')

    if not os.path.exists(VECTORIZER_FILE_PATH):
        raise Exception('vectorizer not found')
    else:
        vectorizer=joblib.load(VECTORIZER_FILE_PATH)

    if not os.path.exists(MODEL_FILE_PATH):
        raise Exception('model not found train the model first')
    else:
        model=joblib.load(MODEL_FILE_PATH)

    def initiate_prediction_pipeline(self,message):
        try:
            text=preprocess_text(message)
            preprocessed_input=self.vectorizer.transform([text])
            prediction=self.model.predict(preprocessed_input)
            return int(prediction[0])
    
        except Exception as e :
            raise CustomException(e,sys)
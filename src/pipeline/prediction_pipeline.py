import os, sys
from src.constants import *
from src.exception import CustomException
from src.logger import logging
import joblib

from src.entity.config_entity import PredictionPipelineConfig
from src.utils.text_preprocessing import preprocess_text


class PredictionPipeline:

    def __init__(self):
        PredictionPipelineConfig_obj = PredictionPipelineConfig()

        self.vectorizer_path = PredictionPipelineConfig_obj.vectorizer_file_path

        self.model_path = PredictionPipelineConfig_obj.model_file_path

        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train the model first."
            )

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)

    def initiate_prediction_pipeline(self, message):
        try:
            text = preprocess_text(message)
            preprocessed_input = self.vectorizer.transform([text])
            prediction = self.model.predict(preprocessed_input)

            return int(prediction[0])

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = PredictionPipeline()
    res = obj.initiate_prediction_pipeline("winning $2000")
    if res == 0:
        print("ham")
    else:
        print("spam")

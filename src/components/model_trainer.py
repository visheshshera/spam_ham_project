import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from src.utils.main_utils import save_object

from src.constants import *
import numpy as np

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_trainer(self,x_train_resampled, y_train_resampled,x_test_arr,y_test_arr): 

        try:
            logging.info("Model Trainer Started")
            model = MultinomialNB()
            model.fit(x_train_resampled,y_train_resampled)

            save_object('artifacts/model.pkl',model)
            y_pred=model.predict(x_test_arr)

            accuracy=accuracy_score(y_test_arr,y_pred)
            logging.info(f"Model Trained with accuracy{accuracy}")
            return accuracy
        except Exception as e:
            raise CustomException(e,sys)


import os,sys

from src.constants import *
from src.utils.main_utils import save_object
from src.utils.text_preprocessing import preprocess_text
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE

from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self):
        pass
    
    def initiate_data_transformation(self,train_file_path:str,test_file_path:str):
        try:
            logging.info('Data Transformation Started') 
            train_df=pd.read_csv(train_file_path)
            test_df=pd.read_csv(test_file_path)
            
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)

            train_df['text']=train_df['text'].apply(preprocess_text)
            test_df['text']=test_df['text'].apply(preprocess_text)


            x_train_df=train_df['text']
            y_train_df=train_df['text_type']

            x_test_df=test_df['text']
            y_test_df=test_df['text_type']

            target_encoder=LabelEncoder()
            y_train_arr=target_encoder.fit_transform(y_train_df)
            y_test_arr=target_encoder.transform(y_test_df)

            #saving target label encoder
            save_object('artifacts/target_encoder.pkl',target_encoder)


            vectorizer_pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
                ])

            x_train_arr = vectorizer_pipeline.fit_transform(x_train_df)
            x_test_arr = vectorizer_pipeline.transform(x_test_df)

            #saving tfidfvectorizer
            save_object('artifacts/tfidf_vectorizer.pkl',vectorizer_pipeline)

            smote = SMOTE(random_state=42)

            x_train_resampled, y_train_resampled = smote.fit_resample(x_train_arr, y_train_arr)

            return x_train_resampled, y_train_resampled,x_test_arr,y_test_arr

        except Exception as e :
            raise CustomException(e,sys)
        
                

    
        

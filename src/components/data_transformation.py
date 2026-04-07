import os,sys

from src.constants import *
from src.utils.main_utils import save_object,save_numpy_array_data
from src.utils.text_preprocessing import preprocess_text
from src.exception import CustomException
from src.logger import logging

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact #, DataValidationArtifact


from scipy import sparse
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import RandomOverSampler

from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
                 #data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            # self.data_validation_artifact = data_validation_artifact
            # self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self):
        try:
            logging.info('Data Transformation Started') 

            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
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

            #making sure directory exists
            os.makedirs(os.path.dirname(self.data_transformation_config.target_encoder_file_path),exist_ok=True)

            #saving target label encoder
            save_object(self.data_transformation_config.target_encoder_file_path,target_encoder)


            vectorizer_pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
                ])

            x_train_arr = vectorizer_pipeline.fit_transform(x_train_df)
            x_test_arr = vectorizer_pipeline.transform(x_test_df)

            #saving tfidfvectorizer
            save_object(self.data_transformation_config.tfidf_vectorizer_file_path,vectorizer_pipeline)


            ros = RandomOverSampler(random_state=DATA_TRANSFORMATION_ROS_RANDOM_STATE)
            x_train_resampled, y_train_resampled = ros.fit_resample(x_train_arr, y_train_arr)

            sparse.save_npz(
                self.data_transformation_config.x_train_resampled_file_path,
                x_train_resampled
            )
            save_numpy_array_data(file_path=self.data_transformation_config.y_train_resampled_file_path,array=y_train_resampled)
  
            sparse.save_npz(
                self.data_transformation_config.x_test_file_path,
                x_test_arr
            )

            save_numpy_array_data(file_path=self.data_transformation_config.y_test_file_path,array=y_test_arr)

            data_transformation_artifact=DataTransformationArtifact(
                x_train_resampled_path=self.data_transformation_config.x_train_resampled_file_path,
                y_train_resampled_path=self.data_transformation_config.y_train_resampled_file_path,
                x_test_path=self.data_transformation_config.x_test_file_path,
                y_test_path=self.data_transformation_config.y_test_file_path
            )

            return data_transformation_artifact
        
        except Exception as e :
            raise CustomException(e,sys)
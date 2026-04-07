import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    raw_file_path: str = os.path.join(data_ingestion_dir, RAW_FILE_NAME)
    train_file_path: str = os.path.join(data_ingestion_dir, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_dir, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataTransformationConfig:
    #for nlp project concatenation is not ised to create train and test arr instead we used x_train_resampled,y_train_resampled,x_test_arr,y_test_arr

    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    tfidf_vectorizer_file_path: str = os.path.join(data_transformation_dir, TFIDF_VECTORIZER_FILE_NAME)
    target_encoder_file_path: str = os.path.join(data_transformation_dir, TARGET_ENCODER_FILE_NAME)
    x_train_resampled_file_path: str = os.path.join(data_transformation_dir, X_TRAIN_RESAMPLED_FILE_NAME)
    y_train_resampled_file_path: str = os.path.join(data_transformation_dir, Y_TRAIN_RESAMPLED_FILE_NAME)
    x_test_file_path: str = os.path.join(data_transformation_dir, X_TEST_FILE_NAME)
    y_test_file_path: str = os.path.join(data_transformation_dir, Y_TEST_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    model_file_path: str = os.path.join(model_trainer_dir, MODEL_FILE_NAME)
    
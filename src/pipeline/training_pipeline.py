import os,sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_transformation_config=DataTransformationConfig()
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_training_pipeline(self):
        try:
            logging.info('Training Pipeline Started')  

            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

            data_transformation=DataTransformation(data_ingestion_artifact=data_ingestion_artifact,data_transformation_config=self.data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()

            model_trainer=ModelTrainer(model_trainer_config=self.model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initiate_model_trainer()

            logging.info('Training Pipeline Completed')
        
        except Exception as e:
            raise CustomException(e,sys)
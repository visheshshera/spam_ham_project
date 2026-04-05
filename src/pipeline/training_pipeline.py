import os,sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_obj=DataIngestion()
        self.data_transformation_obj=DataTransformation()
        self.model_trainer_obj=ModelTrainer()

    def initiate_training_pipeline(self):
        try:
            logging.info('Training Pipeline Started')   
            train_file_path,test_file_Path=self.data_ingestion_obj.initiate_data_ingestion()

            x_train_resampled,y_train_resampled,x_test_arr,y_test_arr=self.data_transformation_obj.initiate_data_transformation(train_file_path,test_file_Path)

            accuracy=self.model_trainer_obj.initiate_model_trainer(x_train_resampled,y_train_resampled,x_test_arr,y_test_arr)

            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=='__main__':
    obj=TrainingPipeline()
    acc=obj.initiate_training_pipeline()
    print(acc)
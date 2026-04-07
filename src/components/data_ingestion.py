import os,sys
from datetime import datetime
from src.exception import CustomException
from src.logger import logging

from src.constants import *
from src.data_access.data_puller import DataPuller
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from sklearn.model_selection import train_test_split
import pandas as pd

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion Started')
            df=DataPuller().import_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)

            x,y=df[['text']],df['text_type']

            train_df,test_df=train_test_split(df,
                                              test_size=self.data_ingestion_config.train_test_split_ratio,
                                              random_state=42,
                                              stratify=y)

            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_file_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_file_path,index=False)

            train_df.to_csv(self.data_ingestion_config.train_file_path,index=False)

            test_df.to_csv(self.data_ingestion_config.test_file_path,index=False)

            logging.info('Data Ingestion Completed')

            data_ingestion_artifact=DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            return data_ingestion_artifact

        
        except Exception as e :
            raise CustomException(e,sys)
        
if __name__=='__main__':
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()
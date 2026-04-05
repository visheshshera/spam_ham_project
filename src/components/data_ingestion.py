import os,sys
from src.exception import CustomException
from src.logger import logging

from src.data_access.data_puller import DataPuller
from src.constants import *

from sklearn.model_selection import train_test_split
import pandas as pd



class DataIngestion:
    def __init__(self):
        pass
    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion Started')
            df=DataPuller().import_collection_as_dataframe(COLLECTION_NAME,DATABASE_NAME)
            x,y=df[['text']],df['text_type']
            train_df,test_df=train_test_split(df,test_size=0.2,random_state=42,stratify=y)

            RAW_FILE_PATH=os.path.join('artifacts','raw.csv')
            raw_dir=os.path.dirname(RAW_FILE_PATH)
            os.makedirs(raw_dir,exist_ok=True)

            TRAIN_FILE_PATH=os.path.join('artifacts','train.csv')
            train_dir=os.path.dirname(TRAIN_FILE_PATH)
            os.makedirs(train_dir,exist_ok=True)
            
            TEST_FILE_PATH=os.path.join('artifacts','test.csv')
            test_dir=os.path.dirname(TEST_FILE_PATH)
            os.makedirs(train_dir,exist_ok=True)
            
            df.to_csv(RAW_FILE_PATH,index=False)
            train_df.to_csv(TRAIN_FILE_PATH,index=False)
            test_df.to_csv(TEST_FILE_PATH,index=False)

            return TRAIN_FILE_PATH,TEST_FILE_PATH

        except Exception as e :
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=DataIngestion()
    a,b=obj.initiate_data_ingestion()
    print(a,b)
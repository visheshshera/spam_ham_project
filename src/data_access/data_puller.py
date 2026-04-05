# class to import data from mongodb and store it in a dataframe locally

import os ,sys
from src.exception import CustomException
from src.logger import logging
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import *

import numpy as np
import pandas as pd


class DataPuller:
    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def import_collection_as_dataframe(self, collection_name: str,database_name: str)-> pd.DataFrame:
        """
        Imports a collection from a MongoDB database as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the collection to import.
        database_name : str
            The name of the MongoDB database to connect to.

        Returns:
        -------
        pd.DataFrame
            The imported collection as a pandas DataFrame.

        Raises:
        ------
        CustomException
            If there is an issue connecting to MongoDB.

        """
        try:
            logging.info(f"Importing {collection_name} collection from {database_name} database")
            
            # Connect to MongoDB and retrieve the collection
            collection = self.mongo_client.database[collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            if '_id' in df.columns:
                df.drop('_id',axis=1,inplace=True)

            df.replace({"na":np.nan},inplace=True)
            return df
        

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    obj=DataPuller()
    a=obj.import_collection_as_dataframe('spam-ham','ml_database')
    print(a)
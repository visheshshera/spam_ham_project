# class to import data from mongodb and store it in a dataframe locally

import os ,sys
from src.exception import CustomException
from src.logger import logging
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import *

import numpy as np
import pandas as pd

from typing import Optional

class DataPuller:
    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def import_collection_as_dataframe(self, collection_name: str,database_name: Optional[str] = None)-> pd.DataFrame:
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
            database_name = database_name if database_name else DATABASE_NAME

            logging.info(
                f"Importing collection '{collection_name}' from database '{database_name}'"
            )

            # Check database exists
            if database_name not in self.mongo_client.client.list_database_names():
                raise Exception(f"Database '{database_name}' does not exist")

            db = self.mongo_client.client[database_name]

            # Check collection exists
            if collection_name not in db.list_collection_names():
                raise Exception(
                    f"Collection '{collection_name}' not found in database '{database_name}'"
                )

            collection = db[collection_name]

            logging.info("Fetching data from MongoDB collection")

            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                logging.warning("Collection exists but contains no documents")

            # Remove MongoDB internal id
            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)

            # Replace 'na' values
            df.replace({"na": np.nan}, inplace=True)

            logging.info(
                f"Data successfully imported. Shape of dataframe: {df.shape}"
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)
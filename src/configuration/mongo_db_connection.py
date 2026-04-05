import os
import sys
import pymongo
import certifi

from src.exception import CustomException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

ca = certifi.where()


class MongoDBClient:
    """
    MongoDBClient establishes and manages MongoDB connection.
    """

    client = None  # shared client instance

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:

            if MongoDBClient.client is None:

                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if mongo_db_url is None:
                    raise Exception(
                        f"Environment variable '{MONGODB_URL_KEY}' is not set."
                    )

                # Create Mongo client
                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tlsCAFile=ca,
                    serverSelectionTimeoutMS=5000  # fail fast if server unreachable
                )

                # Force connection check
                MongoDBClient.client.admin.command("ping")

                logging.info("MongoDB connection established successfully.")

            # reuse existing client
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            logging.error("MongoDB connection failed.")
            raise CustomException(e, sys)
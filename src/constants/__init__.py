# For MongoDB connection
DATABASE_NAME = "ml_database"
COLLECTION_NAME = "spam-ham"
MONGODB_URL_KEY = "MONGODB_URL_KEY"

ARTIFACT_DIR: str = "artifact"

PIPELINE_NAME: str = ""

RAW_FILE_NAME: str = "raw.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "spam-ham"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25
DATA_INGESTION_RANDOM_STATE: int = 42

"""
Data transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_ROS_RANDOM_STATE: int = 42
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_FILE_NAME: str = "preprocessed.pkl"
DATA_TRANSFORMATION_TARGET_ENCODER_OBJECT_FILE_NAME: str = "target_encoder.pkl"
TFIDF_VECTORIZER_FILE_NAME: str = "tfidf_vectorizer.pkl"
TARGET_ENCODER_FILE_NAME:str="target_encoder.pkl"
X_TRAIN_RESAMPLED_FILE_NAME:str="x_train_resampled.npz"
Y_TRAIN_RESAMPLED_FILE_NAME:str="y_train_resampled.npy"
X_TEST_FILE_NAME:str="x_test.npz"
Y_TEST_FILE_NAME:str="y_test.npy"


"""
Model trainer related constant start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_FILE_NAME: str = "model.pkl"


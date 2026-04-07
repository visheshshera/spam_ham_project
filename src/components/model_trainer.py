import os,sys
from scipy import sparse
from src.exception import CustomException
from src.logger import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from src.utils.main_utils import save_object, load_numpy_array_data

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.constants import *
import numpy as np

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def initiate_model_trainer(self): 

        try:
            logging.info("Model Trainer Started")

            #making sure directory exists
            os.makedirs(os.path.dirname(self.model_trainer_config.model_file_path),exist_ok=True)
            
            model = MultinomialNB()


            # Load sparse matrices
            x_train_resampled = sparse.load_npz(
                self.data_transformation_artifact.x_train_resampled_path
            )

            x_test_arr = sparse.load_npz(
                self.data_transformation_artifact.x_test_path
            )

            # Load numpy arrays
            y_train_resampled = load_numpy_array_data(
                self.data_transformation_artifact.y_train_resampled_path
            )

            y_test_arr = load_numpy_array_data(
                self.data_transformation_artifact.y_test_path
            )

            # Flatten target arrays
            y_train_resampled = np.ravel(y_train_resampled)
            y_test_arr = np.ravel(y_test_arr)
                
            model.fit(x_train_resampled,y_train_resampled)

            

            save_object(self.model_trainer_config.model_file_path,model)
            y_pred=model.predict(x_test_arr)

            accuracy=accuracy_score(y_test_arr,y_pred)
            logging.info(f"Model Trained with accuracy : {accuracy}")

            model_trainer_artifact = ModelTrainerArtifact(
                model_file_path=self.model_trainer_config.model_file_path
            )

            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e,sys)



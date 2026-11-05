import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data Ingestion Started")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Data Validation Started")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Model Trainer Started")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Model Evaluation Started")
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_pusher(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:
        try:
            logging.info("Model Pusher Started")
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            logging.info("Training Pipeline Started")

            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            if not data_validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed: {data_validation_artifact.message}"
                )

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("New model is not better than production. Skipping push.")
                return

            self.start_model_pusher(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact,
            )

            logging.info("Training Pipeline Completed")

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()
    print("Training Pipeline Completed")

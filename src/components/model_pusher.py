# src/components/model_pusher.py
import os, sys, json, shutil
from datetime import datetime

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ):
        self.config = model_pusher_config
        self.transformation_artifact = data_transformation_artifact
        self.trainer_artifact = model_trainer_artifact
        self.evaluation_artifact = model_evaluation_artifact

    @staticmethod
    def _safe_copy(src_path: str, dst_path: str) -> None:
        tmp_path = dst_path + ".tmp"
        shutil.copy(src_path, tmp_path)
        os.replace(tmp_path, dst_path)  # atomic swap of this one file

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Model Pusher started")
            os.makedirs(self.config.model_dir, exist_ok=True)

            self._safe_copy(
                self.transformation_artifact.tfidf_vectorizer_file_path,
                self.config.vectorizer_file_path,
            )
            self._safe_copy(
                self.trainer_artifact.model_file_path,
                self.config.model_file_path,
            )

            with open(self.config.metadata_file_path, "w") as f:
                json.dump(
                    {
                        "source_artifact_dir": os.path.dirname(
                            self.trainer_artifact.model_file_path
                        ),
                        "pushed_at": datetime.now().isoformat(),
                        "f1_score": self.evaluation_artifact.new_model_score,
                    },
                    f,
                    indent=2,
                )

            logging.info(f"Model pushed to {self.config.model_dir}/")
            return ModelPusherArtifact(
                model_file_path=self.config.model_file_path,
                vectorizer_file_path=self.config.vectorizer_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e

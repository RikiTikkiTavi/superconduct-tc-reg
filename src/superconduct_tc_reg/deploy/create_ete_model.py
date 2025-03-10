import tempfile
import hydra
import joblib
import mlflow
import mlflow.artifacts
import mlflow.models
import mlflow.models.signature
import mlflow.pyfunc
import mlflow.types
import numpy as np
import pandas as pd
import logging
from superconduct_tc_reg.data.process import DataProcessor
from superconduct_tc_reg.models.dnn import DNNModel

_logger = logging.getLogger(__name__)

class ETEModelWithProcessing(mlflow.pyfunc.PythonModel):  # type: ignore
    """
    A PyFunc wrapper that includes preprocessing and post-processing
    along with a trained model.
    """

    model: mlflow.pyfunc.PyFuncModel | None
    data_processor: DataProcessor | None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):  # type: ignore
        self.data_processor = load_data_processor(
            context.model_config["data_processor_uri"]
        )
        self.model = mlflow.pyfunc.load_model(context.model_config["model_uri"])

    def predict(self, context, model_input: pd.DataFrame, params):
        assert self.data_processor is not None
        assert self.model is not None

        model_input = self.data_processor.transform_features(model_input)
        model_output = self.model.predict(model_input, params)

        if self.data_processor.target_scaler is not None:
            model_output["critical_temp"] = self.data_processor.target_scaler.inverse_transform(
                model_output["critical_temp"]
            )

        return model_output


def load_data_processor(data_processor_uri) -> DataProcessor:
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path_data_processor = mlflow.artifacts.download_artifacts(
            artifact_uri=data_processor_uri, dst_path=tmpdir
        )
        return joblib.load(local_path_data_processor)

@hydra.main(
    config_path="../../../configs",
    config_name="create_ete_model",
    version_base="1.3",
)
def create_ete_model(config):
    """
    Loads a trained model from MLflow, wraps it with preprocessing,
    and logs it as an MLflow PyFunc model.
    """
    assert config.run_id != "???"
    mlflow.set_tracking_uri(config.tracking.tracking_uri,)
    
    path_data_processor = f"{config.data_processor_artifact.path}/{config.data_processor_artifact.name}"
    path_to_model = config.path_to_model
    run_id = config.run_id

    with mlflow.start_run(run_id=run_id):
        wrapped_model = ETEModelWithProcessing()

        model_uri = f"runs:/{run_id}/{path_to_model}"
        data_processor_uri = mlflow.get_artifact_uri(path_data_processor)

        data_processor = load_data_processor(data_processor_uri)

        mlflow.pyfunc.log_model(
            artifact_path=f"{path_to_model}:ete",
            python_model=wrapped_model,
            signature=mlflow.models.signature.ModelSignature(
                outputs=mlflow.types.Schema(
                    [
                        mlflow.types.ColSpec("float", "critical_temp")
                    ]
                ),
                inputs=mlflow.types.Schema(
                    [
                        mlflow.types.ColSpec("float", f)
                        for f in data_processor.original_features
                    ]
                ),
            ),
            input_example=data_processor.example.astype(np.float32),
            model_config={"model_uri": model_uri, "data_processor_uri": data_processor_uri},
            registered_model_name=config.model_name
        )

    _logger.info(f"Wrapped model logged under Run ID: {run_id}")


if __name__ == "__main__":
    create_ete_model()

# pyright: ignore-all

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import json

try:
    import mlflow  # type: ignore
    import mlflow.pytorch as mlflow_pytorch  # type: ignore
    from mlflow.metrics import EvaluationMetric  # type: ignore
    from mlflow.utils.file_utils import TempDir  # type: ignore

    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    mlflow_pytorch = None  # type: ignore
    EvaluationMetric = Any  # type: ignore
    TempDir = None  # type: ignore

    _MLFLOW_AVAILABLE = False

# Keep types permissive (optional dependency)
mlflow: Any = mlflow  # type: ignore
mlflow_pytorch: Any = mlflow_pytorch  # type: ignore
TempDir: Any = TempDir  # type: ignore
EvaluationMetric: Any = EvaluationMetric  # type: ignore


def is_mlflow_available() -> bool:
    return bool(_MLFLOW_AVAILABLE)


from collections.abc import Iterator
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchinfo import summary
import os
import pathlib
import logging
import random
import string
import shutil

from recipe.logging.logger import get_logger_by_name
from recipe.utils.minio_util import MinioManager

DEFAULT_BUCKET = "mlflow-artifacts"

logger = get_logger_by_name("mlflow-util")


def _require_mlflow() -> None:
    if not is_mlflow_available() or mlflow is None:
        raise ImportError(
            "MLflow is required for this operation but is not installed/available. "
            "Install mlflow or disable MLflow usage in config."
        )


def setup_mlflow(experiment: str, mlflow_host: str | None = None):
    """Configure MLflow tracking and MinIO artifact store from environment variables.

    Required environment variables (no credentials are hardcoded in this repo):
      - MLFLOW_HOST:               MLflow tracking server host (e.g. "127.0.0.1")
      - MLFLOW_TRACKING_USERNAME:  MLflow basic-auth username
      - MLFLOW_TRACKING_PASSWORD:  MLflow basic-auth password
      - MINIO_ACCESS_KEY:          MinIO (S3 artifact store) access key
      - MINIO_SECRET_KEY:          MinIO (S3 artifact store) secret key

    Optional environment variables (defaults shown):
      - MLFLOW_PORT:               15123
      - MINIO_PORT:                9000
      - MINIO_SECURE:              "false"
      - MINIO_BUCKET:              "mlflow-artifacts"
      - MLFLOW_ARTIFACT_TIMEOUT:   300 (seconds)

    Args:
        experiment: MLflow experiment name to use/create.
        mlflow_host: Optional explicit host override; when omitted the MLFLOW_HOST
            environment variable is used. Raises RuntimeError if neither is set.
    """
    _require_mlflow()

    # Never hardcode credentials: everything must come from the environment.
    required_env = [
        "MLFLOW_HOST",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required MLflow/MinIO environment variables: "
            + ", ".join(missing)
            + ". Set them in your shell or a .env file (see .env.example). "
            + "Training-time MLflow tracking can be disabled with "
            + "MLFLOW_ENABLE=false or MLflow.enable=False in the recipe config."
        )
    host = mlflow_host if mlflow_host is not None else os.environ["MLFLOW_HOST"]

    mlflow_port = os.getenv("MLFLOW_PORT", "15123")
    minio_port = os.getenv("MINIO_PORT", "9000")
    os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", 
                          os.getenv("MLFLOW_ARTIFACT_TIMEOUT", "300"))
    mlflow.set_tracking_uri(uri=f"http://{host}:{mlflow_port}")
    mlflow.set_experiment(experiment)
    mlflow.enable_system_metrics_logging()
    # set logging level
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.setLevel(logging.WARNING)
    # minio artifact store (credentials are read from the environment only)
    os.environ.setdefault("MINIO_ENDPOINT", f"{host}:{minio_port}")
    os.environ.setdefault("MINIO_SECURE", "false")
    os.environ.setdefault("MINIO_BUCKET", "mlflow-artifacts")
    # disable tracing after setup, which is mandotory!
    if str(os.getenv("ENABLE_TRACE", "false")).lower() != "true":
        logger.info("Disabling MLflow tracing for performance.")
        mlflow.tracing.disable()
    else:
        logger.info("Enabling MLflow tracing.")
        mlflow.tracing.enable()


def load_model(run_id: str, model_name: str) -> torch.nn.Module:
    """Load model from MLflow artifacts."""
    _require_mlflow()
    with mlflow.start_run(run_id=run_id):
        f = pathlib.Path(f"model-{run_id}")
        if not f.exists():
            f.mkdir()
        model_uri = mlflow.get_artifact_uri(model_name)
        model = mlflow_pytorch.load_model(model_uri, dst_path=f"model-{run_id}")
        shutil.rmtree(f.absolute().__str__())
    return model


def load_model_state_dict(run_id: str, model_name: str) -> dict:
    """Load model state dict from MLflow artifacts."""
    _require_mlflow()
    with mlflow.start_run(run_id=run_id):
        f = pathlib.Path(f"model-{run_id}")
        if not f.exists():
            f.mkdir()
        model_uri = mlflow.get_artifact_uri(model_name)
        state_dict = mlflow.pytorch.load_state_dict(
            model_uri, dst_path=f"model-{run_id}"
        )
        shutil.rmtree(f.absolute().__str__())
    return state_dict["model"]


def load_configuration(
    run_id: str, file_name: str = "train-configuration.json"
) -> dict:
    """Load configuration from MLflow artifacts."""
    _require_mlflow()
    with mlflow.start_run(run_id=run_id):
        f = pathlib.Path(f"config-{run_id}")
        if not f.exists():
            f.mkdir()
        config_uri = mlflow.get_artifact_uri(file_name)
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=config_uri, dst_path=f"config-{run_id}"
        )
        with open(local_path, "r") as fp:
            config = json.load(fp)
        shutil.rmtree(f.absolute().__str__())
    return config


def evaluate(
    predictions: List[Any],
    targets: List[Any],
    metric: Any,
    run_id: str,
) -> Any:
    """Evaluate metric for specified model predictions and labels."""
    _require_mlflow()
    eval_df = pd.DataFrame(
        {
            "predictions": predictions,
            "targets": targets,
        }
    )
    with mlflow.start_run(run_id=run_id):
        result = mlflow.evaluate(
            data=eval_df,
            targets="targets",
            predictions="predictions",
            extra_metrics=[metric],
        )
        return result


def create_minio_manager() -> MinioManager:
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    minio_secure = "true" in os.getenv("MINIO_SECURE", "false").lower()
    minio_bucket_env = os.getenv("MINIO_BUCKET")
    if minio_bucket_env is None:
        raise RuntimeError(
            "MINIO_BUCKET env var is not set; required for MLflow artifact storage"
        )
    return MinioManager(
        bucket_name=minio_bucket_env,
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=minio_secure,
    )


def get_experiment_id_by_name(experiment_name: str) -> str:
    _require_mlflow()
    expr = mlflow.get_experiment_by_name(experiment_name)
    if expr is None:
        raise Exception(f"experiment not exists: {experiment_name}")
    return str(expr.experiment_id)


def load_trainer_state(
    minio_manager: MinioManager,
    experiment_id: str,
    artifact_name: str,
    run_id: str,
    map_location: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load a trainer-state checkpoint from MinIO.

    Returns None if the artifact does not exist.
    """
    artifact_stem = pathlib.Path(artifact_name).stem
    with TempDir(remove_on_exit=True) as tmp:
        local_dir = tmp.path()
        minio_manager.download_dir(
            f"{experiment_id}/{run_id}/artifacts/{artifact_stem}", local_dir
        )
        local_file_path = os.path.join(local_dir, artifact_name)
        if not os.path.exists(local_file_path):
            return None
        return torch.load(local_file_path, map_location=map_location)


class MLflowTrackingRun:
    """MLflowTrackingRun is a user-level MLflow run. It starts an active run for tracking.

    Notes:
        This project uses MinIO directly for large artifacts (models/checkpoints)
        instead of the default MLflow artifact APIs for performance and reliability.

        Artifact layout (MinIO):
            All artifacts are stored under `self.base_path`.

            For trainer-state checkpoints we store them as a directory artifact:
                {base_path}/{artifact_stem}/{artifact_name}

            Example (artifact_name='trainer_state_latest.pt'):
                {base_path}/trainer_state_latest/trainer_state_latest.pt

            This keeps consistency with log_model/log_model_state_dict which also upload
            directories.
    """

    def __init__(
        self,
        experiment_name: str,
        target_bucket: str,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
        device: str = "cpu",
        parent_run_id: Optional[str] = None,
    ):
        _require_mlflow()
        self.experiment_id = get_experiment_id_by_name(experiment_name)
        self._device = device
        self._nested = nested
        self.parent_run_id = parent_run_id
        self._run = mlflow.start_run(
            run_id=run_id,
            run_name=run_name,
            nested=self._nested,
            parent_run_id=self.parent_run_id,
        )
        self.base_path = f"{self.experiment_id}/{self._run.info.run_id}/artifacts"
        self.minio_manager = create_minio_manager()
        self._signature = None

    def _start_run(self):
        if (
            mlflow.active_run() is None
            or mlflow.active_run().info.run_id != self._run.info.run_id
        ):
            self._run = mlflow.start_run(
                run_id=self._run.info.run_id,
                run_name=self._run.info.run_name,
                nested=self._nested,
                parent_run_id=self.parent_run_id,
            )

    def log(
        self,
        params: dict,
        model: Optional[nn.Module] = None,
        dataset: Any = None,
        dataset_name: Optional[str] = None,
        dataset_context: str = "training",
        dataset_truncate: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        self._start_run()
        # Log training parameters.
        mlflow.log_params(params)
        # Log tags.
        if tags is not None:
            mlflow.set_tags(tags)
        # Log model summary.
        if model is not None:
            path = f"model_summary.txt"
            with pathlib.Path(path).open("w") as f:
                s = str(summary(model, verbose=0))
                f.write(s)
                logger.info(f"Model summary:\n{s}")
                f.write(f"\n\n{repr(model)}")
            mlflow.log_artifact(path)
            pathlib.Path(path).unlink()
        # Log dataset
        if dataset is not None:
            sample_features_list = []
            sample_labels_list = []
            sample_size = (
                dataset_truncate if dataset_truncate is not None else len(dataset)
            )
            for i in range(min(sample_size, len(dataset))):
                try:
                    X, y = next(dataset)
                except Exception as e:
                    if "too many values to unpack" in str(e):
                        X = next(dataset)
                        y = None
                    else:
                        X = None
                        y = None
                        logger.warning(f"Fail to get dataset sample: {e}")
                sample_features_list.append(X)
                sample_labels_list.append(y)
            sample_features_np = np.array(sample_features_list)
            sample_labels_np = np.array(sample_labels_list)
            mldataset = mlflow.data.from_numpy(
                features=sample_features_np, targets=sample_labels_np, name=dataset_name
            )
            mlflow.log_input(mldataset, context=dataset_context)

    def log_model_state_dict(self, model: nn.Module, model_name: str = "dummy_model"):
        """Log a model state_dict directly to minio server."""
        with TempDir(remove_on_exit=True) as tmp:
            local_path = tmp.path()
            mlflow.pytorch.save_state_dict(
                state_dict={"model": model.state_dict()}, path=local_path
            )
            self.minio_manager.upload_dir(local_path, f"{self.base_path}/{model_name}")
        # mlflow.pytorch.log_state_dict(
        #     {"model": model.state_dict()}, artifact_path=model_name
        # )

    def log_torch_artifact(self, payload: Any, artifact_name: str) -> None:
        """Log an arbitrary torch-serializable payload as a MinIO artifact."""
        artifact_stem = pathlib.Path(artifact_name).stem
        with TempDir(remove_on_exit=True) as tmp:
            local_dir = tmp.path()
            local_file_path = os.path.join(local_dir, artifact_name)
            torch.save(payload, local_file_path)
            self.minio_manager.upload_dir(
                local_dir, f"{self.base_path}/{artifact_stem}"
            )

    def load_torch_artifact(
        self,
        artifact_name: str,
        run_id: str,
        map_location: Optional[str] = None,
    ) -> Optional[Any]:
        """Load a torch-serialized payload from MinIO.

        Returns None if the artifact does not exist.
        """
        artifact_stem = pathlib.Path(artifact_name).stem
        with TempDir(remove_on_exit=True) as tmp:
            local_dir = tmp.path()
            self.minio_manager.download_dir(
                f"{self.experiment_id}/{run_id}/artifacts/{artifact_stem}", local_dir
            )
            local_file_path = os.path.join(local_dir, artifact_name)
            if not os.path.exists(local_file_path):
                return None
            return torch.load(local_file_path, map_location=map_location)

    def log_trainer_state(self, state: Dict[str, Any], artifact_name: str) -> None:
        """Save a full trainer-state checkpoint (epoch-level).

        The checkpoint is stored as a directory artifact (for MinIO parity with models):
            {base_path}/{artifact_stem}/{artifact_name}

        Args:
            state: Serializable state dict (torch.save).
            artifact_name: Filename of the checkpoint (e.g., 'trainer_state_latest.pt').
        """
        artifact_stem = pathlib.Path(artifact_name).stem
        with TempDir(remove_on_exit=True) as tmp:
            local_dir = tmp.path()
            local_file_path = os.path.join(local_dir, artifact_name)
            torch.save(state, local_file_path)
            self.minio_manager.upload_dir(
                local_dir, f"{self.base_path}/{artifact_stem}"
            )

    def load_trainer_state(
        self, artifact_name: str, run_id: str, map_location: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load a trainer-state checkpoint from MinIO.

        Returns None if the artifact does not exist.
        """
        artifact_stem = pathlib.Path(artifact_name).stem
        with TempDir(remove_on_exit=True) as tmp:
            local_dir = tmp.path()
            self.minio_manager.download_dir(
                f"{self.experiment_id}/{run_id}/artifacts/{artifact_stem}", local_dir
            )
            local_file_path = os.path.join(local_dir, artifact_name)
            if not os.path.exists(local_file_path):
                return None
            return torch.load(local_file_path, map_location=map_location)

    def log_model(
        self,
        model: Optional[nn.Module] = None,
        model_name: str = "dummy_model",
        model_input_sample: Optional[Union[torch.Tensor, Tuple[Any, ...]]] = None,
    ):
        """Log a model.

        Args:
            model (nn.Module): The model to log.
            model_name (str): The name of the model in MLflow.
            model_input_sample (torch.Tensor | Tuple): A sample input (or tuple of inputs) to infer the model signature. The signature will be cached after the first non-None input_example.
        """
        self._start_run()
        # Construct the model signature.
        if (
            self._signature is None
            and model is not None
            and model_input_sample is not None
        ):
            model.to(self._device)

            sample = model_input_sample

            # Normalize sample into model_args (tuple of args to pass to model)
            if isinstance(sample, (tuple, list)):
                model_args = tuple(sample)
                input_example = sample
            else:
                model_args = (sample,)
                input_example = sample

            # Move any tensors in args to device
            def _to_device(x):
                return x.to(self._device) if isinstance(x, torch.Tensor) else x

            model_args = tuple(_to_device(x) for x in model_args)

            # Run model on the sample to get an output example
            with torch.no_grad():
                try:
                    y = model(*model_args)
                except TypeError:
                    # Fallback: some models expect a single positional argument that is a tuple/list
                    if isinstance(input_example, (tuple, list)):
                        try:
                            y = model(input_example)
                        except Exception:
                            y = None
                    else:
                        y = None

            # If model returned a tuple/list, take the first element as prediction tensor
            if isinstance(y, (tuple, list)):
                y = y[0]

            # Detach tensor outputs
            if isinstance(y, torch.Tensor):
                y = y.detach()

            # Prepare input example for signature
            if isinstance(input_example, torch.Tensor):
                input_for_sig = input_example.cpu().numpy()
            elif isinstance(input_example, (tuple, list)):
                # Convert multiple inputs into a dict of numpy arrays
                input_for_sig = {
                    f"input_{i}": (
                        v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
                    )
                    for i, v in enumerate(input_example)
                }
            else:
                try:
                    input_for_sig = np.array(input_example)
                except Exception:
                    input_for_sig = input_example

            # Prepare output for signature
            if isinstance(y, torch.Tensor):
                output_for_sig = y.cpu().numpy()
            else:
                output_for_sig = y

            try:
                self._signature = mlflow.models.infer_signature(
                    input_for_sig, output_for_sig
                )
            except Exception:
                # If signature inference fails, leave signature as None
                self._signature = None

        # Save the trained model to MLflow.
        if model is not None:
            with TempDir(remove_on_exit=True) as tmp:
                local_path = tmp.path()
                mlflow.pytorch.save_model(model, local_path, signature=self._signature)
                self.minio_manager.upload_dir(
                    local_path, f"{self.base_path}/{model_name}"
                )
            # if signature is not None:
            #     mlflow.pytorch.log_model(model, model_name, signature=signature)
            # else:
            #     mlflow.pytorch.log_model(model, model_name)

    def close(self, is_exception=False, is_killed=False):
        state = "FINISHED"
        if is_killed:
            state = "KILLED"
        if is_exception:
            state = "FAILED"
        active_run = mlflow.active_run()
        if active_run is not None and self._run.info.run_id == active_run.info.run_id:
            mlflow.end_run(state)
        else:
            logging.warning(
                f"Run {self._run.info.run_id} is not the active run, cannot end it."
            )

# pyright: ignore-all

from functools import partial
import mlflow as _ml
from torch import autocast, distributed
from torch.nn.parallel import DistributedDataParallel
from recipe.utils import ddp
from recipe.utils import mlflow_util
from recipe.utils.ddp import setup_ddp
from torch.amp.grad_scaler import GradScaler
import shutil
import math
import traceback
import os
import pathlib
import random
import string
import threading
import tempfile
from typing import Any, Dict, List, Tuple, Optional, Callable, cast
from collections.abc import Sized

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchtune
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils import log_rank_zero
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtune.training.metric_logging import (
    DiskLogger,
    StdoutLogger,
    MetricLoggerInterface,
)
import torchtune.training
from abc import abstractmethod
import logging

from typing import Any as _Any

import mlflow

# Make the module effectively "Any" for optional-dependency scenarios.
mlflow: _Any = mlflow  # type: ignore

from tqdm import tqdm

from recipe.config import Config
from recipe.utils.train_util import (
    EarlyStopping,
    total_grad_norm,
)
from recipe.utils.mlflow_util import (
    setup_mlflow,
    MLflowTrackingRun,
    MinioManager,
    DEFAULT_BUCKET,
    is_mlflow_available,
    create_minio_manager,
    get_experiment_id_by_name,
)
from torchvision.transforms.v2.functional import to_pil_image

# from dataclasses import dataclass, field, asdict
import concurrent
import concurrent.futures

from recipe.utils.model_util import checkpoint_wrap

DefaultMLflowDatasetTruncate = 1


class BasicRecipe(FTRecipeInterface):
    """Basic trainer base class usable as a torchtune recipe and as a Ray Tune Trainable.

    Subclass BasicRecipe and implement the abstract hooks below. For a full
    reference implementation, see workspace/digit_recognizer/train.py::DigitRecognizer.

    What you must implement and set:
    - _setup_data():
        - self._train_dataset: torch.utils.data.Dataset
        - self._val_dataset: torch.utils.data.Dataset
        - self._dataloader: torch.utils.data.DataLoader (training)
        - self._val_dataloader: torch.utils.data.DataLoader (validation)
        - self._num_labels: int (distinct class count)
    - _setup_model():
        - self._model: nn.Module (moved to self._device)
        - self._loss_fn: torch.nn.modules.loss._Loss (e.g., nn.CrossEntropyLoss)
        - Optionally run a dummy forward with _get_input_sample() to initialize lazy
          layers and help infer the MLflow model signature.
    - _setup_metric():
        - self._metric_fn: callable (stateful torchmetrics is recommended)
    - _setup_optimizer():
        - self._optimizer: torch.optim.Optimizer using self._model.parameters()
    - _setup_lr_scheduler():
        - self._lr_scheduler (optional, can be None)
    - _batch_to_device(batch): move batch fields to self._device and return the same structure
    - _loss_step(batch) -> (logits, loss): compute forward and training loss
    - _metric_step(batch, logits) -> value: compute a per-batch metric value (float-like)
    - _predict(batch): produce class predictions for visualization/evaluation

    Training loop overview:
    1) setup() calls all hooks, validates configuration, optionally initializes DDP,
       sets up MLflow, applies optional activation checkpointing, resumes trainer state,
       and wraps the model with DDP only after optimizer/scheduler state is ready.
    2) train() runs epochs: _train_step() per batch + _validation_step() per epoch.
    3) Early stopping and best-epoch checkpointing are supported when configured.
    4) When cfg.use_ddp=True, checkpoint writes, inference preview, and MLflow run
       mutations happen on rank zero; validation aggregation and resume happen on
       every rank. tqdm bars default to rank zero only, unless
       cfg.Debug.ddp_rank_progress_bars=True.

    Examples
    --------
    Minimal subclass skeleton:
        class MyRecipe(BasicRecipe):
            def _setup_data(self):
                from torch.utils.data import TensorDataset, DataLoader, random_split
                X = torch.randn(1000, 32)
                y = torch.randint(0, 2, (1000,))
                ds = TensorDataset(X, y)
                self._train_dataset, self._val_dataset = random_split(ds, [800, 200])
                self._num_labels = 2
                self._dataloader = DataLoader(self._train_dataset, batch_size=self.cfg.Train.batch_size, shuffle=True)
                self._val_dataloader = DataLoader(self._val_dataset, batch_size=self.cfg.Train.batch_size)

            def _setup_model(self):
                self._model = nn.Sequential(nn.Linear(32, 2)).to(self._device)
                self._loss_fn = nn.CrossEntropyLoss()
                # Optionally initialize lazy modules and aid MLflow signature inference
                self._model(self._get_input_sample().to(self._device))

            def _setup_metric(self):
                from torchmetrics import Accuracy
                self._metric_fn = Accuracy("multiclass", num_classes=self._num_labels)

            def _setup_optimizer(self):
                self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.cfg.Train.learning_rate)

            def _setup_lr_scheduler(self):
                self._lr_scheduler = None

            def _batch_to_device(self, batch):
                x, y = batch
                return x.to(self._device), y.to(self._device)

            def _loss_step(self, batch):
                x, y = batch
                logits = self._model(x)
                loss = self._loss_fn(logits, y)
                return logits, loss

            def _metric_step(self, batch, logits):
                _, y = batch
                preds = logits.argmax(dim=1)
                return self._metric_fn(preds, y)

            def _predict(self, batch):
                logits = self._model(batch).softmax(dim=1)
                return int(logits.argmax(dim=1)[0])

    Enabling the memory profiler (uses torch.profiler.schedule):
        cfg.Debug.enable_profiler = True
        cfg.Debug.skip_steps_to_profile = 10   # schedule(skip_first=10)
        cfg.Debug.wait_steps_between_repeat = 1 # schedule(wait=1)
        cfg.Debug.active_steps_in_one_repeat = 5 # schedule(active=5)
        cfg.Debug.repeats = 2                  # schedule(repeat=2), warmup fixed to 0

    Inference Preview Feature
    -------------------------
    Purpose: Lightweight, periodic qualitative + quantitative snapshot of model predictions
    during training without running a full validation epoch or full generation loop each step.

    Activation:
        cfg.InferencePreview.enable = True (default in some configs)
        cfg.InferencePreview.every_n_steps: log cadence (only when global_step > 0)
        cfg.InferencePreview.num_samples: number of cached batches sampled once from
            each specified dataloader in cfg.InferencePreview.sample_from
         cfg.InferencePreview.sample_from: list of 'train' and/or 'val'
        cfg.InferencePreview.log_as_artifact: if True, a JSON artifact named
            inference_preview_step_<global_step>.json is logged containing structured
            outputs for each cached sample.
        In DDP mode, preview caching and preview logging run on rank zero only.

    Mechanics:
        1. setup() calls _prepare_inference_preview_samples() exactly once which caches
           the first N batches (num_samples) from each chosen dataloader.
        2. At each logging step (in _log_train_step) _maybe_log_inference_preview() is
           invoked; it returns immediately unless:
              - preview enabled
              - global_step > 0
              - global_step % every_n_steps == 0
              - cached samples exist
        3. For each cached sample _inference_preview(sample) is called (subclass must
           override) and should return a dict of scalar + textual fields (e.g., {'bleu': 0.42, 'pred': '...', 'ref': '...'}).
        4. Scalar int/float values are logged as MLflow metrics with key pattern:
           preview_<field>_<sample_index>. Non‑scalar fields are skipped (but included in
           the optional artifact JSON for inspection).
        5. When log_as_artifact is True all sample dicts for that step are bundled into
           the artifact inference_preview_step_<global_step>.json.

    Refreshing Samples:
        If you modify or replace the underlying dataloaders after setup() (e.g., in a
        smoke test or curriculum scenario) call recache_inference_preview_samples() to
        clear the previous cache and repopulate.

    Failure / Disable Logic:
        Any exception thrown while caching or generating preview results causes the
        feature to auto‑disable (sets _inference_preview_disabled=True) to avoid
        interfering with training stability.

    Performance Notes:
        - Samples are cached once; no repeated dataloader iteration overhead each step.
        - Generation / forward logic in _inference_preview should be fast and run under
          torch.no_grad() (enforced here). Heavy decoding should reduce cadence or number
          of samples.
        - Skips at step 0 to avoid logging before any optimization has occurred.

    Customization Points for Subclasses:
        - Override _inference_preview(self, sample) -> Dict[str, Any]
        - Adjust config knobs above for cadence, sample source, artifact logging.
        - Use additional cfg fields (e.g., inference_max_seq_len) inside the override.

    Testing:
        A minimal smoke test can shrink model size, limit dataloader to 1 batch, set
        every_n_steps=1, and assert at least one preview_bleu_* (or other metric) plus
        an inference_preview_step_*.json artifact exists in the MLflow run.

    Basic usage (non‑Tune mode):
        >>> recipe = DigitRecognizer(is_tune_mode=False)
        >>> recipe.setup()
        >>> recipe.safe_train()
        >>> recipe.cleanup()

    Checkpointing & Resume
    ----------------------
    This trainer supports two complementary persistence mechanisms:

    1) Model-weight checkpoints ("model_*.pt")
       - Local (recommended when MLflow is disabled):
         - Enable with `cfg.Checkpoint.enable_local_ckp=True`.
         - Files are written under `cfg.Checkpoint.checkpoint_dir`.
         - If enabled, the directory is created and validated as writable in setup().

       - MLflow model/artifact logging (requires MLflow enabled):
         - When a new best epoch is observed (EarlyStopping), `save_checkpoint(epoch)`
           also logs the payload to MLflow.
         - On cleanup(), the final payload is logged to MLflow as a *_final artifact.
         - In DDP mode these writes are performed on rank zero only.

       Payload customization:
       - Override `_get_model_saving_payload(epoch, is_final=...)`.
       - If it returns an `nn.Module`:
         - local: saves `state_dict()`
         - MLflow: logs a full MLflow model (signature inference when available)
       - If it returns any other torch-serializable object:
         - local: `torch.save(payload, ...)`
         - MLflow: uploads as a torch artifact (not an MLflow model)

    2) Trainer-state checkpoints (epoch-level resume; MLflow-only)
       - Enable upload with `cfg.Checkpoint.enable_state_ckp_to_mlflow=True`.
       - The state is logged each epoch (and once at the end) under
         `cfg.Checkpoint.resume_state_artifact_name` (default: `trainer_state_latest.pt`).
       - Enable restore with `cfg.Checkpoint.resume_from_mlflow=True`.
       - In DDP mode every rank restores trainer state before the model is wrapped with
         `DistributedDataParallel`, but only rank zero uploads fresh trainer-state
         checkpoints and logs "resumed_*" MLflow params.
       - If `cfg.Checkpoint.resume_strict=True`, setup() raises when no state exists.
       - RNG restore is controlled by `cfg.Checkpoint.store_rng_state`.

    Interaction with `cfg.MLflow.enable`:
    - If `cfg.MLflow.enable=False`, all MLflow operations are skipped.
      `enable_state_ckp_to_mlflow` and `resume_from_mlflow` are forced to False.
    - If MLflow is disabled AND `enable_local_ckp=False`, setup() raises because there
      would be no persistence mechanism.

    Interaction with `cfg.use_ddp`:
    - `cfg.device` becomes informational; each process runs on `cuda:LOCAL_RANK`.
    - The training DataLoader must expose a sampler or batch sampler with `set_epoch()`.
    - Rank zero owns metric logging, checkpoint writes, and MLflow run mutations.
      tqdm bars are rank-zero-only by default, or per-rank when
      `cfg.Debug.ddp_rank_progress_bars=True`.
    - Every rank still runs forward/backward/optimizer work, participates in validation
      reduction, and performs resume loading so local state stays aligned.

    Show cases (minimal examples)
    - MLflow ON, no local checkpoints (default-ish):
      ```python
      cfg.MLflow.enable = True
      cfg.Checkpoint.enable_local_ckp = False
      ```

    - MLflow OFF, local checkpoints ON (run without mlflow installed):
      ```python
      cfg.MLflow.enable = False
      cfg.Checkpoint.enable_local_ckp = True
      cfg.Checkpoint.checkpoint_dir = "./checkpoints"
      ```

    - MLflow ON, resume training from last logged state:
      ```python
      cfg.MLflow.enable = True
      cfg.Checkpoint.enable_state_ckp_to_mlflow = True
      cfg.Checkpoint.resume_from_mlflow = True
      cfg.Checkpoint.resume_strict = False
      cfg.Checkpoint.mlflow_run_id = "..."
      ```
    Overwrite _after_resume_from_checkpoint to add any custom logic after resuming from checkpoint.
    For instance:
    ```python
        def _after_resume_from_checkpoint(self):
        # reset early stop
        self._setup_early_stop()
    ```


    See also:
    - recipe.runner.Runner and recipe.tune.AdaptiveTrainableRecipe for Ray Tune usage.

    Args:
        cfg (config.Config): configuration object for training, logging, and tracking.
    """

    def __init__(self, cfg: Config):
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
        warnings.filterwarnings("ignore", category=UserWarning, module="ray")
        super().__init__()
        self.cfg: Config = cfg
        if self.cfg.use_ddp:
            self._local_rank, self._global_rank, self._world_size = setup_ddp()
            self._device = f"cuda:{self._local_rank}"
        else:
            self._device = cfg.device
        self.epochs_run: int = 0
        self.global_step: int = 0
        self._model: Optional[nn.Module] = None
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._num_labels: int = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._loss_fn: Optional[nn.modules.loss._Loss] = None
        self._metric_fn: Optional[Callable[..., Any]] = None
        self._dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None
        self._dist_sampler: Optional[Any] = None
        self._lr_scheduler: Optional[Any] = None
        self._early_stop: Optional[EarlyStopping] = None
        self._mlflow_run: Optional[MLflowTrackingRun] = None
        self._minio_manager: Optional[MinioManager] = None
        self._mlflow_experiment_id: Optional[str] = None
        # memory debugger
        self._profiler: Optional[torch.profiler.profile] = None
        self._mem_record_running = False
        self._profiler_running = False
        self._scaler: GradScaler = GradScaler(
            device=self._device, enabled=self.cfg.Train.enable_amp
        )

        self._metric_logger: Optional[MetricLoggerInterface] = None
        if self.cfg.MetricLogger.enable:
            self._metric_logger = (
                DiskLogger(log_dir=cfg.MetricLogger.log_dir)
                if cfg.MetricLogger.to_disk
                else StdoutLogger()
            )

        self._logger: logging.Logger = cfg.logger or torchtune.utils.get_logger("INFO")

        self._params: Dict[str, Any] = dict()
        self.mlflow_is_exception: bool = False
        self.mlflow_is_killed: bool = False
        self._pool: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
        )
        self._background_tasks: List[concurrent.futures.Future] = []
        self._mlflow_lock: threading.Lock = threading.Lock()

        self._max_metrics: Dict[str, float] = dict()
        # inference preview cache
        self._inference_preview_samples: List[Any] = []
        self._inference_preview_disabled: bool = False
        self._inference_preview_cached: bool = False
        # subclass may override _inference_preview(sample) dynamically; default impl below

    def setup(self, **kwargs):
        """Run all setup hooks and services in the correct order.

        This prepares data, model, metric, optimizer, early stopping, LR scheduler,
        checkpoint cleanup, MLflow tracking, and optional activation checkpointing.
        The metric is moved to the selected device when supported.

        DDP behavior:
        - When cfg.use_ddp=True, setup() expects torch.distributed launcher env vars
          to already exist and requires the training dataloader to expose a sampler
          or batch sampler with set_epoch().
        - Resume happens before DDP wrapping so checkpoints use the underlying module
          state_dict format instead of DDP wrapper state.
        - Rank-zero-only side effects in setup() include checkpoint cleanup, MLflow run
          creation/mutation, memory debugger setup, and inference preview caching.
        - Progress bars are controlled later in train_step: rank-zero-only by default,
          or one bar per rank when cfg.Debug.ddp_rank_progress_bars=True.

        Returns:
            None
        """

        if ddp.is_rank_zero():
            self._setup_memory_debugger()
        self._setup_data()
        if self.cfg.use_ddp:
            self._dist_sampler = self._resolve_epoch_sampler()
            assert self._dist_sampler is not None, (
                "Distributed training requires a train dataloader sampler with set_epoch(); use a DistributedSampler or batch sampler that forwards set_epoch()."
            )
        self._setup_model()
        # Optionally wrap model modules with activation checkpointing to save memory
        if self.cfg.Train.enable_activation_checkpointing:
            self._apply_activation_checkpointing()
            log_rank_zero(
                self._logger, "Enabled activation checkpointing for model modules"
            )
        self._setup_metric()
        # ensure metric on device if applicable
        if self._metric_fn is not None and hasattr(self._metric_fn, "to"):
            try:
                self._metric_fn.to(self._device)  # type: ignore
            except Exception:
                pass
        # self._setup_neptune()
        self._setup_optimizer()
        self._setup_early_stop()
        self._setup_lr_scheduler()
        self._validate_tracking_and_checkpoint_config()
        if ddp.is_rank_zero():
            self._clean_ckp()
        self._setup_mlflow()

        resumed = self._maybe_resume()
        if resumed:
            self._after_resume_from_checkpoint()
        if self.cfg.use_ddp:
            self._wrap_model_for_ddp()
        if ddp.is_rank_zero():
            self._prepare_inference_preview_samples()
        self._check()

    def _apply_activation_checkpointing(self):
        """Wrap all leaf submodules of self._model with activation checkpointing.

        This uses torch.utils.checkpoint to wrap each leaf module (modules without
        children) so that activations are recomputed during the backward pass,
        reducing peak memory usage at the cost of additional compute.

        Notes:
        - Run after _setup_model() so that subclasses can freely construct and move
          models to devices first. We modify the module hierarchy in-place.
        - By default, all leaf modules are wrapped. Use whitelist/blacklist parameters
          in checkpoint_wrap() to control which modules are wrapped.
        - Only leaf modules (modules without children) are wrapped to avoid double
          wrapping composite containers.
        """
        assert self._model is not None
        wrapped_modules = checkpoint_wrap(
            self._model,
            whitelist=self.cfg.Train.activation_checkpoint_white_list,
            blacklist=self.cfg.Train.activation_checkpoint_black_list,
        )
        # record config for tracking
        self.reserve_param(
            "enable_activation_checkpointing",
            self.cfg.Train.enable_activation_checkpointing,
        )
        self.reserve_param(
            "checkpoint_wrapped_modules",
            wrapped_modules,
        )
        log_rank_zero(self._logger, f"Checkpoint wrapped modules: {wrapped_modules}")

    def reserve_param(self, key: str, value: Any):
        """Save MLflow param and report later."""
        self._params[key] = value

    def _unwrap_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        resolved = self._model if model is None else model
        if resolved is None:
            raise RuntimeError("Model is not initialized")
        if isinstance(resolved, DistributedDataParallel):
            return resolved.module
        return resolved

    def _resolve_epoch_sampler(self) -> Optional[Any]:
        if self._dataloader is None:
            return None
        for attr in ("batch_sampler", "sampler"):
            sampler = getattr(self._dataloader, attr, None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                return sampler
        return None

    def _wrap_model_for_ddp(self) -> None:
        if not self.cfg.use_ddp:
            return
        assert self._model is not None
        if isinstance(self._model, DistributedDataParallel):
            return
        self._model = DistributedDataParallel(
            self._model,
            device_ids=[self._local_rank],
            output_device=self._local_rank,
        )
        self._logger.info(
            f"Wrapped model with DistributedDataParallel on device {self._device}"
        )

    def _should_show_progress_bar(self) -> bool:
        if not self.cfg.use_ddp:
            return True
        return bool(self.cfg.Debug.ddp_rank_progress_bars) or ddp.is_rank_zero()

    def _create_progress_bar(self, curr_epoch: int) -> Optional[tqdm]:
        if self._dataloader is None or not self._should_show_progress_bar():
            return None
        desc = f"Epoch {curr_epoch}"
        kwargs: Dict[str, Any] = {}
        if self.cfg.use_ddp and self.cfg.Debug.ddp_rank_progress_bars:
            desc = f"Rank {self._global_rank} Epoch {curr_epoch}"
            kwargs["position"] = self._local_rank
            kwargs["leave"] = True
        return tqdm(total=len(self._dataloader), desc=desc, **kwargs)

    def _validate_tracking_and_checkpoint_config(self) -> None:
        """Validate tracking + persistence config.

        Enforces that at least one persistence mechanism is enabled:
        - MLflow tracking (cfg.MLflow.enabled)
        - local checkpointing (cfg.Checkpoint.enable_local_ckp)

        If MLflow is disabled, this ensures local checkpointing has a valid directory.
        """

        mlflow_enabled = bool(self.cfg.MLflow.enabled)
        local_ckp_enabled = bool(self.cfg.Checkpoint.enable_local_ckp)

        if not mlflow_enabled:
            if self.cfg.Checkpoint.enable_state_ckp_to_mlflow:
                self._logger.warning(
                    "Checkpoint.enable_state_ckp_to_mlflow=True but MLflow is disabled; forcing to False"
                )
                self.cfg.Checkpoint.enable_state_ckp_to_mlflow = False
            if self.cfg.Checkpoint.resume_from_mlflow:
                self._logger.warning(
                    "Checkpoint.resume_from_mlflow=True but MLflow is disabled; forcing to False"
                )
                self.cfg.Checkpoint.resume_from_mlflow = False

        if (not mlflow_enabled) and (not local_ckp_enabled):
            raise ValueError(
                "Invalid configuration: MLflow is disabled (cfg.MLflow.enable=False) and local "
                "checkpointing is disabled (cfg.Checkpoint.enable_local_ckp=False). Enable at least one."
            )

        if local_ckp_enabled:
            checkpoint_dir = pathlib.Path(self.cfg.Checkpoint.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.Checkpoint.resume_from_mlflow:
            if not self.cfg.Checkpoint.mlflow_run_id:
                raise ValueError(
                    f"Checkpoint.resume_from_mlflow=True but no mlflow_run_id set"
                )

    def _check(self):
        """Validate that core components are initialized and log dataset stats.

        Raises:
            AssertionError: If any of datasets, dataloaders, model, loss, metric, or
                optimizer are missing or empty.
        """
        self._logger.info("Checking training setup...")

        if self.cfg.Visualization.vis_enable:
            self._logger.warning("_vis_enable not supported yet, set to False.")
            self.cfg.Visualization.vis_enable = False
        assert self._train_dataset is not None and len(self._train_dataset) > 0, (
            "train dataset is empty, you should set it in _setup_data"
        )  # type: ignore
        if not self.cfg.Train.no_validation:
            assert self._val_dataset is not None and len(self._val_dataset) > 0, (
                "validation dataset is empty, you should set it in _setup_data"
            )  # type: ignore
        assert self._dataloader is not None and len(self._dataloader) > 0, (
            "train dataloader is empty, you should set it in _setup_data"
        )  # type: ignore
        if not self.cfg.Train.no_validation:
            assert self._val_dataloader is not None and len(self._val_dataloader) > 0, (
                "validation dataloader is empty, you should set it in _setup_data"
            )  # type: ignore
        assert self._model is not None, (
            "model is not set, you should set it in _setup_model"
        )
        # assert self._loss_fn is not None, (
        #     "loss function is not set, you should set it in _setup_model"
        # )
        assert self._optimizer is not None, (
            "optimizer is not set, you should set it in _setup_optimizer"
        )

        # print config
        log_rank_zero(
            self._logger,
            f"training set: {len(self._train_dataset)}\nvalidation set: {len(self._val_dataset) if self._val_dataset else None}",  # type: ignore
        )
        log_rank_zero(
            self._logger,
            f"number of steps per epoch: {len(self._dataloader)}",
        )

    def description(self) -> tuple[str, Dict[str, Any]]:
        """Return a human-readable description and a structured dict of core settings.

        The summary covers model, datasets, optimizer, loss, metric, device, batch size,
        LR scheduler, early stopping, and tracking/logging configs.

        Returns:
            tuple[str, Dict[str, Any]]: (summary_string, details_dict)
        """

        def _name(x: Any) -> str:
            return x.__class__.__name__ if x is not None else "None"

        train_size = (
            len(self._train_dataset) if self._train_dataset is not None else None  # type: ignore
        )
        val_size = len(self._val_dataset) if self._val_dataset is not None else None  # type: ignore
        steps_per_epoch = (
            len(self._dataloader) if self._dataloader is not None else None  # type: ignore
        )
        val_steps = (
            len(self._val_dataloader) if self._val_dataloader is not None else None  # type: ignore
        )
        train_bs = (
            getattr(self._dataloader, "batch_size", None)
            if self._dataloader is not None
            else None
        )
        val_bs = (
            getattr(self._val_dataloader, "batch_size", None)
            if self._val_dataloader is not None
            else None
        )

        details: Dict[str, Any] = {
            "device": str(self._device),
            "model": _name(self._model),
            "loss_fn": _name(self._loss_fn),
            "metric_fn": _name(self._metric_fn),
            "optimizer": _name(self._optimizer),
            "lr_scheduler": _name(self._lr_scheduler),
            "num_labels": self._num_labels,
            "datasets": {
                "train": {
                    "type": _name(self._train_dataset),
                    "size": train_size,
                },
                "val": {
                    "type": _name(self._val_dataset),
                    "size": val_size,
                },
            },
            "dataloaders": {
                "train": {"batch_size": train_bs, "steps_per_epoch": steps_per_epoch},
                "val": {"batch_size": val_bs, "steps": val_steps},
            },
            "training": self.cfg.Train.model_dump(),
            "early_stopping": _name(self._early_stop),
            "tracking": {
                "mlflow": self.cfg.MLflow.model_dump(),
                "metric_logger": self.cfg.MetricLogger.model_dump(),
            },
            "checkpoint": self.cfg.Checkpoint.model_dump(),
            "visualization": self.cfg.Visualization.model_dump(),
        }

        if self._early_stop is not None:
            details["early_stopping"] = {
                "patience": self._early_stop._patience,
                "delta": self._early_stop._delta,
            }

        # Build concise summary string
        parts = [
            f"Model={details['model']}",
            f"Loss={details['loss_fn']}",
            f"Metric={details['metric_fn']}",
            f"Optim={details['optimizer']}",
            f"Sched={details['lr_scheduler']}",
            f"Device={details['device']}",
        ]
        if train_size is not None or val_size is not None:
            parts.append(f"Data(train/val)={train_size}/{val_size}")
        if steps_per_epoch is not None:
            parts.append(f"Steps/Epoch={steps_per_epoch}")
        parts.append(
            f"Batch={self.cfg.Train.batch_size} Epochs={self.cfg.Train.total_epochs}"
        )
        parts.append(
            f"MLflow(exp={self.cfg.MLflow.experiment_name}, model={self.cfg.MLflow.mlflow_model_name}, data={self.cfg.MLflow.dataset_name})"
        )

        summary = " | ".join(parts)
        return summary, details

    def _setup_memory_debugger(self):
        """Initialize memory debugging artifacts and optional profiler.

        Behavior:
        - Clears existing profiler output directory and creates a Chrome Trace log dir.
        - Configures a trace handler that writes both Chrome traces and an
          HTML memory timeline per profile capture.
        - If Debug.enable_snapshot is True but device != 'cuda', it is disabled.
        - If Debug.enable_profiler is True, creates a torch.profiler.profile using
          torch.profiler.schedule(skip_first=..., wait=..., warmup=0, active=..., repeat=...).

        Notes:
        - The profiler is entered (__enter__) here and advanced per batch by
          _trace_memory_profile().
        - Snapshotting uses torch.cuda.memory._record_memory_history() and
          _dump_snapshot() in _debug_memory_usage().
        """

        if not self.cfg.Debug.enable_profiler:
            return
        if self.cfg.Debug.enable_snapshot and not self.cfg.device.startswith("cuda"):
            log_rank_zero(
                self._logger,
                f"enable_snapshot is True, but device is not cuda, set to False",
            )
            return
        base_dir = pathlib.Path(self.cfg.Debug.save_memory_timeline_dir)
        if base_dir.exists():
            confirm = input(f"Delete existing profiler output dir {base_dir}? (y/n): ")
            if confirm.lower() == "y":
                shutil.rmtree(base_dir)

        # Chrome trace format - direct output directory without run subdirectory
        import datetime

        chrome_trace_dir = (
            pathlib.Path(self.cfg.Debug.save_memory_timeline_dir) / "chrome_trace"
        )

        if not chrome_trace_dir.exists():
            chrome_trace_dir.mkdir(parents=True)

        # Use Chrome trace format for direct visualization in chrome://tracing
        # Exports .json files that can be loaded directly in Chrome's trace viewer
        def chrome_trace_handler(prof: torch.profiler.profile):
            trace_path = (
                chrome_trace_dir
                / f"trace_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            log_rank_zero(self._logger, f"Exporting Chrome trace to: {trace_path}")
            prof.export_chrome_trace(str(trace_path))
            log_rank_zero(self._logger, f"Exported Chrome trace to: {trace_path}")

        log_rank_zero(
            self._logger,
            f"Chrome profiler traces will be saved to: {chrome_trace_dir}\n"
            f"  Format: Chrome Trace Format (.json files)\n"
            f"  View with: Open chrome://tracing in Chrome browser and load the .json file\n"
            f"  Or use: https://ui.perfetto.dev/ for advanced trace visualization",
        )

        def trace_handler(prof: torch.profiler.profile):
            p = (
                (
                    pathlib.Path(self.cfg.Debug.save_memory_timeline_dir)
                    / f"epoch_{self.epochs_run}_step_{self.global_step}.html"
                )
                .absolute()
                .__str__()
            )
            log_rank_zero(
                self._logger,
                f"Exporting profiler traces:\n"
                f"  - Chrome trace (JSON): {chrome_trace_dir}/*.json\n"
                f"  - Memory timeline (HTML): {p}",
            )
            if self.cfg.Debug.disable_trace_handler:
                log_rank_zero(
                    self._logger, f"Trace handler is disabled, skipping export."
                )
            else:
                chrome_trace_handler(prof)
            # Export memory timeline only when there are recorded events.
            # This avoids ValueError: min() arg is an empty sequence when running on CPU
            # or when no memory events were captured for the specified device.
            export_device = str(self._device)
            should_export = (
                export_device.startswith("cuda") and torch.cuda.is_available()
            )
            if not should_export:
                log_rank_zero(
                    self._logger,
                    f"Skipping export_memory_timeline (device={export_device}, cuda_available={torch.cuda.is_available()})",
                )
            else:
                try:
                    log_rank_zero(self._logger, f"Exporting memory timeline...")
                    prof.export_memory_timeline(
                        p,
                        device=export_device,
                    )
                except ValueError as e:
                    # No events captured; log and continue without failing training
                    log_rank_zero(
                        self._logger,
                        f"Skipping memory timeline export due to empty events: {e}",
                    )
                except Exception as e:
                    log_rank_zero(
                        self._logger,
                        f"Failed to export memory timeline: {e}",
                    )

        if self.cfg.Debug.enable_snapshot:
            if self.cfg.device != "cuda":
                log_rank_zero(
                    self._logger,
                    "enable_snapshot is True, but device is not cuda, set to False",
                )
                self.cfg.Debug.enable_snapshot = False
        if self.cfg.Debug.enable_profiler:
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=self.cfg.Debug.skip_steps_to_profile,
                    wait=self.cfg.Debug.wait_steps_between_repeat,
                    warmup=0,
                    active=self.cfg.Debug.active_steps_in_one_repeat,
                    repeat=self.cfg.Debug.repeats,
                ),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self._profiler.__enter__()
            self._profiler_running = True

    def _setup_mlflow(self):
        """Set up an MLflow run and log initial parameters and configuration.

        Logs:
        - Core hyperparameters and dataset/dataloader sizes
        - Serialized configuration (train-configuration.json)
        - A structured training details summary (train-details.json)

        Also stores reserved params (via reserve_param) as MLflow params.

        DDP behavior:
        - Every rank calls setup_mlflow(...) so MLflow client env vars and MinIO access
          are ready for resume_from_mlflow.
        - Only rank zero creates/attaches the tracked MLflow run and performs logging
          side effects such as params, config JSON, dataset samples, and model summary.
        """

        if not self.cfg.MLflow.enabled:
            self._logger.info("MLflow is disabled, skipping MLflow setup")
            return
        if not is_mlflow_available():
            raise ImportError(
                "MLflow is enabled in config but 'mlflow' is not installed. "
                "Either install mlflow or set cfg.MLflow.enable=False."
            )

        log_rank_zero(
            self._logger,
            f'setting up MLflow with experiment "{self.cfg.MLflow.experiment_name}"',
        )
        setup_mlflow(self.cfg.MLflow.experiment_name)
        if not ddp.is_rank_zero():
            self._minio_manager = create_minio_manager()
            self._mlflow_experiment_id = get_experiment_id_by_name(
                self.cfg.MLflow.experiment_name
            )
            return

        # Guard: if a run is already active and we are NOT nesting, end it to avoid duplicate start_run errors.
        try:
            if not self.cfg.MLflow.nested and _ml.active_run() is not None:
                _ml.end_run()
        except Exception:
            pass
        self._mlflow_run = MLflowTrackingRun(
            target_bucket=DEFAULT_BUCKET,
            experiment_name=self.cfg.MLflow.experiment_name,
            device=self._device,
            nested=self.cfg.MLflow.nested,
            parent_run_id=self.cfg.MLflow.parent_run_id,
            run_id=self.cfg.MLflow.run_id,
        )
        self._mlflow_experiment_id = self._mlflow_run.experiment_id
        self._minio_manager = self._mlflow_run.minio_manager
        assert self._model is not None
        assert self._dataloader is not None
        params = {
            "epochs": self.cfg.Train.total_epochs,
            "batch_size": self.cfg.Train.batch_size,
            "loss_function": self._loss_fn.__class__.__name__
            if self._loss_fn
            else None,
            "metric_function": self._metric_fn.__class__.__name__
            if self._metric_fn is not None
            else None,
            "optimizer": self._optimizer.__class__.__name__,
            "train_dataset_size": len(self._train_dataset),  # type: ignore
            "val_dataset_size": len(self._val_dataset) if self._val_dataset else None,  # type: ignore
            "train_steps_per_epoch": len(self._dataloader),  # type: ignore
            "val_steps_per_epoch": len(self._val_dataloader)
            if self._val_dataloader
            else None,  # type: ignore
        }
        if self._early_stop is not None:
            params.update(
                {
                    "early_stop_patience": self._early_stop._patience,
                    "early_stop_delta": self._early_stop._delta,
                }
            )
        self._mlflow_run.log(
            params=params,
            model=self._unwrap_model(),
            dataset=iter(self._dataloader)
            if not hasattr(self._dataloader, "__len__")
            else self._dataloader,
            dataset_name=self.cfg.MLflow.dataset_name,
            dataset_context="training",
            dataset_truncate=DefaultMLflowDatasetTruncate,
            tags={"mlflow.note.content": self.cfg.MLflow.desc},
        )

        d = self.cfg.model_dump(exclude={"logger"})
        # r = "".join([random.choice(string.ascii_letters) for _ in range(10)])
        if self.cfg.MLflow.enabled and self._mlflow_run is not None:
            mlflow.log_dict(d, f"train-configuration.json")
            mlflow.log_dict(self.description()[1], f"train-details.json")
            for k, v in self._params.items():
                mlflow.log_param(k, v)

    # def _setup_neptune(self):
    #     if not self.cfg.Neptune or not self.cfg.Neptune.enable_neptune:
    #         return
    #     if os.getenv("NEPTUNE_API_TOKEN", None) is None:
    #         raise Exception(f"requires NEPTUNE_API_TOKEN environment to be set")
    #     log_rank_zero(self._logger, f"Initializing neptune")
    # import neptune

    # self._run = neptune.init_run(
    #     project=self.cfg.Neptune.project,
    #     api_token=os.getenv("NEPTUNE_API_TOKEN"),
    #     name=self.cfg.Neptune.name,
    #     tags=self.cfg.Neptune.tags,
    #     dependencies=self.cfg.Neptune.dependencies,
    #     monitoring_namespace=self.cfg.Neptune.monitoring_namespace,
    # )

    def _setup_early_stop(self):
        """Initialize EarlyStopping helper using TrainerCfg parameters.

        Creates an EarlyStopping instance with patience and delta from cfg.Train and
        logs via the rank-zero logger when improvements or stops occur.
        """

        self._early_stop = EarlyStopping(
            patience=self.cfg.Train.early_stop_patience,
            delta=self.cfg.Train.early_stop_delta,
            trace_func=partial(log_rank_zero, self._logger),
        )

    def _get_model_saving_payload(self, epoch: int, *, is_final: bool) -> Any:
        """Return the payload to persist for a model save event.

        Default behavior (fallback): return `self._model` and the framework will:
        - save `self._model.state_dict()` locally
        - log the full model to MLflow (with signature inference when possible)

        To customize saving (e.g., save multiple component weights separately),
        override this method and return any torch-serializable object, for example:
            {
              "embedding": embedding.state_dict(),
              "gru": gru.state_dict(),
            }

        When returning a non-Module payload, the framework will:
        - `torch.save(payload, ...)` for local checkpoints
        - upload the same payload to MLflow/MinIO as a torch artifact (not as an MLflow model)
        """
        return self._unwrap_model()

    def _get_extra_state_for_checkpoint(self) -> Dict[str, Any]:
        """Subclass hook to include additional state in trainer-state checkpoints."""
        return {}

    def _load_extra_state_from_checkpoint(self, extra_state: Dict[str, Any]) -> None:
        """Subclass hook to restore additional state from a trainer-state checkpoint."""
        return

    def _get_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception as e:
                self._logger.warning(f"Failed to get CUDA RNG state: {e}")

        return state

    def _set_rng_state(self, rng_state: Dict[str, Any]) -> None:
        if "python" in rng_state:
            try:
                random.setstate(rng_state["python"])
            except Exception as e:
                self._logger.warning(f"Failed to set Python RNG state: {e}")
        if "numpy" in rng_state:
            try:
                np.random.set_state(rng_state["numpy"])
            except Exception as e:
                self._logger.warning(f"Failed to set NumPy RNG state: {e}")
        if "torch" in rng_state:
            try:
                torch.set_rng_state(rng_state["torch"])
            except Exception as e:
                self._logger.warning(f"Failed to set Torch RNG state: {e}")
        if "cuda" in rng_state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            except Exception as e:
                self._logger.warning(f"Failed to set CUDA RNG state: {e}")

    def _build_trainer_state_dict(self) -> Dict[str, Any]:
        assert self._optimizer is not None
        model = self._unwrap_model()
        state: Dict[str, Any] = {
            "epochs_run": int(self.epochs_run),
            "global_step": int(self.global_step),
            "model": model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": self._lr_scheduler.state_dict()
            if self._lr_scheduler is not None
            and hasattr(self._lr_scheduler, "state_dict")
            else None,
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "early_stop": self._early_stop.state_dict() if self._early_stop else None,
            "extra": self._get_extra_state_for_checkpoint(),
        }
        if self.cfg.Checkpoint.store_rng_state:
            state["rng_state"] = self._get_rng_state()
        return state

    def _load_trainer_state_dict(self, state: Dict[str, Any]) -> None:
        assert self._optimizer is not None
        model = self._unwrap_model()
        model.load_state_dict(state.get("model", {}))
        opt_state = state.get("optimizer", None)
        if opt_state is not None:
            self._optimizer.load_state_dict(opt_state)

        if self._lr_scheduler is not None and hasattr(
            self._lr_scheduler, "load_state_dict"
        ):
            sched_state = state.get("lr_scheduler", None)
            if sched_state is not None:
                try:
                    self._lr_scheduler.load_state_dict(sched_state)
                except Exception as e:
                    raise Exception(f"Failed to load lr_scheduler state: {e}")

        scaler_state = state.get("scaler", None)
        if scaler_state is not None:
            try:
                self._scaler.load_state_dict(scaler_state)
            except Exception as e:
                raise Exception(f"Failed to load GradScaler state: {e}")

        if self._early_stop is not None:
            es_state = state.get("early_stop", None)
            if es_state is not None:
                try:
                    self._early_stop.load_state_dict(es_state)
                except Exception as e:
                    raise Exception(f"Failed to load early_stop state: {e}")

        extra = state.get("extra", {})
        if isinstance(extra, dict):
            try:
                self._load_extra_state_from_checkpoint(extra)
            except Exception as e:
                raise Exception(f"Failed to load extra checkpoint state: {e}")

        self.epochs_run = int(state.get("epochs_run", 0))
        self.global_step = int(state.get("global_step", 0))

        if self.cfg.Checkpoint.store_rng_state:
            rng_state = state.get("rng_state", None)
            if isinstance(rng_state, dict):
                self._set_rng_state(rng_state)

    def _maybe_resume(self) -> bool:
        if self._maybe_resume_from_mlflow():
            return True
        if self._maybe_resume_from_local():
            return True
        return False

    def _maybe_resume_from_local(self) -> bool:
        if not self.cfg.Checkpoint.resume_from_path:
            return False
        if not pathlib.Path(self.cfg.Checkpoint.resume_state_artifact_name).exists():
            if self.cfg.Checkpoint.resume_strict:
                raise FileNotFoundError(
                    f"resume_state_artifact_name file not found: {self.cfg.Checkpoint.resume_state_artifact_name}"
                )
            else:
                self._logger.warning(
                    f"resume_state_artifact_name file not found: {self.cfg.Checkpoint.resume_state_artifact_name}"
                )
                return False
        self._logger.info(
            f"Attempting to resume trainer state from local path: {self.cfg.Checkpoint.resume_state_artifact_name}"
        )

        try:
            state = torch.load(
                self.cfg.Checkpoint.resume_state_artifact_name,
                map_location=self._device,
            )
        except Exception as e:
            if self.cfg.Checkpoint.resume_strict:
                raise
            self._logger.warning(f"Failed to load trainer state from local path: {e}")
            return False

        if state is None:
            msg = f"No trainer-state checkpoint found in local path {self.cfg.Checkpoint.resume_state_artifact_name}"
            if self.cfg.Checkpoint.resume_strict:
                raise FileNotFoundError(msg)
            self._logger.warning(msg)
            return False

        self._load_trainer_state_dict(state)

        if ddp.is_rank_zero():
            if self.cfg.MLflow.enabled and self._mlflow_run is not None:
                mlflow.log_param("resumed", True)
                mlflow.log_param("resumed_epochs_run", self.epochs_run)
                mlflow.log_param("resumed_global_step", self.global_step)
                mlflow.log_param(
                    "resumed_from_path", self.cfg.Checkpoint.resume_state_artifact_name
                )
        self._logger.info(
            f"Resumed trainer state from local path: epochs_run={self.epochs_run}, global_step={self.global_step}"
        )
        return True

    def _maybe_resume_from_mlflow(self) -> bool:
        if not self.cfg.Checkpoint.resume_from_mlflow:
            return False
        if not self.cfg.MLflow.enabled:
            self._logger.warning("resume_from_mlflow=True but MLflow is disabled")
            return False
        if not self.cfg.Checkpoint.mlflow_run_id:
            self._logger.warning("mlflow_run_id must be set to resume from MLflow")
            return False

        artifact_name = self.cfg.Checkpoint.resume_state_artifact_name
        self._logger.info(
            f"Attempting to resume trainer state from MLflow: {artifact_name}"
        )
        try:
            assert self._minio_manager is not None, (
                "MinIO manager is not initialized for MLflow access"
            )
            assert self._mlflow_experiment_id is not None, (
                "MLflow experiment ID is not set for loading checkpoints"
            )
            state = mlflow_util.load_trainer_state(
                self._minio_manager,
                self._mlflow_experiment_id,
                artifact_name=artifact_name,
                run_id=self.cfg.Checkpoint.mlflow_run_id,
                map_location=self._device,
            )
        except Exception as e:
            if self.cfg.Checkpoint.resume_strict:
                raise
            self._logger.warning(f"Failed to load trainer state from MLflow: {e}")
            return False

        if state is None:
            msg = (
                f"No trainer-state checkpoint found in MLflow artifact {artifact_name}"
            )
            if self.cfg.Checkpoint.resume_strict:
                raise FileNotFoundError(msg)
            self._logger.warning(msg)
            return False

        self._load_trainer_state_dict(state)

        if ddp.is_rank_zero():
            if self.cfg.MLflow.enabled and self._mlflow_run is not None:
                mlflow.log_param("resumed", True)
                mlflow.log_param("resumed_epochs_run", self.epochs_run)
                mlflow.log_param("resumed_global_step", self.global_step)
                mlflow.log_param(
                    "resumed_from_run_id", self.cfg.Checkpoint.mlflow_run_id
                )
        self._logger.info(
            f"Resumed trainer state: epochs_run={self.epochs_run}, global_step={self.global_step}"
        )
        return True

    def _get_input_sample(self) -> torch.Tensor:
        """Return a sample batch taken from self._dataloader without mutating its state.

        Notes:
        - We create a short‑lived iterator (iter(self._dataloader)) and take a single
          batch with next(). This does not advance any persistent iterator used by
          training loops, nor does it alter the DataLoader's internal state.
        - The returned value is the "features" part of the batch with a batch
          dimension already present (since it comes from the DataLoader).
        """
        if self._dataloader is None:
            raise RuntimeError("_get_input_sample: self._dataloader is not initialized")

        # New iterator so we don't disturb any ongoing/outer iteration over the dataloader
        it = iter(self._dataloader)
        try:
            batch = next(it)
        except StopIteration:
            raise RuntimeError("_get_input_sample: self._dataloader yielded no data")

        # Extract the feature tensor from common batch structures
        features = batch
        while isinstance(features, (list, tuple)) and len(features) > 0:
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise RuntimeError(
                f"_get_input_sample: unable to extract tensor from batch, got {type(features)}"
            )

        # Ensure we return a single-item batch to mirror previous behavior
        if features.dim() == 0:
            features = features.unsqueeze(0)
        else:
            features = features[:1]
        return features

    def before_get_input_example_check(self):
        """Hook called before _get_input_sample() is used in MLflow signature inference.
        Subclasses can override this method to perform any necessary setup or checks
        before a sample input is retrieved for MLflow model signature inference.
        """
        assert self._dataloader is not None, "self._dataloader is not initialized"

    @abstractmethod
    def _setup_metric(self):
        """Define and assign self._metric_fn.

        Requirements:
        - Must set: self._metric_fn to a callable metric.
        - If using torchmetrics, prefer a stateful metric (e.g., torchmetrics.Accuracy) so
          BasicRecipe can call reset()/compute() during validation.
        - If the metric implements .to(device), it will be moved to self._device in setup().
        - The metric should accept predictions and labels as passed by _metric_step.

        Example (see workspace/digit_recognizer/train.py::DigitRecognizer):
            from torchmetrics import Accuracy
            self._metric_fn = Accuracy("multiclass", num_classes=self._num_labels)
        """

    @abstractmethod
    def _setup_model(self):
        """Construct the model and loss function, and move the model to the device.

        Requirements:
        - Must set:
            - self._model: nn.Module (moved to self._device)
            - self._loss_fn: a torch loss function (e.g., nn.CrossEntropyLoss)
        - If your model contains lazy layers or BatchNorm-like modules, run a forward pass
          once with self._get_input_sample(). This initializes shapes and helps infer
          the MLflow model signature.
        - Typically use self._num_labels (populated in _setup_data) for the classifier head.

        Example (DigitRecognizer):
            self._model = CNN(
                self._num_labels,
                dropout=self.cfg.dropout,
                num_conv_layers=self.cfg.num_conv_layers,
            ).to(self._device)
            self._model(self._get_input_sample().to(self._device))
            self._loss_fn = nn.CrossEntropyLoss()
        """

    def _after_resume_from_checkpoint(self):
        """Hook called after resuming from a checkpoint.
        Subclasses can override this method to perform any necessary setup or adjustments
        after the trainer state has been restored from a checkpoint.
        """
        pass

    @abstractmethod
    def _setup_optimizer(self):
        """Instantiate and assign self._optimizer using self._model.parameters().

        Example:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        """

    def before_setup_optimizer_check(self):
        """Hook called before _setup_optimizer() is used to create the optimizer.
        Subclasses can override this method to perform any necessary setup or checks
        before the optimizer is instantiated.
        """
        assert self._model is not None, "self._model is not initialized"

    @abstractmethod
    def _setup_lr_scheduler(self):
        """Optionally assign self._lr_scheduler (can be None).

        Tips:
        - If you need total steps, compute:
              total_steps = self.cfg.Train.total_epochs * len(self._dataloader)
        - Example (cosine with warmup):
              from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup
              self._lr_scheduler = get_cosine_schedule_with_warmup(
                  self._optimizer,
                  num_training_steps=total_steps,
                  num_warmup_steps=int(0.05 * total_steps),
              )
        """

    @abstractmethod
    def _setup_data(self):
        """Prepare datasets, dataloaders, and number of labels.

        Requirements:
        - Must set:
            - self._train_dataset: torch.utils.data.Dataset
            - self._val_dataset: torch.utils.data.Dataset
            - self._dataloader: torch.utils.data.DataLoader (training)
            - self._val_dataloader: DataLoader (validation)
            - self._num_labels: int (distinct class count)
        - Batch size should use self.cfg.Train.batch_size.

        Example (DigitRecognizer):
            trainset = train_dataset(transform=...)
            self._train_dataset, self._val_dataset = random_split(
                trainset, [1 - cfg.val_ratio, cfg.val_ratio]
            )
            self._num_labels = len({int(y.item()) for _, y in trainset})
            self._dataloader = DataLoader(
                self._train_dataset, batch_size=cfg.Train.batch_size, shuffle=True
            )
            self._val_dataloader = DataLoader(
                self._val_dataset, batch_size=cfg.Train.batch_size, shuffle=False
            )
        """

    @abstractmethod
    def _loss_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute model outputs and loss for a single batch.

        Inputs:
        - batch: object produced by your DataLoader (often a tuple (features, labels)).
                 It should already be on self._device (handled by _batch_to_device()).

        Returns:
        - logits: torch.Tensor of unnormalized scores (e.g., [B, num_classes])
        - loss: scalar torch.Tensor computed using self._loss_fn

        Example (DigitRecognizer):
            features, labels = batch
            logits = self._model(features)
            loss = self._loss_fn(logits, labels)
            return logits, loss
        """

    @abstractmethod
    def _metric_step(self, batch, logits: torch.Tensor) -> torch.FloatTensor:
        """Calculate a per-batch metric given the batch and model logits.

        Notes:
        - If using torchmetrics, you may update internal state (e.g., self._metric_fn(...))
          and return the batch metric value. BasicRecipe will call compute() across the
          validation epoch if available.
        - If returning a value, it must be convertible to float.

        Example (DigitRecognizer):
            _, labels = batch
            preds = logits.softmax(dim=1).argmax(dim=1)
            return self._metric_fn(preds, labels)
        """

    def _predict(self, batch) -> torch.Tensor:
        """Produce predictions for visualization/evaluation.

        Inputs:
        - batch: features tensor or a full batch of features.

        Returns:
        - Predicted class index/indices (int or torch.Tensor of ints).

        Example:
            logits = self._model(batch).softmax(dim=1)
            return int(logits.argmax(dim=1)[0].item())
        """
        raise Exception("Not implemented")

    @abstractmethod
    def _batch_to_device(self, batch) -> Any:
        """Move the given batch to self._device and return it with the same structure.

        Example:
            features, labels = batch
            return features.to(self._device), labels.to(self._device)
        """

    def recache_inference_preview_samples(self):
        """Force re-caching of inference preview samples.

        Use when the underlying dataloader (train or val) has been swapped or
        modified after initial setup(). Safe to call multiple times; errors
        disable preview feature gracefully.
        """
        # Reset flags to allow re-population
        self._inference_preview_cached = False
        self._inference_preview_samples = []
        self._prepare_inference_preview_samples()

    def _prepare_inference_preview_samples(self):
        if not self.cfg.InferencePreview.enable:
            return
        if self._inference_preview_cached:
            return
        try:
            cached = []

            def _extract_primary_tensor(sample):
                x = sample
                # common batch structures: (features, labels), dicts, etc.
                if isinstance(x, dict) and len(x) > 0:
                    # take first value deterministically (sorted for reproducibility)
                    first_key = sorted(x.keys())[0]
                    x = x[first_key]
                while isinstance(x, (list, tuple)) and len(x) > 0:
                    x = x[0]
                return x

            def _is_empty(sample):
                if sample is None:
                    return True
                primary = _extract_primary_tensor(sample)
                if isinstance(primary, torch.Tensor):
                    if primary.numel() == 0:
                        return True
                    if primary.dim() > 0 and primary.shape[0] == 0:
                        return True
                return False

            for source in self.cfg.InferencePreview.sample_from:
                if source == "train":
                    source_loader = self._dataloader
                elif source == "val":
                    source_loader = self._val_dataloader
                else:
                    raise ValueError(f"Unknown sample_from source: {source}")
                if source_loader is None:
                    raise RuntimeError(f"Selected dataloader for {source} is None")
                it = iter(source_loader)
                i = 0
                while i < self.cfg.InferencePreview.num_samples:
                    r = random.randint(0, 5 * self.cfg.InferencePreview.num_samples)
                    if r > self.cfg.InferencePreview.num_samples:
                        continue  # skip some samples randomly to get variety
                    try:
                        batch = next(it)
                    except StopIteration:
                        break
                    # Skip empty batches to avoid downstream errors in _inference_preview
                    if _is_empty(batch):
                        continue
                    cached.append(batch)
                    i += 1
            self._inference_preview_samples = cached
            self._inference_preview_cached = True
            self.reserve_param(
                "inference_preview_enable", self.cfg.InferencePreview.enable
            )
            self.reserve_param(
                "inference_preview_every_n_steps",
                self.cfg.InferencePreview.every_n_steps,
            )
            self.reserve_param(
                "inference_preview_num_samples", self.cfg.InferencePreview.num_samples
            )
            self.reserve_param(
                "inference_preview_sample_from",
                ",".join(self.cfg.InferencePreview.sample_from),
            )
        except Exception as e:
            log_rank_zero(
                self._logger, f"Failed to cache inference preview samples: {e}"
            )
            self._inference_preview_disabled = True

    def _inference_preview(self, sample: Any) -> Dict[str, Any]:  # to be overridden
        raise NotImplementedError("_inference_preview not implemented in subclass")

    def _maybe_log_inference_preview(self):
        if not self.cfg.InferencePreview.enable or self._inference_preview_disabled:
            return
        if not self.cfg.MLflow.enabled or self._mlflow_run is None:
            return
        if self.global_step == 0:
            return
        if self.global_step % self.cfg.InferencePreview.every_n_steps != 0:
            return
        if not self._inference_preview_samples:
            return
        if self._model is None:
            return
        try:
            prev_mode = self._model.training
            self._model.eval()
            results: List[Dict[str, Any]] = []
            with torch.no_grad():
                for s in self._inference_preview_samples:
                    try:
                        r = self._inference_preview(s)
                    except NotImplementedError:
                        self._inference_preview_disabled = True
                        log_rank_zero(
                            self._logger, "Disabling inference preview (NotImplemented)"
                        )
                        return
                    except Exception as ie:
                        r = {"error": str(ie)}
                    results.append(r)
            # Basic artifact: log metrics if present
            for idx, r in enumerate(results):
                for k, v in r.items():
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        mlflow.log_metric(
                            f"preview_{k}_{idx}", float(v), step=self.global_step
                        )
            if self.cfg.InferencePreview.log_as_artifact:
                try:
                    artifact_name = f"inference_preview_step_{self.global_step}.json"
                    mlflow.log_dict(
                        {"step": self.global_step, "samples": results}, artifact_name
                    )
                except Exception as ae:
                    log_rank_zero(
                        self._logger, f"Failed to log inference preview artifact: {ae}"
                    )
            if prev_mode:
                self._model.train()
        except Exception as e:
            log_rank_zero(self._logger, f"Inference preview failed: {e}")
            self._inference_preview_disabled = True

    def _train_step(self, curr_epoch: int):
        """Execute one full training epoch with gradient accumulation, clipping, and logging.

        Steps per batch:
        - Optionally step profiler and collect memory debug info
        - Forward + loss via _loss_step()
        - Accumulate gradients across cfg.Train.gradient_accumulation_steps
        - Clip gradients if configured, then optimizer.step() and lr_scheduler.step()
        - Update global_step and log metrics
        """

        assert self._model is not None
        assert self._dataloader is not None
        assert self._optimizer is not None
        # ensure training mode at the start of each epoch
        self._model.train()
        pbar = self._create_progress_bar(curr_epoch)
        log_rank_zero(self._logger, f"Epoch {curr_epoch}")
        accum_steps = max(1, self.cfg.Train.gradient_accumulation_steps)
        self._optimizer.zero_grad(set_to_none=True)
        accumulation_loss = 0.0
        accumulation_step = 0
        grad_norms: List[Tuple[int, float]] = []
        self._debug_memory_usage(curr_epoch)

        try:
            for idx, batch in enumerate(self._dataloader):
                # memory profiler
                self._trace_memory_profile()
                # cal loss
                batch = self._batch_to_device(batch)
                with autocast(
                    enabled=self.cfg.Train.enable_amp,
                    device_type=torch.device(self._device).type,
                    dtype=self.cfg.Train.amp_dtype,
                ):
                    _, curr_loss = self._loss_step(batch)
                # NaN/Inf diagnosis: check loss before accumulation
                if not torch.isfinite(curr_loss):
                    raise RuntimeError(
                        f"Aborting due to non-finite loss at step {self.global_step}"
                    )
                accumulation_loss += curr_loss.item()
                accumulation_step += 1
                # scale loss for gradient accumulation
                self._scaler.scale(curr_loss / accum_steps).backward(retain_graph=False)

                # step when reaching accumulation boundary or end of epoch
                take_step = ((idx + 1) % accum_steps == 0) or (
                    idx == len(self._dataloader) - 1
                )
                self.global_step += 1

                if take_step:
                    # log grad norm
                    if (
                        self.cfg.Train.log_gradient_norm_per_batch
                        or self.cfg.Train.log_gradient_norm_per_epoch
                    ):
                        self._scaler.unscale_(self._optimizer)
                        grad_norm = total_grad_norm(self._model)
                        grad_norms.append((self.global_step, grad_norm.item()))
                    # Gradient clipping (optional)
                    clip_type = self.cfg.Train.clip_grad_type
                    clip_value = self.cfg.Train.clip_grad_value
                    if clip_type and clip_value is not None:
                        # unscale
                        self._scaler.unscale_(self._optimizer)
                        if clip_type == "norm":
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(clip_value),
                                error_if_nonfinite=self.cfg.Train.error_if_nonfinite_grad,
                            )

                        elif clip_type == "value":
                            torch.nn.utils.clip_grad_value_(
                                self._model.parameters(), clip_value=float(clip_value)
                            )
                        else:
                            self._logger.warning(
                                f"Unknown clip_grad_type '{clip_type}', expected 'norm' or 'value'. Skipping clipping."
                            )
                    # self._optimizer.step()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad(set_to_none=True)
                    # lr scheduler step aligns with optimizer step
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()
                avg_loss = float(accumulation_loss / accumulation_step)
                if pbar is not None:
                    self._log_train_step(
                        pbar,
                        curr_epoch,
                        avg_loss,
                    )
                accumulation_loss = 0.0
                accumulation_step = 0
                self._debug_memory_usage(curr_epoch)
        finally:
            if pbar is not None:
                pbar.close()

        # log grad norm
        if ddp.is_rank_zero():
            if self.cfg.MLflow.enabled and self._mlflow_run is not None:
                if self.cfg.Train.log_gradient_norm_per_epoch and grad_norms:
                    avg_grad_norm = sum(gn for _, gn in grad_norms) / len(grad_norms)
                    mlflow.log_metric("avg_grad_norm", avg_grad_norm, step=curr_epoch)
                if self.cfg.Train.log_gradient_norm_per_batch and grad_norms:
                    for step, grad_norm in grad_norms:
                        mlflow.log_metric("grad_norm", grad_norm, step=step)

    def _log_train_step(
        self,
        pbar: Optional[tqdm],
        curr_epoch: int,
        avg_loss_since_last_log: float,
    ):
        """Log per-step training metrics to tqdm, MetricLogger, and MLflow.

        Shows current loss and LR in tqdm, logs periodic metrics at the cadence
        configured by cfg.log_every_n_steps.
        """
        assert self._optimizer is not None
        lrs = [param_group["lr"] for param_group in self._optimizer.param_groups]
        lr_log_dict = {f"lr_group_{i}": lr for i, lr in enumerate(lrs)}
        lrs[0] if len(lrs) > 0 else 0.0
        if pbar is not None:
            pbar.update(1)
            desc_prefix = ""
            if self.cfg.use_ddp and self.cfg.Debug.ddp_rank_progress_bars:
                desc_prefix = f"R{self._global_rank}|"
            pbar.set_description(
                f"{desc_prefix}{curr_epoch}|{self.global_step}|Loss: {avg_loss_since_last_log:.10f}"
                f"|LR: {lr_log_dict}"
            )
        # periodic logging
        if ddp.is_rank_zero() and self.global_step % self.cfg.log_every_n_steps == 0:
            # Metric logging
            if self._metric_logger and self.cfg.MetricLogger.enable:
                log_dict = {
                    "train_loss": avg_loss_since_last_log,
                    "epoch": curr_epoch,
                }
                log_dict.update(lr_log_dict)
                log_dict.update(
                    torchtune.training.get_memory_stats(
                        device=torch.device(self._device)
                    )
                )
                self._metric_logger.log_dict(log_dict, step=self.global_step)
            # MLflow per-step logging (configurable cadence)
            if self.cfg.MLflow.enabled and self._mlflow_run is not None:
                if math.isnan(avg_loss_since_last_log):
                    log_rank_zero(
                        self._logger,
                        f"train_loss is nan at step {self.global_step}, epoch {curr_epoch}, skipping logging",
                        logging.ERROR,
                    )
                    return
                for lr in lrs:
                    if math.isnan(float(lr)):
                        log_rank_zero(
                            self._logger,
                            f"lr is nan at step {self.global_step}, epoch {curr_epoch}, skipping logging, lrs: {lr_log_dict}",
                            logging.ERROR,
                        )
                        return
                mlflow.log_metric(
                    "train_loss", avg_loss_since_last_log, step=self.global_step
                )
                for i, lr_ in lr_log_dict.items():
                    mlflow.log_metric(
                        i,
                        float(lr_),
                        step=self.global_step,
                    )
            # inference preview hook
            self._maybe_log_inference_preview()

    def _trace_memory_profile(self):
        """Advance the profiler schedule by one step.

        Precondition:
        - Debug.enable_profiler must be True and _profiler is active; otherwise this
          will raise due to None._profiler.
        """

        if self._profiler:
            self._profiler.step()

    def _debug_memory_usage(self, curr_epoch):
        """Collect CUDA memory snapshots when enabled and manage recorder lifetime.

        Behavior:
        - After skipping Debug.skip_steps_to_snapshot steps, takes a snapshot every
          Debug.snapshot_every_n_step steps until reaching Debug.max_steps_to_snapshot.
        - Starts recording with torch.cuda.memory._record_memory_history() on first
          snapshot, and stops recording once max window is exceeded.
        - Snapshots are written to Debug.save_snapshot_dir as pickle files.
        """

        # take memory snapshot
        if (
            self.cfg.Debug.enable_snapshot
            and 0
            < self.global_step - self.cfg.Debug.skip_steps_to_snapshot
            <= self.cfg.Debug.max_steps_to_snapshot
            and self.global_step % self.cfg.Debug.snapshot_every_n_step == 0
        ):
            if not self._mem_record_running:
                torch.cuda.memory._record_memory_history()
                self._mem_record_running = True
            debug_dir = pathlib.Path(self.cfg.Debug.save_snapshot_dir)
            if not debug_dir.exists():
                debug_dir.mkdir(parents=True)
            p = (
                (debug_dir / f"epoch_{curr_epoch}_step_{self.global_step}.pickle")
                .absolute()
                .__str__()
            )
            log_rank_zero(
                self._logger,
                f"Dumping snapshot at step {self.global_step}, epoch: {curr_epoch}, to file {p}",
            )
            torch.cuda.memory._dump_snapshot(
                p,
            )
        if (
            self._mem_record_running
            and self.global_step - self.cfg.Debug.skip_steps_to_snapshot
            > self.cfg.Debug.max_steps_to_snapshot
        ):
            log_rank_zero(
                self._logger,
                f"Reached max_steps_to_snapshot {self.cfg.Debug.max_steps_to_snapshot}, stop recording memory history",
            )
            torch.cuda.memory._record_memory_history(enabled=None)
            self._mem_record_running = False

    def _validation_step(self, curr_epoch: int) -> float:
        """Run a full validation epoch and log metrics.

        In DDP mode, each rank evaluates its local validation shard and the aggregate
        loss / metric totals are reduced across ranks before rank zero performs any
        logging or early-stopping decisions.

        Returns:
            float: Average validation loss per batch across the epoch.
        """
        if self.cfg.Train.no_validation:
            self._logger.info("Skipping validation due to no_validation is set to True")
            return 0.0

        assert self._model is not None
        assert self._val_dataloader is not None
        # assert self._loss_fn is not None
        total_val_loss = torch.zeros(1, device=self._device, dtype=torch.float64)
        val_loss_count = torch.zeros(1, device=self._device, dtype=torch.long)
        metric_total = torch.zeros(1, device=self._device, dtype=torch.float64)
        metric_count = torch.zeros(1, device=self._device, dtype=torch.long)
        self._model.eval()
        with torch.no_grad():
            for batch in self._val_dataloader:
                batch = self._batch_to_device(batch)
                logits, val_loss = self._loss_step(batch)
                if self._metric_fn is not None:
                    mv = self._metric_step(batch, logits)
                    metric_total += torch.as_tensor(
                        mv,
                        device=self._device,
                        dtype=torch.float64,
                    ).reshape(-1).mean()
                    metric_count += 1
                total_val_loss += val_loss.detach().to(dtype=torch.float64).reshape(
                    -1
                ).mean()
                val_loss_count += 1

        # aggregate
        if self.cfg.use_ddp:
            distributed.all_reduce(total_val_loss, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(val_loss_count, op=distributed.ReduceOp.SUM)
            if self._metric_fn is not None:
                distributed.all_reduce(metric_total, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(metric_count, op=distributed.ReduceOp.SUM)
        avg_val_loss = float(
            (total_val_loss / val_loss_count.clamp(min=1)).detach().cpu().item()
        )
        metric_name = None
        metric_value = None
        max_metric_value = None
        if self._metric_fn is not None:
            metric_name = "validation_" + self._metric_fn.__class__.__name__
            if int(metric_count.item()) > 0:
                metric_value = float(
                    (metric_total / metric_count.clamp(min=1)).detach().cpu().item()
                )
            # track running best
            if metric_value is not None:
                max_metric_value = metric_value
                if metric_name not in self._max_metrics:
                    self._max_metrics[metric_name] = float(metric_value)
                if metric_value > self._max_metrics[metric_name]:
                    self._max_metrics[metric_name] = float(metric_value)
                else:
                    max_metric_value = self._max_metrics[metric_name]
        # if self.cfg.Neptune and self.cfg.Neptune.enable_neptune:
        #     self._run["val_loss"].append(avg_val_loss)
        if ddp.is_rank_zero():
            if self.cfg.MetricLogger.enable and self._metric_logger:
                log_dict = {
                    "val_loss_avg_per_batch": avg_val_loss,
                    "epoch": curr_epoch,
                }
                if self._metric_fn is not None:
                    log_dict[metric_name] = metric_value  # type: ignore
                self._metric_logger.log_dict(log_dict, step=self.global_step)
            # mlflow logs
            if self.cfg.MLflow.enabled and self._mlflow_run is not None:
                if math.isnan(float(avg_val_loss)):
                    log_rank_zero(
                        self._logger,
                        f"val_loss is nan at epoch {curr_epoch}, skipping logging",
                        logging.WARNING,
                    )
                    return avg_val_loss
                mlflow.log_metric(
                    "val_loss_avg_per_batch", float(avg_val_loss), step=curr_epoch
                )
                if self._metric_fn is not None:
                    if metric_value is None or math.isnan(float(metric_value)):
                        log_rank_zero(
                            self._logger,
                            f"{metric_name} is nan at epoch {curr_epoch}, skipping logging",  # type: ignore
                            logging.WARNING,
                        )
                        return avg_val_loss
                    if max_metric_value is None or math.isnan(float(max_metric_value)):
                        log_rank_zero(
                            self._logger,
                            f"{metric_name}_max is nan at epoch {curr_epoch}, skipping logging",  # type: ignore
                            logging.WARNING,
                        )
                        return avg_val_loss
                    mlflow.log_metric(metric_name, float(metric_value), step=curr_epoch)  # type: ignore
                    mlflow.log_metric(
                        f"{metric_name}_max",
                        float(max_metric_value),
                        step=curr_epoch,  # type: ignore
                    )
        return avg_val_loss

    def _visualize(self, curr_epoch: int, batch_num: int = 1):
        """Visualize the model predictions (first item of each selected batch)."""
        assert self._model is not None
        assert self._val_dataloader is not None
        self._model.eval()
        _, axes = plt.subplots(batch_num, 1, figsize=(3, 3 * batch_num))
        if batch_num == 1:
            axes = [axes]
        with torch.no_grad():
            for idx, batch in enumerate(self._val_dataloader):
                if idx >= batch_num:
                    break
                features, _ = batch
                features = features.to(self._device)
                pred = self._predict(features[:1])
                ax = axes[idx]
                ax.imshow(to_pil_image(features[0].detach().cpu()))
                ax.set_title(f"Epoch: {curr_epoch} Pred: {pred}")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

    def _clean_ckp(self):
        """Remove all checkpoints."""
        if self.cfg.Checkpoint.enable_local_ckp:
            path = pathlib.Path(self.cfg.Checkpoint.checkpoint_dir)
            if path.exists():
                for file in path.glob("model_*.pt"):
                    file.unlink()
                log_rank_zero(self._logger, f"Removed all checkpoints in {path}")

    def save_checkpoint(self, epoch: int, **kwargs):  # type: ignore
        """Save a training checkpoint.

        Calls the synchronous saver. Provided to satisfy torchtune's recipe interface.

        Args:
            epoch (int): Current epoch index used for naming.

        Returns:
            None
        """

        # self._background_tasks.append(
        #     self._pool.submit(self._save_checkpoint_sync, epoch)
        # )
        self._save_checkpoint_sync(epoch)
        return None

    def _save_checkpoint_sync(self, epoch: int):
        """Synchronously save checkpoint to local disk and log to MLflow if enabled.

        Saves model state_dict under Checkpoint.checkpoint_dir when enabled and logs
        the model artifact to MLflow with an epoch-suffixed name.
        """

        assert self._model is not None
        payload = self._get_model_saving_payload(epoch, is_final=False)

        if self.cfg.Checkpoint.enable_local_ckp:
            file = f"model_{epoch:05}.pt"
            path = pathlib.Path(self.cfg.Checkpoint.checkpoint_dir) / file
            if isinstance(payload, nn.Module):
                torch.save(payload.state_dict(), path)
            else:
                torch.save(payload, path)
            log_rank_zero(self._logger, f"Model saved to {path}")

        if self._mlflow_run is not None and self.cfg.MLflow.enabled:
            log_rank_zero(self._logger, "Save model to mlflow")
            if isinstance(payload, nn.Module):
                self._mlflow_run.log_model(
                    model=payload,
                    model_name=f"{self.cfg.MLflow.mlflow_model_name}_epoch{epoch:05}",
                    model_input_sample=self._get_input_sample(),
                )
            else:
                self._mlflow_run.log_torch_artifact(
                    payload=payload,
                    artifact_name=f"{self.cfg.MLflow.mlflow_model_name}_epoch{epoch:05}.pt",
                )

    def load_checkpoint(self, path: str, **kwargs):  # type: ignore
        """Implement interface of torchtune recipe only."""
        state = torch.load(path, map_location=self._device)
        self._unwrap_model().load_state_dict(state)

    def train(self, **kwargs):
        """Run the training loop over epochs with periodic validation and logging.

        Behavior:
        - Calls torchtune.training.cleanup_before_training(), then iterates epochs
          invoking _train_step() and _validation_step().
        - Optionally visualizes predictions and performs early stopping.
        - Saves a checkpoint when a new best validation metric is observed.
        """

        assert self._optimizer is not None
        assert self._model is not None
        self._optimizer.zero_grad()
        for curr_epoch in range(self.epochs_run, self.cfg.Train.total_epochs):
            if self.cfg.use_ddp:
                self._dist_sampler.set_epoch(curr_epoch)
            # train step
            self._train_step(curr_epoch)
            # validation
            avg_val_loss = self._validation_step(curr_epoch)
            # visualize
            if self.cfg.Visualization.vis_enable and ddp.is_rank_zero():
                self._visualize(
                    curr_epoch,
                    self.cfg.Visualization.vis_batch_num,
                )
            # early stop
            if self._early_stop is not None:
                self._early_stop(avg_val_loss)
                if self._early_stop.is_new_best_score:
                    # save checkpoint at progress
                    log_rank_zero(
                        self._logger,
                        f"Saving model at epoch {curr_epoch}, val_loss: {avg_val_loss}",
                    )
                    if ddp.is_rank_zero():
                        self.save_checkpoint(curr_epoch)
                if self._early_stop.early_stop:
                    log_rank_zero(self._logger, f"Early stopping at epoch {curr_epoch}")
                    break

            # Epoch-level trainer-state checkpoint (resume support)
            if (
                self.cfg.Checkpoint.enable_state_ckp_to_mlflow
                and self._mlflow_run is not None
                and self.cfg.MLflow.enabled
                and ddp.is_rank_zero()
            ):
                try:
                    state = self._build_trainer_state_dict()
                    self._mlflow_run.log_trainer_state(
                        state=state,
                        artifact_name=self.cfg.Checkpoint.resume_state_artifact_name,
                    )
                except Exception as e:
                    self._logger.warning(f"Failed to log trainer-state checkpoint: {e}")

            self.epochs_run += 1

        # Final trainer-state checkpoint at end of training
        if (
            self.cfg.Checkpoint.enable_state_ckp_to_mlflow
            and self._mlflow_run is not None
            and self.cfg.MLflow.enabled
            and ddp.is_rank_zero()
        ):
            try:
                state = self._build_trainer_state_dict()
                self._mlflow_run.log_trainer_state(
                    state=state,
                    artifact_name=self.cfg.Checkpoint.resume_state_artifact_name,
                )
            except Exception as e:
                self._logger.warning(
                    f"Failed to log final trainer-state checkpoint: {e}"
                )

    def safe_train(self):
        """Train with exception safety, ensuring cleanup on failures.

        Any exception raised during train() will be logged, MLflow will be notified,
        cleanup() will be called, and the exception re-raised.

        Raises:
            Exception: Re-raises the underlying exception after cleanup.
        """
        try:
            self.train()
        except Exception as e:
            self._logger.error(
                f"got exception during training, cleanup...\nException: {e.__class__.__name__}: {e} Traceback: {traceback.format_exc()}"
            )
            self.mlflow_is_exception = True
            self.cleanup()
            raise e

    def cleanup(self, **kwargs):
        """Tear down resources and finalize tracking/logging.

        Behavior:
        - Waits for background tasks, shuts down thread pool.
        - Stops CUDA memory recording and exits the profiler if active.
        - Logs final model to MLflow and closes the tracking run, marking whether the
          run ended due to exception or being killed.
        - When cfg.use_ddp=True, rank zero finalizes MLflow and every rank tears down
          the process group in a finally block.
        """

        log_rank_zero(self._logger, "waiting for background tasks to finish...")
        for task in concurrent.futures.as_completed(self._background_tasks):
            task.result(timeout=10)
        # ensure background pool is shutdown
        try:
            self._pool.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            self._logger.warning(f"Failed to shutdown background pool: {e}")
        if self._mem_record_running:
            torch.cuda.memory._record_memory_history(enabled=None)
        if self._profiler and self._profiler_running:
            self._profiler.__exit__(None, None, None)

        # if (
        #     self.cfg.Neptune
        #     and self.cfg.Neptune.enable_neptune
        #     and getattr(self, "_run", None) is not None
        # ):
        #     self._run.stop()
        try:
            if ddp.is_rank_zero():
                if self._mlflow_run is not None and self.cfg.MLflow.enabled:
                    # log final model
                    assert self._model is not None
                    self._logger.info("logging the final model")
                    payload = self._get_model_saving_payload(
                        self.epochs_run, is_final=True
                    )
                    if isinstance(payload, nn.Module):
                        self._mlflow_run.log_model(
                            model=payload,
                            model_name=f"{self.cfg.MLflow.mlflow_model_name}_epoch{self.epochs_run:05}_final",
                            # model_input_sample=self._get_input_sample(),
                        )
                    else:
                        self._mlflow_run.log_torch_artifact(
                            payload=payload,
                            artifact_name=f"{self.cfg.MLflow.mlflow_model_name}_epoch{self.epochs_run:05}_final.pt",
                        )
                    self._mlflow_run.close(
                        is_exception=self.mlflow_is_exception,
                        is_killed=self.mlflow_is_killed,
                    )
        finally:
            if self.cfg.use_ddp:
                ddp.cleanup_ddp()

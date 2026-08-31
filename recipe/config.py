from enum import Enum
import logging
import torch
from typing import Optional, List

from pydantic import BaseModel, Field, field_serializer, field_validator


class MetricLoggerCfg(BaseModel):
    """Metric logging configuration.

    Fields:
    - enable: Toggle metric logging via a MetricLoggerInterface implementation.
    - log_dir: Directory where logs are written when to_disk=True.
    - to_disk: If True, use DiskLogger; otherwise, StdoutLogger.
    """

    #: Toggle metric logging via a MetricLoggerInterface implementation.
    enable: bool = Field(
        default=False,
        description="Toggle metric logging via a MetricLoggerInterface implementation.",
    )
    #: Directory for logs when to_disk=True.
    log_dir: str = Field(
        default="",
        description="Directory where logs are written when to_disk=True.",
    )
    #: If True, use DiskLogger; otherwise, StdoutLogger.
    to_disk: bool = Field(
        default=False,
        description="If True, use DiskLogger; otherwise, StdoutLogger.",
    )


class MLflowCfg(BaseModel):
    """MLflow tracking configuration.

    Fields:
    - experiment_name: Required MLflow experiment name to log under. If it does not
      exist, it will be created by setup_mlflow().
    - mlflow_model_name: Base name used when logging model artifacts. The actual
      saved names may include epoch suffixes (e.g., "<name>_epoch00010").
    - dataset_name: A human-readable name for the dataset used during training;
      stored alongside the model signature for discoverability.
    - desc: Optional long-form description saved as the "mlflow.note.content" tag.
    - nested: Whether this run should be a nested run. In Ray Tune mode, this is
      set to True so that per-trial runs attach to a parent run created by Runner.
    - parent_run_id: Optional MLflow run_id of the parent run. Required in Tune
      mode when nested=True; populated via Runner.on_parent_run_registered.

    Notes:
    - Tracking/Artifact store locations are controlled by standard MLflow settings
      (e.g., MLFLOW_TRACKING_URI) and environment (e.g., AWS credentials for S3).
    - When cfg.use_ddp=True, only rank zero creates or mutates the tracked MLflow
      run (params, metrics, artifacts, final close). Other ranks still initialize
      MLflow client/env state when needed so resume_from_mlflow can load trainer
      state on every rank.
    - See BasicRecipe._setup_mlflow for how parameters, metrics, and artifacts are
      recorded.
    """

    #: Required MLflow experiment name to log under. Created by setup_mlflow() if missing.
    experiment_name: str = Field(
        default="default",
        description="Required MLflow experiment name to log under. Created by setup_mlflow() if missing.",
    )
    #: Base name for logged model artifacts; epoch suffixes may be appended.
    mlflow_model_name: str = Field(
        default="dummy",
        description="Base name for logged model artifacts; epoch suffixes may be appended.",
    )
    #: Human-readable dataset identifier stored with model signature.
    dataset_name: str = Field(
        default="dummy",
        description="Human-readable dataset identifier stored with model signature.",
    )
    #: Long-form run description saved as the "mlflow.note.content" tag.
    desc: Optional[str] = Field(
        default=None,
        description='Long-form run description saved as the "mlflow.note.content" tag.',
    )
    #: Whether this run is nested. In Ray Tune trials this is True.
    nested: bool = Field(
        default=False,
        description="Whether this run is nested. In Ray Tune trials this is True.",
    )
    #: Parent MLflow run_id when nested=True (set by Runner).
    parent_run_id: Optional[str] = Field(
        default=None,
        description="Parent MLflow run_id when nested=True (set by Runner).",
    )
    #: Explicit MLflow run_id to attach to (useful for resume).
    run_id: Optional[str] = Field(
        default=None,
        description="If set, attach to an existing MLflow run_id instead of creating a new run.",
    )

    #: Global switch for all MLflow operations in the framework.
    enable: bool = Field(
        default=True,
        description="If False, disable all MLflow operations. BasicRecipe also forces Checkpoint.enable_state_ckp_to_mlflow and Checkpoint.resume_from_mlflow to False during setup().",
    )

    #: Backward-compatible flag (deprecated). Prefer `enable=False`.
    disable_mlflow: bool = Field(
        default=False,
        description="Deprecated: if True, disables MLflow. Prefer MLflow.enable=False.",
    )

    @property
    def enabled(self) -> bool:
        """True when MLflow operations should be performed."""
        return bool(self.enable) and not bool(self.disable_mlflow)


class CheckpointCfg(BaseModel):
    """Checkpoint configuration for model weights and training state.

    This framework supports two kinds of checkpoints:
    - Model-weight checkpoints: what the code previously saved (model state_dict).
    - Training-state checkpoints: enough state to resume training at epoch boundaries.

    Fields:
    - checkpoint_dir: Directory where local checkpoints are written when enabled.
    - enable_local_ckp: Enables periodic local model weight checkpoint saves.
    - enable_state_ckp_to_mlflow: If True, logs a full training-state checkpoint to MLflow
      at the end of each epoch.
    - resume_from_mlflow: If True, attempts to restore training state from MLflow during
      setup() before starting the epoch loop.
    - resume_state_artifact_name: Artifact name used to store the latest state checkpoint.
    - resume_strict: If True, raises if resume was requested but no checkpoint found.
    - store_rng_state: If True, saves/restores Python/NumPy/Torch RNG states.

    Notes:
    - Resume is epoch-level by design (restores epoch+global_step and continues at the
      next epoch).
    - In Ray Tune mode, local checkpoints are disabled; state checkpoints can still be
      logged to MLflow.
    - When cfg.use_ddp=True, checkpoint writes happen on rank zero only, but resume
      state is loaded on every rank so model/optimizer state stays consistent across
      processes before DDP wrapping.
    """

    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory for local checkpoints (model_*.pt). Must exist and be writable when enabled.",
    )
    enable_local_ckp: bool = Field(
        default=False,
        description="Enable saving local checkpoints. In Ray Tune mode this is forced to False.",
    )

    # MLflow training-state checkpointing
    enable_state_ckp_to_mlflow: bool = Field(
        default=False,
        description="If True, log full training-state checkpoints to MLflow each epoch.",
    )
    resume_from_mlflow: bool = Field(
        default=False,
        description="If True, attempt to resume training state from MLflow during setup(). In DDP mode every rank performs the restore; rank zero remains the only rank that logs new MLflow params/artifacts.",
    )
    resume_from_path: bool = Field(
        default=False, description="Resume from local path (trainer_state_latest.pt)"
    )
    mlflow_run_id: Optional[str] = Field(
        default=None, description="MLflow run id to resume from"
    )
    resume_state_artifact_name: str = Field(
        default="trainer_state_latest.pt",
        description="MLflow artifact filename for the latest training-state checkpoint.",
    )
    resume_strict: bool = Field(
        default=False,
        description="If True, raise when resume_from_mlflow is enabled but no checkpoint exists.",
    )
    store_rng_state: bool = Field(
        default=True,
        description="If True, store and restore RNG state for best-effort reproducibility.",
    )


class DebugCfg(BaseModel):
    """Memory debugger configuration.

    Fields:
    - enable_snapshot: Enable memory snapshots at a fixed cadence.
    - skip_steps_to_snapshot: Number of initial iterations to skip before taking a snapshot.
    - max_steps_to_snapshot: Maximum number of iterations during which snapshots may be taken.
    - snapshot_every_n_step: Take a snapshot every N iterations after the initial skip window.
    - save_snapshot_dir: Directory to save memory snapshots.
    - enable_profiler: Enable the in-epoch memory profiler.
    - skip_steps_to_profile: Maps to torch.profiler.schedule(skip_first=...). Steps to skip before profiling begins.
    - wait_steps_between_repeat: Maps to torch.profiler.schedule(wait=...). Non-profiled steps before each active window.
    - active_steps_in_one_repeat: Maps to torch.profiler.schedule(active=...). Steps captured per profiling window.
    - repeats: Maps to torch.profiler.schedule(repeat=...). Number of profiling cycles to execute.
    - save_memory_timeline_dir: Directory to save memory timeline/traces.
    - ddp_rank_progress_bars: If True, show one tqdm progress bar per rank in DDP runs.

    Notes:
    - The memory profiler uses torch.profiler.schedule(wait=..., warmup=0, active=..., repeat=..., skip_first=...). We set warmup to 0.

    Operation:
    - Memory Snapshots: When enabled, captures memory snapshots starting after 'skip_steps_to_snapshot' training steps, then every 'snapshot_every_n_step' steps, up to 'max_steps_to_snapshot'. Uses torch.cuda.memory._record_memory_history() to record history and torch.cuda.memory._dump_snapshot() to save as pickle files in 'save_snapshot_dir'.
    - Memory Profiler: When enabled, schedules profiling windows using torch.profiler.profile with torch.profiler.schedule (skip_first=skip_steps_to_profile, wait=wait_steps_between_repeat, active=active_steps_in_one_repeat, repeat=repeats). Traces are saved in 'save_memory_timeline_dir'.

    Visualization:
    - To visualize memory snapshots:
      1. Download the visualization script:
         wget https://raw.githubusercontent.com/pytorch/pytorch/main/torch/cuda/_memory_viz.py
      2. Run the script on a snapshot file (e.g., epoch_0_step_4000.pickle):
         python _memory_viz.py trace_plot epoch_0_step_4000.pickle -o snapshot.html
    - To visualize memory profiles, use Chrome tracing:
      1. Open chrome://tracing in Chrome browser
      2. Click "Load" and select the .json trace file from ./memory_timelines/chrome_trace/
      3. Alternatively, use Perfetto UI: https://ui.perfetto.dev/
    """

    # memory snapshot
    enable_snapshot: bool = Field(
        default=False,
        description=(
            "Enable memory snapshots. Takes a snapshot every 'snapshot_every_n_step' after "
            "skipping 'skip_steps_to_snapshot' steps, up to 'max_steps_to_snapshot'."
        ),
    )
    skip_steps_to_snapshot: int = Field(
        default=0,
        ge=0,
        description="Initial number of training steps to skip before taking snapshots.",
    )
    max_steps_to_snapshot: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of training steps during which snapshots may be taken.",
    )
    snapshot_every_n_step: int = Field(
        default=1000,
        gt=0,
        description="Take a memory snapshot every N training steps after the skip window.",
    )
    save_snapshot_dir: str = Field(
        default="./snapshots",
        description="Directory where snapshot artifacts are saved.",
    )
    # memory profiler (uses torch.profiler.schedule)
    enable_profiler: bool = Field(
        default=False,
        description="Enable the memory profiler scheduled via torch.profiler.schedule().",
    )
    skip_steps_to_profile: int = Field(
        default=0,
        ge=0,
        description="Maps to schedule(skip_first=...). Steps to skip before profiling begins.",
    )
    wait_steps_between_repeat: int = Field(
        default=0,
        ge=0,
        description="Maps to schedule(wait=...). Non-profiled steps before each active window in a cycle.",
    )
    active_steps_in_one_repeat: int = Field(
        default=6,
        gt=0,
        description="Maps to schedule(active=...). Steps captured per profiling window.",
    )
    repeats: int = Field(
        default=10,
        ge=0,
        description="Maps to schedule(repeat=...). Number of profiling cycles to run (warmup fixed to 0).",
    )
    save_memory_timeline_dir: str = Field(
        default="./memory_timeline",
        description="Directory where profiler traces (e.g., Chrome trace and TensorBoard) are saved.",
    )
    disable_trace_handler: bool = Field(
        default=False,
        description="Disable chrome trace file export, only generate html.",
    )
    ddp_rank_progress_bars: bool = Field(
        default=False,
        description="If True and use_ddp=True, create one tqdm progress bar per rank using tqdm(position=rank). Intended for interactive debugging; output may be noisy under nohup, redirected logs, or some launchers.",
    )


class LRDecay(str, Enum):
    COSINE = "cosine"
    FACTOR = "factor"


class TrainCfg(BaseModel):
    @field_serializer("amp_dtype")
    def serialize_amp_dtype(self, v: Optional[torch.dtype]):
        if v is None:
            return None
        # return the canonical name (e.g., 'float16') instead of repr('torch.float16')
        return v.__str__().replace("torch.", "")

    @field_validator("amp_dtype", mode="before")
    @classmethod
    def deserialize_amp_dtype(cls, v: object) -> Optional[torch.dtype]:
        if v is None or isinstance(v, torch.dtype):
            return v
        if isinstance(v, str):
            # Accept both "float16" and "torch.float16".
            name = v.removeprefix("torch.")
            try:
                return getattr(torch, name)
            except AttributeError as e:
                raise ValueError(f"Unknown torch.dtype: {v}") from e
        raise TypeError(f"amp_dtype must be torch.dtype | str | None, got {type(v)}")

    """Trainer settings that control core optimization behavior.

    Fields:
    - batch_size: Per-iteration batch size used by DataLoaders.
    - total_epochs: Maximum epochs to run (may finish earlier due to early stop).
    - early_stop_patience: Number of validation epochs without improvement to
      tolerate before stopping.
    - early_stop_delta: Minimum improvement over the best metric to reset patience.
    - early_stop_grace_period: Number of initial epochs exempt from early stopping.
    - gradient_accumulation_steps: Accumulates gradients across this many batches
      before optimizer.step(); effective global batch = batch_size * this value.
    - weight_decay: Weight decay (L2) value to set on optimizer param groups.
    - clip_grad_type: Gradient clipping strategy: "norm", "value", or None.
    - clip_grad_value: Max norm (for type="norm") or absolute value (for
      type="value"). Ignored when clip_grad_type=None.
    - num_warmup_steps: Optional warmup expressed either as a fraction in (0,1]
      of total training steps or as an absolute integer count, depending on your
      scheduler implementation in _setup_lr_scheduler(). If unused, leave default.

    Notes:
    - Early stopping is implemented by EarlyStopping unless running in Ray Tune
      mode, where a tune Stopper controls stopping.
    - Gradient clipping is applied right before each optimizer.step() when enabled.
    """

    learning_rate: float = Field(
        default=1e-5, description="Learning rate for the optimizer."
    )
    #: Per-iteration batch size used by DataLoaders.
    batch_size: int = Field(
        default=64, description="Per-iteration batch size used by DataLoaders."
    )
    #: Maximum number of training epochs to run.
    total_epochs: int = Field(
        default=10, description="Maximum number of training epochs to run."
    )
    #: Validation epochs without improvement to tolerate before stopping.
    early_stop_patience: int = Field(
        default=5,
        description="Validation epochs without improvement to tolerate before stopping.",
    )
    #: Minimum improvement required to reset early stopping patience.
    early_stop_delta: float = Field(
        default=0.0,
        description="Minimum improvement required to reset early stopping patience.",
    )
    #: Initial epochs exempt from early stopping checks.
    early_stop_grace_period: int = Field(
        default=2, description="Initial epochs exempt from early stopping checks."
    )
    #: Number of batches to accumulate gradients before optimizer.step().
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of batches to accumulate gradients before optimizer.step().",
    )
    #: Weight decay (L2) value applied by the optimizer.
    weight_decay: float = Field(
        default=0.01, description="Weight decay (L2) value applied by the optimizer."
    )
    #: Gradient clipping strategy: 'norm', 'value', or None to disable.
    clip_grad_type: Optional[str] = Field(
        default=None,
        description="Gradient clipping strategy: 'norm', 'value', or None to disable.",
    )
    #: Max norm (type='norm') or absolute value (type='value') for gradient clipping.
    clip_grad_value: Optional[float] = Field(
        default=None,
        description="Max norm (type='norm') or absolute value (type='value') for gradient clipping.",
    )
    error_if_nonfinite_grad: bool = Field(
        default=False,
        description="If True, raises an error if non-finite gradients are detected. Only work for 'norm' clipping",
    )
    #: Warmup proportion (0<prop<=1) or absolute count depending on your scheduler.
    num_warmup_steps: float = Field(
        default=0.01,
        description="Warmup proportion (0<prop<=1) or absolute count depending on your scheduler.",
    )
    #: Enable activation checkpointing to reduce activation memory at the cost of extra compute.
    enable_activation_checkpointing: bool = Field(
        default=False,
        description=(
            "If True, wraps all leaf modules of the model with torch.utils.checkpoint so"
            " activations are recomputed in backward to save memory. Increases compute"
            " time but helps avoid CUDA OOMs on large models/batches."
        ),
    )
    activation_checkpoint_white_list: list = Field(
        default_factory=list,
        description="List of module class names to apply activation checkpointing to. If empty, all leaf modules are checkpointed.",
    )
    activation_checkpoint_black_list: list = Field(
        default_factory=list,
        description="List of module class names to exclude from activation checkpointing.",
    )
    enable_amp: bool = Field(
        default=True,
        description="If True, use torch.cuda.amp.autocast for mixed precision training.",
    )
    amp_dtype: Optional[torch.dtype] = Field(
        default=torch.float16,
        description="Dtype for torch.cuda.amp.autocast; torch.float16 or torch.bfloat16.",
    )
    log_gradient_norm_per_batch: bool = Field(
        default=False,
        description="If True, logs the gradient norm per batch as 'grad_norm' metric.",
    )
    log_gradient_norm_per_epoch: bool = Field(
        default=False,
        description="If True, logs the gradient norm per epoch as 'grad_norm' metric.",
    )
    no_validation: bool = Field(
        default=False,
        description="If true, none validation dataset and dataloader is allowed",
    )
    lr_decay: LRDecay = Field(default=LRDecay.COSINE)
    lr_decay_factor: float = Field(default=0.1)

    class Config:
        arbitrary_types_allowed = True


class LRSchedulerCfg(BaseModel):
    """Learning rate scheduler configuration (optional, for subclass use).

    Fields:
    - num_warmup_steps: Warmup proportion (0<prop<=1) or absolute count used by
      custom schedulers in _setup_lr_scheduler(). This class is provided for
      structured configs but is not wired by default in BasicRecipe.
    """

    #: Warmup proportion (0<prop<=1) or absolute count used by custom LR schedulers in _setup_lr_scheduler().
    num_warmup_steps: float | int = Field(
        default=0.001,
        description="Warmup proportion (0<prop<=1) or absolute count used by custom LR schedulers in _setup_lr_scheduler().",
    )


class DatasetCfg(BaseModel):
    """Dataset-related configuration used by concrete recipe subclasses.

    Fields:
    - val_ratio: Fraction of the dataset to reserve for validation, in [0.0, 1.0].
      For example, 0.2 means 20% validation, 80% training.
    """

    #: Fraction [0.0, 1.0] of the dataset reserved for validation.
    val_ratio: float = Field(
        default=0.2,
        description="Fraction [0.0, 1.0] of the dataset reserved for validation.",
    )


class NeptuneCfg(BaseModel):
    """Neptune logging configuration (optional).

    Fields:
    - enable_neptune: Toggle Neptune integration. Currently disabled in this module
      unless you enable and initialize it explicitly.
    - project: Neptune project name.
    - name: Display name for the run.
    - tags: List of tags to attach to the run.
    - dependencies: Dependency capture mode (e.g., "infer").
    - monitoring_namespace: Where system metrics are logged.

    Notes:
    - Requires NEPTUNE_API_TOKEN in the environment when enabled.
    - The integration code is present but commented out by default.
    """

    #: Toggle Neptune integration (requires NEPTUNE_API_TOKEN).
    enable_neptune: bool = Field(
        default=False,
        description="Toggle Neptune integration (requires NEPTUNE_API_TOKEN).",
    )
    #: Neptune project name (e.g., 'user/workspace').
    project: str = Field(
        default="Workspace",
        description="Neptune project name (e.g., 'user/workspace').",
    )
    #: Display name for the run.
    name: str = Field(default="loss", description="Display name for the run.")
    #: Tags to attach to the run.
    tags: List = Field(default_factory=list, description="Tags to attach to the run.")
    #: Dependency capture mode (e.g., 'infer').
    dependencies: str = Field(
        default="infer",
        description="Dependency capture mode (e.g., 'infer').",
    )
    #: Namespace where system metrics are logged.
    monitoring_namespace: str = Field(
        default="monitoring",
        description="Namespace where system metrics are logged.",
    )


class VisualizationCfg(BaseModel):
    """Lightweight visualization controls for validation previews.

    Fields:
    - vis_enable: If True, runs _visualize() each epoch using the validation loader.
    - vis_batch_num: Number of validation batches to visualize per epoch.
    """

    #: If True, runs _visualize() each epoch using the validation loader.
    vis_enable: bool = Field(
        default=False,
        description="If True, runs _visualize() each epoch using the validation loader.",
    )
    #: Number of validation batches to visualize per epoch.
    vis_batch_num: int = Field(
        default=1, description="Number of validation batches to visualize per epoch."
    )


class InferencePreviewCfg(BaseModel):
    """Lightweight periodic inference preview configuration.

    Fields:
    - enable: If True, logs small fixed-set inference previews to MLflow.
    - every_n_steps: Global training step cadence to trigger preview logging.
    - num_samples: Number of samples taken (deterministically) from a chosen loader.
    - sample_from: Which loader to draw samples from: 'val' (default) or 'train'.
    - log_as_artifact: If True, writes a JSON artifact with predictions.

    Behavior:
    - At setup(), the first `num_samples` batches are captured from the selected
      DataLoader (non‑mutating iteration) and reused each preview cycle.
    - During training, when global_step % every_n_steps == 0, the model is
      temporarily set to eval(), inference is run (no grad) on the cached samples
      via the subclass hook `_inference_preview(sample)`, and results are logged.
    - Subclasses should implement `_inference_preview(sample)` returning a dict
      with serializable fields (e.g., prediction/reference/metrics). If not
      implemented and feature is enabled, a NotImplementedError is raised once
      and the feature is auto‑disabled.
    """

    enable: bool = Field(
        default=False, description="Enable periodic inference preview logging."
    )
    every_n_steps: int = Field(
        default=1000, gt=0, description="Global step cadence for preview logging."
    )
    num_samples: int = Field(
        default=2, gt=0, description="Number of cached samples used for preview."
    )
    sample_from: List[str] = Field(
        default=["val"],
        description="Data source for preview samples: 'val' or 'train'.",
    )
    log_as_artifact: bool = Field(
        default=True, description="If True, log prediction JSON as MLflow artifact."
    )


class Config(BaseModel):
    """Top-level configuration consumed by BasicRecipe and its subclasses.

    Fields:
    - device: Device string resolved by torchtune.utils.get_device (e.g., "cuda",
      "cuda:0", or "cpu").
    - use_ddp: If True, BasicRecipe initializes torch.distributed from the launcher
      environment, replaces device with cuda:LOCAL_RANK, requires a training
      sampler that implements set_epoch(), wraps the model with DDP after optimizer
      setup/resume, aggregates validation across ranks, and restricts progress /
      checkpoint / MLflow run mutations to rank zero.
    - Train: TrainerCfg controlling optimization, early stop, and clipping.
    - Checkpoint: CheckpointCfg controlling local checkpoint behavior.
    - MLflow: MLflowCfg controlling experiment, run mode, and artifact naming.
    - Visualization: VisualizationCfg controlling optional previews.
    - logger: Optional logging.Logger to override the default torchtune logger.
      Excluded from model serialization and MLflow config logs.
    - log_every_n_steps: Per-step logging cadence for MLflow metrics during the
      training loop (train_loss and lr). Set to a higher value to reduce logging
      overhead on very small batches.

    Example:
        from workspace.recipe import (
            Config, TrainerCfg, CheckpointCfg, MLflowCfg, VisualizationCfg
        )
        cfg = Config(
            device="cuda",
            Train=TrainerCfg(
                batch_size=128,
                total_epochs=20,
                early_stop_patience=5,
                early_stop_delta=0.0,
                early_stop_grace_period=2,
                gradient_accumulation_steps=1,
            ),
            Checkpoint=CheckpointCfg(
                checkpoint_dir="/tmp/checkpoints/my_model",
                enable_local_ckp=True,
            ),
            MLflow=MLflowCfg(
                experiment_name="my-experiment",
                mlflow_model_name="digit-cnn",
                dataset_name="MNIST",
                desc="Baseline CNN",
                nested=False,
            ),
            Visualization=VisualizationCfg(
                vis_enable=False,
                vis_batch_num=2,
            ),
            log_every_n_steps=10,
        )
    """

    #: Device string resolved by torchtune.utils.get_device (e.g., "cuda", "cuda:0", or "cpu").
    device: str = Field(
        default="cpu",
        description='Device string resolved by torchtune.utils.get_device (e.g., "cuda", "cuda:0", or "cpu"). If use_ddp=True, BasicRecipe ignores this at runtime and uses cuda:LOCAL_RANK for the current process.',
    )
    #: Trainer settings controlling optimization, early stopping, and clipping.
    Train: TrainCfg = Field(
        default_factory=TrainCfg,
        description="Trainer settings controlling optimization, early stopping, and clipping.",
    )
    #: Local checkpoint configuration.
    Checkpoint: CheckpointCfg = Field(
        default_factory=CheckpointCfg, description="Local checkpoint configuration."
    )
    #: MLflow tracking and artifact naming configuration.
    MLflow: MLflowCfg = Field(
        default_factory=MLflowCfg,
        description="MLflow tracking and artifact naming configuration.",
    )
    #: Visualization settings for epoch previews.
    Visualization: VisualizationCfg = Field(
        default_factory=VisualizationCfg,
        description="Visualization settings for epoch previews.",
    )
    MetricLogger: MetricLoggerCfg = Field(
        description="Metric logger configuration.", default_factory=MetricLoggerCfg
    )
    Debug: DebugCfg = Field(
        description="Debug configuration.", default_factory=DebugCfg
    )
    InferencePreview: InferencePreviewCfg = Field(
        description="Periodic inference preview configuration.",
        default_factory=InferencePreviewCfg,
    )
    #: Optional logger to override default. Excluded from MLflow config logs.
    logger: Optional[logging.Logger] = Field(
        default=None,
        description="Optional logger to override default. Excluded from MLflow config logs.",
    )
    #: Cadence for per-step MLflow logging of train_loss and lr.
    log_every_n_steps: int = Field(
        default=1,
        description="Cadence for per-step MLflow logging of train_loss and lr.",
    )
    use_ddp: bool = Field(
        default=False,
        description="Enable Distributed Data Parallel. Requires a launcher that sets LOCAL_RANK/RANK/WORLD_SIZE and a training DataLoader sampler (or batch sampler) with set_epoch(). Progress logs, checkpoint writes, and MLflow run mutations stay on rank zero; validation reduction and resume happen on every rank.",
    )

    # validation step logging cadence
    # val_log_every_n_steps: int = Field(default=1)
    # neptune
    # Neptune: Optional[NeptuneCfg] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

import os

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

from torchtune.utils import log_rank_zero
import random
from typing import Type, Optional, Dict, Any, Callable, List

from ray import tune as tune
from ray.tune import Stopper
import torchtune

from recipe.tune import AdaptiveTrainableRecipe
from recipe.utils.mlflow_util import setup_mlflow
from recipe.utils.train_util import SharedStopperStore


class Runner:
    """Reusable Ray Tune runner for AdaptiveTrainableRecipe subclasses.

    Runner encapsulates the common "train_tune" orchestration:
    - Allocates Ray resources per-trial (cpu/gpu)
    - Wires a Stopper and a SharedStopperStore for cross-trial best-epoch signaling
    - Starts a parent MLflow run, logs config, and closes it on experiment end
    - Optionally disables parameter search (single-trial)
    - Forwards Tune parameters to the recipe via AdaptiveTrainableRecipe.setup

    Parameters:
      trainable_cls: Concrete subclass of AdaptiveTrainableRecipe to run.
      experiment_name: MLflow experiment name used for the parent run.
      no_param_search: If True, disables param search (runs with param_space=None).
      store: SharedStopperStore actor. If None, a new one is created.
      stopper: Ray Tune Stopper. If None, a BestScorePlateauStopper is created.
      cpu: CPUs per trial. Defaults to os.cpu_count().
      gpu: GPUs per trial. Can be fractional (e.g., 0 or 0.5) when supported.
      is_tune_mode: Passed into the recipe so it configures itself for Tune.
      param_space: Ray Tune parameter space dict (e.g., grid_search(), uniform()).
      num_samples: Samples for Tune (when using stochastic search methods).
      max_concurrent_trials: Concurrency for the Tune scheduler.
      metric_name: Metric name for early stopping (default: "val_loss").
      parent_run_name: Optional MLflow parent run name override.
      on_parent_run_registered: Callback receiving the MLflow parent run_id; use this
        to set cfg.MLflow.parent_run_id before trials start.
      additional_callbacks: Extra Ray Tune callbacks to attach to the run.
      excludes: Paths to exclude from Ray runtime_env syncing (default ['.git']).
      auto_exclude_large: If True, scan working dir at run() and exclude files >= large_size_threshold.
      large_size_threshold: Size in bytes used to classify a file as large (default 10MB).

    Example:
        runner = Runner(
            trainable_cls=MyRecipeSubclass,
            experiment_name="my-experiment",
            param_space={"lr": tune.grid_search([3e-4, 1e-4])},
            is_tune_mode=True,
            on_parent_run_registered=lambda rid: setattr(cfg.MLflow, "parent_run_id", rid),
        )
        runner.run()
    """

    def __init__(
        self,
        trainable_cls: Type[AdaptiveTrainableRecipe],
        *,
        experiment_name: str,
        no_param_search: bool = False,
        store: Optional[SharedStopperStore] = None,
        stopper: Optional[Stopper] = None,
        cpu: Optional[int] = None,
        gpu: float = 1.0,
        is_tune_mode: bool = True,
        param_space: Optional[Dict[str, Any]] = None,
        num_samples: int = 1,
        max_concurrent_trials: int = 1,
        metric_name: str = "val_loss",
        parent_run_name: Optional[str] = None,
        on_parent_run_registered: Optional[Callable[[str], None]] = None,
        additional_callbacks: Optional[List[tune.Callback]] = None,
        # excludes: Optional[List[str]] = None,
        # auto_exclude_large: bool = False,
        # large_size_threshold: int = 10 * 1024 * 1024,
    ) -> None:
        self.trainable_cls = trainable_cls
        self.experiment_name = experiment_name
        self.no_param_search = no_param_search
        self.store = store
        self.stopper = stopper
        self.cpu = cpu if cpu is not None else os.cpu_count()
        self.gpu = gpu
        self.is_tune_mode = is_tune_mode
        self.param_space = param_space
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.metric_name = metric_name
        self.parent_run_name = parent_run_name
        self.on_parent_run_registered = on_parent_run_registered
        self.additional_callbacks = additional_callbacks or []
        # self.excludes = excludes or [".git"]
        # self.auto_exclude_large = auto_exclude_large
        # self.large_size_threshold = large_size_threshold

        # import ray
        # Defer ray.init to run(); keep excludes relative for Ray's packager
        # self._formatted_excludes = []
        # for p in self.excludes:
        #     # Convert absolute paths back to relative to working_dir if needed
        #     if p.startswith("/"):
        #         rel = os.path.relpath(p, os.getcwd())
        #         self._formatted_excludes.append(rel)
        #     else:
        #         self._formatted_excludes.append(p)

        # internal logger for traces
        self._logger = torchtune.utils.get_logger("INFO")

    def run(self) -> None:
        # import ray
        # if ray.is_initialized():
        #     self._logger.info("Ray is already initialized, shutting down...")
        #     ray.shutdown()
        # Auto-exclude large files/directories if requested. We do this here (not __init__) so
        # working directory contents reflect latest state at run time.
        # if self.auto_exclude_large:
        #     large_paths: List[str] = []
        #     cwd = os.getcwd()
        #     for root, dirs, files in os.walk(cwd):
        #         # Skip already excluded roots early
        #         rel_root = os.path.relpath(root, cwd)
        #         if any(rel_root.startswith(e.rstrip("/")) for e in self._formatted_excludes):
        #             continue
        #         try:
        #             for f in files:
        #                 fpath = os.path.join(root, f)
        #                 try:
        #                     size = os.path.getsize(fpath)
        #                 except OSError:
        #                     continue
        #                 if size >= self.large_size_threshold:
        #                     rel = os.path.relpath(fpath, cwd)
        #                     large_paths.append(rel)
        #         except Exception:
        #             # Best-effort: ignore traversal errors
        #             pass
        #     # De-duplicate and add parent directories if many files share same top-level dir.
        #     # Simple heuristic: if >5 large files under a directory, exclude that directory instead.
        #     dir_counts: Dict[str, int] = {}
        #     for p in large_paths:
        #         parts = p.split(os.sep)
        #         if len(parts) > 1:
        #             top = parts[0]
        #             dir_counts[top] = dir_counts.get(top, 0) + 1
        #     for top, count in dir_counts.items():
        #         if count > 5 and top not in self._formatted_excludes:
        #             self._formatted_excludes.append(top)
        #     for p in large_paths:
        #         if p not in self._formatted_excludes:
        #             self._formatted_excludes.append(p)
        # Always ensure .git internals are excluded even if caller forgot.
        # for git_path in [".git", ".git/objects", ".git/lfs"]:
        #     if git_path not in self._formatted_excludes:
        #         self._formatted_excludes.append(git_path)
        # self._logger.info(f"Runner: final excludes count={len(self._formatted_excludes)} sample={self._formatted_excludes[:10]}")
        # ray.init(runtime_env={"excludes": self._formatted_excludes})
        from functools import partial as _partial

        # Shared store and stopper
        if self.store is None:
            self.store = SharedStopperStore.remote()
        if self.stopper is None:
            # Lazy import to avoid tight coupling at type level; any Stopper works.
            from recipe.utils.train_util import (
                BestScorePlateauStopper as _DefaultStopper,
            )

            self.stopper = _DefaultStopper(
                metric_name=self.metric_name,
                patience=10,
                delta=0,  # int for type consistency
                grace_period=5,
                trace_func=_partial(log_rank_zero, self._logger),
                store=self.store,
            )

        # Wrap trainable and inject parameters for setup()
        trainable = tune.with_resources(
            self.trainable_cls, {"cpu": float(self.cpu), "gpu": float(self.gpu)}
        )
        trainable = tune.with_parameters(
            trainable,
            is_tune_mode=self.is_tune_mode,
            stopper=self.stopper,
            store=self.store,
        )

        # Parameter space handling
        pspace = None if self.no_param_search else self.param_space

        # Start MLflow parent run
        if mlflow is None:
            raise ImportError(
                "Runner requires mlflow but it is not installed. Install mlflow or avoid using Runner."
            )
        setup_mlflow(self.experiment_name)
        parent_run = mlflow.start_run(
            run_name=self.parent_run_name
            or f"{self.experiment_name}-Parent-{self.trainable_cls.__name__}-{random.randint(0, 10000)}"
        )
        run_id = parent_run.info.run_id

        # Allow caller to propagate parent run id into the recipe's cfg
        if self.on_parent_run_registered is not None:
            try:
                self.on_parent_run_registered(run_id)
            except Exception as e:
                log_rank_zero(
                    self._logger,
                    f"on_parent_run_registered callback failed: {e.__class__.__name__}: {e}",
                )

        # Log basic tuner configuration to the parent run
        mlflow.log_params(
            {
                "num_samples": self.num_samples,
                "max_concurrent_trials": self.max_concurrent_trials,
                "no_param_search": self.no_param_search,
                "metric_name": self.metric_name,
                # Store stringified values to avoid MLflow type issues
                "param_space": str(pspace),
                "stopper": f"{self.stopper.__class__.__name__}(patience={getattr(self.stopper, '_patience', None)}, delta={getattr(self.stopper, '_delta', None)}, grace={getattr(self.stopper, '_grace_period', None)})",
                "resources": str({"cpu": self.cpu, "gpu": self.gpu}),
            }
        )

        # Ensure the MLflow run is ended at experiment end
        logger = self._logger

        class _CloseParentRunCallback(tune.Callback):
            def on_experiment_end(self, trials, **info):
                log_rank_zero(logger, "Runner: closing MLflow parent run")
                if mlflow is not None:
                    mlflow.end_run()

        callbacks = [
            _CloseParentRunCallback(),
            *self.additional_callbacks,
        ]

        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                num_samples=self.num_samples,
                max_concurrent_trials=self.max_concurrent_trials,
            ),
            run_config=tune.RunConfig(
                failure_config=tune.FailureConfig(fail_fast=True),
                stop=self.stopper,
                callbacks=callbacks,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=0,
                ),
            ),
            param_space=pspace,
        )
        tuner.fit()

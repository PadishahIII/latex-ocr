import pathlib
from typing import Any, Dict, Optional

from recipe.utils import ddp

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

import ray
import torch
from ray import tune as tune
from torchtune.utils import log_rank_zero

from recipe.basic import BasicRecipe
from recipe.config import Config
from recipe.logging.logger import get_logger_by_name
from recipe.utils.train_util import BestScorePlateauStopper, SharedStopperStore


class TrainableRecipe(BasicRecipe, tune.Trainable):
    def __init__(self, cfg: Config, *args, **kwargs):
        # Determine tune mode first (default True for raw TrainableRecipe usage)
        is_tune_mode = kwargs.pop("is_tune_mode", True)
        BasicRecipe.__init__(self, cfg)
        self._cur_epoch = 0  # zero-based, to sync with training_iteration
        self.config: dict = {}
        self._is_tune_mode = is_tune_mode
        # Nested MLflow runs ONLY when in tune mode
        self.cfg.MLflow.nested = bool(self._is_tune_mode)
        self._stopper: Optional[BestScorePlateauStopper] = None
        self._wait_gpu = False
        self._setup_status = False
        if self._is_tune_mode:
            # Initialize Ray Tune base which will call setup(); Adaptive subclass will manage alternative path
            tune.Trainable.__init__(self, *args, **kwargs)

    def train(self):
        """To avoid method conflict."""
        if self._is_tune_mode:
            return tune.Trainable.train(self)
        else:
            BasicRecipe.train(self)

    def setup(
        self, config: dict, stopper: BestScorePlateauStopper, store: SharedStopperStore
    ):
        """Ray Tune setup hook.

        Args:
            config: Trial configuration dict produced by Ray Tune param sampling. All
                entries are logged to MLflow with the "tune_" prefix.
            stopper: BestScorePlateauStopper shared across trials for early stopping
                coordination and "best score" checkpoint signaling.
            store: SharedStopperStore used by stopper to share best-epoch signals.

        Notes:
        - Requires cfg.MLflow.parent_run_id to be set by the caller (a parent run).
        - Local checkpoints are disabled in Tune mode; model state is saved on MLflow
          when the stopper flags a best score.
        - If cfg.use_ddp=True, BasicRecipe's DDP rules still apply: every rank trains,
          but only rank zero writes checkpoints or mutates the trial MLflow run.
        - This calls BasicRecipe.setup() to build data/model/optimizer/etc.
        """

        self.config = config
        if self.cfg.MLflow.parent_run_id is None or self.cfg.MLflow.parent_run_id == "":
            raise Exception("MLflow parent_run_id is required")
        if self.cfg.Checkpoint.enable_local_ckp:
            self.cfg.Checkpoint.enable_local_ckp = False
            self._logger.warning(
                "local checkpoint not supported in Tune mode, set Checkpoint.enable_local_ckp to False"
            )
        self._stopper = stopper
        self._store = store

        super().setup()

    def _setup_early_stop(self):
        self._early_stop = None

    def _setup_mlflow(self):
        super()._setup_mlflow()
        if not self.cfg.MLflow.enabled:
            return
        if mlflow is None:
            raise ImportError(
                "MLflow is enabled in config but 'mlflow' is not installed. "
                "Either install mlflow or set cfg.MLflow.enable=False."
            )

        if self.config is not None and ddp.is_rank_zero():
            d = dict()
            for k, v in self.config.items():
                d[f"tune_{k}"] = v
            mlflow.log_params(d)
        if self._stopper is not None and ddp.is_rank_zero():
            d = {
                "stopper_patience": self._stopper._patience,
                "stopper_metric": self._stopper.metric,
                "stopper_delta": self._stopper._delta,
                "stopper_grace_period": self._stopper._grace_period,
            }
            mlflow.log_params(d)

    def _check(self):
        super()._check()
        if self._stopper is None:
            raise Exception("stopper should not be None, you should set it in setup")
        if self._is_tune_mode:
            assert self.config is not None, (
                "config is not set, you should set it in setup"
            )
            assert hasattr(self, "_store") and self._store is not None, (
                "store is not set, you should set it in setup"
            )

    def step(self):
        """One Ray Tune training iteration (one epoch).

        Behavior:
        - If the previous epoch was marked as a new best by BestScorePlateauStopper,
          save the checkpoint for that previous epoch before training this one.
        - Run one training epoch and one validation pass.
        - In DDP mode, checkpoint writes remain rank-zero-only while validation metrics
          are already aggregated by BasicRecipe before this method returns them.
        - Return a result dict with at least {"val_loss": float}.
        """
        # save checkpoint for the previous epoch, step()(omit the ckp for the first epoch) => stopper() => step()(save ckp) ... => stopper() => cleanup()(save final ckp)
        if self._cur_epoch > 0 and ray.get(
            self._store.get.remote(self.trial_id, self._cur_epoch)
        ):
            log_rank_zero(
                self._logger,
                f"Saving model of epoch {self._cur_epoch - 1}",
            )
            if ddp.is_rank_zero():
                self.save_checkpoint(self._cur_epoch - 1)
        # train step
        self._train_step(self._cur_epoch)
        # validation
        avg_val_loss = self._validation_step(self._cur_epoch)
        self._cur_epoch += 1

        return {"val_loss": avg_val_loss}

    def save_checkpoint(self, epoch_or_ckp_dir: int | str):
        if isinstance(epoch_or_ckp_dir, str):
            # tuner checkpoint
            p = pathlib.Path(epoch_or_ckp_dir) / "model.pth"
            torch.save(self._unwrap_model().state_dict(), p.absolute().__str__())
            return
        super().save_checkpoint(epoch_or_ckp_dir)

    def load_checkpoint(self, ckp_dir):
        # tuner checkpoint
        p = pathlib.Path(ckp_dir) / "model.pth"
        self._unwrap_model().load_state_dict(torch.load(p))

    def cleanup(self):
        # Log final epoch metadata on the tracked MLflow run from rank zero only.
        if (
            self.cfg.MLflow.enabled
            and mlflow is not None
            and self._mlflow_run is not None
            and ddp.is_rank_zero()
        ):
            mlflow.log_param("epochs_run", self._cur_epoch)
        return super().cleanup()


class AdaptiveTrainableRecipe(TrainableRecipe):
    """Adaptive trainable recipe that can be in tune mode or not.

    Default is non-tune mode (plain training). Explicitly pass is_tune_mode=True for Ray Tune usage.
    """

    def __init__(self, cfg: Config, *args, **kwargs):
        is_tune_mode = kwargs.pop("is_tune_mode", False)
        # Pass through explicit flag to parent
        super().__init__(cfg, is_tune_mode=is_tune_mode, *args, **kwargs)
        self._is_tune_mode = is_tune_mode
        # Ensure MLflow nested flag mirrors tune mode
        self.cfg.MLflow.nested = bool(self._is_tune_mode)
        if not cfg.logger:
            cfg.logger = get_logger_by_name("AdaptiveTrainableRecipe")
        cfg.logger.info(f"Use device: {self._device}")

    def setup(
        self,
        param_config: Optional[Dict] = None,
        is_tune_mode: bool = False,
        **kwargs,
    ):
        """Unified setup that works for both Tune and non-Tune modes.

        Args:
            param_config: Hyperparameter sample dict (from Ray Tune when tuning) that
                you may use to modify self.cfg before delegating to the base setup.
            is_tune_mode: If True, call TrainableRecipe.setup to register stopper/store
                and log tune parameters; if False, call BasicRecipe.setup.
            **kwargs: Additional keyword args forwarded to TrainableRecipe.setup in
                Tune mode (e.g., stopper, store).
        """
        self._is_tune_mode = is_tune_mode
        self.cfg.MLflow.nested = bool(self._is_tune_mode)
        if self._is_tune_mode:
            super().setup(param_config, **kwargs)
        else:
            # Direct BasicRecipe.setup path
            super(TrainableRecipe, self).setup()

    def cleanup(self):
        if self._is_tune_mode:
            super().cleanup()
        else:
            super(TrainableRecipe, self).cleanup()

    def _check(self):
        if self._is_tune_mode:
            super()._check()
        else:
            super(TrainableRecipe, self)._check()

    def _setup_early_stop(self):
        if self._is_tune_mode:
            super()._setup_early_stop()
        else:
            super(TrainableRecipe, self)._setup_early_stop()

    def _setup_metric(self):
        self._metric_fn = None

    def _metric_step(self, batch, logits) -> torch.Tensor:
        return torch.FloatTensor([1])

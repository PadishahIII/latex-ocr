# pyright: ignore-all

from typing import Dict, Any

import torch
import ray
from ray.tune import Stopper


@ray.remote
class SharedStopperStore:
    def __init__(self):
        self._best_scores: Dict[str, Dict[int, bool]] = dict()

    def put(self, trial_id, iteration, result: bool):
        if trial_id not in self._best_scores:
            self._best_scores[trial_id] = dict()
        self._best_scores[trial_id][iteration] = result

    def get(self, trial_id, iteration):
        if (
            trial_id not in self._best_scores
            or iteration not in self._best_scores[trial_id]
        ):
            return False
        return self._best_scores[trial_id][iteration]


class EarlyStopping:
    """Early stops the training if least validation loss doesn't shrink after a given patience."""

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self._patience,
            "counter": self._counter,
            "best_score": self._best_score,
            "last_score": self._last_score,
            "early_stop": self.early_stop,
            "delta": self._delta,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._patience = int(state.get("patience", self._patience))
        self._counter = int(state.get("counter", 0))
        self._best_score = state.get("best_score", None)
        self._last_score = state.get("last_score", None)
        self.early_stop = bool(state.get("early_stop", False))
        self._delta = float(state.get("delta", self._delta))
        self.is_new_best_score = False

    def __init__(self, patience=7, delta=0.0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: print
        """
        self._patience = patience
        self._counter = 0
        self._best_score = None
        self._last_score = None
        self.early_stop = False
        self._delta = delta
        self._trace_func = trace_func
        self.is_new_best_score = False

    def __call__(self, val_loss, no_counter_update=False):
        """Track new value.

        Args:
            val_loss (any): _description_
            no_counter_update (bool, optional): if True, the counter will not increase, fit to grace period. Defaults to False.
        """
        score = -val_loss
        self.is_new_best_score = False

        if self._best_score is None:
            self._best_score = score
        if self._last_score is None:
            self._last_score = score
        elif score < self._best_score + self._delta:
            # val loss not decreased
            if not no_counter_update:
                self._counter += 1
                self._trace_func(
                    f"EarlyStopping counter: {self._counter} out of {self._patience}, score: {score}, last: {self._last_score}, best: {self._best_score}"
                )
                if self._counter >= self._patience:
                    self.early_stop = True
        else:
            # val loss decreased, reset counter
            self._counter = 0
            if score > self._best_score:
                self._best_score = score
                self._trace_func(
                    f"EarlyStopping new best score: {self._best_score}, score: {self._best_score}"
                )
                self.is_new_best_score = True
        self._last_score = score


class BestScorePlateauStopper(Stopper):
    def __init__(
        self,
        metric_name: str,
        store: SharedStopperStore,
        patience=7,
        delta=0,
        grace_period=4,
        trace_func=print,
    ):
        super().__init__()
        self.trials: Dict[str, EarlyStopping] = dict()
        self._patience = patience
        self._delta = delta
        self._trace_func = trace_func
        self._grace_period = grace_period
        self.store: SharedStopperStore = store
        self.metric = metric_name

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        iteration = int(result["training_iteration"])
        # self._trace_func(f"iteration: {iteration}, grace_period: {self._grace_period}")
        if trial_id not in self.trials:
            self.trials[trial_id] = EarlyStopping(
                patience=self._patience, delta=self._delta, trace_func=self._trace_func
            )
        es = self.trials[trial_id]
        if iteration < self._grace_period:
            es(result[self.metric], no_counter_update=True)
        else:
            es(result[self.metric])
        if es.is_new_best_score:
            ray.get(self.store.put.remote(trial_id, iteration, True))
            self._trace_func(
                f"Stopper new best score, iteration: {iteration}, trial_id: {trial_id}, stopper: {self}"
            )
        if iteration < self._grace_period:
            return False
        return es.early_stop

    def stop_all(self):
        return False


def total_grad_norm(model: torch.nn.Module, norm_type=2) -> torch.Tensor:
    # same idea as torch.nn.utils.clip_grad_norm_
    norms = [
        p.grad.detach().norm(norm_type)
        for p in model.parameters()
        if p.grad is not None
    ]
    return torch.norm(torch.stack(norms), norm_type)


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


class ReduceLROnPlateauWithWarmup:
    """Combine linear warmup (per optimizer step) with ReduceLROnPlateau (per validation epoch).

    Usage pattern:
        scheduler = ReduceLROnPlateauWithWarmup(optimizer, warmup_steps=500)
        # In training loop per optimizer step:
        scheduler.step()  # handles warmup scaling only until warmup complete
        # After validation epoch:
        scheduler.plateau_step(val_loss)  # triggers plateau logic once warmup finished

    Warmup behavior:
        Linearly scales each param group's lr from 0 -> base_lr over `warmup_steps` optimizer steps.
        Base LRs are captured at construction time.

    Plateau behavior:
        Delegates to an internal torch.optim.lr_scheduler.ReduceLROnPlateau instance once
        warmup is complete. Its reductions operate on the actual current LR values.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | list[float] | None = None,
        eps: float = 1e-8,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.step_count = 0
        # capture base lrs once
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        # prepare plateau scheduler
        # Allow list min_lr or scalar
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr if min_lr is not None else 0.0,
            eps=eps,
        )

    def _set_lrs(self, lrs):
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def step(self):
        """Per optimizer step. Applies linear warmup scaling while warmup incomplete."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps and self.warmup_steps > 0:
            scale = float(self.step_count) / float(self.warmup_steps)
            new_lrs = [base * scale for base in self.base_lrs]
            self._set_lrs(new_lrs)

    def plateau_step(self, val_loss: float):
        """Call after a validation epoch with the average validation loss."""
        # Skip plateau logic during warmup period
        if self.step_count < self.warmup_steps:
            return
        prev_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self._plateau.step(val_loss)
        new_lrs = [g["lr"] for g in self.optimizer.param_groups]
        # Optionally could log here; delegate to caller for structured logging
        return prev_lrs, new_lrs

    # Expose selected diagnostic attributes mirroring ReduceLROnPlateau
    @property
    def best(self):
        return getattr(self._plateau, "best", None)

    @property
    def num_bad_epochs(self):
        return getattr(self._plateau, "num_bad_epochs", None)

    @property
    def patience(self):
        return getattr(self._plateau, "patience", None)

    @property
    def factor(self):
        return getattr(self._plateau, "factor", None)

    @property
    def cooldown(self):
        return getattr(self._plateau, "cooldown", None)

    @property
    def threshold(self):
        return getattr(self._plateau, "threshold", None)

    def warmup_done(self) -> bool:
        return self.step_count >= self.warmup_steps

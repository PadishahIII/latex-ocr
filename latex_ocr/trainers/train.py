# pyright: ignore-all

import os
import math
import click
import logging
from typing import Any, Dict, Optional, Tuple, cast

_visible_cuda_devices = os.environ.get("LATEX_OCR_VISIBLE_CUDA_DEVICES")
if _visible_cuda_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
    # Must be set before importing torch so CUDA only enumerates the intended GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible_cuda_devices

_disable_nccl_p2p = os.environ.get("LATEX_OCR_DISABLE_NCCL_P2P", "").strip().lower()
if _disable_nccl_p2p in {"1", "true", "yes", "on"} and "NCCL_P2P_DISABLE" not in os.environ:
    # Allow a repo-local escape hatch for boxes with unstable NVLink / GPU P2P.
    os.environ["NCCL_P2P_DISABLE"] = "1"

import torch

from torch import Tensor
from ray import tune
from torch.nn import CrossEntropyLoss
from torchtune.utils import log_rank_zero
from recipe.logging.logger import get_logger_by_name
from recipe.runner import Runner
from recipe.tune import AdaptiveTrainableRecipe
from recipe.utils import ddp
from recipe.utils.mlflow_util import setup_mlflow
from recipe.utils.train_util import get_device
from recipe import config

from latex_ocr.data.synth.data import (
    ConcatTokenizedDataset,
    TokenizedDataset,
    create_dataloader,
)
from latex_ocr.trainers.config import (
    CoCaCfg,
    DataCfg,
    ModelCfg,
    TrainerCfg,
    TransformerDecoderCfg,
    GRUDecoderCfg,
    ImageToSeqModel,
    ModelType,
)
from latex_ocr.trainers.model import get_model
from latex_ocr.models.coca.model import CoCaSwinOCR
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
logger = get_logger_by_name("latex_ocr_trainer")


def _log_gpu_runtime_overrides(cfg: TrainerCfg) -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        logger.info(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    if os.environ.get("NCCL_P2P_DISABLE") == "1":
        logger.warning("NCCL_P2P_DISABLE=1; disabling NCCL GPU P2P/NVLink transport")

    if (
        not cfg.use_ddp
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and not visible_devices
    ):
        logger.warning(
            "use_ddp=False but multiple CUDA devices are visible; training will use only "
            f"{cfg.device}. If peer/NVLink is unstable, set LATEX_OCR_VISIBLE_CUDA_DEVICES "
            "or CUDA_VISIBLE_DEVICES before launch."
        )


xception_gru_data_cfg = DataCfg(
    config_name="styled",
    both=False,
)
xception_gru_train_cfg = config.TrainCfg(
    learning_rate=3e-4,
    batch_size=64,  # 200 for 96GB VRAM, 64 for 16GB VRAM
    total_epochs=100,  # 10-100 for styled
    early_stop_patience=15,
    early_stop_delta=0.00001,
    early_stop_grace_period=5,
    gradient_accumulation_steps=2**0,
    weight_decay=0.01,  # drop if underfit
    num_warmup_steps=0.0,  # no warmup since resumed from checkpoint
    enable_activation_checkpointing=False,  # Checkpoint all leaf modules
    enable_amp=True,
    clip_grad_type="norm",
    clip_grad_value=1,  # mandatory for RNN decoder
    error_if_nonfinite_grad=False,
    log_gradient_norm_per_epoch=False,
    lr_decay=config.LRDecay.FACTOR,
    lr_decay_factor=0.3,
)
xception_gru_model_cfg = ModelCfg(
    model_type=ModelType.XCEPTION_GRU,
    dropout=0.1,  # basic
    max_seq_length=324,  # Reduced from 5000 (99th percentile is 271)
    encoder_pretrained=True,
    decoder_pretrained=True,
    # transformer_decoder_cfg=TransformerDecoderCfg(
    #     ffn_dim=384,
    #     nhead=12,
    #     layers=4,
    # ),
    vocab_size=1122,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    gru_decoder_cfg=GRUDecoderCfg(
        emb_dim=512,
        dec_dim=256,
        attn_dim=256,
        freeze_encoder=False,
        use_gradient_checkpointing=True,
    ),
    encoder_lr=1e-5,  # 1e-5 to 1e-4
    decoder_lr=3e-4,  # 1e-4 to 3e-4
)
xception_gru_ckp_cfg = config.CheckpointCfg(
    enable_local_ckp=False,
    enable_state_ckp_to_mlflow=True,
    store_rng_state=False,
    # NOTE: resume defaults are off in the open-source repo. To resume a run,
    # set these via your own config or env (see recipe.config.CheckpointCfg).
    resume_from_mlflow=False,
    resume_strict=False,
    mlflow_run_id=None,
)
xception_gru_mlflow_cfg = config.MLflowCfg(
    experiment_name="latex-ocr",
    mlflow_model_name="latex-ocr",
    dataset_name="",
    desc="xception gru",
    disable_mlflow=False,
)

coca_pretrain_data_cfg = DataCfg(
    config_name="plain",
    both=False,
)
coca_finetune_data_cfg = DataCfg(
    config_name="plain",
    both=False,
    plain_proportion=0.5,
)

# Train config
coca_pretrain_train_cfg = config.TrainCfg(
    learning_rate=3e-4,
    batch_size=40,  # 200 for 96GB VRAM, 64 for 16GB VRAM
    total_epochs=50,
    early_stop_patience=10,
    early_stop_delta=0.00001,
    early_stop_grace_period=10,
    gradient_accumulation_steps=2**4,
    weight_decay=0.01,  # drop if underfit
    # If <1 treated as ratio of total steps, else absolute steps
    num_warmup_steps=0.05,  # 2.5 epochs
    enable_activation_checkpointing=False,  # Checkpoint all leaf modules
    enable_amp=False,  # V100 does not support bfloat16; float16 overflows in early attention layers
    amp_dtype=torch.float32,
    clip_grad_type="norm",
    clip_grad_value=1,
    error_if_nonfinite_grad=False,
    log_gradient_norm_per_epoch=False,
    lr_decay=config.LRDecay.COSINE,
)
FINETUNE_STAGE_A_EPOCHS = 5  # decoder/head-only (0.25–1 epoch suggested)
FINETUNE_STAGE_B_EPOCHS = 45  # end-to-end (3–10 epochs suggested)
FINETUNE_STAGE_A_WARMUP_RATIO = 0.10
FINETUNE_STAGE_B_WARMUP_RATIO = 0.05
FINETUNE_STAGE_B_MIN_LR_RATIO = 0.05  # 1–10% of peak suggested

coca_finetune_train_cfg = config.TrainCfg(
    learning_rate=1e-5,
    batch_size=40,  # 200 for 96GB VRAM, 64 for 16GB VRAM
    total_epochs=FINETUNE_STAGE_A_EPOCHS + FINETUNE_STAGE_B_EPOCHS,
    early_stop_patience=10,
    early_stop_delta=0.00001,
    early_stop_grace_period=10,
    gradient_accumulation_steps=2**4,
    weight_decay=0.01,  # drop if underfit
    # Kept for backwards compatibility; finetune uses per-stage warmup instead.
    num_warmup_steps=FINETUNE_STAGE_B_WARMUP_RATIO,
    enable_activation_checkpointing=False,  # Checkpoint all leaf modules
    enable_amp=False,  # V100 does not support bfloat16; float16 overflows in early attention layers
    amp_dtype=torch.float32,
    clip_grad_type="norm",
    clip_grad_value=1,
    error_if_nonfinite_grad=False,
    log_gradient_norm_per_epoch=False,
    lr_decay=config.LRDecay.COSINE,
)

# Model config
coca_pretrain_model_cfg = ModelCfg(
    model_type=ModelType.COCA,
    dropout=0.2,
    max_seq_length=354,  # Reduced from 5000 (99th percentile is 271)
    encoder_pretrained=True,
    vocab_size=1122,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    coca_cfg=CoCaCfg(
        dim=384,
        heads=6,
        unimodal_depth=3,
        multimodal_depth=3,
        num_img_queries=256,
        dim_latents=384 * 3,  # 1152
        lambda_cap=2.0,
        lambda_con=1.0,
        label_smoothing=0.1,
        swin_name="swin_small_patch4_window7_224.ms_in22k",
    ),
    encoder_lr=5e-5,
    decoder_lr=3e-4,  # 2e-4 to 3e-4
)
coca_finetune_model_cfg = ModelCfg(
    model_type=ModelType.COCA,
    dropout=0.2,
    max_seq_length=354,  # Reduced from 5000 (99th percentile is 271)
    encoder_pretrained=True,
    # load_weight_from_mlflow_run="2e8391d13bbd448f9882c9fa545c3701",
    # mlflow_model_name="latex-ocr-coca-pretrain_epoch00020_final",
    coca_pretrained=True,
    vocab_size=1122,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    coca_cfg=CoCaCfg(
        dim=384,
        heads=6,
        unimodal_depth=3,
        multimodal_depth=3,
        num_img_queries=256,
        dim_latents=384 * 3,  # 1152
        lambda_cap=2.0,
        lambda_con=1.0,
        label_smoothing=0.1,
        swin_name="swin_small_patch4_window7_224.ms_in22k",
    ),
    encoder_lr=1e-6,  # 5-10x lower than decoder lr
    decoder_lr=1e-5,  # 5-20x lower than pretraining lr
)

# Checkpoint config
coca_pretrain_ckp_cfg = config.CheckpointCfg(
    enable_local_ckp=False,
    enable_state_ckp_to_mlflow=True,
    store_rng_state=False,
    # resume_strict=True,
    # resume_from_mlflow=True,
    # mlflow_run_id="2e87f94365874042af449ff84b1311e1",
)
coca_finetune_ckp_cfg = config.CheckpointCfg(
    enable_local_ckp=False,
    enable_state_ckp_to_mlflow=True,
    store_rng_state=False,
    # resume_from_mlflow=False,
    # resume_from_path=True,
    # resume_strict=True,
    # mlflow_run_id="fece6f9cb3574aeca0262a9b9d024da5",
)


# MLFlow config
coca_pretrain_mlflow_cfg = config.MLflowCfg(
    experiment_name="latex-ocr-coca-pretrain",
    mlflow_model_name="latex-ocr-coca-pretrain",
    dataset_name="",
    desc="pretrain on v2.0 dataset, use swin-small as encoder",
    disable_mlflow=False,
)
coca_finetune_mlflow_cfg = config.MLflowCfg(
    experiment_name="latex-ocr-coca-finetune",
    mlflow_model_name="latex-ocr-coca-finetune",
    dataset_name="",
    desc="styled finetune, 50 epochs total, 0.2 dropout, 0.01 weight decay",
    disable_mlflow=False,
)

cfg = TrainerCfg(
    use_ddp=False,
    pin_memory=False,  # for large memory machine
    label_smoothing=0.1,
    model_cfg=coca_pretrain_model_cfg,
    data_cfg=coca_pretrain_data_cfg,
    model_factory=get_model,
    device=get_device(),
    log_every_n_steps=int(2e2),
    Debug=config.DebugCfg(
        enable_snapshot=False,
        skip_steps_to_snapshot=int(0),
        max_steps_to_snapshot=12,
        snapshot_every_n_step=12,
        save_snapshot_dir="./snapshots",
        enable_profiler=False,
        skip_steps_to_profile=0,
        wait_steps_between_repeat=0,
        active_steps_in_one_repeat=2,
        repeats=10,
        save_memory_timeline_dir="./memory_timeline",
        disable_trace_handler=True,
        ddp_rank_progress_bars=True,
    ),
    Train=coca_pretrain_train_cfg,
)


class Trainer(AdaptiveTrainableRecipe):
    def __init__(self, cfg: TrainerCfg, *args, **kwargs):
        cfg.logger = logger
        super().__init__(cfg, *args, **kwargs)
        if cfg.use_ddp and not ddp.is_rank_zero():
            logger.setLevel(logging.WARNING)
        # Keep base-class `cfg` attribute for recipe framework.
        self.cfg = cfg
        # Use a strongly-typed alias for local access.
        self._cfg: TrainerCfg = cfg
        self._finetune_encoder_frozen: bool = False

    def _is_coca_finetune(self) -> bool:
        return bool(
            self._cfg.model_cfg.model_type == ModelType.COCA
            and self._cfg.model_cfg.coca_pretrained
        )

    def _set_encoder_trainable(self, trainable: bool) -> None:
        if trainable == (not self._finetune_encoder_frozen):
            return

        model = cast(ImageToSeqModel, self._unwrap_model())
        for p in model.encoder_parameters():
            p.requires_grad = trainable

        self._finetune_encoder_frozen = not trainable
        logger.info(
            f"Finetune encoder trainable={trainable} (frozen={self._finetune_encoder_frozen})"
        )

    def setup(self, param_config: Optional[Dict] = None, is_tune_mode=False, **kwargs):
        logger.info("=== Starting setup ===")
        if param_config is not None and len(param_config) > 0:
            pass
            # if "lr" in param_config:
            #     self.cfg.Train.learning_rate = param_config["lr"]
            # if "clip_grad_value" in param_config:
            #     self.cfg.Train.clip_grad_value = param_config["clip_grad_value"]
        super().setup(param_config=param_config, **kwargs)

        # Stage A: decoder/head-only finetune warm-start.
        if self._is_coca_finetune():
            self._set_encoder_trainable(False)

        logger.info("=== Finished setup ===")

    def _setup_data(self):
        logger.info(">>> Starting _setup_data")
        # IMPORTANT: keep dataset token lengths aligned with model max_seq_length.
        # This reduces padding/compute (faster epochs) and avoids training on sequences
        # that are longer than the model was configured for.
        self._train_dataset, self._dataloader = create_dataloader(
            config_name=self._cfg.data_cfg.config_name,
            split="train",
            batch_size=self._cfg.Train.batch_size,
            shuffle=True,
            num_workers=8,
            both=self._cfg.data_cfg.both,
            pin_memory=self._cfg.pin_memory,
            max_seq_len=self._cfg.model_cfg.max_seq_length,
            plain_proportion=self._cfg.data_cfg.plain_proportion,
            use_distributed_sampler=self.cfg.use_ddp,
            # Only the Transformer-decoder model uses a (T,T) causal mask.
            # GRU and CoCa use internal attention masking.
            return_attention_mask=(
                self._cfg.model_cfg.model_type == ModelType.SWIN_TRANSFORMER
            ),
        )
        # Dataset can be TokenizedDataset, ConcatTokenizedDataset, or MixedTokenizedDataset.
        # Log filtered samples count if available
        if hasattr(self._train_dataset, "filtered_count"):
            logger.info(
                f"Train: filtered {self._train_dataset.filtered_count} samples "
                f"exceeding max_seq_len={self._cfg.model_cfg.max_seq_length}"
            )
        logger.info(
            f"Train dataset size: {len(self._train_dataset)}, Train dataloader batches: {len(self._dataloader)}"
        )
        self._val_dataset, self._val_dataloader = create_dataloader(
            config_name=self._cfg.data_cfg.config_name,
            split="validation",
            batch_size=self._cfg.Train.batch_size,
            shuffle=False,
            pin_memory=self._cfg.pin_memory,
            plain_proportion=self._cfg.data_cfg.plain_proportion,
            num_workers=8,
            max_seq_len=self._cfg.model_cfg.max_seq_length,
            use_distributed_sampler=self.cfg.use_ddp,
        )
        # Log filtered samples count if available
        if hasattr(self._val_dataset, "filtered_count"):
            logger.info(
                f"Val: filtered {self._val_dataset.filtered_count} samples "
                f"exceeding max_seq_len={self._cfg.model_cfg.max_seq_length}"
            )
        logger.info(
            f"Val dataset size: {len(self._val_dataset)}, Val dataloader batches: {len(self._val_dataloader)}"
        )

        # Update model config with actual tokenizer vocab size
        actual_vocab_size = self._train_dataset.tokenizer.vocab_size
        if self._cfg.model_cfg.vocab_size != actual_vocab_size:
            logger.warning(
                f"Updating model vocab_size from {self._cfg.model_cfg.vocab_size} to {actual_vocab_size} (from tokenizer)"
            )
            self._cfg.model_cfg.vocab_size = actual_vocab_size
        logger.info("<<< Finished _setup_data")

    def _get_input_sample(self) -> torch.Tensor:
        self.before_get_input_example_check()
        batch = next(iter(self._dataloader))
        return batch

    def _setup_model(self):
        logger.info(">>> Starting _setup_model")
        logger.info(
            f'setting up MLflow with experiment "{self.cfg.MLflow.experiment_name}"'
        )
        setup_mlflow(self.cfg.MLflow.experiment_name)

        self._model = self._cfg.model_factory(self._cfg.model_cfg)
        logger.info(f"Model created with vocab_size={self._cfg.model_cfg.vocab_size}")
        self._model.to(self._device)
        logger.info(f"Model moved to device: {self._device}")
        self._loss_fn = CrossEntropyLoss(
            label_smoothing=self._cfg.label_smoothing,
            ignore_index=self._train_dataset.tokenizer.pad_token_id,
        )
        logger.info(
            f"Loss function initialized with label_smoothing={self._cfg.label_smoothing}, ignore_index={self._train_dataset.tokenizer.pad_token_id}"
        )
        logger.info("<<< Finished _setup_model")

    def _setup_optimizer(self):
        logger.info(">>> Starting _setup_optimizer")
        self.before_setup_optimizer_check()
        model = cast(ImageToSeqModel, self._unwrap_model())
        self._optimizer = torch.optim.AdamW(
            # self._model.parameters(),
            [
                {
                    "params": model.encoder_parameters(),
                    "lr": self._cfg.model_cfg.encoder_lr,
                },
                {
                    "params": model.decoder_parameters(),
                    "lr": self._cfg.model_cfg.decoder_lr,
                },
                {
                    "params": model.other_parameters(),
                    "lr": self._cfg.Train.learning_rate,
                },
            ],
            lr=self._cfg.Train.learning_rate,
            weight_decay=self._cfg.Train.weight_decay,
        )
        logger.info(
            f"AdamW optimizer initialized with lr={self._cfg.Train.learning_rate}, weight_decay={self._cfg.Train.weight_decay}"
        )
        logger.info("<<< Finished _setup_optimizer")

    def _setup_lr_scheduler(self):
        logger.info(">>> Starting _setup_lr_scheduler")
        assert self._dataloader is not None
        assert self._optimizer is not None

        accum_steps = max(1, self._cfg.Train.gradient_accumulation_steps)
        steps_per_epoch = len(self._dataloader) // accum_steps
        total_steps = int(self._cfg.Train.total_epochs * steps_per_epoch)

        if self._is_coca_finetune():
            stage_a_epochs = int(FINETUNE_STAGE_A_EPOCHS)
            stage_b_epochs = int(FINETUNE_STAGE_B_EPOCHS)
            stage_a_steps = max(1, stage_a_epochs * steps_per_epoch)
            stage_b_steps = max(1, stage_b_epochs * steps_per_epoch)

            warmup_a_steps = max(1, int(FINETUNE_STAGE_A_WARMUP_RATIO * stage_a_steps))
            warmup_b_steps = max(1, int(FINETUNE_STAGE_B_WARMUP_RATIO * stage_b_steps))

            def lr_lambda(step: int) -> float:
                if step < stage_a_steps:
                    # Stage A: warmup then constant LR.
                    if step < warmup_a_steps:
                        return float(step) / float(warmup_a_steps)
                    return 1.0

                # Stage B: short warmup/hold then cosine decay to min LR.
                s = step - stage_a_steps
                if s < warmup_b_steps:
                    return 1.0

                decay_steps = max(1, stage_b_steps - warmup_b_steps)
                progress = min(1.0, float(s - warmup_b_steps) / float(decay_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_lr = float(FINETUNE_STAGE_B_MIN_LR_RATIO)
                return min_lr + (1.0 - min_lr) * cosine

            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lr_lambda
            )
            self._reduce_on_plateau = None
            log_rank_zero(
                logger,
                f"Finetune schedule: stage_a_steps={stage_a_steps} (warmup={warmup_a_steps}), "
                f"stage_b_steps={stage_b_steps} (warmup={warmup_b_steps}), "
                f"min_lr_ratio={FINETUNE_STAGE_B_MIN_LR_RATIO}, total_steps={total_steps}",
            )
            self.reserve_param("finetune_stage_a_steps", stage_a_steps)
            self.reserve_param("finetune_stage_b_steps", stage_b_steps)
            self.reserve_param("finetune_warmup_a_steps", warmup_a_steps)
            self.reserve_param("finetune_warmup_b_steps", warmup_b_steps)
            self.reserve_param("finetune_min_lr_ratio", FINETUNE_STAGE_B_MIN_LR_RATIO)
            self.reserve_param("total steps", total_steps)
            logger.info("Finetune LambdaLR scheduler initialized (2-stage)")
            logger.info("<<< Finished _setup_lr_scheduler")
            return

        if self._cfg.Train.num_warmup_steps < 1:
            warmup_steps = int(self._cfg.Train.num_warmup_steps * total_steps)
        else:
            warmup_steps = int(self._cfg.Train.num_warmup_steps)
        log_rank_zero(
            logger,
            f"Warmup steps: {warmup_steps}, total steps: {total_steps}\n"
            + f"Total steps: {total_steps}",
        )
        self.reserve_param("warmup steps", warmup_steps)
        self.reserve_param("total steps", total_steps)
        self.once = False

        if self._cfg.Train.lr_decay == config.LRDecay.FACTOR:

            def factor_lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                if not self.once:
                    self.once = True
                    self._lr_scheduler = None
                return 1.0

            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=factor_lr_lambda
            )
            logger.info("LambdaLR scheduler initialized with warmup")
            self._reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                "min",
                factor=self._cfg.Train.lr_decay_factor,
                patience=1,
                threshold=1e-6,
                cooldown=0,
                threshold_mode="abs",
                eps=0.0,
            )
            logger.info(
                "ReduceLROnPlateau scheduler initialized with factor=0.3, patience=1"
            )
        else:
            from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

            self._lr_scheduler = get_cosine_schedule_with_warmup(
                self._optimizer,
                num_training_steps=total_steps,
                num_warmup_steps=warmup_steps,
            )
            self._reduce_on_plateau = None
            logger.info("Cosine LR scheduler initialized with warmup")

        logger.info("<<< Finished _setup_lr_scheduler")

    def _loss_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None, "_loss_step: model is not initialized"
        image = batch["images"].to(self._device)
        text = batch["labels"].to(self._device)

        # CoCa computes caption + contrastive loss internally.
        if self._cfg.model_cfg.model_type == ModelType.COCA:
            if not isinstance(self._unwrap_model(), CoCaSwinOCR):
                raise RuntimeError("COCA model_type requires CoCaSwinOCR")
            if self._cfg.model_cfg.coca_cfg is None:
                raise RuntimeError("COCA model_type requires model_cfg.coca_cfg")
            out = self._model(
                image,
                text,
                return_loss=True,
                lambda_cap=self._cfg.model_cfg.coca_cfg.lambda_cap,
                lambda_con=self._cfg.model_cfg.coca_cfg.lambda_con,
            )
            if not torch.isfinite(out["cap_loss"]):
                logger.warning(
                    f"cap_loss non-finite: {out['cap_loss'].item()}, step={self.global_step}"
                )
            if not torch.isfinite(out["con_loss"]):
                logger.warning(
                    f"con_loss non-finite: {out['con_loss'].item()}, step={self.global_step}"
                )
            return out["logits"], out["loss"]

        attention_mask = None
        # Note: attention_mask is only used for Transformer decoder models
        if self._cfg.model_cfg.model_type == ModelType.SWIN_TRANSFORMER:
            attention_mask = batch["attention_mask"].to(self._device)

        logits = self._model(
            image,
            text,
            tgt_mask=attention_mask,
        )
        # Reshape logits and labels for CrossEntropyLoss
        # For GRU decoder: logits shape is (B, T-1, V) where it predicts tokens 1..T-1
        # given input tokens 0..T-2. Target should be text[:, 1:] to match.
        # For Transformer decoder: logits shape is (B, T, V), so target is text as-is.
        if logits.size(1) == text.size(1) - 1:
            # GRU decoder case: shift target to match logits
            target = text[:, 1:].contiguous()  # Remove BOS, predict tokens 1..T-1
        else:
            # Transformer decoder case: no shift needed
            target = text
        loss = self._loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return logits, loss

    def _validation_step(self, curr_epoch: int) -> float:
        avg_val_loss = super()._validation_step(curr_epoch)

        # Switch to Stage B (end-to-end) right after Stage A completes.
        if self._is_coca_finetune() and curr_epoch + 1 >= int(FINETUNE_STAGE_A_EPOCHS):
            self._set_encoder_trainable(True)

        if self._reduce_on_plateau is not None:
            self._reduce_on_plateau.step(avg_val_loss, epoch=curr_epoch)
            self._logger.info(
                f"best: {self._reduce_on_plateau.best}, num_bad_epochs: {self._reduce_on_plateau.num_bad_epochs}, cooldown counter: {self._reduce_on_plateau.cooldown_counter}, patience: {self._reduce_on_plateau.patience}"
            )
        return avg_val_loss

    def _setup_metric(self):
        pass

    def _metric_step(self, batch, logits) -> torch.Tensor:
        return torch.tensor(0.0)

    def _batch_to_device(self, batch) -> Any:
        if not self._cfg.pin_memory:
            batch = {
                k: v.to(self._device) if v is not None else None
                for k, v in batch.items()
            }
        else:
            batch = {
                k: v.to(self._device, non_blocking=True) if v is not None else None
                for k, v in batch.items()
            }
        return batch


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--no-param-search",
    is_flag=True,
    default=False,
    help="Disable hyperparameter search",
)
def train_tune(no_param_search: bool) -> None:
    # Define the search space (Runner will disable it if no_param_search=True)
    param_space = {
        # "dropout": tune.grid_search([0.1, 0.3, 0.5]),
        "lr": tune.grid_search([1e-3, 2e-3, 5e-4]),
        "clip_grad_value": tune.grid_search([0.5, 1.0]),
        # "batch_size": tune.grid_search([128]),
        # "d_model": tune.grid_search([256, 512]),
        # "d_ff": tune.grid_search([128, 256, 512]),
        # "num_head": tune.grid_search([2, 4, 8]),
        # "num_speaker": tune.grid_search([600]),
        # "encoder_layers": tune.grid_search([3, 4]),
    }

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    _log_gpu_runtime_overrides(cfg)

    runner = Runner(
        trainable_cls=Trainer,
        experiment_name=cfg.MLflow.experiment_name,
        is_tune_mode=True,
        cpu=os.cpu_count(),
        gpu=1,
        no_param_search=no_param_search,
        param_space=param_space,
        on_parent_run_registered=lambda rid: setattr(cfg.MLflow, "parent_run_id", rid),
    )
    runner.run()


@cli.command()
@click.option(
    "--recipe-name",
    help="Recipe to select",
    type=click.Choice(["xception_gru", "coca_pretrain", "coca_finetune"]),
)
def train(recipe_name: str):
    match recipe_name:
        case "xception_gru":
            cfg.Train = xception_gru_train_cfg
            cfg.model_cfg = xception_gru_model_cfg
            cfg.Checkpoint = xception_gru_ckp_cfg
            cfg.MLflow = xception_gru_mlflow_cfg
            cfg.data_cfg = xception_gru_data_cfg
        case "coca_pretrain":
            cfg.Train = coca_pretrain_train_cfg
            cfg.model_cfg = coca_pretrain_model_cfg
            cfg.Checkpoint = coca_pretrain_ckp_cfg
            cfg.MLflow = coca_pretrain_mlflow_cfg
            cfg.data_cfg = coca_pretrain_data_cfg
        case "coca_finetune":
            cfg.Train = coca_finetune_train_cfg
            cfg.model_cfg = coca_finetune_model_cfg
            cfg.Checkpoint = coca_finetune_ckp_cfg
            cfg.MLflow = coca_finetune_mlflow_cfg
            cfg.data_cfg = coca_finetune_data_cfg
        case _:
            raise ValueError(f"Not supported recipe: {recipe_name}")
    _log_gpu_runtime_overrides(cfg)
    recipe = Trainer(
        cfg=cfg,
        is_tune_mode=False,
    )
    recipe.setup(is_tune_mode=False)
    recipe.safe_train()
    recipe.cleanup()


if __name__ == "__main__":
    """
    PYTHONPATH=. uv run python latex_ocr/trainers/train.py train_tune --no-param-search
    PYTHONPATH=. uv run python latex_ocr/trainers/train.py train --recipe-name coca_finetune
    PYTHONPATH=. uv run python latex_ocr/trainers/train.py train --recipe-name xception_gru
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone --nproc_per_node=2 latex_ocr/trainers/train.py train --recipe-name coca_finetune
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone --nproc_per_node=2 latex_ocr/trainers/train.py train --recipe-name xception_gru
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone --nproc_per_node=2 latex_ocr/trainers/train.py train --recipe-name coca_pretrain
    LATEX_OCR_DISABLE_NCCL_P2P=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone --nproc_per_node=2 latex_ocr/trainers/train.py train --recipe-name coca_pretrain
    LATEX_OCR_VISIBLE_CUDA_DEVICES=1 PYTHONPATH=. nohup uv run python latex_ocr/trainers/train.py train --recipe-name coca_pretrain 2>&1 &> pretrain.log &
    PYTHONPATH=. nohup uv run python latex_ocr/trainers/train.py train --recipe-name xception_gru 2>&1 &> train.log &
    PYTHONPATH=. nohup uv run python latex_ocr/trainers/train.py train --recipe-name coca_finetune 2>&1 &> train.log &
    PYTHONPATH=. nohup uv run python latex_ocr/trainers/train.py train --recipe-name coca_pretrain 2>&1 &> pretrain.log &
    PYTHONPATH=. nohup uv run python latex_ocr/trainers/train.py train 2>&1 &> train.log &
    """
    cli()

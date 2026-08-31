from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple, cast

import click
import torch
from torch.nn import CrossEntropyLoss
from torchtune.utils import log_rank_zero


from recipe.logging.logger import get_logger_by_name
from recipe.tune import AdaptiveTrainableRecipe
from recipe.utils.train_util import get_device
from latex_ocr.data.synth.data import TokenizedDataset, create_dataloader
from latex_ocr.trainers.config import (
    DataCfg,
    ModelCfg,
    ModelType,
    TrainerCfg,
    TransformerDecoderCfg,
)

from recipe import config
from latex_ocr.trainers.decoder.transformer_pretrain_model import TransformerDecoder

warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

logger = get_logger_by_name("latex_ocr_transformer_decoder_pretrain")


def get_transformer_decoder_model(cfg: ModelCfg) -> TransformerDecoder:
    if cfg.transformer_decoder_cfg is None:
        cfg.transformer_decoder_cfg = TransformerDecoderCfg()

    return TransformerDecoder(
        vocab_size=cfg.vocab_size,
        decoder_nhead=cfg.transformer_decoder_cfg.nhead,
        decoder_layers=cfg.transformer_decoder_cfg.layers,
        decoder_ffn_dim=cfg.transformer_decoder_cfg.ffn_dim,
        decoder_model_dim=768,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout,
    )


data_cfg = DataCfg(
    config_name="styled",
    both=True,
)

cfg = TrainerCfg(
    pin_memory=False,  # for large memory machine
    label_smoothing=0.1,
    model_cfg=ModelCfg(
        model_type=ModelType.SWIN_TRANSFORMER,
        dropout=0.1,
        max_seq_length=324,  # Reduced from 5000 (99th percentile is 271)
        encoder_pretrained=False,  # Not used in decoder-only pretraining
        vocab_size=1122,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        transformer_decoder_cfg=TransformerDecoderCfg(
            ffn_dim=384,
            nhead=12,
            layers=4,
        ),
    ),
    data_cfg=data_cfg,
    model_factory=get_transformer_decoder_model,
    device=get_device(),
    log_every_n_steps=int(10),  # about 1hr per epoch
    Train=config.TrainCfg(
        no_validation=True,
        learning_rate=1e-4,
        batch_size=256,
        total_epochs=10,
        early_stop_patience=3,
        early_stop_delta=0.00001,
        early_stop_grace_period=0,
        gradient_accumulation_steps=2**0,
        weight_decay=0.01,
        # If <1 treated as ratio of total steps, else absolute steps
        num_warmup_steps=0.05,
        enable_activation_checkpointing=False,  # Checkpoint all leaf modules
        enable_amp=True,
        # clip_grad_type="norm",
        # clip_grad_value=1,
        error_if_nonfinite_grad=True,
        log_gradient_norm_per_epoch=False,
    ),
    MLflow=config.MLflowCfg(
        experiment_name="latex-ocr-transformer-decoder",
        mlflow_model_name="latex-ocr-transformer-decoder",
        dataset_name="both",
        desc="pretrain transformer decoder",
        disable_mlflow=False,
    ),
    Checkpoint=config.CheckpointCfg(
        enable_local_ckp=False,
        enable_state_ckp_to_mlflow=True,
        store_rng_state=False,
        # resume_from_mlflow=True,
        # resume_strict=True,
        # mlflow_run_id="<run_id>",
    ),
)


class TransformerDecoderPretrainTrainer(AdaptiveTrainableRecipe):
    def __init__(self, cfg: TrainerCfg, *args, **kwargs):
        cfg.logger = logger
        super().__init__(cfg, *args, **kwargs)
        self._cfg = cfg

    def setup(
        self, param_config: Optional[Dict] = None, is_tune_mode: bool = False, **kwargs
    ):
        super().setup(param_config=param_config, is_tune_mode=is_tune_mode, **kwargs)

    def _setup_data(self):
        # For decoder-only LM pretraining we do not need images.
        train_ds, train_dl = create_dataloader(
            config_name=self._cfg.data_cfg.config_name,
            split="train",
            batch_size=self._cfg.Train.batch_size,
            shuffle=True,
            num_workers=2,
            both=self._cfg.data_cfg.both,
            pin_memory=self._cfg.pin_memory,
            max_seq_len=self._cfg.model_cfg.max_seq_length,
            formula_only=True,
            return_attention_mask=True,
        )
        self._train_dataset = cast(TokenizedDataset, train_ds)
        self._dataloader = cast(Any, train_dl)

        # val_ds, val_dl = create_dataloader(
        #     config_name=self._cfg.data_cfg.config_name,
        #     split="validation",
        #     batch_size=self._cfg.Train.batch_size,
        #     shuffle=False,
        #     num_workers=2,
        #     both=self._cfg.data_cfg.both,
        #     pin_memory=self._cfg.pin_memory,
        #     max_seq_len=self._cfg.model_cfg.max_seq_length,
        #     formula_only=True,
        #     return_attention_mask=True,
        # )
        # self._val_dataset = cast(TokenizedDataset, val_ds)
        # self._val_dataloader = cast(Any, val_dl)

        # Keep config vocab in sync with tokenizer.
        actual_vocab_size = self._train_dataset.tokenizer.vocab_size  # type: ignore[attr-defined]
        if self._cfg.model_cfg.vocab_size != actual_vocab_size:
            logger.warning(
                f"Updating model vocab_size from {self._cfg.model_cfg.vocab_size} to {actual_vocab_size} (from tokenizer)"
            )
            self._cfg.model_cfg.vocab_size = actual_vocab_size

    def _get_input_sample(self) -> torch.Tensor:
        # For signature inference, return token tensor
        assert self._dataloader is not None
        batch = next(iter(self._dataloader))
        return batch["labels"]

    def _setup_model(self):
        self._model = self._cfg.model_factory(self._cfg.model_cfg)
        self._model.to(self._device)

        pad_id = self._train_dataset.tokenizer.pad_token_id  # type: ignore[attr-defined]
        self._loss_fn = CrossEntropyLoss(
            label_smoothing=self._cfg.label_smoothing,
            ignore_index=pad_id,
        )

    def _setup_metric(self):
        self._metric_fn = None

    def _setup_optimizer(self):
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),  # type: ignore[union-attr]
            lr=self._cfg.Train.learning_rate,
            weight_decay=self._cfg.Train.weight_decay,
        )

    def _setup_lr_scheduler(self):
        assert self._dataloader is not None
        assert self._optimizer is not None

        total_steps = self.cfg.Train.total_epochs * (
            len(self._dataloader) // self.cfg.Train.gradient_accumulation_steps
        )
        if self.cfg.Train.num_warmup_steps < 1:
            warmup_steps = int(self.cfg.Train.num_warmup_steps * total_steps)
        else:
            warmup_steps = int(self.cfg.Train.num_warmup_steps)
        log_rank_zero(
            logger,
            f"Warmup steps: {warmup_steps}, total steps: {total_steps}\n"
            + f"Total steps: {total_steps}",
        )
        self.reserve_param("warmup steps", warmup_steps)
        self.reserve_param("total steps", total_steps)
        from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

        self._lr_scheduler = get_cosine_schedule_with_warmup(
            self._optimizer,
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
        )

    def _after_resume_from_checkpoint(self):
        # reset early stop
        self._setup_early_stop()

    def _batch_to_device(self, batch) -> Any:
        # batch is a dict from collate_fn
        if not self._cfg.pin_memory:
            return {
                k: v.to(self._device) if v is not None else None
                for k, v in batch.items()
            }
        return {
            k: v.to(self._device, non_blocking=True) if v is not None else None
            for k, v in batch.items()
        }

    def _loss_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None
        assert self._loss_fn is not None

        labels = batch["labels"].to(self._device)
        attentioin_mask = batch["attention_mask"].to(self._device)
        logits = self._model(None, labels, tgt_mask=attentioin_mask)

        loss = self._loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

    def _metric_step(self, batch, logits) -> torch.Tensor:
        return torch.tensor(0.0)




@click.group()
def cli():
    pass


@cli.command()
def train() -> None:
    recipe = TransformerDecoderPretrainTrainer(cfg=cfg, is_tune_mode=False)
    recipe.setup(is_tune_mode=False)
    recipe.safe_train()
    recipe.cleanup()


@cli.command(name="smoke_cpu")
def smoke_cpu() -> None:
    """CPU-only smoke test that does not touch the HF dataset.

    Runs a tiny forward/backward/optimizer step on random token data to validate
    shapes and basic training plumbing.
    """

    device = "cpu"
    local_cfg = cfg.model_copy(deep=True)
    local_cfg.device = device
    local_cfg.MLflow.enable = False
    local_cfg.MLflow.disable_mlflow = True
    local_cfg.Checkpoint.enable_local_ckp = False
    local_cfg.Train.total_epochs = 1
    local_cfg.Train.batch_size = 2
    local_cfg.Train.gradient_accumulation_steps = 1
    local_cfg.Train.enable_amp = False

    local_cfg.model_cfg.transformer_decoder_cfg = TransformerDecoderCfg(
        ffn_dim=256,
        nhead=4,
        layers=2,
    )

    train_ds, train_dl = create_dataloader(
        config_name="styled",
        split="train",
        batch_size=2,
        shuffle=False,
        num_workers=2,
        both=False,
        max_seq_len=324,
        formula_only=True,
        return_attention_mask=True,
    )
    dp = next(iter(train_dl))

    model = get_transformer_decoder_model(local_cfg.model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss(ignore_index=0)

    torch.manual_seed(0)
    labels = dp['labels']
    mask = dp['attention_mask']

    model.train()
    logits = model(None, labels, tgt_mask=mask)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    optimizer.step()

    click.echo(
        f"smoke-cpu ok: logits={tuple(logits.shape)} loss={loss.detach().item():.4f}"
    )


if __name__ == "__main__":
    """
    CPU smoke test:
      PYTHONPATH=. uv run python latex_ocr/trainers/decoder/transformer_pretrain.py smoke_cpu

    Full pretraining (downloads HF dataset):
      PYTHONPATH=. uv run python latex_ocr/trainers/decoder/transformer_pretrain.py train
      PYTHONPATH=. nohup uv run python latex_ocr/trainers/decoder/transformer_pretrain.py train 2>&1 &> pretrain.log &
    """

    cli()

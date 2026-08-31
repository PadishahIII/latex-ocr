"""Training entrypoints for latex-ocr.

Same recipes as the original workspace, now importable as a standalone package:

    python -m latex_ocr.trainers.train train --recipe-name coca_pretrain
    python -m latex_ocr.trainers.train train --recipe-name coca_finetune
    python -m latex_ocr.trainers.train train --recipe-name xception_gru
    python -m latex_ocr.trainers.train train-tune --no-param-search

Multi-GPU (DDP):
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 \
        torchrun --standalone --nproc_per_node=2 \
        -m latex_ocr.trainers.train train --recipe-name coca_pretrain
"""

from latex_ocr.trainers.train import cli

if __name__ == "__main__":
    cli()

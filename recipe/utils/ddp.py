import os

import torch
import torch.distributed as dist


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_ddp() -> tuple[int, int, int]:
    """
    Sets up Distributed Data Parallel (DDP) environment for a single rank.
    Return: local_rank, global_rank, world_size
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("DDP mode requires CUDA, but no GPU found.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not is_initialized():
        dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    if is_initialized():
        dist.destroy_process_group()


def is_rank_zero() -> bool:
    """
    Checks if the current process is the rank 0 process.
    """
    if not is_initialized():
        return True
    return dist.get_rank() == 0

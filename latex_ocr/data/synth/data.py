from __future__ import annotations
import concurrent
import os
from pathlib import Path
import threading

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from latex_ocr.models.tokenizer.latex_tokenizer import Tokenizer

# Disable tokenizers parallelism warning before any imports that use tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from functools import partial
from pydantic import BaseModel, Field
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from latex_ocr.data.synth.loader import (
    SyntheticLaTeXDataset,
    get_synthetic_transforms,
)
import torch
import numpy as np
from typing import Optional, Tuple, Iterator, List, Union, cast


class DataPoint(BaseModel):
    image: Optional[torch.Tensor] = Field()
    labels: torch.Tensor = Field()

    class Config:
        arbitrary_types_allowed = True


def generate_mask(tgt: torch.Tensor) -> torch.Tensor:
    """Generate target mask for Transformer decoder.

    Creates a causal (autoregressive) mask where each position can only attend
    to previous positions. Uses float format expected by PyTorch's TransformerDecoder:
    - 0.0 = position is allowed to attend
    - -inf = position is masked (not allowed to attend)
    """
    # tgt shape: (N, tgt_len)
    # Returns: (seq_len, seq_len) 2D float tensor for the batch

    seq_len = tgt.size(1)
    # Create a causal mask using PyTorch's standard format
    # Upper triangular matrix with -inf above diagonal
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        seq_len, device=tgt.device
    )

    return tgt_mask


class PretrainedTokenizer:
    def __init__(self, add_tokens: list, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            # "OleehyO/TexTeller",
            "gpt2",
            bos_token="<s>",
            add_bos_token=True,
            eos_token="</s>",
            add_eos_token=True,
        )
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
            }
        )
        if add_tokens:
            self.tokenizer.add_tokens(add_tokens)
        self.max_length = max_length

    @property
    def bos_token_id(self) -> int:
        """Get the begin-of-sequence token ID from the underlying tokenizer."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID from the underlying tokenizer."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Get the pad token ID from the underlying tokenizer."""
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size including all added tokens."""
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        t = self.tokenizer.encode(
            text,
            truncation=True,  # NOTE: dummy
            padding="longest",
            max_length=self.max_length - 2,  # account for bos/eos
        )
        # t = [self.tokenizer.bos_token_id] + t + [self.tokenizer.eos_token_id]

        if len(t) > self.max_length:
            raise Exception(
                f"Tokenized length {len(t)} exceeds max length {self.max_length}"
            )
        return t

    def decode(self, toks: list[int]) -> str:
        s = self.tokenizer.decode(toks)
        return s


special_tokens = {
    "mathbb": "\\mathbb",
    "mathcal": "\\mathcal",
    "mathbf": "\\mathbf",
    "mathit": "\\mathit",
    "mathrm": "\\mathrm",
    "mathsf": "\\mathsf",
    "mathtt": "\\mathtt",
    "mathfrak": "\\mathfrak",
    "mathscr": "\\mathscr",
}


class LengthSnapshot:
    def __init__(
        self,
        dataset: Union[
            TokenizedDataset, "ConcatTokenizedDataset", "MixedTokenizedDataset"
        ],
        lengths_snapshot: Path | None = None,
    ):
        # Determine snapshot path
        if lengths_snapshot is None:
            max_seq_len = getattr(dataset, "max_seq_len", None)
            max_suffix = f"_max{int(max_seq_len)}" if max_seq_len is not None else ""
            snapshot_path = Path(__file__).parent / (
                f"lengths_{dataset.config_name}_{dataset.split}{max_suffix}.npy"
            )
        else:
            snapshot_path = lengths_snapshot

        def _compute_lengths() -> list[int]:
            lengths_local: list[int] = []
            bar = tqdm(
                total=len(dataset),
                desc=f"Computing sequence lengths for {len(dataset)} samples...",
            )
            pool = ThreadPoolExecutor(max_workers=16)
            mu = threading.Lock()
            tasks = []

            def process_index(i: int) -> None:
                item = dataset.get_formula(i)
                mu.acquire()
                lengths_local.append(len(item.labels))
                mu.release()

            for i in range(len(dataset)):
                tasks.append(pool.submit(process_index, i))

            for task in as_completed(tasks):
                task.result()
                bar.update(1)
            pool.shutdown()

            return lengths_local

        # Try to load precomputed lengths; recompute if stale/mismatched.
        if snapshot_path.exists():
            print(f"Loading precomputed lengths from {snapshot_path}")
            lengths = np.load(snapshot_path).tolist()

            if len(lengths) != len(dataset):
                print(
                    f"Lengths snapshot size mismatch (snapshot={len(lengths)}, dataset={len(dataset)}). "
                    "Recomputing to avoid out-of-range sampler indices."
                )
                lengths = _compute_lengths()
                np.save(snapshot_path, np.array(lengths))
                print(f"Saved lengths snapshot to {snapshot_path}")
            else:
                print(f"Loaded {len(lengths)} precomputed lengths")
        else:
            print(
                f"Computing sequence lengths (snapshot will be saved to {snapshot_path})"
            )
            lengths = _compute_lengths()
            np.save(snapshot_path, np.array(lengths))
            print(f"Saved lengths snapshot to {snapshot_path}")
        self._lengths = lengths

    @property
    def lengths(self) -> list[int]:
        return self._lengths


class BucketBatchSampler(Sampler):
    """
    Batch sampler that groups sequences of similar lengths together.

    This reduces padding waste and memory usage by ensuring sequences in the
    same batch have similar lengths. Sequences are sorted by length and grouped
    into buckets, then batches are shuffled for training.

    Args:
        dataset: TokenizedDataset instance
        batch_size: Number of samples per batch
        drop_last: Whether to drop the last incomplete batch
        shuffle_batches: Whether to shuffle batch order (recommended for training)
        lengths_snapshot: Optional path to save/load precomputed lengths, if None, use Path(__file__).parent / "lengths_{config_name}_{split}.npy", if the file is empty, it would be created and lengths would be computed.
    """

    def __init__(
        self,
        batch_size: int,
        lengths: list[int],
        drop_last: bool = False,
        sampler: DistributedSampler | None = None,
        shuffle_batches: bool = True,
    ):
        self.batch_size = batch_size
        self.lengths = lengths
        self.sampler = sampler
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Created buckets with batch_size={batch_size}")

    def __iter__(self) -> Iterator[List[int]]:
        # Rebuild ordering for each iteration so epoch-level shuffling still works.
        indices = list(self.sampler) if self.sampler else list(range(len(self.lengths)))
        sorted_indices = sorted(indices, key=lambda i: self.lengths[i], reverse=True)
        batch_starts = list(range(0, len(sorted_indices), self.batch_size))
        if self.shuffle_batches:
            batch_starts = np.random.permutation(batch_starts).tolist()
        # Generate batches on-the-fly
        for start_idx in batch_starts:
            end_idx = start_idx + self.batch_size
            batch_indices = sorted_indices[start_idx:end_idx]

            # Skip incomplete batches if drop_last is True
            if len(batch_indices) == self.batch_size or not self.drop_last:
                yield batch_indices

    def __len__(self) -> int:
        num_samples = len(self.sampler) if self.sampler is not None else len(self.lengths)
        if self.drop_last:
            return num_samples // self.batch_size
        return (num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle batches for a new epoch (if shuffle_batches is True)."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class TokenizedDataset(Dataset):
    """Dataset that tokenizes LaTeX formulas and filters out sequences that exceed max_seq_len.

    Unlike truncating long sequences (which creates incomplete/invalid training data),
    this dataset discards samples that are too long during initialization.

    Args:
        config_name: Dataset configuration ("plain" or "styled")
        split: Dataset split ("train" or "validation")
        max_length: Maximum token length for the underlying tokenizer
        formula_only: If True, don't load images (faster for length computation)
        max_seq_len: Maximum sequence length. Samples exceeding this are DISCARDED.
    """

    def __init__(
        self,
        config_name: str = "plain",
        split: str = "train",
        max_length: int = 1000000,
        formula_only: bool = False,
        max_seq_len: int = 1000,  # Filter extreme outliers (99.5th percentile)
    ):
        self.ds = SyntheticLaTeXDataset(
            config_name=config_name,
            split=split,
            transform=get_synthetic_transforms(is_train=(split == "train")),
            formula_only=formula_only,
        )
        self.config_name = config_name
        self.split = split
        self.tokenizer = Tokenizer()
        self.tokenizer.load(None)
        self.max_seq_len = max_seq_len

        # Pre-filter: build list of valid indices (samples within max_seq_len)
        # This is computed once at initialization to avoid repeated filtering
        self._valid_indices: Optional[List[int]] = None
        self._filtered_count = 0

        # Determine cache path for valid indices
        self._cache_path = (
            Path(__file__).parent
            / f"valid_indices_{config_name}_{split}_max{max_seq_len}.npy"
        )

        # Try to load cached valid indices
        if self._cache_path.exists():
            print(f"Loading cached valid indices from {self._cache_path}")
            loaded = cast(List[int], np.load(self._cache_path).tolist())
            self._valid_indices = loaded
            self._filtered_count = len(self.ds) - len(loaded)
            print(
                f"Loaded {len(loaded)} valid samples "
                f"(filtered {self._filtered_count} samples exceeding max_seq_len={max_seq_len})"
            )
        else:
            # Compute valid indices
            self._compute_valid_indices()

    def _compute_valid_indices(self) -> None:
        """Compute and cache the list of valid indices (samples within max_seq_len)."""
        print(
            f"Computing valid indices for {self.config_name}/{self.split} "
            f"(max_seq_len={self.max_seq_len})..."
        )

        valid_indices = []
        filtered_count = 0

        def process_sample(idx):
            try:
                item = self.ds[idx]
                toks = self.tokenizer.encode(item["text"], add_special_tokens=True)

                if len(toks) <= self.max_seq_len:
                    return idx, True
                else:
                    return idx, False
            except Exception as e:
                return idx, e

        # Run in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            # Submit all tasks
            futures = [pool.submit(process_sample, idx) for idx in range(len(self.ds))]

            # Process results as they complete
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Filtering by sequence length",
            ):
                idx, result = future.result()

                if result is True:
                    valid_indices.append(idx)
                elif result is False:
                    filtered_count += 1
                else:
                    # Exception case
                    filtered_count += 1
                    print(f"Warning: Skipping sample {idx} due to error: {result}")

        # Sort indices to maintain deterministic order (since threads finish out of order)
        valid_indices.sort()

        self._valid_indices = valid_indices
        self._filtered_count = filtered_count

        # Cache for future use
        np.save(self._cache_path, np.array(valid_indices))
        print(
            f"Kept {len(valid_indices)} samples, filtered {filtered_count} "
            f"(saved to {self._cache_path})"
        )

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.ds)

    def _get_underlying_idx(self, idx: int) -> int:
        """Map external index to underlying dataset index."""
        if self._valid_indices is not None:
            if idx < 0 or idx >= len(self._valid_indices):
                raise IndexError(
                    f"Index {idx} out of range [0, {len(self._valid_indices)})"
                )
            return self._valid_indices[idx]
        return idx

    def get_formula(self, idx: int) -> DataPoint:
        """Get the formula text for a given index (without image)."""
        underlying_idx = self._get_underlying_idx(idx)
        item = self.ds[underlying_idx]
        toks = self.tokenizer.encode(item["text"], add_special_tokens=True)

        # Note: No truncation needed since we pre-filtered
        # But add a safety check just in case
        if len(toks) > self.max_seq_len:
            raise ValueError(
                f"Sample {idx} (underlying {underlying_idx}) has {len(toks)} tokens, "
                f"exceeding max_seq_len={self.max_seq_len}. This should not happen "
                f"if valid_indices was computed correctly."
            )

        toks_tensor = torch.tensor(toks, dtype=torch.long)
        return DataPoint(
            image=None,
            labels=toks_tensor,
        )

    def __getitem__(self, idx: int) -> DataPoint:
        underlying_idx = self._get_underlying_idx(idx)
        item = self.ds[underlying_idx]
        toks = self.tokenizer.encode(item["text"], add_special_tokens=True)

        # Note: No truncation needed since we pre-filtered
        # But add a safety check just in case
        if len(toks) > self.max_seq_len:
            raise ValueError(
                f"Sample {idx} (underlying {underlying_idx}) has {len(toks)} tokens, "
                f"exceeding max_seq_len={self.max_seq_len}. This should not happen "
                f"if valid_indices was computed correctly."
            )

        toks_tensor = torch.tensor(toks, dtype=torch.long)
        if self.ds.formula_only:
            image = None
        else:
            image = item["image"]
        return DataPoint(
            image=image,
            labels=toks_tensor,
        )

    @property
    def filtered_count(self) -> int:
        """Return the number of samples that were filtered out due to exceeding max_seq_len."""
        return self._filtered_count

    def clear_cache(self) -> None:
        """Clear the cached valid indices file."""
        if self._cache_path.exists():
            self._cache_path.unlink()
            print(f"Cleared cache: {self._cache_path}")


class ConcatTokenizedDataset(Dataset):
    """
    Dataset that concatenates multiple TokenizedDataset instances.

    This is useful for combining datasets with different configurations
    (e.g., plain and styled) into a single dataset for training.

    Args:
        datasets: List of TokenizedDataset instances to concatenate
    """

    def __init__(self, datasets: List[TokenizedDataset]):
        self.datasets = datasets
        self.cumulative_sizes = self._calculate_cumulative_sizes()

        # Use the tokenizer from the first dataset (all should have the same tokenizer)
        self.tokenizer = datasets[0].tokenizer
        self.config_name = "_".join([ds.config_name for ds in datasets])
        self.split = datasets[0].split  # Assume all datasets have the same split

    def _calculate_cumulative_sizes(self) -> List[int]:
        """Calculate cumulative sizes for efficient indexing."""
        cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cumulative_sizes.append(total)
        return cumulative_sizes

    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> DataPoint:
        """
        Get item from the appropriate dataset based on index.

        Args:
            idx: Global index across all concatenated datasets

        Returns:
            DataPoint from the appropriate dataset
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f"Absolute value of index {idx} should not exceed dataset length {len(self)}"
                )
            idx = len(self) + idx

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break

        # Calculate the local index within the found dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][local_idx]

    def get_formula(self, idx: int) -> DataPoint:
        """
        Get formula (without image) from the appropriate dataset.

        Args:
            idx: Global index across all concatenated datasets

        Returns:
            DataPoint with formula only
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f"Absolute value of index {idx} should not exceed dataset length {len(self)}"
                )
            idx = len(self) + idx

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break

        # Calculate the local index within the found dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_formula(local_idx)

    @property
    def filtered_count(self) -> int:
        """Return total number of samples filtered out across all underlying datasets."""
        return sum(
            ds.filtered_count for ds in self.datasets if hasattr(ds, "filtered_count")
        )


class MixedTokenizedDataset(Dataset):
    """Dataset-level mixing that keeps *all* styled samples.

    Semantics for this repo:
        - Always include the full `styled` partition.
        - Add enough `plain` samples so that the final plain proportion equals
          `plain_proportion`.

    If `styled` has N samples and `plain_proportion = p`, then we pick the total
    dataset length T such that:
        N / T = (1 - p)
        => T = N / (1 - p)
    and the number of plain samples is:
        T - N.

    Plain samples are taken with replacement if needed (via modular indexing).
    """

    def __init__(
        self,
        plain: TokenizedDataset,
        styled: TokenizedDataset,
        plain_proportion: float,
    ):
        if plain_proportion < 0 or plain_proportion > 1:
            raise ValueError(
                f"plain_proportion must be in [0, 1]; got {plain_proportion}"
            )

        self.plain = plain
        self.styled = styled
        self.plain_proportion = float(plain_proportion)

        # Expose common attributes used elsewhere.
        self.tokenizer = styled.tokenizer
        self.split = styled.split
        self.config_name = f"mixed_{plain_proportion:.2f}"

        n_styled = len(styled)
        if self.plain_proportion >= 1:
            self.num_styled = 0
            self.num_plain = len(plain)
        else:
            self.num_styled = n_styled
            total_len = int(np.ceil(n_styled / (1.0 - self.plain_proportion)))
            self.num_plain = max(0, total_len - n_styled)

        self.total_length = self.num_styled + self.num_plain

    def __len__(self) -> int:
        return self.total_length

    def _map_index(self, idx: int) -> tuple[str, int]:
        if idx < self.num_styled:
            return "styled", idx

        plain_rank = idx - self.num_styled
        return "plain", (plain_rank % len(self.plain))

    def __getitem__(self, idx: int) -> DataPoint:
        source, underlying_idx = self._map_index(idx)
        if source == "styled":
            return self.styled[underlying_idx]
        return self.plain[underlying_idx]

    def get_formula(self, idx: int) -> DataPoint:
        source, underlying_idx = self._map_index(idx)
        if source == "styled":
            return self.styled.get_formula(underlying_idx)
        return self.plain.get_formula(underlying_idx)

    @property
    def filtered_count(self) -> int:
        """Return total number of samples filtered out from both plain and styled datasets."""
        count = 0
        if hasattr(self.plain, "filtered_count"):
            count += self.plain.filtered_count
        if hasattr(self.styled, "filtered_count"):
            count += self.styled.filtered_count
        return count


def collate_fn(
    batch: list[DataPoint],
    padding_value: int = 0,
    return_attention_mask: bool = True,
) -> dict:
    """
    Collate function that pads labels to the same length within a batch.

    Args:
        batch: List of DataPoint objects
        padding_value: Value to use for padding
        return_attention_mask: If True, include a (T, T) causal mask.
            GRU decoders do not use this mask, so disabling it saves CPU time.

    Returns:
        Dictionary with:
            - images: Tensor of shape (B, C, H, W) or None
            - labels: Tensor of shape (B, max_seq_len) padded with padding_value
            - attention_mask: Tensor of shape (max_seq_len, max_seq_len) or None
            - label_lengths: Tensor of shape (B,) containing original lengths
    """
    # Get label lengths
    label_lengths = torch.tensor([len(item.labels) for item in batch], dtype=torch.long)

    # Pad labels to max_seq_len using torch.nn.utils.rnn.pad_sequence
    labels_list = [item.labels for item in batch]
    labels_tensor = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=padding_value
    )  # (B, max_seq_len)

    # Generate attention masks for the batch (Transformer only)
    attention_mask = (
        generate_mask(labels_tensor) if return_attention_mask else None
    )  # (max_seq_len, max_seq_len) or None

    # Stack images if available
    images = None
    if batch[0].image is not None:
        # Filter out None images and stack
        image_list = [item.image for item in batch if item.image is not None]
        if image_list:
            images = torch.stack(image_list)  # (B, C, H, W)

    return {
        "images": images,
        "labels": labels_tensor,
        "attention_mask": attention_mask,
        "label_lengths": label_lengths,
    }


class CombinedDataLoader:
    """Combine multiple dataloaders into a single iterable.

    This is primarily used to support `both=True` in `create_dataloader` while
    still allowing each underlying dataset/sampler to keep its own length snapshot
    (e.g. `lengths_plain_train.npy` and `lengths_styled_train.npy`).

    Notes:
        - `shuffle=True` shuffles at the *batch source* level (i.e. which loader a
          batch comes from). Each underlying loader is responsible for shuffling
          within itself.
        - Implements the minimal DataLoader protocol used in this repo
          (`__iter__`, `__len__`).
    """

    def __init__(self, dataloaders: List[DataLoader], shuffle: bool):
        if len(dataloaders) < 1:
            raise ValueError("CombinedDataLoader requires at least one DataLoader")
        self.dataloaders = dataloaders
        self.shuffle = shuffle

    def __len__(self) -> int:
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        if not self.shuffle:
            for dl in self.dataloaders:
                yield from dl
            return

        # Build a per-epoch schedule that chooses which loader to draw the next
        # batch from, then yield from each loader's iterator.
        schedule: list[int] = []
        for i, dl in enumerate(self.dataloaders):
            schedule.extend([i] * len(dl))

        iters = [iter(dl) for dl in self.dataloaders]
        for loader_idx in np.random.permutation(schedule).tolist():
            yield next(iters[loader_idx])


def create_dataloader(
    config_name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    both: bool = False,  # whether to load both plain and styled configs
    plain_proportion: float = -1,  # mix plain and styled, proportion of plain samples when both=False, ignored when both=True
    num_workers: int = 4,
    formula_only: bool = False,
    max_seq_len: int = 1000,  # Filter extreme outliers; should usually match model max_seq_length
    use_bucket_sampler: bool = True,  # Enable bucketing for memory efficiency
    use_distributed_sampler: bool = False,  # Enable DistributedSampler for multi-GPU training
    return_attention_mask: bool = True,  # GRU decoders can set False
    pin_memory: bool = False,
) -> Tuple[
    Union[TokenizedDataset, ConcatTokenizedDataset, MixedTokenizedDataset],
    Union[DataLoader, CombinedDataLoader],
]:
    def _create_single_dataloader(
        local_config_name: str,
    ) -> Tuple[TokenizedDataset, DataLoader]:
        dataset = TokenizedDataset(
            config_name=local_config_name,
            split=split,
            formula_only=formula_only,
            max_seq_len=max_seq_len,
        )
        length_snapshot = LengthSnapshot(dataset)
        dist_sampler: DistributedSampler | None = None
        if use_distributed_sampler:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "DistributedSampler requires an initialized distributed environment"
                )
            dist_sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                drop_last=False,
            )

        # Use bucket sampler to group similar-length sequences together
        # This significantly reduces padding waste and memory usage
        if use_bucket_sampler:
            batch_sampler = BucketBatchSampler(
                batch_size=batch_size,
                sampler=dist_sampler,
                lengths=length_snapshot.lengths,
                drop_last=False,
                shuffle_batches=shuffle,
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=partial(
                    collate_fn,
                    padding_value=dataset.tokenizer.pad_token_id,
                    return_attention_mask=return_attention_mask,
                ),
            )
        else:
            # Fallback to standard DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle if dist_sampler is None else False,
                sampler=dist_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=partial(
                    collate_fn,
                    padding_value=dataset.tokenizer.pad_token_id,
                    return_attention_mask=return_attention_mask,
                ),
            )

        return dataset, dataloader

    if not both:
        # If plain_proportion is explicitly set, ignore config_name and mix datasets.
        if plain_proportion >= 0:
            p_plain = float(plain_proportion)
            if p_plain > 1:
                raise ValueError(
                    f"plain_proportion must be in [0, 1] when set; got {plain_proportion}"
                )

            ds_plain = TokenizedDataset(
                config_name="plain",
                split=split,
                formula_only=formula_only,
                max_seq_len=max_seq_len,
            )
            ds_styled = TokenizedDataset(
                config_name="styled",
                split=split,
                formula_only=formula_only,
                max_seq_len=max_seq_len,
            )

            dataset = MixedTokenizedDataset(
                plain=ds_plain,
                styled=ds_styled,
                plain_proportion=p_plain,
            )
            length_snapshot = LengthSnapshot(dataset)
            dist_sampler: DistributedSampler | None = None
            if use_distributed_sampler:
                if not dist.is_available() or not dist.is_initialized():
                    raise RuntimeError(
                        "DistributedSampler requires an initialized distributed environment"
                    )
                dist_sampler = DistributedSampler(
                    dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=shuffle,
                    drop_last=False,
                )

            if use_bucket_sampler:
                batch_sampler = BucketBatchSampler(
                    batch_size=batch_size,
                    sampler=dist_sampler,
                    lengths=length_snapshot.lengths,
                    drop_last=False,
                    shuffle_batches=shuffle,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=partial(
                        collate_fn,
                        padding_value=dataset.tokenizer.pad_token_id,
                        return_attention_mask=return_attention_mask,
                    ),
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle if dist_sampler is None else False,
                    sampler=dist_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=partial(
                        collate_fn,
                        padding_value=dataset.tokenizer.pad_token_id,
                        return_attention_mask=return_attention_mask,
                    ),
                )

            return dataset, dataloader

        return _create_single_dataloader(config_name)

    # Create two independent datasets/samplers/loaders so we can reuse their
    # per-config lengths snapshots.
    ds_plain, dl_plain = _create_single_dataloader("plain")
    ds_styled, dl_styled = _create_single_dataloader("styled")

    # Returned dataset is still a concat so callers can see total size/tokenizer.
    dataset = ConcatTokenizedDataset([ds_plain, ds_styled])
    dataloader = CombinedDataLoader([dl_plain, dl_styled], shuffle=shuffle)

    return dataset, dataloader


if __name__ == "__main__":
    """
    PYTHONPATH=. uv run python latex_ocr/data/synth/data.py

    Example output statistics with percentiles:
    
    plain-train:
    Min: 4
    Max: 4585
    Mean: 68.94
    Median: 34
    50th percentile (median): 34.0
    75th percentile: 82.0
    90th percentile: 159.0
    95th percentile: 221.0
    99th percentile: 485.0
    99.5th percentile: 687.0

    plain-validation:
    Min: 4
    Max: 2926
    Mean: 120.57
    Median: 67
    50th percentile (median): 67.0
    75th percentile: 138.0
    90th percentile: 265.0
    95th percentile: 378.0
    99th percentile: 812.0
    99.5th percentile: 1045.0

    styled-train:
    Min: 8
    Max: 1731
    Mean: 56.54
    Median: 41
    50th percentile (median): 41.0
    75th percentile: 68.0
    90th percentile: 111.0
    95th percentile: 152.0
    99th percentile: 331.0
    99.5th percentile: 453.0

    """
    import matplotlib.pyplot as plt

    # tokenizer = PretrainedTokenizer(list(special_tokens.values()))
    tokenizer = Tokenizer()
    tokenizer.load(None)
    toks = tokenizer.encode(
        "\\mathbf is \\mathbb . \\tilde \\mathring \\hat \\dot", add_special_tokens=True
    )
    s = ""
    for i in toks:
        s += f"'{tokenizer.decode([i], skip_special_tokens=False)}' "
    print(s)

    print(toks)
    print(tokenizer.decode(toks, skip_special_tokens=False))

    ds, loader = create_dataloader(
        config_name="plain",
        both=False,
        split="train",
        batch_size=2,
        shuffle=True,
    )
    dp = next(iter(loader))
    print(f"image: {dp['images'].shape}")  # [2, 3, 192, 672]
    print(f"labels: {dp['labels'].shape}")  # [2, 6]
    print(f"attention_mask: {dp['attention_mask'].shape}")  # [6, 6]
    print(f"attention_mask: {dp['attention_mask']}")  # [6, 6]
    print(f"bos: {ds.tokenizer.bos_token_id}")  # gpt2:50257 TexTeller: 0
    print(f"eos: {ds.tokenizer.eos_token_id}")  # gpt2:50258 TexTeller: 2
    print(f"pad: {ds.tokenizer.pad_token_id}")  # gpt2:50259 TexTeller: 15000
    print(f"vocab_size: {ds.tokenizer.vocab_size}")  # gpt2: 50269 TexTeller: 15010
    exit(0)
    config_name = "plain"
    split = "train"
    ds = TokenizedDataset(
        config_name=config_name,
        split=split,
        formula_only=False,
    )
    # ds = Subset(ds, range(100))

    # Build histogram on labels' length
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def get_label_length(idx):
        item = ds[idx]
        return len(item.labels)

    print(f"Collecting label lengths from {len(ds)} samples...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        label_lengths = list(
            tqdm(
                executor.map(get_label_length, range(len(ds))),
                total=len(ds),
                desc="Processing samples",
            )
        )

    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(label_lengths, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Label Length (number of tokens)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Label Lengths in Dataset")
    plt.grid(True, alpha=0.3)

    # Print statistics
    import numpy as np

    label_lengths_array = np.array(label_lengths)

    print("\nLabel Length Statistics:")
    print(f"Min: {min(label_lengths)}")
    print(f"Max: {max(label_lengths)}")
    print(f"Mean: {sum(label_lengths) / len(label_lengths):.2f}")
    print(f"Median: {sorted(label_lengths)[len(label_lengths) // 2]}")
    print(f"50th percentile (median): {np.percentile(label_lengths_array, 50):.1f}")
    print(f"75th percentile: {np.percentile(label_lengths_array, 75):.1f}")
    print(f"90th percentile: {np.percentile(label_lengths_array, 90):.1f}")
    print(f"95th percentile: {np.percentile(label_lengths_array, 95):.1f}")
    print(f"99th percentile: {np.percentile(label_lengths_array, 99):.1f}")
    print(f"99.5th percentile: {np.percentile(label_lengths_array, 99.5):.1f}")

    plt.tight_layout()
    plt.savefig(
        f"label_length_histogram_{config_name}_{split}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("\nHistogram saved to 'label_length_histogram.png'")
    plt.show()

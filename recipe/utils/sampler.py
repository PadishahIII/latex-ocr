import random
import math
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    """
    A sampler that groups samples into buckets based on their lengths and creates batches within those buckets.
    """
    def __init__(self, lengths, batch_size=64, boundaries=(32,64,128,256,512), shuffle=True, sort_batches:bool=False, sort_order:str='ascending'):
        """
        Initialize the BucketBatchSampler.

        Args:
            lengths (list[int]): List of lengths for each sample.
            batch_size (int): Size of each batch.
            boundaries (tuple): Boundaries for bucket creation.
            shuffle (bool): Whether to shuffle within buckets and batches.
            sort_batches (bool): Whether to sort batches by bucket id.
            sort_order (str): Order for sorting batches, 'ascending' or 'descending'.
        """
        self.lengths = lengths  # list[int], length per sample (tokenized)
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.sort_batches = sort_batches
        self.sort_order = sort_order
        # assign indices to buckets
        self.buckets = {i: [] for i in range(len(self.boundaries)+1)}
        for idx, L in enumerate(self.lengths):
            # bucket id: first boundary > L
            bid = 0
            while bid < len(self.boundaries) and L >= self.boundaries[bid]:
                bid += 1
            self.buckets[bid].append(idx)

    def __iter__(self):
        """
        Iterate over the batches.

        Yields:
            list: A batch of indices.
        """
        # shuffle within buckets
        for b in self.buckets.values():
            if self.shuffle:
                random.shuffle(b)
        # make batches per bucket
        batches = []
        batches_id = []
        for k, b in self.buckets.items():
            for i in range(0, len(b), self.batch_size):
                batch = b[i:i+self.batch_size]
                if batch:
                    batches.append(batch)
                    batches_id.append((k, batch))
        if self.shuffle:
            random.shuffle(batches)
        if self.sort_batches:
            reverse = self.sort_order == 'descending'
            batches_id = sorted(batches_id, key=lambda x: x[0], reverse=reverse)
            batches = [b for _, b in batches_id]
        for batch in batches:
            yield batch

    def __len__(self):
        """
        Return the number of batches.

        Returns:
            int: Number of batches.
        """
        return math.ceil(len(self.lengths) / self.batch_size)

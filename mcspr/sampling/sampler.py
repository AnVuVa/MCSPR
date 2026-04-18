import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, List


class StratifiedContextSampler(Sampler):

    def __init__(
        self,
        context_labels: np.ndarray,
        context_weights: np.ndarray,
        batch_size: int,
        n_contexts: int,
        k_min: float = 30.0,
        drop_last: bool = True,
        cross_slide: bool = True,
    ):
        self.context_labels = context_labels
        self.context_weights = context_weights
        self.batch_size = batch_size
        self.n_contexts = n_contexts
        self.k_min = k_min
        self.drop_last = drop_last
        self.cross_slide = cross_slide

        # Precompute per-context index pools
        self.context_indices = {
            t: np.where(context_labels == t)[0] for t in range(n_contexts)
        }
        self.per_context = batch_size // n_contexts
        self.remainder = batch_size - self.per_context * n_contexts

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle each context's index pool at epoch start
        pools = {}
        pointers = {}
        for t in range(self.n_contexts):
            pool = self.context_indices[t].copy()
            np.random.shuffle(pool)
            pools[t] = pool
            pointers[t] = 0

        active_contexts = [
            t for t in range(self.n_contexts) if len(pools[t]) > 0
        ]
        if not active_contexts:
            return

        while True:
            # Check if we can fill another batch without wrapping any pool
            if self.drop_last:
                can_fill = all(
                    pointers[t] + min(self.per_context, len(pools[t]))
                    <= len(pools[t])
                    for t in active_contexts
                )
                if not can_fill:
                    break

            batch: List[int] = []
            for t in active_contexts:
                n_to_sample = min(self.per_context, len(pools[t]))
                if n_to_sample == 0:
                    continue

                # CRITICAL: Never call np.random.choice(..., replace=True).
                # Undersampled contexts are dropped from MCSPR loss, not inflated.
                if pointers[t] + n_to_sample > len(pools[t]):
                    # Reshuffle that context's pool and reset pointer
                    np.random.shuffle(pools[t])
                    pointers[t] = 0

                batch.extend(
                    pools[t][pointers[t] : pointers[t] + n_to_sample].tolist()
                )
                pointers[t] += n_to_sample

            if len(batch) == 0:
                break

            # Shuffle the assembled batch before yielding
            np.random.shuffle(batch)
            yield batch

            # For non-drop_last: stop after one full pass through all spots
            if not self.drop_last:
                total_spots = sum(len(pools[t]) for t in active_contexts)
                all_exhausted = all(
                    pointers[t] >= len(pools[t]) for t in active_contexts
                )
                if all_exhausted:
                    break

    def __len__(self) -> int:
        active_pools = [
            len(self.context_indices[t])
            for t in range(self.n_contexts)
            if len(self.context_indices[t]) > 0
        ]
        if not active_pools or self.per_context == 0:
            return 0
        if self.drop_last:
            min_pool = min(active_pools)
            return min_pool // self.per_context
        else:
            total = sum(active_pools)
            return (total + self.batch_size - 1) // self.batch_size

    def get_context_eff_n(self, batch_context_weights: np.ndarray) -> np.ndarray:
        return batch_context_weights.sum(axis=0)  # (T,)

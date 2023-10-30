import torch
from torch.distributions import Categorical
from torch_scatter import scatter_max, scatter_log_softmax


def index_mapping(src_key, tgt_key):
    return torch.full((max(src_key, default=0) + 1,), -1, dtype=torch.long).scatter_(
        dim=0, index=src_key, src=tgt_key
    )


def index_ranking(indices):
    """
    input : [3, 3, 4, 7, 7, 0]
    output: [1, 1, 2, 3, 3, 0]
    """
    u = indices.unique(sorted=True)
    s = torch.arange(len(u))
    return index_mapping(u, s)[indices]


def index_mask(indices, size):
    # input : [2, 3], 5
    # output: [False, False, True, True, False]
    src = torch.ones(size, dtype=torch.bool)
    output = torch.zeros(size, dtype=torch.bool)
    return output.scatter_(dim=0, index=indices, src=src)


def batch_offset(indices):
    """
    compute offset that is useful for batch operation

    input : [0, 0, 0, 1, 2, 2]
    output: [0, 3, 4]
    """
    count = torch.bincount(indices)
    return torch.cumsum(count, 0) - count


def repeated_arange(sizes):
    return torch.cat([torch.arange(i) for i in sizes], 0)


def delete_elems(tensor, indices):
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def insert_value(tensor, values, indices):
    assert values.size(0) == indices.size(0)
    output = torch.empty(len(tensor) + len(values), dtype=tensor.dtype)
    output[indices] = values
    mask = torch.ones_like(output, dtype=torch.bool)
    mask[indices] = False
    output[mask] = tensor
    return output


def gumbel_max(logits, indices, eps=1e-16):
    unif = torch.rand_like(logits) + eps
    gumbel = -(-unif.log() + eps).log()
    _, samples = scatter_max(logits + gumbel, indices)
    return samples


class DefaultCategorical(Categorical):
    def __init__(self, logits):
        """
        Set default value when `logits` is empty.
        """
        self.logits = logits
        self._is_empty = len(logits) == 0
        self.size = len(logits)

        if not self._is_empty:
            super().__init__(logits=logits)

    def sample(self):
        if self._is_empty:
            return torch.empty_like(self.logits, dtype=torch.long).flatten()
        return super().sample()

    def log_prob(self, value):
        if self._is_empty:
            return torch.empty_like(self.logits, dtype=torch.float).flatten()
        return super().log_prob(value)


class VariadicCategorical:
    def __init__(self, logits, indices):
        assert logits.dtype == torch.float, "`logits` should be torch.float data type"
        assert indices.dtype == torch.long, "`indices` should be torch.long data type"
        assert (
            logits.shape == indices.shape
        ), f"The shape of `logits` and `indices` should be the same, got logits={logits.shape}, indices={indices.shape}"

        batch = index_ranking(indices)

        # we don't assume indices are non-decreasing.
        self.batch, _sort_indices = torch.sort(batch, stable=True)
        self.logits = logits[_sort_indices]
        self._offset = batch_offset(self.batch)
        self.size = max(self.batch.tolist(), default=-1) + 1

    def log_prob(self, value):
        assert self.size == len(
            value
        ), f"size doesn't match, size {self.size}, got {value.shape}"
        log_probs = scatter_log_softmax(self.logits, self.batch)
        return log_probs[value + self._offset]

    @torch.no_grad()
    def sample(self):
        sample_ids = gumbel_max(self.logits, self.batch)
        return sample_ids - self._offset

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class PlackettLuce:
    def __init__(self, logits, sizes):
        assert logits.dtype == torch.float, "`logits` should be torch.float data type"
        assert sizes.dtype == torch.long, "`sizes` should be torch.long data type"
        assert (
            logits.size(0) == sizes.sum().item()
        ), f"The size of `logits` should be the sum of `sizes`, got logits={logits.shape}, sizes={sizes.shape}"
        self.logits = logits
        self.sizes = sizes

    def log_prob(self, value):
        # iterative sampling / faster implementation?
        logit_mask = torch.ones(len(self.logits), dtype=torch.bool)
        done_mask = torch.ones(len(self.logits), dtype=torch.bool)
        total_log_prob = torch.zeros(len(self.sizes), dtype=torch.float)
        ptr = offset = torch.cumsum(self.sizes, 0) - self.sizes
        max_ptr = offset + self.sizes - 1
        done_mask = ptr < max_ptr

        for _ in range(self.sizes.max().item() - 1):
            masked_logits = torch.where(logit_mask, self.logits, -torch.inf)
            log_prob = scatter_log_softmax(
                masked_logits, torch.repeat_interleave(self.sizes)
            )

            r = offset[done_mask] + value[ptr[done_mask]]
            total_log_prob[done_mask] = total_log_prob[done_mask] + log_prob[r]
            logit_mask[r] = False
            ptr = ptr + 1
            done_mask = ptr < max_ptr
        return total_log_prob

    @torch.no_grad()
    def sample(self):
        logits = self.logits.data.clone()
        samples = []
        indices = torch.repeat_interleave(self.sizes)
        nelem = sum(self.sizes).item()  # default value when there's no choice but -inf

        for i in range(self.sizes.max().item()):
            sample = gumbel_max(logits, indices)
            logits[sample[sample != nelem]] = -torch.inf
            samples.append(sample)

        stack = torch.vstack(samples)
        mask = (stack != nelem).t()
        offset = torch.cumsum(self.sizes, 0) - self.sizes
        permu = (stack - offset).t()
        return permu[mask]

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes={self.sizes})"

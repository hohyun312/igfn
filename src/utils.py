import torch
from torch_scatter import scatter_max


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

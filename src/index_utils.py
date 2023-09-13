import torch


def rearange(indices):
    # input : [3, 3, 4, 7, 7, 0]
    # output: [1, 1, 2, 3, 3, 0]
    u = indices.unique(sorted=True)
    s = torch.arange(len(u))
    mapping = torch.full((max(u, default=0) + 1,), -1, dtype=torch.long).scatter_(
        dim=0, index=u, src=s
    )
    return mapping


def inv_rearange(mapping):
    n = max(mapping)
    inv_mapping = torch.empty(n + 2, dtype=torch.long)  # last index is for -1 indices
    index = torch.where(mapping == -1, n + 1, mapping)
    inv_mapping = inv_mapping.scatter_(
        dim=0, index=index, src=torch.arange(len(index))
    )[:-1]
    return inv_mapping


def compute_offset(indices):
    """
    compute offset that is useful for batch operation

    input : [0, 0, 0, 1, 2, 2]
    output: [0, 3, 4]
    """
    count = torch.bincount(indices)
    return torch.cumsum(count, 0) - count


def transpose(indices):
    """
    transpose indices that permutes array

    input : [1, 2, 3, 0]
    output: [3, 0, 1, 2]
    """
    o = torch.empty_like(indices)
    s = torch.arange(len(indices))
    # o[indices] = s : This is bit slower for some reason
    return o.scatter_(dim=0, index=indices, src=s)


def to_mask(indices, size):
    # input : [2, 3], 5
    # output: [False, False, True, True, False]
    src = torch.ones(size, dtype=torch.bool)
    output = torch.zeros(size, dtype=torch.bool)
    return output.scatter_(dim=0, index=indices, src=src)

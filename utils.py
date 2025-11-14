import torch
import torch.nn.functional as F


def grad_y(y):
    Mdx_y = F.pad(torch.diff(y, 1, dim=-1), (1, 0))
    Mdy_y = F.pad(torch.diff(y, 1, dim=-2), (0, 0, 1, 0))
    return Mdx_y, Mdy_y
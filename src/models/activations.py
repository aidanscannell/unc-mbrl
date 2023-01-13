#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch import Tensor


class Sin(nn.Module):
    def __init__(self) -> None:
        super(Sin, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)

    # def extra_repr(self) -> str:
    #     inplace_str = ", inplace=True" if self.inplace else ""
    #     return "alpha={}{}".format(self.alpha, inplace_str)

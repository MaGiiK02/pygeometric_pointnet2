# -*- coding: utf-8 -*-
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Conv2d

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(
            Lin(channels[i - 1], channels[i]),
            ReLU(),
            BN(channels[i])
        )
        for i in range(1, len(channels))
    ])
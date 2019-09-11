# -*- coding: utf-8 -*-
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from Layers.sampling import GlobalSortPool,SAModuleMSG
from Layers.MPLs import MLP


class PointNet2MSGSortPoolClass(torch.nn.Module):
    def __init__(self, class_count, n_feature=3, sort_pool_k=32):
        super(PointNet2MSGSortPoolClass, self).__init__()

        self.sa1_module = SAModuleMSG(512, [0.1,0.2,0.4], [16,32,128], [
            MLP([n_feature, 32,32,64]),
            MLP([n_feature, 64,64,128]),
            MLP([n_feature, 64,96,128])
        ])

        #Because we concat the outout of each layer as a feature of each point
        n_features_l2 = 3 + 64 + 128 + 128
        self.sa2_module = SAModuleMSG(128, [0.2,0.4,0.8], [32,64,128], [
            MLP([n_features_l2, 64, 64, 128]),
            MLP([n_features_l2, 128, 128, 256]),
            MLP([n_features_l2, 128, 128, 256])
        ])

        n_features_l3 = 3 + 128 + 256 + 256
        self.sa3_module = GlobalSortPool(MLP([n_features_l3, 256, 512, 1024]), sort_pool_k)

        #Classification Layers
        classification_point_feature = 1024*sort_pool_k
        self.lin1 = Lin(classification_point_feature, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = Lin(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = Lin(256, class_count)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
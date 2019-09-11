# -*- coding: utf-8 -*-
import os.path as osp

import torch
import torch.nn.functional as F
from Layers.sampling import GlobalSAModule,SAModuleMSG
from Layers.MPLs import MLP
from Layers.features_propagation import FPModule


class PointNet2MSGSeg(torch.nn.Module):
    def __init__(self, class_count, nfeatures=3):
        super(PointNet2MSGSeg, self, ).__init__()

        self.sa1_module = SAModuleMSG(512, [0.1,0.2,0.4], [16,32,128], [
            MLP([nfeatures, 32,32,64]),
            MLP([nfeatures, 64,64,128]),
            MLP([nfeatures, 64,96,128])
        ])

        #Because we concat the out of each layer as a feature of each point
        nFeaturesL2 = 3 + 64 + 128 + 128
        self.sa2_module = SAModuleMSG(128, [0.2,0.4,0.8], [32,64,128], [
            MLP([nFeaturesL2, 64, 64, 128]),
            MLP([nFeaturesL2, 128, 128, 256]),
            MLP([nFeaturesL2, 128, 128, 256])
        ])

        nFeaturesL3 = 3 + 128 + 256 + 256
        self.sa3_module = GlobalSAModule(MLP([nFeaturesL3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, class_count)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

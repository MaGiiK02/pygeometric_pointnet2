# -*- coding: utf-8 -*-
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from Layers.sampling import GlobalSAModule,SAModule
from Layers.MPLs import MLP

#TODO Implement weight decay on the classification part
class PointNet2Class(torch.nn.Module):
    r'''
    '''
    def __init__(self, class_count, bn_momentum=0.1):
        super(PointNet2Class, self).__init__()

        self.sa1_module = SAModule(512, 0.2, 32, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(128, 0.4, 64, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        
        #Classification Layers
        self.lin1 = Lin(1024, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=bn_momentum)
        self.lin2 = Lin(512, 256)
        self.bn2 = nn.BatchNorm1d(256, momentum=bn_momentum)
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
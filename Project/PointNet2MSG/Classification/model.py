# -*- coding: utf-8 -*-
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from Project.Layers.Sampling import GlobalSAModule,SAModuleMSG
from Project.Layers.MPLs import MLP


class PointNet2MSGClass(torch.nn.Module):
    def __init__(self,class_count):
        super(PointNet2MSGClass, self).__init__()
                                                                            #serve un mpl per raggio!!!!!!
        self.sa1_module = SAModuleMSG(512, [0.1,0.2,0.4], [16,32,128], [
            MLP([32,32,64]),
            MLP([64,64,128]),
            MLP([64,96,128])
        ])
        self.sa2_module = SAModuleMSG(128, [0.2,0.4,0.8], [32,64,128], [
            MLP([64, 64, 128]),
            MLP([128, 128, 256]),
            MLP([128, 128, 256])
        ])
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        
        #Classification Layers
        self.lin1 = Lin(1024, 512)  #aggiungi dropout
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, class_count)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
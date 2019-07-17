# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import PointConv, fps, radius
from torch_geometric.utils import scatter_

class SAModule(torch.nn.Module):
    def __init__(self, sample_points, r, nn):
        super(SAModule, self).__init__()
        self.sample_points = sample_points
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        print(x.dev)
        idx = fps(pos, batch, ratio=self.sample_points/len(pos))
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class SAModuleMSG(torch.nn.Module):
    r'''

    '''
    def __init__(self, sample_points, r_list, group_sample_size_list, nn_list):
        super(SAModuleMSG, self).__init__()
        assert(len(nn_list) == len(group_sample_size_list)
               and len(r_list) == len(nn_list)) #all have to have the same size

        self.sample_points = sample_points
        self.r_list = r_list
        self.group_sample_size = group_sample_size_list
        self.conv_list = torch.nn.ModuleList()

        for i in range(len(nn_list)):
            #create a pointConv for each radius
            self.conv_list.append(PointConv(nn_list[i]))

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, self.sample_points/len(pos))
        x_list = []
        for i in range(len(self.r_list)):
            row, col = radius(
                pos, pos[idx], self.r_list[i], batch, batch[idx], max_num_neighbors=self.group_sample_size[i])
            edge_index = torch.stack([col, row], dim=0)
            group_x = self.conv_list[i](x, (pos, pos[idx]), edge_index)
            x_list.append(group_x)

        new_x = torch.cat(x_list, 1)
        return new_x, pos[idx], batch[idx]

class GlobalSAModule(torch.nn.Module):
    r'''

    '''
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = scatter_('max', x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
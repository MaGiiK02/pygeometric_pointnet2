# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import PointConv, fps, radius, global_sort_pool, DeepGraphInfomax, knn_interpolate
from torch_geometric.utils import scatter_

class SAModule(torch.nn.Module):
    def __init__(self, sample_points, r, sample_size,nn):
        super(SAModule, self).__init__()
        self.sample_points = sample_points
        self.r = r
        self.sample_size = sample_size
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.sample_points/len(pos))
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.sample_size)
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


class SAModuleFullPoint(torch.nn.Module):
    def __init__(self, r, sample_size, nn):
        super(SAModuleFullPoint, self).__init__()
        self.r = r
        self.sample_size = sample_size
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        row, col = radius(
            pos, pos, self.r, batch, batch, max_num_neighbors=self.sample_size)
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(x, (pos, pos), edge_index)
        pos, batch = pos, batch
        return x, pos, batch


class SAModuleMRG(torch.nn.Module):
    def __init__(self, start_point, sampled_point_ammount, layers):
        super(SAModuleMRG, self).__init__()

        self.upsample_needed = sampled_point_ammount < start_point
        self.downsample = DownsampleMRG(sampled_point_ammount / start_point)
        self.layers = torch.nn.ModuleList()
        if(self.upsample_needed):
            self.upsample = UpsampleMRG(start_point, sampled_point_ammount)

        for i in range(len(layers)):
            self.layers.append(layers[i])


    def forward(self, x, pos, batch):

        out_lr = self.downsample(x, pos, batch)

        for i in range(len(self.layers)):
            out_lr = self.layers[i](*out_lr)

        x_lr, pos_lr, batch_lr = out_lr

        if(self.upsample_needed):
            x, pos, batch = self.upsample(x, pos, batch, x_lr, pos_lr, batch_lr)
        else:
            x = x_lr, pos = pos_lr, batch = batch_lr

        return x, pos, batch

class DownsampleMRG(torch.nn.Module):
    def __init__(self, scale_factor):
        assert (scale_factor <= 1)
        assert (scale_factor > 0)

        super(DownsampleMRG, self).__init__()

        self.scale_factor = scale_factor

    def forward(self, x, pos, batch):

        if( self.scale_factor < 1 ):
            downsampled_idx = fps(pos, batch, self.scale_factor)
            x = None if x is None else x[downsampled_idx]
            pos = pos[downsampled_idx]
            batch = batch[downsampled_idx]

        return x, pos, batch

class UpsampleMRG(torch.nn.Module):
    def __init__(self, high_res_points, low_res_points):
        assert (high_res_points > low_res_points)

        super(UpsampleMRG, self).__init__()

        self.high_res_points = high_res_points
        self.low_res_points = low_res_points

    def forward(self, x_hr, pos_hr, batch_hr, x_lr, pos_lr, batch_lr):
        out_x = x_hr
        out_pos = pos_hr
        out_batch = batch_hr
        if(out_x is not None):
            out_x = torch.cat([out_x, out_pos], dim=1)
        else:
            out_x = out_pos

        lr_x = x_lr
        lr_pos = pos_lr
        lr_batch = batch_lr
        lr_x = torch.cat([lr_x, lr_pos], dim=1)

        #return a tensor where the feature of each point of lr_x are appended in a new array in
        # the point of the upsampled version is in his knn
        lr_x = knn_interpolate(lr_x, lr_pos, out_pos, lr_batch, out_batch, k=1)
        out_x = torch.cat([out_x, lr_x], dim=1)

        return out_x, out_pos.new_zeros((out_x.size(0), 3)), out_batch



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

class GlobalSAModuleX(torch.nn.Module):
    r'''

    '''
    def __init__(self, nn):
        super(GlobalSAModuleX, self).__init__()
        self.nn = nn

    def forward(self, x,  batch):
        x = self.nn(x)
        x = scatter_('max', x, batch)
        batch = torch.arange(x.size(0), device=batch.device)
        return x, batch


class GlobalSortPool(torch.nn.Module):
    r'''

    '''
    def __init__(self, nn, k):
        super(GlobalSortPool, self).__init__()
        self.nn = nn
        self.k = k

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_sort_pool(x, batch, self.k)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
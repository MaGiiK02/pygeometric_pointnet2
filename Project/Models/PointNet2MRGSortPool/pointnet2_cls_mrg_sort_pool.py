# -*- coding: utf-8 -*-
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from Layers.sampling import  SAModuleMRG, SAModuleFullPoint, GlobalSortPool
from Layers.MPLs import MLP


class PointNet2MRGSortPoolClass(torch.nn.Module):
	def __init__(self, class_count, n_features=3, num_points=1024, sort_pool_k=32):
		super(PointNet2MRGSortPoolClass, self).__init__()

		nFeaturesL2 = 3 + 128

		shared_mpls = [
			SAModuleFullPoint(0.4, 16, MLP([n_features, 64, 64, 128])),
			SAModuleFullPoint(0.9, 32, MLP([nFeaturesL2, 128, 128, 256]))
		]

		# The mpls are shared to lower the model memory footprint
		self.high_resolution_module = SAModuleMRG(num_points, 512, shared_mpls)
		self.mid_resolution_module = SAModuleMRG(num_points, 256, shared_mpls)
		self.low_resolution_module = SAModuleMRG(num_points, 128, shared_mpls)

		self.readout = GlobalSortPool(MLP([789, 1024, 1024, 1024]), k=sort_pool_k)

		# Classification Layers
		sort_pool_out = 1024*sort_pool_k
		self.lin1 = Lin(sort_pool_out, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.lin2 = Lin(512, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.lin3 = Lin(256, class_count)

	def forward(self, data):
		sa_out = (data.x, data.pos, data.batch)

		hr_x, hr_pos, hr_batch = self.high_resolution_module(*sa_out)
		mr_x, mr_pos, mr_batch = self.mid_resolution_module(*sa_out)
		x = torch.cat([hr_x, mr_x], dim=1)

		lr_x, lr_pos, lr_batch = self.low_resolution_module(*sa_out)
		x = torch.cat([x, lr_x], dim=1)

		batch = data.batch

		x, pos, batch = self.readout(x, data.pos, batch)

		x = F.relu(self.bn1(self.lin1(x)))
		x = F.dropout(x, p=0.4, training=self.training)
		x = F.relu(self.bn2(self.lin2(x)))
		x = F.dropout(x, p=0.4, training=self.training)
		x = self.lin3(x)
		return F.log_softmax(x, dim=-1)
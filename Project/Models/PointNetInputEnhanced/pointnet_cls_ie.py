# -*- coding: utf-8 -*-
import os.path as osp

import torch
from torch.nn import Sequential, ReLU, Conv1d, BatchNorm1d
import torch.nn.functional as F
from torch.nn import Linear as Lin
from Layers.sampling import GlobalSAModule
from Layers.data_enhacment import AddNeightboursCount

class PointNetInputEnhanced(torch.nn.Module):
	r'''
	'''

	def __init__(self, class_count, nfeatures, batch_size, bn_momentum=0.1, nPoints=1024):
		super(PointNetInputEnhanced, self).__init__()

		self.nPoints = nPoints
		self.nFeatures = nfeatures + 4 # 4 is number of radii used to count the points
		self.batch_size = batch_size

		self.de_layer = AddNeightboursCount(
			max_points=[1024, 1024, 1024, 1024],
			radii=[0.1, 0.2, 0.4, 0.8]
		)

		self.net = Sequential(
			Conv1d(self.nFeatures, 64, 1, stride=1),
			ReLU(),
			Conv1d(64, 64, 1, stride=1),
			ReLU(),
			Conv1d(64, 128, 1, stride=1),
			ReLU(),
			Conv1d(128, 1024, 1, stride=1),
			ReLU()
		)

		self.sa3_module = GlobalSAModule()  # Maxpool

		# Classification Layers
		self.lin1 = Lin(1024, 512)
		self.bn1 = BatchNorm1d(512, momentum=bn_momentum)
		self.lin2 = Lin(512, 256)
		self.bn2 = BatchNorm1d(256, momentum=bn_momentum)
		self.lin3 = Lin(256, class_count)

	def forward(self, data):
		x_size = int(data.pos.size(0)/self.batch_size)
		x, pos, batch = self.de_layer(data.x, data.pos, data.batch)
		x = x.view(self.batch_size, x_size, self.nFeatures).transpose(1, 2)

		x = self.net(x)
		x = x.transpose(1, 2).contiguous().view(-1, 1024)
		x, pos, batch = self.sa3_module(x, data.pos, data.batch)

		x = F.relu(self.bn1(self.lin1(x)))
		x = F.dropout(x, p=0.4, training=self.training)
		x = F.relu(self.bn2(self.lin2(x)))
		x = F.dropout(x, p=0.4, training=self.training)
		x = self.lin3(x)
		return F.log_softmax(x, dim=-1)
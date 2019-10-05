# -*- coding: utf-8 -*-
import os.path as osp

import torch
from torch.nn import Sequential, ReLU, Conv1d, BatchNorm1d
import torch.nn.functional as F
from torch.nn import Linear as Lin
from Layers.sampling import GlobalSAModule
from Layers.data_enhacment import AddNeightboursCount

class PointNetInputEnhancedClass(torch.nn.Module):
	r'''
		A pointNet where we enhance the data with the infromation we belive the Pointnet++
		obtains form the ball operation
	'''

	def __init__(self, class_count, nfeatures, bn_momentum=0.1, nPoints=1024):
		super(PointNetInputEnhancedClass, self).__init__()

		self.nPoints = nPoints
		self.nFeatures = nfeatures

        self.data_enhance_module = AddNeightboursCount(max_points=[0.1,0.2,0.4,0.8], radii=[16,32,128,128])

		self.net = Sequential(
			Conv1d(nfeatures, 64, 1, stride=1),
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
        x = self.input_enhance_layer(data.pos, data.batch)
		x_1_out = self.net(x).view(-1, 1024)
		sa2_out = self.sa3_module(x_1_out, data.pos, data.batch)
		x, pos, batch = sa2_out

		x = F.relu(self.bn1(self.lin1(x)))
		x = F.dropout(x, p=0.4, training=self.training)
		x = F.relu(self.bn2(self.lin2(x)))
		x = F.dropout(x, p=0.4, training=self.training)
		x = self.lin3(x)
		return F.log_softmax(x, dim=-1)
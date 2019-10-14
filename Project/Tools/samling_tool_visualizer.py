import argparse
import os
import os.path as osp
import random
from off_parser import OffParser
from collections import namedtuple

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', default=None)
parser.add_argument('--sampling_method', default='ImportanceSampling', help='The type of samling to use.')
parser.add_argument('--out_dir', default="./processed_data", help='The directory where to save the processed data')
parser.add_argument('--models_count', default=10, help="Number to model to process!")
ARGS = parser.parse_args()

SOURCEDIR = ARGS.source_dir
SAMPLING_METHOD = ARGS.sampling_method
OUT_DIR_ROOT = ARGS.out_dir
MODELS_TO_PROCESS = int(ARGS.models_count)


def getTransform():
	transform = None

	if(SAMPLING_METHOD == 'ImportanceSampling'):
		transform = T.SamplePoints(1024, remove_faces=True, include_normals=False)

	return transform

if __name__ == '__main__':
	'''
	out_dir = osp.join(
		osp.dirname(osp.realpath(__file__)), OUT_DIR_ROOT, SAMPLING_METHOD)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		os.chmod(out_dir, 0o777)'''

	transform = getTransform()
	assert (transform != None)

	# DATASET preprocessing
	pre_transform = T.NormalizeScale()

	classes = os.listdir(SOURCEDIR)
	files = []
	for _class in classes:
		folder = osp.join(SOURCEDIR, _class, 'train')
		if(osp.isdir(folder)):
			folder_file = os.listdir(folder)
			for f in folder_file:
				files.append(osp.join(folder, f))

	random.shuffle(files)
	files = files[:MODELS_TO_PROCESS]




	for filename in files:
		if osp.isdir(filename):
			continue

		off_obj = OffParser(filename)
		data = Data(
			pos = torch.Tensor(off_obj.points).float(),
			face = torch.Tensor(off_obj.faces).transpose(0,1).long()
		)

		data = pre_transform(data)
		data = transform(data)
		pos = data.pos.transpose(0,1)

		### Plot Sampled PointCloud
		fig_point = plt.figure()
		axp =fig_point.add_subplot(111,
			 projection='3d',
			 xlabel='X',
			 ylabel='Y',
			 zlabel='Z',
			 title=osp.basename(filename)
			 )
		axp.scatter(pos[0], pos[1], pos[2], c='r', marker='o')


		### Plot mesh
		fig_mesh = plt.figure()
		mx = off_obj.points.max(axis=0)
		c = 0.5 * (mx + off_obj.points.min(axis=0))
		r = 1.1 * np.max(mx - c)
		xlim, ylim, zlim = np.column_stack([c - r, c + r])
		axm = fig_mesh.add_subplot(111,
			 projection='3d',
			 xlim=xlim,
			 ylim=ylim,
			 zlim=zlim,
			 xlabel='X',
			 ylabel='Y',
			 zlabel='Z',
			 title=osp.basename(filename)
			 )
		off_obj.plot(axm)

	plt.show()













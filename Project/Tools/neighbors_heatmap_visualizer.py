import argparse
import os
import os.path as osp
import random
from off_parser import OffParser


import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as NormalizeColor
import numpy as np
import Sampling.poisson_disk_sampling as Poisson
import Normalization.normalization as N
from Layers.data_enhacment import AddNeightboursCount

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', default=None)
parser.add_argument('--sample_num', default=1024, help='The number of points to sample.')
parser.add_argument('--sampling_method', default='ImportanceSampling', help='The type of samling to use.')
parser.add_argument('--out_dir', default="./processed_data", help='The directory where to save the processed data')
parser.add_argument('--models_count', default=1, help="Number to model to process!")
ARGS = parser.parse_args()

SAMPLE_NUM = ARGS.sample_num
SOURCEDIR = ARGS.source_dir
SAMPLING_METHOD = ARGS.sampling_method
OUT_DIR_ROOT = ARGS.out_dir
MODELS_TO_PROCESS = int(ARGS.models_count)


def getTransform():
	transform = None

	if(SAMPLING_METHOD == 'ImportanceSampling'):
		transform = T.SamplePoints(SAMPLE_NUM, remove_faces=True, include_normals=False)

	elif (SAMPLING_METHOD == 'PoissonDiskSampling'):
		transform = Poisson.PoissonDiskSampling(SAMPLE_NUM, remove_faces=True)

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
	#pre_transform = T.NormalizeScale()
	pre_transform = N.Normalize()

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


	#Setup Neightbour count level
	radii = [0.1, 0.2, 0.4, 0.8]
	max_points = [SAMPLE_NUM, SAMPLE_NUM, SAMPLE_NUM, SAMPLE_NUM]
	add_neightbours_layer = AddNeightboursCount(
		max_points=max_points,
		radii=radii
	)

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

		x, pos, batch = add_neightbours_layer.forward(
			None,
			data.pos,
			torch.zeros(data.pos.size(0))
		)

		x = x.transpose(0,1) # to be esaier to plot

		for i in range(len(radii)):
			neightbours = x[i+3]

			### Plot Sampled PointCloud
			fig_point = plt.figure()
			axp = fig_point.add_subplot(111,
				projection='3d',
				xlabel='X',
				ylabel='Y',
				zlabel='Z',
				xlim=[-1, 1],
				ylim=[-1, 1],
				zlim=[-1, 1],
				title=osp.basename(filename) + ' Radious:{}'.format(radii[i])
			)

			axp.scatter(x[0], x[1], x[2],
						c=neightbours.numpy(),
						marker='o'
			)

	plt.show()













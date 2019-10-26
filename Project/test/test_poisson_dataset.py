import argparse
import os
import os.path as osp
import random
from DatasetLoader.loader import LoadDataset
from torch_geometric.data import DataLoader


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='PoissonModelNet40')
ARGS = parser.parse_args()

DATASET = ARGS.dataset


if __name__ == '__main__':

	(train_dataset, test_dataset) = LoadDataset(DATASET)

	for data in train_dataset:

		pos = data.pos.transpose(0, 1)

		### Plot Sampled PointCloud
		fig_point = plt.figure()
		axp = fig_point.add_subplot(111,
									projection='3d',
									xlabel='X',
									ylabel='Y',
									zlabel='Z',
									xlim=[-1.5, 1.5],
									ylim=[-1.5, 1.5],
									zlim=[-1.5, 1.5]
									)
		axp.scatter(pos[0], pos[1], pos[2], c='r', marker='o')

		plt.show()
		break








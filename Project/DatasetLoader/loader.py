from torch_geometric.datasets import ShapeNet
from torch_geometric.datasets import ModelNet
from DatasetLoader.poisson_modelnet40  import PoissonModelNet

import os.path as osp
import os
import logging


DATASET_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')

def LoadDataset(dataset_train, dataset_test, pre_transform_train=None, transform_train=None, pre_transform_test=None, transform_test=None,  category=None):
	path_train = osp.join(DATASET_PATH, dataset_test)
	path_test = osp.join(DATASET_PATH, dataset_train)

	if not osp.exists(path_train):
		os.makedirs(path_train)

	if not osp.exists(path_test):
		os.makedirs(path_test)

	_train_dataset = None
	if(dataset_train == 'ModelNet10'):
		_train_dataset = ModelNet(path_train, '10', True, transform_train, pre_transform_train)

	elif(dataset_train == 'ModelNet40'):
		_train_dataset = ModelNet(path_train, '40', True, transform_train, pre_transform_train)

	elif (dataset_train == 'PoissonModelNet40'):
		_train_dataset = PoissonModelNet(path_train, '40', True)

	elif (dataset_train == 'PoissonModelNet10'):
		_train_dataset = PoissonModelNet(path_train, '10', True)

	elif(dataset_train == 'ShapeNet'):
		_train_dataset = ShapeNet(
			path_train, category, train=True, transform=transform_train, pre_transform=pre_transform_train)

	_test_dataset = None
	if (dataset_test == 'ModelNet10'):
		_test_dataset = ModelNet(path_test, '10', False, transform_test, pre_transform_test)

	elif (dataset_test == 'ModelNet40'):
		_test_dataset = ModelNet(path_test , '40', False, transform_test, pre_transform_test)

	elif (dataset_test == 'PoissonModelNet40'):
		_test_dataset = PoissonModelNet(path_test, '40', False)

	elif (dataset_test == 'PoissonModelNet10'):
		_test_dataset = PoissonModelNet(path_test, '10', False)

	elif (dataset_test == 'ShapeNet'):
		_test_dataset = ShapeNet(
			path_test, category, train=False, pre_transform=pre_transform_test)


	if _train_dataset is None or _test_dataset is None:
		print("Invalid dataset requested!")
		return (None, None)
	else:
		print('Dataset Train: {}'.format(dataset_train))
		logging.info('Dataset Train: {}'.format(dataset_train))

		print('Dataset Test: {}'.format(dataset_test))
		logging.info('Dataset Test: {}'.format(dataset_test))

		return(_train_dataset, _test_dataset)


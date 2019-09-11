from torch_geometric.datasets import ShapeNet
from torch_geometric.datasets import ModelNet
import os.path as osp
import os
import logging

DATASET_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')

def LoadDataset(dataset_name, pre_transform, transform, category=None):
	path = osp.join(DATASET_PATH, dataset_name)
	if not osp.exists(path):
		os.makedirs(path)

	if(dataset_name == 'ModelNet10'):
		_train_dataset = ModelNet(path, '10', True, transform, pre_transform)
		_test_dataset = ModelNet(path, '10', False, transform, pre_transform)
		print('Dataset: ModelNet10')
		logging.info('Dataset: ModelNet10')
		return (_train_dataset, _test_dataset)

	elif(dataset_name == 'ModelNet40'):
		_train_dataset = ModelNet(path, '40', True, transform, pre_transform)
		_test_dataset = ModelNet(path, '40', False, transform, pre_transform)
		print('Dataset: ModelNet40')
		logging.info('Dataset: ModelNet40')
		return (_train_dataset, _test_dataset)

	elif(dataset_name == 'ShapeNet'):
		_train_dataset = ShapeNet(
			path,
			category,
			train=True,
			transform=transform,
			pre_transform=pre_transform)
		_test_dataset = ShapeNet(
			path, category, train=False, pre_transform=pre_transform)
		print('Dataset: ShapeNet')
		logging.info('Dataset: ShapeNet')
		return (_train_dataset, _test_dataset)



	print("Invalid dataset requested!")
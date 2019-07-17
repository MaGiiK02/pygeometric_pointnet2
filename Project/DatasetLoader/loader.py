from torch_geometric.datasets import ShapeNet
from torch_geometric.datasets import ModelNet
import os.path as osp
import os

DATASET_PATH = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')

def LoadDataset(datasetName, preTransform, transform):
	path = osp.join(DATASET_PATH, datasetName)
	if not osp.exists(path):
		os.makedirs(path)

	if(datasetName == 'ModelNet10'):
		_train_dataset = ModelNet(path, '10', True, transform, preTransform)
		_test_dataset = ModelNet(path, '10', False, transform, preTransform)
		print('Dataset: ModelNet10')
		return (_train_dataset, _test_dataset)
	elif(datasetName == 'ModelNet40'):
		_train_dataset = ModelNet(path, '40', True, transform, preTransform)
		_test_dataset = ModelNet(path, '40', False, transform, preTransform)
		print('Dataset: ModelNet40')
		return (_train_dataset, _test_dataset)

	print("Invalid dataset requested!")
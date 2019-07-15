import sys

import torch

if __name__ == '__main__':

	if sys.argv !=3:
		print('Wrong arguments count, only a file at time can be opened!')

	filePath = sys.argv[1]

	savedModel = torch.load(filePath)
	epoch = savedModel['epoch']
	loss = savedModel['loss']

	print('epoch:{:03d}  loss:{:.4f}'.format(epoch, loss))
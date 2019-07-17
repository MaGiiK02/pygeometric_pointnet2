import sys

import torch

if __name__ == '__main__':

	filePath = sys.argv[1]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	savedModel = torch.load(filePath, map_location=device
							)
	epoch = savedModel['epoch']
	loss = savedModel['loss']

	print('epoch:{:03d}  loss:{:.4f}'.format(epoch, loss))
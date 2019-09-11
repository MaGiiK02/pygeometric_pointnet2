import sys

import torch

if __name__ == '__main__':

	filePath = sys.argv[1]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	savedModel = torch.load(filePath, map_location=device
							)
	epoch = savedModel['epoch']
	loss = savedModel['loss']
	avg_loss = savedModel['loss_avg']
	current_lr = savedModel['current_lr']
	train_time_hms = savedModel['train_time_hms']

	print('{} :: Epoch: {:03d}, Test: {:.4f}, Last 10 AVG: {:.4f}, LR: {:.6f}'.format(
		train_time_hms, epoch, loss, avg_loss, current_lr))
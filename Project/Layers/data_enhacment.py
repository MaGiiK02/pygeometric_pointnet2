import torch
from torch_geometric.nn import radius

class AddNeightboursCount(torch.nn.Module):
	def __init__(self, radii, max_points):
		super(AddNeightboursCount, self).__init__()
		self.max_points = max_points
		self.radii = radii

	def forward(self, x, pos, batch):
		if(x is None):
			x = pos
		else:
			x = torch.cat([x, pos], dim=1)

		# extend the feature dimension in order to accommodate the new features
		for i in range(len(self.radii)) :
			centers_index, unused  = radius(
                pos, pos, self.radii[i], batch, batch, max_num_neighbors=self.max_points[i])

			#Prepare for concat
			neighboors_count = centers_index.bincount().float().unsqueeze(-1)
			neighboors_count /= self.max_points[i] #normaliation

			x = torch.cat((x, neighboors_count), dim=1)


		return x, pos, batch

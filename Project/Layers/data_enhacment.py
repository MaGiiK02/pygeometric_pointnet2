import torch
from torch_geometric.nn import radius

class AddNeightboursCount(torch.nn.Module):
    def __init__(self, radii, max_points):
        super(AddNeightboursCount, self).__init__()
        self.max_points = max_points
        self.radii = radii

    def forward(self, x, pos, batch):
		if(x is None):
			x = torch.tensor(device= pos.getDevice())
		for i in range(self.radii) :
			row, col = radius(
                pos, pos[idx], self.r_list[i], batch, batch[idx], max_num_neighbors=self.group_sample_size[i])
			x[row][i] = len(col)

		return x
import torch
import math

class Normalize(object):
	def __call__(self, data):
		pos = data.pos.transpose(0,1)

		bbox = torch.tensor([
			[pos[0].min(), pos[1].min(), pos[2].min()],
			[pos[0].max(), pos[1].max(), pos[2].max()]
		])

		bbox_size = torch.tensor([
			[bbox[1][0] - bbox[0][0]], #x_size
			[bbox[1][1] - bbox[0][1]], #y_size
			[bbox[1][2] - bbox[0][2]], #z_size
		])

		center = (bbox_size / 2)
		center[0] += bbox[0][0]
		center[1] += bbox[0][1]
		center[2] += bbox[0][2]

		pos = pos - center.expand_as(pos)

		scale = 2/bbox_size.pow(2).sum().sqrt()
		data.pos = (pos * scale).transpose(0,1)

		return data
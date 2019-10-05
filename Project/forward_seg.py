import argparse
import logging
import os
import os.path as osp
import time
from collections import deque

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u

from DatasetLoader.loader import LoadDataset
from Utils.generics import saveModelCheckpoint, loadFromCheckpoint
from Utils.lr_schedulers import limitedExponentialDecayLR as customExpDecayLambda
from Utils.helper import time_to_hms_string

from Models.PointNet2MSG.pointnet2_seg_msg import PointNet2MSGSeg as PointNet2MSG
from Models.PointNet2MSGSortPool.pointnet2_seg_msg_sort_pool import PointNet2MSGSortPoolSeg as PointNet2MSGSortPool
from Models.PointNet2MSGFPSortPool.pointnet2_seg_msgfp_sort_pool import PointNet2MSGFPSortPoolSeg as PointNet2MSGFPSortPool

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, help='Model name (PointNet2MSG)')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number sampled from each object [default: 1024]')
parser.add_argument('--dataset', default='ShapeNet', help='The name of the dataset to use (ShapeNet)[default: ShapeNet]')
parser.add_argument('--weights_path', default=None, help='The path to the checkpoint file from which continue the train')
parser.add_argument('--use_normals', type=bool, default=False, help='Specufy if train the model using point normals from the data sets[default: False]')
ARGS = parser.parse_args()

NUM_POINT = ARGS.num_point #should be moved in the model weight save
DATASET_NAME = ARGS.dataset
MODEL_NAME = ARGS.model
N_FEATURES = 6 if ARGS.use_normals else 3
SORT_POOL_K = ARGS.sort_pool_k #should be moved in the model weight save
WWIGHT_PATH = ARGS.weights_path



def eval(model, test_loader, device):
    model.eval()

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes
        i, u = i_and_u(pred, data.y, test_dataset.num_classes, data.batch)
        intersections.append(i.to(torch.device('cpu')))
        unions.append(u.to(torch.device('cpu')))
        categories.append(data.category.to(torch.device('cpu')))

    category = torch.cat(categories, dim=0)
    intersection = torch.cat(intersections, dim=0)
    union = torch.cat(unions, dim=0)

    ious = [[] for _ in range(len(test_loader.dataset.categories))]
    for j in range(len(test_loader.dataset)):
        i = intersection[j, test_loader.dataset.y_mask[category[j]]]
        u = union[j, test_loader.dataset.y_mask[category[j]]]
        iou = i.to(torch.float) / u.to(torch.float)
        iou[torch.isnan(iou)] = 1
        ious[category[j]].append(iou.mean().item())

    for cat in range(len(test_loader.dataset.categories)):
        ious[cat] = torch.tensor(ious[cat]).mean().item()

    return correct_nodes / total_nodes, torch.tensor(ious).mean().item()


def getModel(name, input_features, class_count):
    model = None
    if (MODEL_NAME == 'PointNet2MSG'):
        model = PointNet2MSG(class_count, nfeatures=input_features)

    elif (MODEL_NAME == 'PointNet2MSGSortPool'):
    	model = PointNet2MSGSortPool(class_count, nfeatures=input_features, sort_pool_k=SORT_POOL_K)

    elif (MODEL_NAME == 'PointNet2MSGFPSortPool'):
    	model = PointNet2MSGFPSortPool(class_count, nfeatures=input_features, sort_pool_k=SORT_POOL_K)

    return model


if __name__ == '__main__':
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    modelWeightsPath = osp.join(
        osp.dirname(osp.realpath(__file__)), '.', 'weights/', MODEL_NAME + '_seg/')
    if not os.path.exists(modelWeightsPath):
        os.makedirs(modelWeightsPath)
        os.chmod(modelWeightsPath, 0o777)

    # DATASET preparing
    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    (train_dataset, test_dataset) = LoadDataset(
        DATASET_NAME, transform=transform, pre_transform=pre_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)


    # TRAIN SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_10_loss_values = deque([], 10)



    train_start_time = time.time()

    if (CHECKPOINT != None):
        (model, optimizer, optimizer_scheduler, start_epoch, train_checkpoint_time) = loadFromCheckpoint(
            CHECKPOINT, model, optimizer, optimizer_scheduler, device)



    #Model Evaluation
	(test_loss, test_acc, output_seg) = eval(model, test_loader, device)
	optimizer_scheduler.step()


	current_train_time = time.time() - train_start_time

	epoch_to_print = 'Test: {:.4f}, Last 10 AVG: {:.4f}, LR: {:.8f}, Loss: {:.4f}'.format(
		 test_acc, loss_avg, optimizer_scheduler.get_lr()[0], test_loss)

	print(epoch_to_print)

	# TODO add better system to save checkpoint, and maybe a better naming (with start and end feature count)
	if (epoch % 10 == 0):
		saveModelCheckpoint(
			modelWeightsPath, model, optimizer, test_acc, epoch, optimizer_scheduler, loss_avg, current_train_time)


    # Final trained Model store for inference
    inferenceReadyModel = osp.join( modelWeightsPath, 'trained/model_inference.pt')

    torch.save(model.state_dict(), inferenceReadyModel)
import argparse
import logging
import os
import os.path as osp
import time
import re
from collections import deque

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from DatasetLoader.loader import LoadDataset
from Utils.helper import time_to_hms_string
from Utils.generics import saveModelCheckpoint, loadFromCheckpoint
from Utils.lr_schedulers import limitedExponentialDecayLR as customExpDecayLambda
from Normalization.normalization import Normalize
from Sampling.poisson_disk_sampling import PoissonDiskSampling

from Models.PointNet2.pointnet2_cls_ssg import PointNet2Class as PointNet2
from Models.PointNet2MSG.pointnet2_cls_msg import PointNet2MSGClass as PointNet2MSG
from Models.PointNet2MRG.pointnet2_cls_mrg import PointNet2MRGClass as PointNet2MRG
from Models.PointNet2MRGSortPool.pointnet2_cls_mrg_sort_pool import PointNet2MRGSortPoolClass as PointNet2MRGSortPool
from Models.PointNet2MSGSortPool.pointnet2_cls_msg_sort_pool import PointNet2MSGSortPoolClass as PointNet2MSGSortPool
from Models.PointNet2MSGFPSortPool.pointnet2_cls_msgfp_sort_pool import PointNet2MSGFPSortPoolClass as PointNet2MSGFPSortPool
from Models.PointNet2MRGLight.pointnet2_cls_mrg_light import PointNet2MRGLightClass as PointNet2MRGLight
from Models.PointNetVanilla.pointnet_cls_vanilla import PointNetVanillaClass  as PointNetVanilla
from Models.PointNetInputEnhanced.pointnet_cls_ie import PointNetInputEnhanced  as PointNetInputEnhanced

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, help='Model name (PointNet2, PointNet2MSG, PointNet2MSGSortPool, PointNet2MRG, PointNet2MRGSortPool, PointNet2MSGFPSortPool)')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number sampled from each object [default: 1024]')
parser.add_argument('--epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='The size of the batch to use [default: 16]')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='The weight decay to use for the training, the original PointNet2 1e-5 [default:1e-5]')
parser.add_argument('--lr_decay', type=float, default=0.7, help='The weight decay used as to start the exponential LR decay, the step is defined from the Exponential class of pytorch.[default:0.7]')
parser.add_argument('--lr_decay_step', type=float, default=2e5, help='Learning rate decay step [default: 2e5]');
parser.add_argument('--train_dataset', default='ModelNet40', help='The name of the dataset to use to train the model (ModelNet10, ModelNet40, ShapeNet)[default: ModelNet40]')
parser.add_argument('--test_dataset', default='ModelNet40', help='The name of the dataset to use to test the model (ModelNet10, ModelNet40, ShapeNet)[default: ModelNet40]')
parser.add_argument('--checkpoint', default=None, help='The path to the checkpoint file from which continue the train')
parser.add_argument('--log_file', default='./train_log.txt', help='The file where to print the logs [default: ./train_log.txt]')
parser.add_argument('--use_normals', type=bool, default=False, help='Specufy if train the model using point normals from the data sets[default: False]')
parser.add_argument('--sort_pool_k', type=int, default=32, help='The number of point the sort_pool should keep <only needed with the sort_pool model> [default: 32]')
parser.add_argument('--sampling_method_train', default='ImportanceSampling', help='The type of sampling to use. [PoissonDiskSampling or ImportanceSampling]')
parser.add_argument('--sampling_method_test', default='ImportanceSampling', help='The type of sampling to use. [PoissonDiskSampling or ImportanceSampling]')
ARGS = parser.parse_args()

BATCH_SIZE = ARGS.batch_size
NUM_POINT = ARGS.num_point
EPOCH = ARGS.epoch
DATASET_TRAIN = ARGS.train_dataset
DATASET_TEST = ARGS.train_dataset
MODEL_NAME = ARGS.model
CHECKPOINT = ARGS.checkpoint
WEIGHT_DECAY = ARGS.weight_decay
LR_DECAY = ARGS.lr_decay
LR_DECAY_STEP = ARGS.lr_decay_step
LOG_FILE = ARGS.log_file
USE_NORMALS = ARGS.use_normals
N_FEATURES = 6 if ARGS.use_normals else 3
SORT_POOL_K = ARGS.sort_pool_k
SAMPLING_TRAIN = ARGS.sampling_method_train
SAMPLING_TEST = ARGS.sampling_method_test

def train(model, train_loader, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        if data.norm is not None:
            data.x = data.norm
        optimizer.zero_grad()
        test_values = data.category if DATASET_TRAIN == 'ShapeNet' else data.y
        loss = F.nll_loss(model(data), test_values)
        loss.backward()
        optimizer.step()

def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        if data.norm is not None:
            data.x = data.norm
        test_result = data.category if DATASET_TEST == 'ShapeNet' else data.y
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(test_result).sum().item()
    return correct / len(test_loader.dataset)

def setupTrain(model_name, n_features, class_number, device, w_decay, lr_decay, batch_size, decay_step):
    model = getModel(model_name, n_features, class_number).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=w_decay)
    start_epoch = 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lr_lambda=customExpDecayLambda(lr_decay, batch_size, decay_step))

    return model, optimizer, start_epoch , scheduler

def getModel(name, input_features, class_count):
    model = None
    if (MODEL_NAME == 'PointNetVanilla'):
        model = PointNetVanilla(class_count, nfeatures=input_features, nPoints=NUM_POINT)

    elif (MODEL_NAME == 'PointNetInputEnhanced'):
        model = PointNetInputEnhanced(class_count, nfeatures=input_features, batch_size=BATCH_SIZE, nPoints=NUM_POINT)

    elif (MODEL_NAME == 'PointNet2'):
        model = PointNet2(class_count, bn_momentum=0.1)

    elif (MODEL_NAME == 'PointNet2MSG'):
        model = PointNet2MSG(class_count, nfeatures=input_features)

    elif (MODEL_NAME == 'PointNet2MRG'):
        model = PointNet2MRG(class_count, nfeatures=input_features)

    elif (MODEL_NAME == 'PointNet2MRGLight'):
        model = PointNet2MRGLight(class_count, n_features=input_features)

    elif (MODEL_NAME == 'PointNet2MSGSortPool'):
        model = PointNet2MSGSortPool(class_count, n_feature=input_features, sort_pool_k=SORT_POOL_K)

    elif (MODEL_NAME == 'PointNet2MSGFPSortPool'):
        model = PointNet2MSGFPSortPool(class_count, n_feature=input_features, sort_pool_k=SORT_POOL_K)

    elif (MODEL_NAME == 'PointNet2MRGSortPool'):
        model = PointNet2MRGSortPool(class_count, n_features=input_features, sort_pool_k=SORT_POOL_K)

    return model

def getSampler(name, dataset_name):
    transform = None

    if(dataset_name == 'ShapeNet'):
        transform = T.FixedPoints(NUM_POINT)

    elif(name == 'ImportanceSampling'):
        transform = T.SamplePoints(NUM_POINT, remove_faces=True, include_normals=USE_NORMALS)

    elif (name == 'PoissonDiskSampling'):
        transform = PoissonDiskSampling(NUM_POINT, remove_faces=True)

    return transform

if __name__ == '__main__':
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    modelWeightsPath = osp.join(
        osp.dirname(osp.realpath(__file__)), '.', 'weights/', MODEL_NAME + '_cls/')
    if not os.path.exists(modelWeightsPath):
        os.makedirs(modelWeightsPath)
        os.chmod(modelWeightsPath, 0o777)

    # DATASET preprocessing
    if(USE_NORMALS and ('Poisson' in DATASET_TRAIN or 'Poisson' in DATASET_TEST)):
        print("Poisson sampled data do not support normals(at the moment).")
        exit(1)

    pre_transform = Normalize(),
    transform_train = getSampler(SAMPLING_TRAIN, DATASET_TRAIN)  # POINT SAMPLING
    transform_test = getSampler(SAMPLING_TEST, DATASET_TEST)
    (train_dataset, test_dataset) = LoadDataset(
        DATASET_TRAIN, DATASET_TEST,
        transform_train=transform_train, pre_transform_train=pre_transform,
        transform_test=transform_test, pre_transform_test=pre_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)


    # TRAIN SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_10_loss_values = deque([], 10)

    (model, optimizer, start_epoch, optimizer_scheduler) = setupTrain(
        model_name = MODEL_NAME,
        n_features = N_FEATURES,
        class_number = train_dataset.num_classes,
        w_decay = WEIGHT_DECAY,
        lr_decay = LR_DECAY,
        decay_step=LR_DECAY_STEP,
        batch_size = BATCH_SIZE,
        device = device
	)

    train_start_time = time.time()

    if (CHECKPOINT != None):
        (model, optimizer, optimizer_scheduler, start_epoch, train_checkpoint_time) = loadFromCheckpoint(
            CHECKPOINT, model, optimizer, optimizer_scheduler, device)
        train_start_time-=train_checkpoint_time


    # TRAIN CYCLE
    for epoch in range(start_epoch, start_epoch + EPOCH):
        train(model, train_loader, optimizer, device)
        test_acc = test(model, test_loader, device)
        optimizer_scheduler.step()

        # the older item are automatically removed from the deque when full
        last_10_loss_values.append(test_acc)
        loss_avg = sum(last_10_loss_values) / float(len(last_10_loss_values))
        current_train_time = time.time() - train_start_time

        epoch_to_print = '{} :: Epoch: {:03d}, Test: {:.4f}, Last 10 AVG: {:.4f}, LR: {:.8f}'.format(
            time_to_hms_string(current_train_time) ,epoch, test_acc, loss_avg, optimizer_scheduler.get_lr()[0])

        print(epoch_to_print)
        logging.info(epoch_to_print)

        # TODO add better system to save checkpoint, and maybe a better naming (with start and end feature count)
        if (epoch % 10 == 0):

            saveModelCheckpoint(
                modelWeightsPath, model, optimizer, test_acc, epoch, optimizer_scheduler, loss_avg, current_train_time)

    # Final trained Model store for inference
    inferenceReadyModel = osp.join( modelWeightsPath, 'trained/model_inference.pt')

    torch.save(model.state_dict(), inferenceReadyModel)


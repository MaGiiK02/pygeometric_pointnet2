import sys
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from Project.DatasetLoader.loader import LoadDataset
from PointNet2.Classification.model import PointNet2Class as Net



def train(model, train_loader, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)

def load_arguments():
    args = sys.argv
    dataset='ModelNet10'
    modelName='PointNet2'
    for idx,arg in enumerate(args):
        if(arg == '-d'):
            data = args[idx+1]
        if(arg == '-m'):
            modelName = args[idx+1]

    return (dataset, modelName)



def saveModel(folder_path, model, optimizer, loss, epoch):
    modelWeightsPathEpoch = osp.join(
        folder_path, 'checkpoint/train_checkpoint' + str(epoch) + '.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, modelWeightsPathEpoch)


def trainModel(datasetName, batchSize=32):

    modelWeightsPath = osp.join(
        osp.dirname(osp.realpath(__file__)), '.', 'weights')

    #DATASET preparing
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #POINT SAMPLING
    (train_dataset, test_dataset) = LoadDataset(
        datasetName, transform=transform, preTransform=pre_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batchSize, shuffle=True, num_workers=6)
    test_loader = DataLoader(
        test_dataset, batch_size=batchSize, shuffle=False, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   #to test use 6 epoch, to train at least 201
    for epoch in range(1, 251):
        train(model, train_loader, optimizer, device)
        test_acc = test(model, test_loader, device)
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))

        if( epoch%5 == 0):
            saveModel(
                modelWeightsPath, model, optimizer, test_acc, epoch)


    #Final trained Model store
    inferenceReadyModel =  osp.join(
        modelWeightsPath, 'inference/model_inference.pt')
            
    torch.save(model.state_dict(), inferenceReadyModel)
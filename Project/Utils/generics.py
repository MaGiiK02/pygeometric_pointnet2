import torch
import os.path as osp
from Utils.helper import time_to_hms_string

def saveModelCheckpoint(folder_path, model, optimizer, loss, epoch, scheduler, loss_avg, elapsed_time):
    modelWeightsPathEpoch = osp.join(
        folder_path, 'train_checkpoint' + str(epoch) + '.pt')



    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'loss_avg': loss_avg,
        'current_lr': scheduler.get_lr()[0],
        'train_time_ms': elapsed_time,
        'train_time_hms': time_to_hms_string(elapsed_time)
    }, modelWeightsPathEpoch)

def loadFromCheckpoint(checkpointPath, model, optimizer, lr_scheduler, device):
    checkpoint = torch.load(checkpointPath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    return model, optimizer, lr_scheduler, checkpoint['epoch'] +1, checkpoint['train_time_ms']
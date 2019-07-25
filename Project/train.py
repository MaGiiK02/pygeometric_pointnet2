from PointNet2.Classification.train import trainModel as TrainPointNet2Classification
from PointNet2MSG.Classification.train import trainModel as TrainPointNet2MSGClassification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, help='Model name (PointNet2, PointNet2MSG)')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number sampled from each object [default: 1024]')
parser.add_argument('--epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='The size of the batch to use [default: 16]')
parser.add_argument('--dataset', default='ModelNet40', help='The name of the dataset to use (ModelNet10, ModelNet40, ShapeNet)[default: ModelNet40]')
parser.add_argument('--checkpoint', default=None, help='The path to the checkpoint file from which continue the train')
ARGS = parser.parse_args()

BATCH_SIZE = ARGS.batch_size
NUM_POINT = ARGS.num_point
EPOCH = ARGS.epoch
DATASET_NAME = ARGS.dataset
MODEL_NAME = ARGS.model
CHECKPOINT = ARGS.checkpoint


if __name__ == '__main__':
    if(MODEL_NAME == 'PointNet2'):
        TrainPointNet2Classification(
            DATASET_NAME,
            batchSize=BATCH_SIZE,
            nPoints=NUM_POINT,
            train_epoch=EPOCH,
            checkpoint=CHECKPOINT
        )
    elif (MODEL_NAME == 'PointNet2MSG'):
        TrainPointNet2MSGClassification(
            DATASET_NAME,
            batchSize=BATCH_SIZE,
            nPoints=NUM_POINT,
            train_epoch=EPOCH,
            checkpoint=CHECKPOINT
        )


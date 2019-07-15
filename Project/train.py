from PointNet2.Classification.train import trainModel as TrainPointNet2Classification
from PointNet2MSG.Classification.train import trainModel as TrainPointNet2MSGClassification
import sys

def load_arguments():
    args = sys.argv
    dataset = 'ModelNet40'
    modelName = 'PointNet2'
    for idx, arg in enumerate(args):
        if (arg == '-d'):
            dataset = args[idx + 1]
        elif (arg == '-m'):
            modelName = args[idx + 1]

    return (dataset, modelName)

if __name__ == '__main__':
    (datasetName, modelName) = load_arguments()
    if(modelName == 'PointNet2'):
        TrainPointNet2Classification(datasetName, 32)
    if (modelName == 'PointNet2MSG'):
        TrainPointNet2MSGClassification(datasetName, 32)



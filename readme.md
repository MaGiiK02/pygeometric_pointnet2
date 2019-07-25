# PointNet++ (PointNet2) on pytorch_geometric

A remake of the pointnet2 model using the popular pytorch_geometric framework, starting form the pytorc_geometric's example of the PointNet2.

## Run the code

All the models can be run from the 3 main files:

1. *test.py* to launch a test on a given pre-trained model weights(to be implemented).
2. *train.py* to train the model.
3. *forward.py* to launch the model on an input (to be implemented).

### train.py

Here is explained how launch the train.py to train the models, giving some examples and explain it's parameters.

#### Parameters

1. --model: the model name relative to the wanted model (accepts: PointNet2, PointNet2MSG).
2. --num_point: the nuber of point to sample from the original mash, the resulting model will be used for train (default:1024).
3. --epoch: numer of train iteration over the dataset, the default value is enought to make the model learning to stale (default: 251).
4. --batch_size: the batch size to use for the train, the default is 32 but the results are better with 16 (default:32).
5. --dataset: the dataset to use for the training (accepts: ModelNet10, ModelNet40) (default:ModelNet40).
6. --checkpoint: the path to a train checkpoint, it has to contain the loss function status so a forward only checkpoint will return an error, oblivously the checkpoint will only work if it's obtained from a train run with same num_point and same dataset(the classes are infered from the dataset). 

#### Examples

* Default run (ModelNet40, 1024 points, batch_size 32, epoch 251, checkpoint None)
  
``` Shell
python3 train.py --model PointNet2
```

-or- for the MSG variant

``` Shell
python3 train.py --model PointNet2MSG
```

* With the checkpoint Default run (ModelNet40, 1024 points, batch_size 32, epoch 251, checkpoint GivenPath)
  
``` Shell
python3 train.py --model PointNet2 --checkpoint ./PointNet2/Classification/weights/train_checkpoint/train_checkpoint_100.pt
```

If you want use a checkpoint

## Code structure
The Project folder contains all the code, each model have it's own subfolder and the model definintion can be found in the **model.py** file in each of the 2 implementation (segmentation classification).
All the layers defined for this project can be found in the Layers Module, like the MSG abstraction layer or the standard one, as well as the redount layer (*GloabalSAModule*).
The checkpoint will be automatically stored in the folder relatives to the models, so for example the weights of the PointNet2 Classification will be stored in this subfolder:

``` shell
./PointNet2/Classification/weights/train_checkpoint/
```

## Prerequisites

To run the project you need python3, and the following libraries:

1. pytorch(https://pytorch.org/)
2. pytorch_geometric(https://github.com/rusty1s/pytorch_geometric)
3. maybe something else need to check

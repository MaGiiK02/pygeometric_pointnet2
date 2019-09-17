import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('Paths', metavar='Path',  nargs='+', help='Path to each file with segmentation data to plot')
ARGS = parser.parse_args()

Axes3D.scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)

#Devi creare un sistema che date le classi genera un array di colori, rifacendoti alla distaza dei punti e delle classi tra loro!!!
#Praticamente una specie di map coloring!!




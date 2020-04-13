import os, sys
#from utils.utils import *
import nomen 
import yaml
import pandas as pd 
import random 
import torch

from Street2Vec import Street2Vec
from Locations import Locations, load_datasets # dataset


# Total train size 94440
# Total valSize 9565
# Total testSize 9760

# 58007 are from pittsburgh

config = """

modelPath : $experiments/v1 
datasetPath : $datasets/streetlearn
previsualizeData: false
trainSize : 93767
valSize : 5000
testSize : 5000
flip: true
l2_normalization: true
scale: 3
numEpochs : 30
clusters: 64
examples_by_location : 5
trainZoom : [18, 19]
testZoom : [18]
embedding_dim : 32
l1 : 1.0
l2 : 1.0
l3 : 0.5
l4 : 0.5
batchSize : 30
alpha : 2
lr : 0.00001
seed: 396
useGPU: true
workers : 4
pca : false
"""

dictionary = yaml.safe_load(config)
for key in dictionary.keys():
    print(key, dictionary[key])

#a = input('It\'s a good idea to double check configuration ..., Is it correct? y/n ')
#if a == 'n':
#    sys.exit('Aborted ...')
cfg = nomen.Config(dictionary)
cfg.parse_args()

# Save a copy of configuration file to model dir
filename = os.path.join(cfg.modelPath, 'config.yml')
with open(filename, 'w') as config_file:
    yaml.dump(dictionary, config_file, default_flow_style=True)

# Load dataset
datasets, loaders = load_datasets(cfg)

# Call train function 
s2v = Street2Vec(cfg, mode='train')
s2v.train(datasets, loaders)

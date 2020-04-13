import os, sys
from utils.utils import *
import nomen 
import yaml
import pandas as pd 
import random 
import torch
from Street2Vec import Street2Vec
from datasets.Locations import load_datasets, load_test_dataset, load_val_dataset
from utils.visualize_CNN import Zeiler

mode = 'train'  # train, predict, evaluate, predict_all
save_dir = os.path.join(os.environ['experiments'], 'EmbeddingsVLAD', 's2v700k_v2')

# config
# In training takes the configuration below 
# In any other mode the configuration will be read from the config file in  the model's directory
# if you run train this will create or override the file in dir

config = """
train_dataset : locations_10_19
val_dataset : scattered_london
test_dataset : scattered_london
previsualize_data: false
train_size : 
val_size : 
test_size :
flip: true
l2_normalization: true
scale: 3
num_epochs : 2
clusters: 64
examples_by_location : 5
train_zoom : [19, 20]
test_zoom : [19]
embedding_dim : 32
l1 : 0.2
l2 : 1.0
l3 : 0.2
l4 : 0.2
batch_size : 20
alpha : 2
lr : 0.00001
seed: 403
use_gpu: true
workers : 3
arch : resnet50
"""
   
# Parse config ----------------------------------------------------------------------------
if mode is 'train':
    dictionary = yaml.safe_load(config)
    print(dictionary)
    a = input('Continue y/n ? ')
    if a == 'n':
        sys.exit('Aborted ...')

    cfg = nomen.Config(dictionary)
    cfg.parse_args()
    filename = os.path.join(save_dir, 'config.yml')
    with open(filename, 'w') as config_file:
        yaml.dump(dictionary, config_file, default_flow_style=True)

else:
    print("Reading parameters from yaml file")
    filename = os.path.join(save_dir, 'config.yml')
    with open(filename, 'r') as config:
        try:
            dictionary = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            sys.exit('configuration file not found')
    cfg = nomen.Config(dictionary)
    cfg.parse_args()

cfg.dir = save_dir
cfg = moreconfig(cfg, mode)


# Set the seed ----------------------------------------------------------------------------------------
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(cfg.seed)

#train and evaluate ----------------------------------------------
s2v = Street2Vec(cfg)

if cfg.mode == "train":
    s2v.summary()
    dataset, loader = load_datasets(cfg)
    s2v.train(dataset, loader)

elif cfg.mode == "predict":
    dataset, loader = load_val_dataset(cfg)
    # Predict validation dataset and save predictions in model's directory
    s2v.predict(dataset, loader)

elif cfg.mode == 'evaluate':
    s2v.evaluate()
    #figures(cfg)

elif cfg.mode == 'predict_all':
    datasets = ['london_10_19', 
                'edinburgh_10_19', 
                'newyork_10_19', 
                'toronto_v1', 
                'luton_v4',
                'scattered_london']

    #datasets = ['manchester_10_19', 'scattered_london']
    #datasets=['scattered_london']
    for dset in datasets:
        cfg.val_dataset = dset 
        cfg.val_size = None
        cfg.previsualize_data = False
        cfg = moreconfig(cfg, mode)
        dataset, loader = load_val_dataset(cfg)
        s2v.predict(dataset, loader)

elif cfg.mode == 'visualize_filters':
    cfg.val_dataset = 'luton_v4'
    cfg.val_size = None
    cfg.previsualize_data = False
    cfg = moreconfig(cfg, mode)
    dataset, loader = load_val_dataset(cfg)
    z = Zeiler(s2v.model, cfg, dataset)
    z.save_heat_maps(980)
    #z.save_heat_maps_given_x_y(219,942)


else:
    print('Option not found')
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time

def create_network(protofile, weightfile):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.train()
    return net

protofile = "deploy_googlenet_places365.prototxt"
weightfile = "googlenet_places365.caffemodel"

net = create_network(protofile, weightfile)


'''
 @article{zhou2017places,
   title={Places: A 10 million Image Database for Scene Recognition},
   author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2017},
   publisher={IEEE}
   website={http://places2.csail.mit.edu}
 }
'''

import argparse
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchnet.meter as meter
from torchstat import stat
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch BSD Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet161',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use imagenet pre-trained model')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='use checkpoint model')
parser.add_argument('--num_classes',default=2, type=int, help='num of class in the model')
parser.add_argument('--check_interval', default=500, type=int, metavar='N',
                    help='interval of each checkpoint')
parser.add_argument('--num_save', default=0, type=int, metavar='N',
                    help='inital number of the checkpoint')
parser.add_argument('--num_checkpoints', default=5, type=int, metavar='N',
                    help='number of saved checkpoints')

writer = SummaryWriter('runs/vgg/jc_all_index')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def main():
    global args
    args = parser.parse_args()
    print(args)

    if not args.pretrained:
        model = models.__dict__[args.arch](num_classes=args.num_classes)
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % args.arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # gpu to cpu
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        if args.arch is "alexnet":
            state_dict = {str.replace(k,'classifier.6.bias' ,'fc1.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'classifier.6.weight' ,'fc1.weight'): v for k,v in state_dict.items()}
        if args.arch is "resnet18" or "resnet50":      
            state_dict = {str.replace(k,'fc.bias' ,'fc1.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'fc.weight' ,'fc1.weight'): v for k,v in state_dict.items()}
        if args.arch is "densenet161":
            state_dict = {str.replace(k,'classifier.bias' ,'fc1.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'classifier.weight' ,'fc1.weight'): v for k,v in state_dict.items()}       
        model.load_state_dict(state_dict, strict=False)  
    else:
        if args.arch is "googlenet":  
            model = models.googlenet(pretrained=args.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,args.num_classes)
        if args.arch is "vgg":
            model = models.vgg11_bn(pretrained=args.pretrained)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,args.num_classes)
    
    stat(model, (3, 224, 224))

if __name__ == '__main__':
    main()

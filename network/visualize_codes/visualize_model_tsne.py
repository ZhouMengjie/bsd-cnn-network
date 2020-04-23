# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchnet.meter as meter
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import pdb
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch BSD Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=2, type=int, help='num of class in the model')
parser.add_argument('--check_interval', default=500, type=int, metavar='N',
                    help='interval of each checkpoint')
parser.add_argument('--num_save', default=0, type=int, metavar='N',
                    help='inital number of the checkpoint')
parser.add_argument('--num_checkpoints', default=5, type=int, metavar='N',
                    help='number of saved checkpoints')

writer = SummaryWriter('runs/t-SNE')

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0),-1)

            x = module(x)
            # print(name)
            if self.extracted_layers is None or name in self.extracted_layers:
                outputs[name] = x

        return outputs

def main():
    args = parser.parse_args()
    print(args)

    # load model
    model_file = 'model_junction/resnet18_best.pth.tar'
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print(model) 

    # Data loading code
    data_dir = 'data/t-sne/JUNCTIONS' # or GAPS
    subarea = 'hudsonriver5k'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    data_transforms = {
        subarea: transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in [subarea]}

    val_loader = {x: torch.utils.data.DataLoader(image_datasets[x], 
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
                for x in [subarea]}

    classes = image_datasets[subarea].classes
    image_paths = image_datasets[subarea].imgs
    targets = image_datasets[subarea].targets

    # Images
    # images = load_images(image_paths)
    images = []

    # class labels
    class_labels = [classes[target] for target in targets]

    # extract features
    features = []
    model.eval()

    for i, (input, target) in enumerate(val_loader[subarea]):
        print("Images:")
        print("\t#{}: {}".format(i, image_paths[i]))
        # a batch image add to t-sne
        fc_etxractor = FeatureExtractor(model, 'avgpool')
        extracted_results = fc_etxractor(input)
        features.append(extracted_results['avgpool'].data.numpy())
        images.append(input.data.numpy())

    features = np.asarray(features)
    images = np.asarray(images)

    writer.add_embedding(features.squeeze(),metadata=class_labels, label_img=images.squeeze())
    

if __name__ == '__main__':
    main()

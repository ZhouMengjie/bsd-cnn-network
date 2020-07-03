
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

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


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

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

def main():
    global args
    args = parser.parse_args()
    # print(args)

    # load model
    model_file = 'model_junction/resnet18_accuracy.pth.tar'
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Data loading code
    data_dir = 'data/JUNCTIONS' # or GAPS
    valdir = os.path.join(data_dir, 'hudsonriver5k')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    print(len(val_loader))

    validate(val_loader, model)


def validate(val_loader, model):
    extract_list = ["layer1","layer2","layer3","layer4"]
    class_names = ["junctions", "non_junctions"]

    # switch to evaluate mode
    model.eval()
    dst = './feauture_maps'
    therd_size = 224

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)

        # generate feature map
        fc_etxractor = FeatureExtractor(model, extract_list)
        with torch.no_grad():
            extracted_results = fc_etxractor(input)

        for k, v in extracted_results.items():
            features = v[0]
            iter_range = features.shape[0]
            for j in range(iter_range):
                if "fc" in k:
                    continue
            
                feature = features.data.numpy()
                feature_img = feature[j, :, :]
                feature_img = np.asarray(feature_img * 256, dtype = np.uint8)
                dst_path = os.path.join(dst, k)

                make_dirs(dst_path)
                feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
                if feature_img.shape[0] < therd_size:
                    tmp_file = os.path.join(dst_path, str(j) + '_' + str(therd_size) + '.png')
                    tmp_img = feature_img.copy()
                    tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(tmp_file, tmp_img)

                # dst_file = os.path.join(dst_path, str(i) + '.png')
                # cv2.imwrite(dst_file, feature_img)
            
        out = torchvision.utils.make_grid(input)
        print(target)
        imshow(out, title=[class_names[target]]) 

if __name__ == '__main__':
    main()

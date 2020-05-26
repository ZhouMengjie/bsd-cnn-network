# grad_cam for jc

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
import matplotlib.cm as cm
from grad_cam import(BackPropagation, Deconvnet, GradCAM, GuidedBackPropagation, occlusion_sensitivity)
from tensorboardX import SummaryWriter
import pdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch BSD Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg',
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

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def preprocess(image_path, arch):
    image_path = image_path[0]
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    if arch is "alexnet":
        image = transforms.Compose(
        [
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        )(raw_image[..., ::-1].copy())
    else:
        image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        )(raw_image[..., ::-1].copy())

    return image, raw_image

def load_images(image_paths, arch):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path, arch)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

def main():
    global args
    args = parser.parse_args()
    # print(args)
    
    # load model
    main_directory = 'model_junction_vgg/'
    output_dir = 'Grad_CAM/vgg/jc'
    data_dir = 'data/jc' # or GAPS
    subarea = 'val'
    model_file = main_directory + args.arch + '_recall.pth.tar'

    if not args.pretrained:
        model = models.__dict__[args.arch](num_classes=args.num_classes)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        if args.arch is "googlenet":  
            model = models.googlenet(pretrained=args.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,args.num_classes)
        else:
            model = models.vgg11_bn(pretrained=args.pretrained)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,args.num_classes)

        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    
    print(model)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Data loading code
    image_datasets = torchvision.datasets.ImageFolder(os.path.join(data_dir, subarea))
    image_paths = image_datasets.imgs
    # targets = image_datasets.targets

    # Images
    images, raw_images = load_images(image_paths, args.arch)
    images = torch.stack(images).to(device)
          
    # load classes
    classes= ["junctions", "non_junctions"]
    make_dirs(output_dir)

    # switch to evaluate mode
    model.eval()

    # the four resisual layers
    # target_layers = ["layer1", "layer2", "layer3", "layer4"]
    target_layers = ['features'] # vgg
    target_class = 0 # 0-jc/njc, 1-njc/nbd

    gcam = GradCAM(model = model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids = ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))
        
        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            p = probs[j][ids[j][:] == target_class]
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(p)
                )
            )

            save_gradcam(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet18", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

       
if __name__ == '__main__':
    main()

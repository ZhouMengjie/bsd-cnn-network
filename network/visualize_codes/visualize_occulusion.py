
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

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def preprocess(image_path):
    image_path = image_path[0]
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

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
    data_dir = 'data/jc' # or GAPS
    subarea = 'val'
    image_datasets = torchvision.datasets.ImageFolder(os.path.join(data_dir, subarea))
    image_paths = image_datasets.imgs
    # targets = image_datasets.targets

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)
          
    # load classes
    classes= ["junctions", "non_junctions"]
    # classes= ["gaps", "non_gaps"]
    output_dir = 'Occulusion/jc'
    make_dirs(output_dir)

    # switch to evaluate mode
    model.eval()

    print("Occlusion Sensitivity:")
    patche_sizes = [15, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    topk = 1
    stride = 10

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=args.batch_size
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=os.path.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, args.arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


       
if __name__ == '__main__':
    main()

# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter
import pdb
import scipy.io
import re
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


def main():
    global args
    args = parser.parse_args()
    # print(args)

    # load model
    classes = ('junctions', 'non_junctions')
    # classes = ('gaps', 'non_gaps')
    model_file = 'model_junction_vgg/vgg_recall.pth.tar'
    # model_file = 'model_gap_vgg/vgg_recall.pth.tar'
    
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

    # print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Data loading code
    data_dir = 'data/JUNCTIONS' # JUNCTIONS or GAPS
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) 
    if args.arch is "alexnet":  
        transform = [transforms.Resize(227),transforms.ToTensor(),normalize]  
    else:
        transform = [transforms.ToTensor(),normalize]                              

    sub_area = 'unionsquare5k'            
    valdir = os.path.join(data_dir, sub_area)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(
            transform
        )),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(len(val_loader))

    # define loss function (criterion) and pptimizer
    data_transforms = {sub_area: transforms.Compose([transforms.ToTensor(),normalize,]),}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                    for x in [sub_area]}  
    img_paths = image_datasets[sub_area].imgs

    panoids = [None] * len(val_loader)
    features = [None] * len(val_loader)
    criterion = nn.CrossEntropyLoss()
    features, panoids = validate(val_loader, model, criterion, classes, features, img_paths, panoids)

    
    scipy.io.savemat('uq_junctions_features.mat', mdict={'features': features})
    scipy.io.savemat('uq_junctions_ids_labels.mat', mdict={'panoids': panoids})

    # scipy.io.savemat('ws_gaps_features.mat', mdict={'features': features})
    # scipy.io.savemat('ws_gaps_ids_labels.mat', mdict={'panoids': panoids})
    


def validate(val_loader, model, criterion, classes, features, img_paths, panoids):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        
        # convert output probabilities to predicted class
        _, pred_tensor = torch.max(output, 1)
        preds = np.squeeze(pred_tensor.cpu().numpy()) 
        features[i] = preds  
        re_str = img_paths[i][0]   
        result = re.findall('[^/]+',re_str)
        panoids[i] = result[4]
        # print(panoids[i])
        # pred_lable = classes[preds]
        # print(preds)
        # print(pred_lable)

        # measure accuracy and record loss
        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return features, panoids

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

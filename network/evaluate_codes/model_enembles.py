import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num_classes',default=2, type=int, help='num of class in the model')

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
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    main_directory = 'model_junction/'
    file_name = 'hd_junctions_features.mat' 

    # Data loading code
    data_dir = 'data/JUNCTIONS' # JUNCTIONS or GAPS
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) 

    transform = [transforms.ToTensor(),normalize]                              

    sub_area = 'hudsonriver5k'            
    valdir = os.path.join(data_dir, sub_area)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(
            transform
        )),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(len(val_loader))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()


    # load model parameters
    model_file = main_directory + 'resnet18_accuracy.pth.tar'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    features_acc = [None] * len(val_loader)
    features_acc = validate(val_loader, model, criterion, classes, features_acc)
    torch.cuda.empty_cache()
   
    # load model parameters
    model_file = main_directory + 'resnet18_precision.pth.tar'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    features_prec = [None] * len(val_loader)   
    features_prec = validate(val_loader, model, criterion, classes, features_prec)
    torch.cuda.empty_cache()

    # load model parameters
    model_file = main_directory + 'resnet18_recall.pth.tar'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    features_rec = [None] * len(val_loader) 
    features_rec = validate(val_loader, model, criterion, classes, features_rec)
    torch.cuda.empty_cache()

    # load model parameters
    model_file = main_directory + 'resnet18_F1.pth.tar'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    features_F1 = [None] * len(val_loader) 
    features_F1 = validate(val_loader, model, criterion, classes, features_F1)
    torch.cuda.empty_cache()

    # load model parameters
    model_file = main_directory + 'resnet18_loss.pth.tar'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)
    features_loss = [None] * len(val_loader) 
    features_loss = validate(val_loader, model, criterion, classes, features_loss)
    torch.cuda.empty_cache()

    features = [None] * len(val_loader) 
    features = (features_acc + features_prec + features_rec + features_loss + features_F1) / 5

    for i in range(len(val_loader)):
        output = (features_acc[i] + features_prec[i] + features_rec[i] + features_loss[i] + features_F1[i])/5
        _, pred_tensor = torch.max(output, 1)
        preds = np.squeeze(pred_tensor.cpu().numpy()) 
        features[i] = preds 
  
    scipy.io.savemat(file_name, mdict={'features': features})



def validate(val_loader, model, criterion, classes, features):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input)

        # convert output to softmax
        s = F.softmax(output, dim=1)

        # convert output probabilities to predicted class
        # _, pred_tensor = torch.max(output, 1)
        # preds = np.squeeze(pred_tensor.cpu().numpy()) 
        # features[i] = preds  
        features[i] = s
        # re_str = img_paths[i][0]   
        # result = re.findall('[^/]+',re_str)
        # panoids[i] = result[4]
        # print(panoids[i])
        # pred_lable = classes[preds]
        # print(preds)
        # print(pred_lable)

        # measure accuracy and record loss
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            i, len(val_loader), batch_time=batch_time, loss=losses,
        #            top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return features

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

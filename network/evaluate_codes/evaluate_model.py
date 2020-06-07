# plot roc and pr curve for gap models

import argparse
import os
import shutil
import time
import numpy as np
import torch
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
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pdb
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch BSD Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
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

    # Data loading code
    data_dir = 'data/GAPS' # JUNCTIONS or GAPS
    valdir = os.path.join(data_dir, 'hudsonriver5k')
    ROC_names = 'ROC_bd.png'
    PR_names = 'PR_bd.png'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

              
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # load alexnet model
    model_file = 'model_gap_alexnet/' + 'alexnet_accuracy.pth.tar'
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(227),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = models.__dict__['alexnet'](num_classes=args.num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)    
    # print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    precision_alexnet, recall_alexnet, fpr_alexnet = validate(val_loader, model, criterion)
    torch.cuda.empty_cache()


    # load vgg model
    model_file = 'model_gap_vgg/' + 'vgg_recall.pth.tar'

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)   

    model = models.vgg11_bn(pretrained=True)
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

    precision_vgg, recall_vgg, fpr_vgg = validate(val_loader, model, criterion)
    torch.cuda.empty_cache()

    # load googlenet model
    model_file = 'model_gap_googlenet/' + 'googlenet_recall.pth.tar'

    model = models.googlenet(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,args.num_classes)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    precision_googlenet, recall_googlenet, fpr_googlenet = validate(val_loader, model, criterion)
    torch.cuda.empty_cache()

    # load resnet18 model
    model_file = 'model_gap/' + 'resnet18_recall.pth.tar'

    model = models.__dict__['resnet18'](num_classes=args.num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    precision_resnet18, recall_resnet18, fpr_resnet18 = validate(val_loader, model, criterion)
    torch.cuda.empty_cache()


    # load resnet50 model
    model_file = 'model_gap_resnet50/' + 'resnet50_recall.pth.tar'

    model = models.__dict__['resnet50'](num_classes=args.num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    precision_resnet50, recall_resnet50, fpr_resnet50 = validate(val_loader, model, criterion)
    torch.cuda.empty_cache()

    # load densenet161 model
    model_file = 'model_gap_densenet161/' + 'densenet161_recall.pth.tar'

    model = models.__dict__['densenet161'](num_classes=args.num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # print(model) 

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    precision_densenet161, recall_densenet161, fpr_densenet161 = validate(val_loader, model, criterion)
    torch.cuda.empty_cache()


    # plot ROC curve
    plt.figure(figsize=(10,6))
    plt.plot(fpr_alexnet,recall_alexnet,label="Alexnet",linewidth=2)
    plt.plot(fpr_vgg,recall_vgg,label="Vgg",linewidth=2)
    plt.plot(fpr_googlenet,recall_googlenet,label="Googlenet",linewidth=2)  
    plt.plot(fpr_resnet18,recall_resnet18,label="Resnet18",linewidth=2)
    plt.plot(fpr_resnet50,recall_resnet50,label="Resnet50",linewidth=2)
    plt.plot(fpr_densenet161,recall_densenet161,label="Densenet161",linewidth=2)
    plt.xlabel("False Positive Rate",fontsize=16)
    plt.ylabel("True Positive Rate",fontsize=16)
    plt.title("ROC Curve",fontsize=16)
    plt.legend(loc="lower right",fontsize=16)
    plt.savefig(ROC_names)
    # plt.show()

    # plot P-R curve
    plt.figure(figsize=(10,6))
    plt.plot(recall_alexnet,precision_alexnet,label="Alexnet",linewidth=2)
    plt.plot(recall_vgg,precision_vgg,label="Vgg",linewidth=2)
    plt.plot(recall_googlenet,precision_googlenet,label="Googlenet",linewidth=2)
    plt.plot(recall_resnet18,precision_resnet18,label="Resnet18",linewidth=2)  
    plt.plot(recall_resnet50,precision_resnet50,label="Resnet50",linewidth=2) 
    plt.plot(recall_densenet161,precision_densenet161,label="Densenet161",linewidth=2) 
    plt.xlabel("Recall",fontsize=16)
    plt.ylabel("Precision",fontsize=16)
    plt.title("Precision Recall Curve",fontsize=17)
    plt.legend(fontsize=16)
    plt.savefig(PR_names)
    # plt.show()



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    # define a confusion_matrix
    confusion_matrix_0 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_1 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_2 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_3 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_4 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_5 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_6 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_7 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_8 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_9 = meter.ConfusionMeter(args.num_classes)
    confusion_matrix_10 = meter.ConfusionMeter(args.num_classes)

    end = time.time()
    cm = []

    # define threshold to plot ROC and PR curve
    threshold = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # convert output to softmax
        s = F.softmax(output, dim=1)

        # update confusion matrix
        s_p = torch.sub(s,torch.tensor([0.0]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_0.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.1]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_1.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.2]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_2.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.3]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_3.add(s_p.data.squeeze(),target.long()) 

        s_p = torch.sub(s,torch.tensor([0.4]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_4.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.5]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_5.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.6]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_6.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.7]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_7.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.8]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_8.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([0.9]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_9.add(s_p.data.squeeze(),target.long())

        s_p = torch.sub(s,torch.tensor([1.0]).to(device))
        s_p = torch.index_fill(s_p, 1, torch.tensor([1]).to(device),0)
        confusion_matrix_10.add(s_p.data.squeeze(),target.long())
    
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            i, len(val_loader), batch_time=batch_time, loss=losses,
        #            top1=top1))

    # print(' * Prec@1 {top1.avg:.3f}\t'
    #         'Loss {loss.avg:.4f}'
    #         .format(top1=top1, loss=losses))
    
    cm.append(confusion_matrix_0) 
    cm.append(confusion_matrix_1) 
    cm.append(confusion_matrix_2)   
    cm.append(confusion_matrix_3) 
    cm.append(confusion_matrix_4) 
    cm.append(confusion_matrix_5) 
    cm.append(confusion_matrix_6)  
    cm.append(confusion_matrix_7) 
    cm.append(confusion_matrix_8) 
    cm.append(confusion_matrix_9) 
    cm.append(confusion_matrix_10) 

    acc = []
    precision = []
    recall = []
    F1 = []
    fpr = []
    for i in range(len(threshold)):
        cm_value = cm[i].value()
        acc.append((cm_value[0][0]+cm_value[1][1]) / (cm_value.sum()))  
        precision.append(cm_value[0][0] / (cm_value[0][0] + cm_value[1][0]))
        recall.append(cm_value[0][0] / (cm_value[0][0] + cm_value[0][1]))
        F1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
        # tpr = recall fpr = 1 - specificity
        fpr.append(cm_value[1][0] / (cm_value[1][0] + cm_value[1][1]))

    print(' * Accuracy@1 {top1:.3f}\t'
        'Loss {loss.avg:.4f}\t'
        'Presion {p:.3f}\t'
        'Recall {r:.3f}\t'
        'F1 score {F1:.3f}\t'
        'Specificity {s:.3f}'
        .format(top1=acc[5], loss=losses, p=precision[5], r=recall[5], F1=F1[5], s=1-fpr[5]))  

    return precision, recall, fpr

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

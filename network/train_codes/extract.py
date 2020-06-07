import nomen 
import os
import yaml
import time
import torch
import copy
import torchvision
import torch.nn as nn
import torchnet.meter as meter
import scipy.io
import re
import numpy as np
from load_data import load_datasets, load_test_dataset
from my_model import MyModel
from tensorboardX import SummaryWriter


config = """
previsualizeData: False
batchSize : 1
workers : 4
lr : 3e-4
start_epoch: 0
epochs: 90
print_freq: 10
arch: resnet18
check_interval: 500
mode: test
"""

class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main():
    global cfg, device

    # load parameters
    dictionary = yaml.safe_load(config)
    for key in dictionary.keys():
        print(key, dictionary[key])
    cfg = nomen.Config(dictionary)
    cfg.parse_args()

    # load data
    area = 'unionsquare5k'
    _, test_loader = load_test_dataset(cfg, area)
    file_name = 'uq_features,mat'

    # load model
    model = MyModel(cfg) # design own model

    model_file = 'model_combined/resnet18_accuracy.pth.tar'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), "GPUs!") 
        model = _CustomDataParallel(model)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPU!")
    #     model = nn.DataParallel(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()  

    # panoids = [None] * len(test_loader)
    features = [None] * len(test_loader)
    criterion = nn.CrossEntropyLoss()        
    # evaluate on validation set
    features = validate(test_loader, model, criterion, features)

    scipy.io.savemat(file_name, mdict={'features': features})

def validate(val_loader, model, criterion, features):
    batch_time = AverageMeter()    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, sample in enumerate(val_loader):        
        Yf, Yl, Yr, Yb = sample['y0'], sample['y1'], sample['y2'], sample['y3'] #[batch,1,3,224,224]
        Label = sample['label'] 
        
        Yf = Yf.to(device)
        Yf = Yf.view(-1,3,224,224)
        Yl = Yl.to(device)
        Yl = Yl.view(-1,3,224,224)
        Yr = Yr.to(device)
        Yr = Yr.view(-1,3,224,224)                
        Yb = Yb.to(device)
        Yb = Yb.view(-1,3,224,224)                
        target = Label.to(device,dtype=torch.int64).view(-1)  

        # compute output
        output = model.forward(Yf, Yl, Yb, Yr)

        # convert output probabilities to predicted class
        _, pred_tensor = torch.max(output, 1)
        preds = np.squeeze(pred_tensor.cpu().numpy()) 
        features[i] = preds  
        # print(preds)

        # measure accuracy and record loss
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0], target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

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


    
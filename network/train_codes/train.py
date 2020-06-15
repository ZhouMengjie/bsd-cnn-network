import nomen 
import os
import yaml
import time
import torch
import copy
import torchvision
import torch.nn as nn
import torchnet.meter as meter
from load_data import load_datasets, load_test_dataset
from my_model import MyModel
from tensorboardX import SummaryWriter


config = """
previsualizeData: False
batchSize : 128
workers : 4
start_epoch: 0
epochs: 20
print_freq: 10
arch: resnet18
check_interval: 500
num_classes: 4
seed: 396
mode: train
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
    global cfg, device, writer, best_loss, best_acc

    writer = SummaryWriter('runs/resnet18_combined_bce')
    best_acc = 0
    best_loss = 100

    # load parameters
    dictionary = yaml.safe_load(config)
    for key in dictionary.keys():
        print(key, dictionary[key])
    cfg = nomen.Config(dictionary)
    cfg.parse_args()

    # load data
    _, loader = load_datasets(cfg)

    # load model
    model = MyModel(cfg) # design own model

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
    # criterion = nn.CrossEntropyLoss().cuda()  
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4) # default:1e-3



    for epoch in range(cfg.start_epoch, cfg.epochs):   
        # train and evaluate for one epoch
        train(loader['train'], model, criterion, optimizer, epoch)  
        
        # evaluate on validation set
        with torch.no_grad():
            acc, loss = validate(loader['val'], model, criterion, epoch)

        # remember best prec and best loss and save checkpoint
        is_acc = acc > best_acc
        best_acc = max(acc, best_acc)
        
        is_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        if (is_acc) | (is_loss):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'best_loss': best_loss
            }, is_acc, is_loss, cfg.arch.lower()) 


def train(train_loader,model,criterion,optimizer,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # switch to train mode
    model.train()
                
    # Iterate over data.
    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # All in GPU                    
        Yf, Yr, Yb, Yl = sample['y0'], sample['y1'], sample['y2'], sample['y3'] #[batch,1,3,224,224]
        label = sample['label'] #[batch, 10]
                    
        # tiles to gpu and reshape
        Yf = Yf.to(device)
        Yf = Yf.view(-1,3,224,224)
        Yl = Yl.to(device)
        Yl = Yl.view(-1,3,224,224)
        Yr = Yr.to(device)
        Yr = Yr.view(-1,3,224,224)                
        Yb = Yb.to(device)
        Yb = Yb.view(-1,3,224,224)                
        # target = Label.to(device,dtype=torch.int64).view(-1)
        target = label.to(device)

                
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # Fordward pass for tiles
        output = model.forward(Yf, Yr, Yb, Yl)
        m = nn.Sigmoid()    
        # s = m(output)
        loss = criterion(m(output).double(), target.double())

        # backward + optimize only if in training phase
        loss.backward()
        # 'clip_grap_norm' help prevent the exploding gradient problem in RNNs / LSTMs
        # torch.nn.utils.clip_grad_norm_(model.gsv_model.parameters(),0.5)
        # torch.nn.utils.clip_grad_norm_(model.tile_model.parameters(),0.5)
        # torch.nn.utils.clip_grad_norm_(model.embedding_model.parameters(),0.5)
        optimizer.step()

        # statistics
        # prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), target.size(0)) #input.size(0) = batch_size
        # top1.update(prec1[0], target.size(0))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        t_step = epoch*len(train_loader) + i         
        writer.add_scalar('traning loss', losses.avg, t_step)
        # writer.add_scalar('traning accuracy', top1.avg, t_step) 

        if t_step % cfg.check_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'best_loss': best_loss
            }, False, False,'checkpoint') 


def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):        
        Yf, Yr, Yb, Yl = sample['y0'], sample['y1'], sample['y2'], sample['y3'] #[batch,1,3,224,224]
        label = sample['label'] 
        
        Yf = Yf.to(device)
        Yf = Yf.view(-1,3,224,224)
        Yl = Yl.to(device)
        Yl = Yl.view(-1,3,224,224)
        Yr = Yr.to(device)
        Yr = Yr.view(-1,3,224,224)                
        Yb = Yb.to(device)
        Yb = Yb.view(-1,3,224,224)                
        # target = Label.to(device,dtype=torch.int64).view(-1)  
        target = label.to(device)

        # compute output
        output = model.forward(Yf, Yr, Yb, Yl)
        m = nn.Sigmoid()
        loss = criterion(m(output).double(), target.double())

        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), target.size(0))
        # top1.update(prec1[0], target.size(0))

    print('Epoch: [{0}]\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        epoch, loss=losses))
    
    writer.add_scalar('validation loss', losses.avg, epoch)
    # writer.add_scalar('validation accuracy', top1.avg, epoch)
    acc = 0

    return acc, losses.avg  


def save_checkpoint(state, is_acc, is_loss, filename='checkpoint.pth.tar'):
    if (not is_acc) & (not is_loss):
        torch.save(state, filename + '_latest.pth.tar')
    if is_acc:
        torch.save(state, filename + '_accuracy.pth.tar') 
    if is_loss:
        torch.save(state, filename + '_loss.pth.tar')

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


    

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
from tensorboardX import SummaryWriter

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
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
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

best_acc = 0
best_prec = 0
best_rec = 0
best_loss = 100
best_F1 = 0

def main():
    global args, device, writer, best_prec, best_loss, best_acc, best_rec, best_F1
    args = parser.parse_args()
    print(args)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
            transform = [transforms.Resize(227),transforms.ToTensor(),normalize]
        if args.arch is "resnet18" or "resnet50":      
            state_dict = {str.replace(k,'fc.bias' ,'fc1.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'fc.weight' ,'fc1.weight'): v for k,v in state_dict.items()}
            transform = [transforms.ToTensor(),normalize]
        if args.arch is "densenet161":
            state_dict = {str.replace(k,'classifier.bias' ,'fc1.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'classifier.weight' ,'fc1.weight'): v for k,v in state_dict.items()}
            transform = [transforms.ToTensor(),normalize]
       
        model.load_state_dict(state_dict, strict=False)  

    else:
        if args.arch is "googlenet":  
            model = models.googlenet(pretrained=args.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,args.num_classes)
            transform = [transforms.ToTensor(),normalize]
        if args.arch is "vgg":
            model = models.vgg11_bn(pretrained=args.pretrained)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,args.num_classes)
            transform = [transforms.ToTensor(),normalize]

    print(model)

    # if args.resume:
    #     model_file = 'net5_latest.pth.tar'
    #     # checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    #     checkpoint = torch.load(model_file)
    #     args.start_epoch = checkpoint['epoch']
    #     # best_prec = checkpoint['best_prec']
    #     # best_loss = checkpoint['best_loss']
    #     # best_acc = checkpoint['best_acc']
    #     # best_rec = checkpoint['best_rec']
    #     # best_F1 = checkpoint['best_F1']
    #     model.load_state_dict(checkpoint['state_dict'])        
             
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Data loading code
    data_dir = 'data/JUNCTIONS' # JUNCTIONS or GAPS
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'hudsonriver5k')

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose(
            transform
        )),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(
            transform
        )),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()  
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters())

    # set tf logger for tensorboard
    for epoch in range(args.start_epoch, args.epochs):   
        # train for one epoch
        # adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        with torch.no_grad():
            acc, prec, rec, F1, loss = validate(val_loader, model, criterion, epoch)

        # remember best prec and best loss and save checkpoint
        is_acc = acc > best_acc
        best_acc = max(acc, best_acc)
        
        is_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        is_prec = prec > best_prec
        best_prec = max(prec, best_prec)

        is_rec = rec > best_rec
        best_rec = max(rec, best_rec)

        is_F1 = F1 > best_F1
        best_F1 = max(F1, best_F1)

        if (is_acc) | (is_loss) | (is_prec) | (is_rec) | (is_F1):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'best_prec': best_prec,
                'best_rec': best_rec,
                'best_F1': best_F1,
                'best_loss': best_loss
            }, is_acc, is_prec, is_rec, is_F1, is_loss, args.arch.lower())     

       
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    is_acc = False
    is_prec = False
    is_rec = False
    is_F1 = False
    is_loss = False

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0)) # input.size(0) = batch_size
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
        t_step = epoch*len(train_loader) + i         
        writer.add_scalar('traning loss', losses.avg, t_step)
        writer.add_scalar('traning accuracy', top1.avg, t_step)
        # random mini-batch
        # classes = ('junctions', 'non_junctions')
        # writer.add_figure('predictions vs. actuals',
        #                 plot_classes_preds(output, input, target, classes),
        #                 global_step=epoch * len(train_loader) + i)
        
        # if t_step % args.check_interval == 0:
        #     args.num_save += 1
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc': losses.avg,
        #         'best_loss': top1.avg
        #     }, is_acc, is_prec, is_rec, is_F1, is_loss,'net'+ str(args.num_save))
        #     if args.num_save == 5:
        #         args.num_save = 0
        if t_step % args.check_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'best_prec': best_prec,
                'best_rec': best_rec,
                'best_F1': best_F1,
                'best_loss': best_loss
            }, is_acc, is_prec, is_rec, is_F1, is_loss,'checkpoint')

def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # define a confusion_matrix
    confusion_matrix = meter.ConfusionMeter(args.num_classes)

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        confusion_matrix.add(output.squeeze(),target.long())

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

    print('Epoch: [{0}]\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        epoch, loss=losses, top1=top1))
    
    writer.add_scalar('validation loss', losses.avg, epoch)
    writer.add_scalar('validation accuracy', top1.avg, epoch)

    cm_value = confusion_matrix.value()
    acc = (cm_value[0][0]+cm_value[1][1]) / (cm_value.sum()) 
    precision = cm_value[0][0] / (cm_value[0][0] + cm_value[1][0])
    recall = cm_value[0][0] / (cm_value[0][0] + cm_value[0][1])
    F1 = 2 * precision * recall / (precision + recall)

    return acc, precision, recall, F1, losses.avg

def save_checkpoint(state, is_acc, is_prec, is_rec, is_F1, is_loss, filename='checkpoint.pth.tar'):
    if (not is_acc) & (not is_prec) & (not is_rec) & (not is_loss) & (not is_F1):
        torch.save(state, filename + '_latest.pth.tar')
    if is_acc:
        torch.save(state, filename + '_accuracy.pth.tar')
    if is_prec:
        torch.save(state, filename + '_precision.pth.tar') 
    if is_rec:
        torch.save(state, filename + '_recall.pth.tar')          
    if is_F1:
        torch.save(state, filename + '_F1.pth.tar')  
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(output):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(output, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(output)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


if __name__ == '__main__':
    main()

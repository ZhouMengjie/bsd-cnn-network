
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

import torch
from torch.autograd import Variable as V
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import functional as F
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/bsd_experiment')

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights 
model_file = 'junctions_latest.pth.tar'


model = models.__dict__[arch](num_classes=2)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
print(model) 
model.eval()


# load the test images
data_dir = 'data/hymenoptera_data' # or GAPS
testdir = os.path.join(data_dir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
test_loader = torch.utils.data.DataLoader(
datasets.ImageFolder(testdir, transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=4, shuffle=False,
    num_workers=2)

# load the class label
classes = ('ants','bees')

class_probs = []
class_preds = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        output = model(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds_batch.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)



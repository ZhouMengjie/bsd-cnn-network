import os, sys 
import numpy as np
import pandas as pd
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nomen 
import yaml
from transforms import ToTensor, Normalize
import matplotlib.pyplot as plt

config = """
previsualizeData: true
batchSize : 32
workers : 4
seed: 396
"""

class Locations(Dataset):
    def __init__(self, cfg, transforms=None, dataset_type='train', datasetName=None):
        self.transforms = transforms
        self.batch_size = cfg.batchSize
        self.img_dir_jc = os.path.join('data','JUNCTIONS',datasetName,'junctions')
        self.img_dir_njc = os.path.join('data','JUNCTIONS',datasetName,'non_junctions')         
        self.img_dir_bd = os.path.join('data','GAPS',datasetName,'gaps') 
        self.img_dir_nbd = os.path.join('data','GAPS',datasetName,'non_gaps')  

        names = ["pano_id", "gsv_lat", "gsv_lon", "gsv_yaw", "front", "right", "back", "left", "label", "city"]
        
        if dataset_type == 'train':
            self.csvFile = os.path.join('csv', 'train.csv')
            self.frame = pd.read_csv(self.csvFile, names=names) # Nodes in dataframe
            self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) # Shuffle dataframe
        elif dataset_type == 'val':
            self.datasetName = 'hudsonriver5k'
            self.csvFile = os.path.join('csv', 'hudsonriver5k.csv')
            self.frame = pd.read_csv(self.csvFile, names=names) # Nodes in dataframe
            self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) # Shuffle dataframe       
        else:
            self.datasetName = datasetName
            self.csvFile = os.path.join('csv', datasetName  + '.csv')
            self.frame = pd.read_csv(self.csvFile, names=names) # Nodes in dataframe
            # self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) # Shuffle dataframe       


        self.cfg = cfg
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # Metadata of location
        row = self.frame.loc[idx, :]
        # Given a location get the pano snaps and tile and save them into a dictionary
        sample = {}
        
        # Read snaps
        pano_id = row['pano_id']
        front = row['front']
        right = row['right']
        back = row['back']
        left = row['left']  

        if front == 1:
            filename_front = os.path.join(self.img_dir_jc, pano_id + '_front.jpg') 
        else:
            filename_front = os.path.join(self.img_dir_njc, pano_id + '_front.jpg') 
            
        if back == 1:
            filename_back = os.path.join(self.img_dir_jc, pano_id + '_back.jpg')
        else:
            filename_back = os.path.join(self.img_dir_njc, pano_id + '_back.jpg')
            
        if left == 1:
            filename_left = os.path.join(self.img_dir_bd, pano_id + '_left.jpg') 
        else:
            filename_left = os.path.join(self.img_dir_nbd, pano_id + '_left.jpg') 

        if right == 1:
            filename_right = os.path.join(self.img_dir_bd, pano_id + '_right.jpg') 
        else:
            filename_right = os.path.join(self.img_dir_nbd, pano_id + '_right.jpg')


        sample['y0'] = cv2.imread(filename_front) / 255.0
        sample['y1'] = cv2.imread(filename_right) / 255.0
        sample['y2'] = cv2.imread(filename_back) / 255.0
        sample['y3'] = cv2.imread(filename_left) / 255.0
        # sample['label'] = row['label']
        sample['label'] = [front, right, back, left]
        sample['id'] = pano_id

        if self.transforms:
            sample = self.transforms(sample)
        return sample  

def denormalize_image(img):
    img = img.squeeze()
    img = img.data.numpy()
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    mean = mean.reshape((-1,1,1))
    sd = sd.reshape((-1,1,1))
    img = ((img * sd) + mean)
    img = img.transpose(2,1,0)
    img = img[...,::-1]
    return img

def show_dataset(dst, cfg):
    for j in range(4):
        sample = dst[j]
        images = []
        for key in ['y0','y1','y2','y3']:
            y = sample[key]
            y = denormalize_image(y)
            images.append(y)
        images = np.concatenate(images, 1)
        plt.imshow(images)
        plt.pause(3)

def load_datasets(cfg):
    # Dataset
    train_transforms = transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    train_dataset = Locations(cfg, transforms=train_transforms, dataset_type='train',datasetName='train')
    val_dataset = Locations(cfg, transforms=val_transforms, dataset_type='val',datasetName='hudsonriver5k')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batchSize,
        num_workers=cfg.workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batchSize,
        num_workers=cfg.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    dataset = {'train': train_dataset, 'val':val_dataset}
    loader = {'train': train_loader, 'val':val_loader}

    for ds in ['train', 'val']:
        print('{} dataset len: {}'.format(ds,len(dataset[ds])))

    if cfg.previsualizeData is True:
        show_dataset(dataset['train'], cfg)
        show_dataset(dataset['val'], cfg)

    # for ld in ['train','val']:
    #     ldr = iter(loader[ld])
    #     sample = next(ldr)
    #     print('first sample in {}: shape {}, min {}, max {}, label_shape {}'.format(
    #         ld, sample['y0'].size(), sample['y0'].detach().min(), sample['y0'].detach().max(), sample['y0'].size())) 

    return dataset, loader

def load_test_dataset(cfg, datasetName):
    # Dataset
    test_transforms = transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_dataset = Locations(cfg, transforms=test_transforms, dataset_type='test', datasetName=datasetName)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batchSize,
        num_workers=cfg.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    dataset = test_dataset
    loader = test_loader

    print('{} dataset len: {}'.format("Dataset length",len(dataset)))

    if cfg.previsualizeData is True:
        show_dataset(dataset, cfg)

    ldr = iter(loader)
    sample = next(ldr)
    print('first sample in: shape {}, min {}, max {}, label_shape {}'.format(sample['y0'].size(), sample['y0'].detach().min(), sample['y0'].detach().max(), sample['label'].size())) 
    return dataset, loader

if __name__ == "__main__":
    dictionary = yaml.safe_load(config)
    for key in dictionary.keys():
        print(key, dictionary[key])
    cfg = nomen.Config(dictionary)
    cfg.parse_args()

    area = 'hudsonriver5k'
    dataset, loader = load_test_dataset(cfg,area)
    # dataset, loader = load_datasets(cfg)
    print('finish and check please')

import os, sys 
import numpy as np
import pandas as pd
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader
from Transforms import RandomZoom, FixedZoom, CenterCrop, ToTensor, Normalize
from torchvision import transforms

sys.path.append( os.path.join( os.environ['dev'], 'CVLoc'))

from utils.Location import Location

class Locations(Dataset):
    def __init__(self, cfg, transforms=None, dataset_type='train', datasetName=None):
        self.transforms = transforms
        self.batch_size = cfg.batchSize

        names = ["pano_id", "yaw", "lat", "lon", "city", "neighbor", "bearing", "index"]
        
        if dataset_type == 'train':
            self.csvFile = os.path.join( 'data', 'train.csv')
            self.frame = pd.read_csv(self.csvFile, names=names, nrows=cfg.trainSize) # Nodes in dataframe
            self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) # Shuffle dataframe

        elif dataset_type == 'val':
            self.datasetName = 'hudsonriver5k'
            self.csvFile = os.path.join( 'data', 'hudsonriver5k.csv')
            self.frame = pd.read_csv(self.csvFile, names=names, nrows=cfg.valSize) # Nodes in dataframe
            self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) # Shuffle dataframe
            #if cfg.mode != 'predict' and cfg.mode != 'predict_all' and cfg.mode != 'visualize_filters':
            #    self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
        
        else:
            self.datasetName = datasetName
            self.csvFile = os.path.join( 'data', datasetName  + '.csv')
            self.frame = pd.read_csv(self.csvFile, names=names, nrows=cfg.testSize) # Nodes in dataframe
            #self.frame = self.frame.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) # Shuffle dataframe

        self.cfg = cfg
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # Metadata of location
        row = self.frame.loc[idx, :]
        # Given a location get the pano snaps and tile and save them into a dictionary
        sample = {}

        # Zoom level depend on the dataset type
        if self.dataset_type == 'train':
            xzoom = self.cfg.trainZoom
        else:
            xzoom = self.cfg.testZoom

        
        # Get a flip probability
        flip_prob = np.random.rand(1)[0]

        # Read tiles 
        loc = Location(self.datasetName, row['pano_id'], row['city'])
        for z in xzoom:
            sample['x'+ str(z)]= cv2.cvtColor(loc.getTile(zoom=z), cv2.COLOR_BGR2RGB) / 255.0
    
            if self.dataset_type == 'train' and self.cfg.flip and flip_prob >= 0.5: # Flipping
                sample['x'+ str(z)]= cv2.flip(sample['x'+ str(z)], 1)

        # Read snaps
        sample['pano'] = cv2.cvtColor(loc.getPano(), cv2.COLOR_BGR2RGB)
        if self.dataset_type == 'train' and self.cfg.flip and flip_prob >= 0.5: 
            sample['pano'] = cv2.flip(sample['pano'], 1)

        sample['label'] = row['index']
        sample['zoom']= int(np.ceil(sample['pano'].shape[0] / 512))
        sample['row'] = row

        if self.transforms:
            sample = self.transforms(sample)
        return sample  

def denormalize_image(img):
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
    for j in range(5):
        sample = dst[j]
        tiles = sample['x']
        for i in range(tiles.shape[0]):
            images = []
            tile = tiles[i]
            tile = denormalize_image(tile)
            images.append(tile)
            for key in ['y0','y1','y2','y3']:
                y = sample[key][i]
                y = denormalize_image(y)
                images.append(y)

            images = np.concatenate(images, 1)
            cv2.imshow(dst.dataset_type, images)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_datasets(cfg):
    # Dataset
    train_transforms = transforms.Compose([RandomZoom(224, cfg), ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([CenterCrop(224, cfg), ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    #test_transforms = transforms.Compose([CenterCrop(224, cfg), ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_dataset = Locations(cfg, transforms=train_transforms, dataset_type='train')
    val_dataset = Locations(cfg, transforms=val_transforms, dataset_type='val')
    #test_dataset = Locations(cfg, transforms=test_transforms, dataset_type='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batchSize,
        num_workers=cfg.workers,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batchSize * cfg.examples_by_location,
        num_workers=cfg.workers,
        shuffle=False,
        drop_last=True
    )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=cfg.batchSize * cfg.examples_by_location,
    #     num_workers=cfg.workers,
    #     shuffle=False,
    #     drop_last=False
    # )

    dataset = {'train': train_dataset, 'val':val_dataset} #, 'test':test_dataset}
    loader = {'train': train_loader, 'val':val_loader} #, 'test':test_loader}

    for ds in ['train', 'val']:
        print('{} dataset len: {}'.format(ds,len(dataset[ds])))

    if cfg.previsualizeData is True:
        show_dataset(dataset['train'], cfg)
        show_dataset(dataset['val'], cfg)
        #show_dataset(dataset['test'], cfg)

    for ld in ['train','val']:
        ldr = iter(loader[ld])
        sample = next(ldr)
        print('first sample in {}: shape {}, min {}, max {}, label_shape {}'.format(
            ld, sample['y0'].size(), sample['y0'].detach().min(), sample['y0'].detach().max(), sample['label'].size())) 

    return dataset, loader

def load_test_dataset(cfg, datasetName):
    # Dataset
    test_transforms = transforms.Compose([CenterCrop(224, cfg), ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_dataset = Locations(cfg, transforms=test_transforms, dataset_type='test', datasetName=datasetName)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batchSize * cfg.examples_by_location,
        num_workers=cfg.workers,
        shuffle=False,
        drop_last=False
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
    dataset = "hudsonriver5k"
    dirPath = os.path.join( os.environ['datasets'], 'streetlearn')
    trainFile = os.path.join(dirPath, 'train.csv')
    names = ["pano_id", "yaw", "lat", "lon", "city"]
    frame = pd.read_csv(trainFile, names=names)
    for i in range(4):
        row = frame.loc[i, :]
        loc = Location(dataset, row['pano_id'], row['city'])
        loc.showLocation()

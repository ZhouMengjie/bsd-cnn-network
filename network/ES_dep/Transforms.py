import sys, os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
#sys.path.append('/home/os17592/dev/Equirec2Perspec')
sys.path.append(os.path.join(os.environ['datasets'], 'Equirec2Perspec'))
import Equirec2Perspec as E2P
import random

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).reshape(1,-1,1,1)
        self.std = torch.FloatTensor(std).reshape(1,-1,1,1)

    def __call__(self, sample):
        for key in sample:
            if key[0] in ['x','y']:
                sample[key] = (sample[key] - self.mean) / self.std
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key in sample:
            if key[0] in ['x', 'y']:
                sample[key] = torch.from_numpy(sample[key])
                sample[key] = sample[key].permute(0,3,2,1)
                sample[key] = sample[key].float()
            elif key == 'label':
                sample[key] = torch.tensor(sample[key])
        return sample

class RandomZoom(object):
    def __init__(self, output_size, cfg):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.cfg = cfg

    def add_noise(self, image):
        row,col,ch= image.shape
        mean = 0
        var = 0.001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy

    
    def __call__(self, sample):

        new_h, new_w = self.output_size
        new_dict = dict()
        # Lets randomly generate n tiles
           
        tiles = [sample[key] for key in sample.keys() if key[0] == 'x'] #
        new_tiles = []

        for i in range(self.cfg.examples_by_location):
            #zoom = np.random.randint(0,len(tiles)-1)
            zoom = np.random.randint(0,len(tiles))            
            tile = tiles[zoom]
            h, w = tile.shape[:2]
            h_t = np.random.randint(224, 256)
            w_t = h_t
            new_h, new_w = self.output_size
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            temp_tile = tile[top:top+h_t, left:left+w_t]
            tile = cv2.resize(temp_tile, (new_h,new_w))
            #tile = self.add_noise(tile)
            new_tiles.append(tile)

        # stack tiles
        tiles = np.stack(new_tiles,0)
        # Add tiles to dict
        new_dict['x'] = tiles

        # Crop 10 random GSV images
        pano = sample['pano']
        if  pano.shape[0] < 512*np.power(2,sample['zoom']-1):
            panowidth = 416 * np.power(2,sample['zoom'])
            panoheight = 416 * np.power(2,sample['zoom']-1)
            pano = pano[0:panoheight,0:panowidth,:]                
        equ = E2P.Equirectangular(pano)

        for i,tetha in enumerate([0,-90,90,180]):
            snaps = []
            for _ in range(self.cfg.examples_by_location):
                fov_shift = np.random.randint(-20,20)
                pitch_shift = np.random.randint(-15,15)
                tetha_shift = np.random.randint(-10,10)
                #snap = equ.GetPerspective(100 + fov_shift, tetha + tetha_shift, sample['row']['tilt_pitch_deg']-10 + pitch_shift,new_h,new_w) / 255.0
                snap = equ.GetPerspective(100 + fov_shift, tetha + tetha_shift, 0 + pitch_shift,new_h,new_w) / 255.0        
                snaps.append(snap)
            new_dict['y'+str(i)] = np.stack(snaps, 0)
        
        # Tile the labels 10 times
        labels = np.repeat(sample['label'], self.cfg.examples_by_location)
        new_dict['label'] = labels

        return new_dict

class FixedZoom(object):
    def __init__(self, output_size, cfg):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.cfg = cfg
    
    def __call__(self, sample):
        new_h, new_w = self.output_size
        new_dict = dict()

        tiles = [sample[key] for key in sample.keys() if key[0] == 'x'] #
        new_tiles = []

        zooms = np.repeat(np.array([0,1]),5)
        h_ts = np.array([224,230,236,242,248])
        #h_ts = np.array([224,255])        
        h_ts = np.tile(h_ts,2)

        for i in range(self.cfg.examples_by_location):
            zoom = zooms[i]
            tile = tiles[zoom]
            h, w = tile.shape[:2]
            h_t = h_ts[i]
            w_t = h_t
            new_h, new_w = self.output_size
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            temp_tile = tile[top:top+h_t, left:left+w_t]
            tile = cv2.resize(temp_tile, (new_h,new_w))
            new_tiles.append(tile)
            
        # stack tiles
        tiles = np.stack(new_tiles,0)
        # Add tiles to dict
        new_dict['x'] = tiles

        # Crop 5 different gsvs
        pano = sample['pano']
        if  pano.shape[0] < 512*np.power(2,sample['zoom']-1):
            panowidth = 416 * np.power(2,sample['zoom'])
            panoheight = 416 * np.power(2,sample['zoom']-1)
            pano = pano[0:panoheight,0:panowidth,:]                
        equ = E2P.Equirectangular(pano)

        for i,tetha in enumerate([0,-90,90,180]):
            snaps = []
            fov_shift = np.arange(-10,10,2)
            tetha_shift = np.arange(-10,10,2)
            pitch_shift = np.arange(-10,10,2)
            for k in range(10):
                snap = equ.GetPerspective(100 + fov_shift[k], tetha + tetha_shift[k], 0 + pitch_shift[k],new_h,new_w) / 255.0               
                snaps.append(snap)
            new_dict['y'+str(i)] = np.stack(snaps, 0)

        new_dict['label'] = np.repeat(sample['label'],10)
        return new_dict

class CenterCrop(object):
    def __init__(self, output_size, cfg):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.cfg = cfg

    def __call__(self, sample):
        new_h, new_w = self.output_size
        new_dict = dict()
        
        # Crop tiles
        tile = sample['x{}'.format(self.cfg.testZoom[0])] 
        h, w = tile.shape[:2]
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        new_tile = tile[top: top + new_h, left: left + new_w, :]
        new_tile = np.expand_dims(new_tile,0)
        new_dict['x'] = new_tile # Must have shape [1, 224,224,3]

        # Crop gsv 
        pano = sample['pano']
        if  pano.shape[0] < 512*np.power(2,sample['zoom']-1):
            panowidth = 416 * np.power(2,sample['zoom'])
            panoheight = 416 * np.power(2,sample['zoom']-1)
            pano = pano[0:panoheight,0:panowidth,:]                
        equ = E2P.Equirectangular(pano)

        for i,tetha in enumerate([0,-90,90,180]):
            snap = equ.GetPerspective(100, tetha, -10, new_h, new_w) / 255.0  
            new_dict['y'+str(i)] = np.expand_dims(snap, 0) # Must have shape [1, 224,224,3]

        new_dict['label'] = sample['label']
        #new_dict['row'] = sample['row'] 
        return new_dict

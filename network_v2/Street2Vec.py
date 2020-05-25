import os, sys
import numpy as np
import h5py 
import csv
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms
from torchsummary import summary 

from architecture import MyModel

# Losses
from softtriplet import Embeddings # Check carrefully this file

# sklearn 
from sklearn.decomposition import PCA

import cv2 
import matplotlib.pyplot as plt
import time
import copy 
import nomen 
import yaml
import pandas as pd 
import random
import pathlib

#from utils.utils import *
from Metrics import NumpyMetrics

# Plot
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits import mplot3d

#from tensorboardX import SummaryWriter
#from utils.visualize_CNN import Zeiler
#from utils.figures import MyPlotter


#class MyModelParallel(nn.DataParallel):
#    def __getattr__(self, name):
#        return getattr(self.module, name)

class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Street2Vec():
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.device = torch.device("cuda:0" if self.cfg.useGPU else "cpu")
        #self.model = MyModel(self.cfg)

        if torch.cuda.device_count() > 1:
            model = MyModel(self.cfg)
            #modelp = nn.DataParallel(model)
            print("Let's use ", torch.cuda.device_count(), "GPUs!") 
            self.model = _CustomDataParallel(model)

        self.model.to(self.device)
        self.criterion = Embeddings(cfg, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr) 

        # Color map for visualization
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_names = [name for name, _ in colors.items()]
        random.shuffle(color_names)
        self.color_names = color_names
        
        if os.path.isfile(os.path.join(self.cfg.modelPath, 'checkpoint.pth.tar')):

            checkpoint = torch.load(os.path.join(self.cfg.modelPath,'checkpoint.pth.tar'))
            state_dict = checkpoint['state_dict']
            
            if self.mode is 'train':
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                except:
                    a = input('incompatible output size ... change embedding dim to {} y/n?   '.format(cfg.embedding_dim))
                    if a == 'y':
                        compatible_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'embedding' not in k }
                        self.model.load_state_dict(compatible_dict, strict=False)
                    else:
                        sys.exit('Incompatible dimensions')

            else:
                self.model.load_state_dict(state_dict, strict=True)
            
            self.epoch = 0
            #self.epoch = checkpoint['epoch'] + 1
            self.best_recall = checkpoint['best_recall']
            print('best starting recall: ', self.best_recall)
        else:
            self.epoch = 0
            self.best_recall = 0.0

    def train(self, dataset, loader):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        metrics = NumpyMetrics()

        logBatchFile = os.path.join( self.cfg.modelPath, 'logbatch.csv')
        logStepFile = os.path.join( self.cfg.modelPath, 'logstep.csv')

        with open( logBatchFile, 'a', encoding='utf8') as f:
            f.write('{}\t{}\t{}\t{}\t{}\n'.format('phase', 'epoch', 'loss', 'recall', 'precision'))

        for epoch in range(self.epoch, self.cfg.numEpochs):
            print('Epoch {}/{}'.format(epoch, self.cfg.numEpochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train','val']:
                if phase == 'train':
                    self.model.train()
                    for param in self.model.gsv_model.parameters():
                        param.requires_grad = False
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_rank = 0.0
                running_recall = 0.0
                running_precision = 0.0

                # Iterate over data.
                for i, sample in enumerate(loader[phase]):
                    # All in GPU
                    #X18, X19, X20 = sample['x10'], sample['x20'], sample['x30'] # [batch,10,3,224,224]
                    
                    X = sample['x']
                    

                    #Yl, Yf, Yr, Yb = sample['y0'], sample['y1'], sample['y2'], sample['y3'] #[batch,10,3,224,224]
                    Yf, Yl, Yr, Yb = sample['y0'], sample['y1'], sample['y2'], sample['y3'] #[batch,10,3,224,224]

                    Label = sample['label'] #[batch, 10]
                    
                    # tiles to gpu and reshape
                    X = X.to(self.device)
                    X = X.view(-1,3,224,224)

                    Yf = Yf.to(self.device)
                    Yf = Yf.view(-1,3,224,224)
                    Yl = Yl.to(self.device)
                    Yl = Yl.view(-1,3,224,224)
                    Yr = Yr.to(self.device)
                    Yr = Yr.view(-1,3,224,224)                
                    Yb = Yb.to(self.device)
                    Yb = Yb.view(-1,3,224,224)                
                    Labels = Label.to(self.device).view(-1)
                
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Fordward pass for tiles
                        #X_out, Y_out = self.model.forward(X, Yl, Yf, Yr, Yb)
                        X_out, Y_out = self.model.forward(X, Yf, Yl, Yr, Yb)

                        #_, preds = torch.max(outputs, 1)
                        loss = self.criterion.batch_all(X_out, Y_out, Labels, 'mean')

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()

                            # 'clip_grap_norm' help prevent the exploding gradient problem in RNNs / LSTMs
                            torch.nn.utils.clip_grad_norm_(self.model.gsv_model.parameters(),0.5)
                            torch.nn.utils.clip_grad_norm_(self.model.tile_model.parameters(),0.5)
                            torch.nn.utils.clip_grad_norm_(self.model.embedding_model.parameters(),0.5)
                            self.optimizer.step()

                    # statistics
                    #average_X_Y_rank = rank(X_out,Y_out,Labels,False).sum().item() /Labels.size(0)
                    x = X_out.detach().cpu().data.numpy()
                    y = Y_out.detach().cpu().data.numpy()
                    labels = Labels.detach().cpu().data.numpy()
                    _rank = metrics.rank(y,x,labels,labels)
                    _recall = (_rank <= int(_rank.shape[0] * 0.01) ).sum() # Top 1 recall
                    _precision = metrics.precision_at_k(y,x,labels,labels,self.cfg.examples_by_location).sum()

                    running_loss += loss.item()
                    running_rank += _rank
                    running_recall += _recall 
                    running_precision += _precision

                    print('{} Loss: {:.6f} Recall: {:.3f} Precision: {:.3f}'.format(
                    i, loss.item(), _recall/_rank.shape[0], _precision/_rank.shape[0]), end="\r")

                    with open(logStepFile, 'a', encoding='utf8') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(phase, epoch, i, loss.item(), _recall/_rank.shape[0], _precision/_rank.shape[0]))

                    if ((i+1) % 10 == 0) and (self.cfg.pca == True): #and phase == 'train':
                        self.visualize_model(x,y,labels, block=False)

                num_batches = len(dataset[phase]) // loader[phase].batch_size
                epoch_loss = running_loss / num_batches
                epoch_recall = running_recall / (num_batches*_rank.shape[0])
                epoch_precision = running_precision / (num_batches*_rank.shape[0])

                print('{} Loss: {:.6f} Recall: {:.3f} Precision: {:.3f}'.format(
                    phase, epoch_loss, epoch_recall, epoch_precision))

                with open(logBatchFile, 'a', encoding='utf8') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(phase, epoch, epoch_loss, epoch_recall, epoch_precision))

                # deep copy the model
                if phase == 'val':
                    #visualize_model(x,y,labels, block=False)
                    if epoch_recall > self.best_recall:
                        self.best_recall = epoch_recall
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        save_dict = {
                            'epoch': epoch,
                            'loss': loss,
                            'state_dict': self.model.state_dict(),
                            'best_recall': self.best_recall
                        }
                        torch.save(save_dict, os.path.join(self.cfg.modelPath, 'checkpoint.pth.tar'))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best recall: {:2f}'.format(self.best_recall))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def visualize_model(self, x_out, y_out, labels, block = True):
        # plot PCA to visualize a batch
        data = np.concatenate((x_out, y_out), axis=0)
        labels = np.tile(np.repeat(np.arange(0,self.cfg.batchSize),self.cfg.examples_by_location),2)
        #labels = np.tile(labels, 2)
        pca = PCA(n_components=3)
        data = pca.fit_transform(data)
        
        markers = ['o'] * self.cfg.batchSize * self.cfg.examples_by_location
        markers.extend(['x'] * self.cfg.batchSize * self.cfg.examples_by_location )
        color = self.color_names[0:self.cfg.batchSize]

        #color = color_names[0:34]
        # 3d plot
        fig = plt.axes(projection='3d')
        for i in range(self.cfg.batchSize*self.cfg.examples_by_location*2):
            fig.scatter3D([data[i,0]],[data[i,1]], [data[i,2]],alpha=0.8,c=color[labels[i]], marker=markers[i])
        #plt.savefig('model_visualization.png')        
        plt.draw()
        plt.pause(0.001)
        if block is True:
            input("Press [enter] to continue ...")

    def predict(self, dataset, loader, datasetName):
        """
            This method computes the descriptors of all pairwise points in specified dataset.
            Results are save in three formats: hdf5, npz, and mat.
        """
        self.model.eval()


        # -------------First part save results in an HDF5 file --------------------------------
        # Create a hdf5 file to store results
        hdf5_file = os.path.join(self.cfg.modelPath, 'z'+str(self.cfg.testZoom[0]),datasetName + '.hdf5')
        print('Predicting {} dataset and saving hdf5 file in path {}'.format( datasetName, hdf5_file))
        f = h5py.File(hdf5_file, 'w')
        
        for i, sample in enumerate(loader):
            # All in GPU
            print("Predicting batch ", i, end='\r')
            #X18, X19, X20 = sample['x10'], sample['x20'], sample['x30'] # [batch,10,3,224,224]
            X = sample['x']
            Yf, Yl, Yr, Yb = sample['y0'], sample['y1'], sample['y2'], sample['y3'] #[batch,10,3,224,224]
            Label = sample['label'] #[batch, 10]
            
            # tiles to gpu and reshape
            X = X.to(self.device)
            X = X.view(-1,3,224,224)

            Yf = Yf.to(self.device)
            Yf = Yf.view(-1,3,224,224)
            Yl = Yl.to(self.device)
            Yl = Yl.view(-1,3,224,224)
            Yr = Yr.to(self.device)
            Yr = Yr.view(-1,3,224,224)                
            Yb = Yb.to(self.device)
            Yb = Yb.view(-1,3,224,224)                
            Label = Label.to(self.device).view(-1)           

            # Predict and save in a numpy array
            with torch.set_grad_enabled(False):
                # Fordward pass
                X_out, Y_out = self.model.forward(X, Yf, Yl, Yr, Yb)

            # statistics
            x = X_out.detach().cpu().data.numpy()
            y = Y_out.detach().cpu().data.numpy()
            label = Label.detach().cpu().data.numpy()
            
            for k in range(x.shape[0]):
                xpath = str(label[k].item()) + '/x'
                ypath = str(label[k].item()) + '/y'  

                f.create_dataset(xpath, data=x[k,:], dtype=float)
                f.create_dataset(ypath, data=y[k,:], dtype=float)
        
        f.close()


        # ----Second part save results in numpy and mat format
        
        print("Saving .mat and .npz files in {}".format(self.cfg.modelPath))
        names = ["pano_id", "yaw", "lat", "lon", "city", "neighbor", "bearing", "index"]
        location_file = os.path.join( 'data', datasetName + '.csv')
        df = pd.read_csv(location_file, names=names)
        n = len(df)
        print('Total number of points to be saved: ', n)
        
        X = np.zeros((n,self.cfg.embedding_dim), dtype=float)
        Y = np.zeros((n,self.cfg.embedding_dim), dtype=float)
        loc_ids = np.zeros((n), dtype=int)

        f = h5py.File(hdf5_file, 'r')
        for i in range(n):
            loc_ids[i] = df.loc[i,'index']    
            X[i] = f[str(loc_ids[i])+'/x']
            Y[i] = f[str(loc_ids[i])+'/y']

        f.close()
        df_dict = df.to_dict(orient='list')
        df_dict['X'] = X
        df_dict['Y'] = Y

        mat_filename = os.path.join(self.cfg.modelPath, 'z'+str(self.cfg.testZoom[0]), datasetName + '.mat')
        np_filename = os.path.join(self.cfg.modelPath, 'z'+str(self.cfg.testZoom[0]) ,datasetName + '.npz')
        sio.savemat(mat_filename, df_dict)
        np.savez(np_filename, X=X, Y=Y, loc_ids=loc_ids) #loc_id is index
        print("All finished")

    # def evaluate(self):
    #     plotter = MyPlotter(self.cfg)
    #     plt.ion()
    #     plt.show()
    #     plotter.histogram()
    #     #plotter.PCA()

    #     val_filename = os.path.join(self.cfg.modelPath, datasetName + '.hdf5') 
    #     frame = pd.read_csv(self.cfg.csv_file_val)
    #     x,y,loc_ids = read_data(val_filename, frame, self.cfg.val_size)
    #     print('Testing on {} points'.format(x.shape[0]))
    #     mtr = NumpyMetrics()
    #     #rank_x_y = mtr.rank(x,y,loc_ids, loc_ids)
    #     rank_y_x = mtr.rank(y,x,loc_ids, loc_ids)
    #     top1_recall = ( rank_y_x < int( self.cfg.val_size * 0.01) ).sum() * 100 / self.cfg.val_size
    #     print('Top 1 "%" recall', top1_recall)
    #     print('Average y-x rank: ', rank_y_x.mean())

    def summary(self):
        summary(self.model, [(3,224,224),(3,224,224),(3,224,224),(3,224,224),(3,224,224)])

    # def figures(self):
    #     plotter = MyPlotter(self.cfg)
    #     #plotter.dimensions()
    #     plotter.recall_curve_both_zooms('Model 1')
    #     #plotter.recall_curve_both_models('z19')
    #     #plotter.recall_curve_individual('Model 2', 'z20')

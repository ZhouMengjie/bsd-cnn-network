import os, sys 
sys.path.append( os.path.join(os.environ['dev'], 'CVLoc') )
import pandas as pd 
import numpy as np
import scipy.io as sio 
import ast

def getFrame(config):
    path = os.path.join('data/', config['dataset'] + '_roadlabels.csv')
    frame = pd.read_csv(path)
    return frame

def SaveFlagsMatFile(config):
    frame = getFrame(config)
    highways = frame['highway']
    filterList = ['motorway', 'motorway_link']

    flags = np.zeros(shape=(5000,), dtype=int)

    for i, row_string in enumerate(highways):
        row_list = ast.literal_eval(row_string)    
        for way in row_list:
            if way in filterList:
                flags[i] = 1

    indices = np.argwhere( flags == 1)
    filename = os.path.join(os.environ['dev'], 'route-finder/Data/streetlearn', config['dataset'] + '_highwayflags.mat')
    sio.savemat(filename, {'highway_flag': flags})
    print('Number of points to exclude from routes: {} porcentage {}'.format(indices.size, indices.size/5000))

def saveDiscardedDataFrame(config):
    frame = getFrame(config)
    # Get the flags array
    filename = os.path.join(os.environ['dev'], 'route-finder/Data/streetlearn', config['dataset'] + '_highwayflags.mat')
    flags = sio.loadmat(filename)['highway_flag'].reshape(-1)
    discardedPointIndices = np.argwhere(flags == 1).reshape(-1)
    discarded_df = frame.loc[discardedPointIndices, :]
    fileName = os.path.join( os.environ['dev'], 'CVLoc/data',  config['dataset'] + '_discarded.csv')
    discarded_df.to_csv(fileName, index=False)    

config = {
    'dataset': 'wallstreet5k'
}

frame = getFrame(config)
#SaveFlagsMatFile(config, frame)
saveDiscardedDataFrame(config)





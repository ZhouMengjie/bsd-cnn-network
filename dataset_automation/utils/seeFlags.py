import os, sys 
sys.path.append( os.path.join(os.environ['dev'], 'CVLoc') )
import pandas as pd 
import numpy as np
import scipy.io as sio 
from Notebooks.commun import getDataFrame, getRoutesMatrix, getHighwayFlags
import ast


config = {
    'dataset': 'wallstreet5k',
    'm': 20
}

path = os.path.join('data/', config['dataset'] + '_roadlabels.csv')
frame = pd.read_csv(path)

routes = getRoutesMatrix(config)
route = routes[9]
flags = getHighwayFlags(config)

for pointIndex in route:
    highway = frame.loc[pointIndex - 1, 'highway']
    flag = flags[pointIndex -1 ]
    print(pointIndex, highway, flag)








# highways = frame['highway']
# filterList = ['motorway', 'motorway_link']

# flags = np.zeros(shape=(5000,), dtype=int)

# for i, row_string in enumerate(highways):
#     row_list = ast.literal_eval(row_string)    
#     for way in row_list:
#         if way in filterList:
#             flags[i] = 1

# indices = np.argwhere( flags == 1)
# filename = os.path.join(os.environ['dev'], 'route-finder/Data/streetlearn', config['dataset'] + '_highwayflags.mat')
# sio.savemat(filename, {'highway_flag': flags})
# print('Number of points to exclude from routes: {} porcentage {}'.format(indices.size, indices.size/5000))

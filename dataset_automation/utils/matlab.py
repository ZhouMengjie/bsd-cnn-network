import os, sys 
import pandas as pd 
import scipy.io as sio

def convertTXT2mat():
    """
    Converts txt metadata files from streeatlearn dataset to matfiles.
    streetlearn dataset should be in $datasets/streetlearn path
    Mat files are saved in same directory as the txt files
    """

    dirPath = os.path.join( os.environ['datasets'], 'streetlearn')
    cities = ['manhattan', 'pittsburgh']

    for city in cities:
        # nodes frame
        nodesFile = os.path.join(dirPath, 'jpegs_' + city +'_2019', 'nodes.txt')
        names = ["pano_id", "yaw", "lat", "lon"]
        nodesFrame = pd.read_csv(nodesFile, names=names)

        # links frame
        linksFile = os.path.join(dirPath, 'jpegs_' + city +'_2019', 'links.txt')
        names = ["src", "bearing", "dst"]
        linksFrame = pd.read_csv(linksFile, names=names)

        # Convert pandas frames to dictionaries and save them as mat files 

        nodes_dict = nodesFrame.to_dict(orient='list')
        filename = dirPath = os.path.join( os.environ['datasets'], 'streetlearn', city + '_nodes.mat')
        sio.savemat(filename, nodes_dict)

        links_dict = linksFrame.to_dict(orient='list')
        filename = dirPath = os.path.join( os.environ['datasets'], 'streetlearn', city + '_links.mat')
        sio.savemat(filename, links_dict)

def roadFlag2MAT(path):
    frame = pd.read_csv(path)
    subframe = frame.loc[frame['highway_flag'] == 1, :]
    filename = 'temp.csv'
    subframe.to_csv(filename)
    subframe = frame.loc[:,'highway_flag']
    flags = subframe.values
    filename = 'hudosnriver5k_highwayflags.mat'
    sio.savemat(filename, {'highway_flag' : flags})


filename = 'data/hudsonriver5k_roadlabels.csv'
roadFlag2MAT(filename)
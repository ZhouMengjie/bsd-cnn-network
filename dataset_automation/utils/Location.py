import os, sys
import numpy as np 
import pandas as pd
import cv2 
sys.path.append(os.path.join( os.environ['dev'],'CVLoc'))
from utils.Tile import Tile
from utils.Pano import Pano

class Location(Tile, Pano):
    def __init__(self, dataset, ID, city, base_index='local'):
        
        self.datasetPath = os.path.join( os.environ['datasets'], 'streetlearn')
        self.city = city
        
        tileDir = os.path.join( self.datasetPath, 'tiles_' + city + '_2019') # pano Directory
        panoDir = os.path.join( self.datasetPath, 'jpegs_' + city + '_2019') # pano Directory

        if type(ID) is str:
            pano_id = ID
        
        elif type(ID) is int and base_index == 'global':
            filename = os.path.join( panoDir, 'nodes.txt' )
            names = ["pano_id", "yaw", "lat", "lon"]
            frame = pd.read_csv(filename, names=names)
            pano_id = frame.loc[ID-1, 'pano_id']

        elif type(ID) is int and base_index == 'local':
            filename = os.path.join( 'data', dataset + '.csv' )
            names = ["pano_id", "yaw", "lat", "lon", "city", "neighbor", "bearing", "index", "loc_id"]
            frame = pd.read_csv(filename, names=names)
            pano_id = frame.loc[ID-1, 'pano_id']
        
        else:
            sys.exit("Pano ID not found")
                   
        Tile.__init__(self, tileDir, pano_id + '.png')
        Pano.__init__(self, panoDir, pano_id + '.jpg')
        self.pano_id = pano_id
        self.neighbors, self.bearings = self.getNeighbours()

    def getLocationWithInfo(self, zoom=18, size=224, colour=(255,255,255)):
        snaps = self.getSnapswithInfo(size=size, colour=colour)
        tile = self.getTilewithInfo(zoom=zoom, size=size, colour=colour)
        snaps.append(tile)
        img = np.concatenate(snaps, axis=1)
        return img
    
    def showLocation(self):
        img = self.getLocationWithInfo()
        cv2.imshow("Location", img)
        cv2.waitKey(0)


    def getNeighbours(self):
        linkFile = os.path.join( self.datasetPath, 'jpegs_' + self.city + '_2019', 'links.txt')
        names = ['pano_id', 'bearing', 'neighbor']
        frame = pd.read_csv(linkFile, names=names)
        subframe = frame[ frame['pano_id'] == self.pano_id ]
        neighbors = subframe['neighbor'].to_list()
        bearings = subframe['bearing'].to_list()
        return neighbors, bearings

if __name__ == "__main__":
    loc = Location("hudsonriver5k", 'PreXwwylmG23hnheZ__zGw', 'manhattan', base_index='local')
    loc.showLocation()

import os 
import sys 
import cv2 
import numpy as np
#import pandas as pd 
from Location import Location 

#from commun import getDataFrame

class Route():
    
        """ A Route is a list of nodes"""
    
    def __init__(self, dataset, city, nodes=[], base_index='local'):
        """Create a route"""
        self.dataset = dataset
        self.city = city
        self.nodes = nodes
        self.base_index = base_index


    def showRoute(self, option='single'):
        """ Show a route by concatenating map tiles """
        if option == 'single':
            for node in self.nodes:
                loc = Location(self.dataset, node, self.city,base_index=self.base_index)
                loc.showLocation()
        
        else:
            tiles = []
            for node in self.nodes:
                loc = Location(self.dataset, node, self.city,base_index=self.base_index)
                tile = loc.getLocationWithInfo()
                tiles.append(tile)
            route = np.concatenate(tiles, axis=0)
            cv2.imshow("route", route)
            cv2.waitKey(0)


if __name__ == "__main__":
    nodes = [4142,4067,3996,4068,4143]
    route = Route('unionsquare5k', 'manhattan', nodes=nodes)
    route.showRoute(option='single')
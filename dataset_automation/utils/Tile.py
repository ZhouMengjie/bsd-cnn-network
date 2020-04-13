import os, sys
import numpy as np
import pandas as pd
import cv2 


class Tile():
    "A tile is a png image with matlab index ID stored in tileDir with the same name as the panorama asociated"

    def __init__(self, tileDir, ID):
        """ Read the tile specified by ID
            if ID is an int then it is assumed to be the global matlab index of the location
            if ID is a string then it is assumed to be the name of the tile file
            """
        self.tileDir = tileDir
        
        if type(ID) is str:
            self.tileName = ID
        
        elif type(ID) is int:
            filename = os.path.join( self.tileDir, 'nodes.txt' )
            names = ["pano_id", "yaw", "lat", "lon"]
            frame = pd.read_csv(filename, names=names)
            pano_id = frame.loc[ID-1, 'pano_id']
            self.tileName = pano_id + '.png'
        
        else:
            sys.exit("Pano ID not found")

        self.index, self.lat, self.lon, self.yaw = self.getCoordinates()
    
    def __str__(self):
        return "Tile centered at ({},{})".format(self.lat, self.lon)

    def getCoordinates(self):
        """Get the coordinates of the tile"""
        filename = os.path.join( self.tileDir, 'nodes.txt' )
        names = ["pano_id", "yaw", "lat", "lon"]
        frame = pd.read_csv(filename, names=names)
        row = frame.loc[frame['pano_id'] == self.tileName.split('.')[0]]
        index = row.index[0]
        yaw, lat, lon = row['yaw'].values[0], row['lat'].values[0], row['lon'].values[0]
        return (index, lat, lon, yaw)
    
    def getTile(self, zoom=19):
        """Returns tile image with specified zoom"""
        path = os.path.join( self.tileDir, 'z'+str(zoom), self.tileName)
        img = cv2.imread(path)
        return img

    def showTile(self):
        """Shows the tile and wait for a key to be pressed"""
        window_name = "({},{})".format(self.lat, self.lon)
        cv2.imshow(window_name, self.getTilewithInfo())
        cv2.waitKey(0)

    def getTilewithInfo(self, zoom=19, size=256, colour=(255,255,255), text=True):
        """ Get tiles with an index and zoom label. Also it shows an arrow to indicate heading direction"""
        thick = int(0.05 * size) # Thickness is 5 % of size
        tile = self.getTile(zoom=zoom)
        tile = cv2.resize(tile, (size, size))
        tile = cv2.copyMakeBorder(tile, thick,thick,thick,thick, cv2.BORDER_CONSTANT, None, colour)
        #text = '({},{})'.format(self.lat, self.lon)          
        if text == True:
            text = 'ID: {}'.format(self.index + 1) #Show ID
            cv2.putText(tile, text, (10,size), cv2. FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.arrowedLine(tile, (size//2, size//2), ((size//2, size//2 - 15)), (255,0,0), 1, cv2.LINE_4, 0, 0.4)
        return tile

if __name__ == "__main__":
    tileDir = os.path.join(os.environ['datasets'], 'streetlearn', 'tiles_manhattan_2019')
    tile = Tile(tileDir, 53160) 
    tile.showTile()

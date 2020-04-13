import os, sys 
import numpy as np 
import cv2
import pandas as pd
sys.path.append(os.path.join(os.environ['dev'], 'Equirec2Perspec'))
import Equirec2Perspec as E2P

class Pano():
    """ Create object panorama"""
    def __init__(self, panoDir, ID):
        self.panoDir = panoDir
        
        if type(ID) is str:
            self.panoName = ID
        
        elif type(ID) is int:
            filename = os.path.join( self.panoDir, 'nodes.txt' )
            names = ["pano_id", "yaw", "lat", "lon"]
            frame = pd.read_csv(filename, names=names)
            pano_id = frame.loc[ID-1, 'pano_id']
            self.panoName = pano_id + '.jpg'

        else:
            sys.exit("Pano ID not found")

        self.path = self.getPath()
        self.pano = self.getPano()
        self.shape = self.pano.shape
        try:
            self.index, self.lat, self.lon, self.yaw = self.getCoordinates()
        except:
            self.index, self.lat, self.lon, self.yaw = None, None, None, None

    def getPath(self):
        path = os.path.join( self.panoDir, self.panoName)
        return path

    def getPano(self):
        pano = cv2.imread(self.path)
        pano = cv2.resize(pano, (1024,512))
        return pano
    
    def showPano(self):
        cv2.imshow(self.panoName, self.pano)
        cv2.waitKey(0)

    def getZoom(self):
        """Returns pano's zoom level"""
        return int(np.ceil(self.pano.shape[0] / 512))

    def getSnaps(self, size=224):
        """ Returns a list with snaps in directions 0, 90, -90, 180"""
        snaps = []
        equ = E2P.Equirectangular(self.pano)
        views = [0,-90,90,180]
        snaps = [equ.GetPerspective(100, t, 0.0, size, size) for t in views]
        return snaps

    def getSnapswithInfo(self, size=224, colour = (255,255,255), text=True):
        """ Returns a list with snaps in directions 0, 90, -90, 180"""
        thick = int(0.05 * size) # Thichness is 5 % 
        snaps = self.getSnaps(size)
        snaps = [cv2.copyMakeBorder(snap, thick,thick,thick,thick, cv2.BORDER_CONSTANT, None, colour) for snap in snaps] 
        directions = ['F', 'L', 'R', 'B']
        if text == True:
            for i, direction in enumerate(directions):
                text = 'ID: ' + str(self.index + 1) + ' ' + direction
                cv2.putText(snaps[i], text, (10,size), cv2. FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
        return snaps 

    def cropPano(self, fov, theta, pitch, h, w):
        """ Returns a pano the the specified parameters"""
        equ = E2P.Equirectangular(self.pano)
        snap = equ.GetPerspective(fov, theta, pitch, h, w)
        return snap

    def saveSnaps(self, size=224, directory=None, option='group'):
        savedir = os.getcwd() if directory == None else directory
        basename = os.path.join(savedir, self.panoName.split('.')[0]) 

        if option == 'group':
            snaps = self.getSnapswithInfo(size=size, text=False)
            row1 = np.concatenate([snaps[0], snaps[2]], axis=1) # FR
            row2 = np.concatenate([snaps[3], snaps[1]], axis=1) # BL
            image = np.concatenate([row1, row2], axis=0)
            filename = basename + '.jpg'
            cv2.imwrite(filename, image)    

        elif option == 'individual':
            snaps = self.getSnapswithInfo(size=size, text=False)
            directions = ['F', 'L', 'R', 'B']
            for i, snap in enumerate(snaps):
                direction = directions[i]
                filename = basename + '_' + direction + '.jpg'
                cv2.imwrite(filename, snap)
        else:
            print("Option not found, image not saved")

    def getCoordinates(self):
        filename = os.path.join( self.panoDir, 'nodes.txt' )
        names = ["pano_id", "yaw", "lat", "lon"]
        frame = pd.read_csv(filename, names=names)
        row = frame.loc[frame['pano_id'] == self.panoName.split('.')[0]]
        index = row.index[0]
        yaw, lat, lon = row['yaw'].values[0], row['lat'].values[0], row['lon'].values[0]
        return (index, lat, lon, yaw)

    def __str__(self):
        return "Pano name: {}, shape: {}, coordinates: ({},{},{})".format(self.panoName, self.pano.shape, self.lat, self.lon, self.yaw)

if __name__ == "__main__":

    panoDir = os.path.join( os.environ['datasets'],'streetlearn', 'jpegs_manhattan_2019')
    pano = Pano(panoDir, 11796)
    snaps = pano.getSnapswithInfo(size=256)
    #pano.saveSnaps(size=256, directory=None, option='individual')
    img = np.concatenate(snaps, axis=1)
    cv2.imshow("pano", img)
    cv2.waitKey(0)

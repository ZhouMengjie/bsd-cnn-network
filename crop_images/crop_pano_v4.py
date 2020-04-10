import numpy as np
import os, sys
import cv2
import pandas as pd
import Equirec2Perspec as E2P
from scipy.io import loadmat
import time

def crop_pano(pano):
    # Crop gsv 
    zoom = int(np.ceil(pano.shape[0] / 512))
    snaps = []
    if  pano.shape[0] < 512*np.power(2,zoom-1):
        panowidth = 416 * np.power(2,zoom)
        panoheight = 416 * np.power(2,zoom-1)
        pano = pano[0:panoheight,0:panowidth,:]                
    equ = E2P.Equirectangular(pano)

    for i,tetha in enumerate([0,-90,90,180]):
        snap = equ.GetPerspective(90, tetha, 0, 224, 224)  
        snaps.append(snap) # Must have shape [1, 224,224,3]
    return snaps 

start = time.clock()
if __name__ == '__main__':
    directory = os.getcwd()
    dataset = 'unionsquare5k'
    snaps_directory = os.path.join(directory, 'images', dataset) 
    if not os.path.isdir(snaps_directory):
        os.mkdir(snaps_directory)

    snaps_directory_jc = os.path.join(directory, 'images', dataset, 'junctions') 
    snaps_directory_njc = os.path.join(directory, 'images', dataset, 'non_junctions') 
    snaps_directory_bd = os.path.join(directory, 'images', dataset, 'gaps') 
    snaps_directory_nbd = os.path.join(directory, 'images', dataset, 'non_gaps') 
    if not os.path.isdir(snaps_directory_jc):
        os.mkdir(snaps_directory_jc)

    if not os.path.isdir(snaps_directory_njc):
        os.mkdir(snaps_directory_njc)

    if not os.path.isdir(snaps_directory_bd):
        os.mkdir(snaps_directory_bd)

    if not os.path.isdir(snaps_directory_nbd):
        os.mkdir(snaps_directory_nbd)

    # Open csv file
    filename = os.path.join(os.getcwd(), 'data', dataset + '.csv')
    names = ["pano_id", "gsv_lat", "gsv_lon", "gsv_yaw", "front", "right", "back", "left", "city"]
    routes = pd.read_csv(filename,names=names)

    # Read all the pano_ids and lables to process
    for i in range(routes.shape[0]):
        row = routes.loc[i, :]
        pano_id = row['pano_id']
        front = int(row['front'])
        right = int(row['right'])
        back = int(row['back'])
        left = int(row['left'])

        path = os.path.join(os.getcwd(), 'images', 'jpegs_manhattan_2019', pano_id + '.jpg')
        if not os.path.exists(path):
            path = os.path.join(os.getcwd(), 'images', 'jpegs_pittsburgh_2019', pano_id+'.jpg')

        pano = cv2.imread(path)
        if front == 1:
            filename_front = os.path.join(snaps_directory_jc, pano_id + '_front.jpg') 
        else:
            filename_front = os.path.join(snaps_directory_njc, pano_id + '_front.jpg') 
            
        if back == 1:
            filename_back = os.path.join(snaps_directory_jc, pano_id + '_back.jpg')
        else:
            filename_back = os.path.join(snaps_directory_njc, pano_id + '_back.jpg')
            
        if left == 1:
            filename_left = os.path.join(snaps_directory_bd, pano_id + '_left.jpg') 
        else:
            filename_left = os.path.join(snaps_directory_nbd, pano_id + '_left.jpg') 

        if right == 1:
            filename_right = os.path.join(snaps_directory_bd,pano_id + '_right.jpg') 
        else:
            filename_right = os.path.join(snaps_directory_nbd,pano_id + '_right.jpg') 

        if os.path.isfile(filename_front):
            msg = "Exists..."    
        else:
            snaps = crop_pano(pano)
            cv2.imwrite(filename_front, snaps[0]) 
            # cv2.imshow('snaps0', snaps[0])
            # cv2.waitKey(0)
            cv2.imwrite(filename_left, snaps[1]) 
            cv2.imwrite(filename_right, snaps[2]) 
            cv2.imwrite(filename_back, snaps[3])    
            msg = filename_front + " saved"

        print(pano_id, msg)

elapsed = (time.clock() - start)
print(elapsed)
print("All finished")
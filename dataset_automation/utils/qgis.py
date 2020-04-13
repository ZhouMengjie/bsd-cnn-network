import os, sys 
import csv 
import cv2
import numpy as np
import pandas as pd

sys.path.append('/home/os17592/dev/CVLoc')
from utils.Pano import Pano 

def generateCSV(dataset):
    datasetPath = os.path.join(os.environ['datasets'], dataset)
    csvPath = os.path.join(datasetPath, 'locations.csv')
    with open(csvPath, 'w') as csvFile:
        writer = csv.writer(csvFile)
        # Lets write the following fields [name, lat, lon]
        header = ['name', 'lat', 'lon', 'time', 'path']
        writer.writerow(header)
        for name in os.listdir(datasetPath):
            if name.split('.')[1] == 'JPG' or name.split('.')[1] == 'jpg':
                try:
                    pano = Pano(dataset, name)
                    values = [pano.name, pano.lat, pano.lon, pano.getTimeStamp(), name]
                    writer.writerow(values)
                except:
                    print("Error with image: ", name)
    print("Done! Locations file written to {}".format(csvPath))

def cropImages(dataset, size=224):
    datasetPath = os.path.join(os.environ['datasets'], dataset)
    savePath = os.path.join(os.environ['datasets'], dataset + '_snaps')
    os.mkdir(savePath)
    
    for name in os.listdir(datasetPath):
        if  (name.split(' ')[-1] == 'JPG' or name.split('.')[-1] == 'jpg') and name[-5] == 'E':
            try:
                pano = Pano(dataset, name)
                snaps = pano.getSnapswithInfo(size=size)
                # concatenate snaps
                img = np.concatenate(snaps, axis=1)
                filename = os.path.join(savePath, name)
                cv2.imwrite(filename, img)
            except:
                print("Error in image: ", name)

    print("Done! Cropped images written to {}".format(savePath))

def generateCSV_v2(dataset):
    datasetPath = os.path.join(os.environ['datasets'], dataset)
    csvPath = os.path.join(datasetPath, 'locations.csv')
    with open(csvPath, 'w') as csvFile:
        writer = csv.writer(csvFile)
        # Lets write the following fields [name, lat, lon, 'time', 'path']
        header = ['name', 'lat', 'lon', 'time', 'path']
        writer.writerow(header)
        for name in os.listdir(datasetPath):
            # Split the name by space character
            if  (name.split('.')[-1] == 'JPG' or name.split('.')[-1] == 'jpg') and name[-5] == 'E':
                try:
                    pano = Pano(dataset, name ) # Pano class requires the dataset and the name 
                    values = [pano.name, pano.lat, pano.lon, pano.getTimeStamp(), name]
                    writer.writerow(values)
                except:
                    print("There was an error with ", name)
    print("Done! Locations file written to {}".format(csvPath))

def findNearest(dataset):
    from bisect import bisect, bisect_left #operate as sorted container
    datasetPath = os.path.join(os.environ['datasets'], dataset)
    csvPath1 = os.path.join(datasetPath, 'locations.csv')
    csvPath2 = os.path.join(datasetPath, 'locationsiphone.csv')
    frame1 = pd.read_csv( csvPath1 )
    frame2 = pd.read_csv( csvPath2 )

    
    #dt = pd.to_datetime('2020-01-20 13:02:40.000 +0000')
    timestamps = pd.to_datetime(np.array(frame2['loggingTime']))
    s = sorted(timestamps)
    frame3 = frame1

    for i in range(len(frame1)):
        timeStamp = frame1.loc[i, 'time'].split(' ')
        timeStamp[0] = timeStamp[0].replace(':','-')
        timeStamp = ' '.join(timeStamp) + '.000 +0000'
        #print(timeStamp)
        dt = pd.to_datetime( timeStamp )
        index = bisect_left(s, dt)
        nearest = min(s[max(0, index-1): index+2], key=lambda t: abs(dt - t))
        print(index-1, dt, nearest)

        # create 
        frame3.loc[i,'lat'] = frame2.loc[index-1,'locationLatitude(WGS84)']
        frame3.loc[i,'lon'] = frame2.loc[index-1,'locationLongitude(WGS84)']
    
    csvPath3 = os.path.join(datasetPath, 'locations3.csv')
    frame3.to_csv(csvPath3)


if __name__ == "__main__":
    #findNearest('ricoh_test4')
    #generateCSV('gsv_app_22_01_20')
    cropImages('gsv_app_22_01_20')

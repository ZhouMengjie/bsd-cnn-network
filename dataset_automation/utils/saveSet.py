import os
import scipy.io as sio 
import folium 
import pandas as pd 

dirPath = os.path.join( os.environ['datasets'], 'streetlearn', 'oursubregions')
city = 'manhattan'
area = 'wallstreet5k'

def saveTestSet(city, area):
    # Load indices from mat file 
    matFile = os.path.join(dirPath, city + '_' + area + '.mat')
    content = sio.loadmat(matFile)['routes']
    columns = ['pano_id', 'yaw', 'lat', 'lon', 'city', 'neighbor', 'bearing', 'index']
    frame = pd.DataFrame(columns=columns)

    for rowidx in range(0, content.shape[1]):
        row = content[0,rowidx] # For extract row
        data = {
            'pano_id' : row[0][0],
            'lat': row[1][0][0],
            'lon': row[1][0][1],
            'yaw' : row[2][0][0],
            'neighbor' : row[3],
            'bearing' : row[4],
            'city': city,
            'index' : row[5][0][0]
        }
        frame = frame.append(data, ignore_index=True)
        print(rowidx, end="\r")
    # # Save data in new dataframe
    filename = os.path.join( dirPath, city + '_' +area + '.csv')    
    frame.to_csv(filename, index=False, header=False)

def saveTrainSet():
    # Load indices from mat file 
    matFile = os.path.join(dirPath, 'trainstreetlearn.mat')
    content = sio.loadmat(matFile)['routes']
    columns = ['pano_id', 'yaw', 'lat', 'lon', 'city', 'neighbor', 'bearing', 'index']
    frame = pd.DataFrame(columns=columns)

    for rowidx in range(0, content.shape[1]):
        row = content[0,rowidx] # For extract row
        data = {
            'pano_id' : row[0][0],
            'lat': row[1][0][0],
            'lon': row[1][0][1],
            'yaw' : row[2][0][0],
            'neighbor' : row[3],
            'bearing' : row[4],
            'city': row[6][0],
            'index' : row[5][0][0]
        }
        frame = frame.append(data, ignore_index=True)
        print(rowidx, end="\r")

    filename = os.path.join( dirPath, 'train.csv')    
    frame.to_csv(filename, index=False, header=False)

saveTrainSet()
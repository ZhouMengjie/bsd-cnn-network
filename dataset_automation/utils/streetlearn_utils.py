import os, sys
import numpy as np 
import pandas as pd
import cv2 
sys.path.append('/home/os17592/dev/CVLoc')
from utils.Tile import Tile
from utils.Pano import Pano


city = 'manhattan'
datasetPath = os.path.join( os.environ['datasets'] , 'streetlearn')

filename = os.path.join(datasetPath, 'jpegs_' + city + '_2019', 'nodes.txt')
names = ["pano_id","yaw","lat","lon"]
frame = pd.read_csv(filename, names=names)
frame['city'] = city


# subframes, bbox = split_by_latitude(frame, sections=3) # A list of tuples (min, max)
# for i, df in enumerate(subframes):
#     print(bbox[i])
#     print(i, len(df))

# conv = 40.7540
# range1 = (40.701149, conv)
# range2 = (conv,40.76569175)
# range3 = (40.76569175, 40.787206)

range1 = (40.701149, 40.717400)
range2 = (40.717400, 40.729400)
range3 = (40.729400, 40.787206)
bbox = [range1, range2, range3]

subframes = split_by_latitude_manually(frame, bbox) # A list of tuples (min, max)

for i, df in enumerate(subframes):
    print(bbox[i])
    print(i, len(df))
    name = os.path.join(datasetPath, 'ny{}.csv'.format(i))
    df.to_csv(name, index=False, header=False)

# for i in range(0,len(frame)):
#     pano_id = frame.loc[i, 'pano_id']
#     loc = Location( datasetPath, pano_id + '.jpg', city)
#     loc.showLocationinDir()
#     if i == 10:
#         sys.exit("ya")


city = 'pittsburgh'
datasetPath = os.path.join( os.environ['datasets'] , 'streetlearn')

filename = os.path.join(datasetPath, 'jpegs_' + city + '_2019', 'nodes.txt')
names = ["pano_id","yaw","lat","lon"]
frame = pd.read_csv(filename, names=names)
frame['city'] = city
name = os.path.join(datasetPath, 'pittsburgh.csv')
frame.to_csv(name, index=False, header=False)
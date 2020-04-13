import os
import sys
sys.path.append(os.path.join(os.environ['dev'], 'CVLoc'))
import cv2
import pandas as pd 
import numpy as np
from Location import Location 
from commun import getDataFrame


def saveImageList(index, img_list, save_Dir='images/'):
    for i, img in enumerate(img_list):
        name = 'idx_{}_{}.jpg'.format(idx, i)
        fileName = os.path.join(saveDir, name) 
        cv2.imwrite(fileName, img)

def savePano(config, matlab_index, option):
    index = matlab_index - 1
    datasetPath = os.path.join(os.environ['datasets'], 'streetlearn')
    savedir = os.path.join(os.environ['dev'], 'CVLoc', 'images_and_maps')
    frame = getDataFrame(config)
    pano_id = frame.loc[index, 'pano_id']
    city = frame.loc[index, 'city']
    loc = Location(config['dataset'], pano_id, city)
    loc.saveSnaps(size=224, directory=savedir, option=option)


def saveRoute(config, route_indices, name, size=128):
    datasetPath = os.path.join(os.environ['datasets'], 'streetlearn')
    savedir = os.path.join(os.environ['dev'], 'CVLoc', 'images_and_maps')
    frame = getDataFrame(config)
    
    route_images = []
    for matlab_idx in route_indices:
        index = matlab_idx -1
        pano_id = frame.loc[index, 'pano_id']
        city = frame.loc[index, 'city']
        snaps = Location(config['dataset'], pano_id, city).getSnapswithInfo(size=size, text=False)
        row1 = np.concatenate([snaps[0], snaps[2]],axis=1)
        row2 = np.concatenate([snaps[3], snaps[1]],axis=1)
        loc_img = np.concatenate([row1, row2], axis=0)
        route_images.append(loc_img)

    image = np.concatenate(route_images, axis=1)
    filename = os.path.join(savedir, name)
    cv2.imwrite(filename, image)


# #locations = [4858, 4871, 4789, 4707, 4626, 4545, 4466, 4385, 4307, 4226] #Matlab indices
# #locations = [911,946,981,1019,1057,1094,1133,1174,1218,1262]

config={
    'dataset': 'hudsonriver5k',
    'model': 'v1',
    'zoom': 'z18'
}


# save panos given index 
#pano_id = frame.iloc[idx-1, 'pano_id']
#Pano

#savePano(config, matlab_index=746, option='group')

################################################
#Save routes

# name = 'Route_320_gt_unionsquare.jpg'
# route1 = [223,207,191,176,190,206]
# saveRoute(config, route1, name, size=224)

# name = 'Route_top2_of_320_unionsquare.jpg'
# route1 = [4639,4557,4476,4399,4475,4556]
# saveRoute(config, route1, name, size=224)

# name = 'Route_top3_of_320_unionsquare.jpg'
# route1 = [2321,2261,2202,2145,2203,2262]
# saveRoute(config, route1, name, size=224)

# name = 'Route_33_gt_unionsquare_turns_false.jpg'
# route1 = [2881,2814,2742,2670,2602]
# saveRoute(config, route1, name, size=224)

# name = 'Route_top1_of_route33_unionsquare_turns_false.jpg'
# route1 = [730,697,664,630,599]
# saveRoute(config, route1, name, size=224)

# name = 'Route_top2_of_route33_unionsquare_turns_false.jpg'
# route1 = [3756,3682,3609,3531,3451]
# saveRoute(config, route1, name, size=224)

# name = 'Route_top3_of_route33_unionsquare_turns_false.jpg'
# route1 = [2999,2932,2867,2799,2728]
# saveRoute(config, route1, name, size=224)

###################################################
locations = [407, 290, 2748, 2996]
frame = getDataFrame(config)
datasetPath = os.path.join(os.environ['datasets'], 'streetlearn')
saveDir = 'images/'


images = []
for idx in locations:
    pano_id = frame.loc[idx-1, 'pano_id']
    city = frame.loc[idx-1, 'city']
    loc = Location(config['dataset'], pano_id, city)
    loc.showLocation()
    snaps = loc.getSnapswithInfo(size=256)
    img = np.concatenate(snaps, axis=1)
    images.append(img)
    tile = loc.getTilewithInfo(zoom=18,size=256)
    snaps.append(tile)
    saveImageList(idx, snaps, saveDir)




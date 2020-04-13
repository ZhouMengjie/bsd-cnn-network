import os
import subprocess
import cv2
import numpy as np

def generateSnaps(mapfile, lat, lon, elevation, shift=0.0005, size=256):
    lat_shift = shift
    lon_shift = shift
    lookatList = [(lat + lat_shift, lon, elevation), # Front
                  (lat, lon - lon_shift, elevation), # Left
                  (lat, lon + lon_shift, elevation), # Right
                  (lat - lat_shift, lon, elevation)] # Back
   
    pview_pos = "+{},{},+{}".format(lat, lon, elevation)

    #configPath = os.path.join( os.environ['HOME'], 'soft', 'OSM2world', 'texture_config.properties')
    configPath = os.path.join( os.environ['HOME'], 'soft', 'OSM2World', 'standard.properties')
   
    for i, direction in enumerate(lookatList):
        lat, lon , elevation = direction
        look_at = "+{},{},+{}".format(lat, lon, elevation)
        outFile = "out_{}.png".format(i)
        subprocess.call( ['java', '-jar', jarPath, '-i', mapFilePath, '-o', outFile, '--pview.pos', pview_pos, '--pview.lookAt', look_at, '--config', configPath])
   
    snaps = []
    for i in range(4):
        imgPath = 'out_' + str(i) + '.png'
        img = cv2.imread( imgPath )
        img = cv2.resize( img, (size, size))
        snaps.append(img)
   
    image = np.concatenate( snaps, axis=1)
    cv2.imshow( "Snaps", image )
    cv2.waitKey(0)







#osm2world
jarPath = os.path.join(os.environ['HOME'], 'soft', 'OSM2world', 'OSM2World.jar', )
mapFile = 'college_green.osm'


mapFilePath = os.path.join( os.environ['datasets'], 'map_data', mapFile)
print(mapFilePath)

#camera settings

lat, lon =  51.4526064, -2.6000576
elevation = 2
pview_pos = "+{},{},+{}".format(lat, lon, elevation)


generateSnaps( mapFilePath, lat, lon, elevation)

# #Look at
# lat, lon = 51.4526704, -2.6000899
# elevation = 2
# look_at = "+{},{},+{}".format(lat, lon, elevation)


# #Call subprocess
# image = subprocess.call( ['java', '-jar', jarPath, '-i', mapFilePath, '-o', 'osm2world.png', '--pview.pos', pview_pos, '--pview.lookAt', look_at])



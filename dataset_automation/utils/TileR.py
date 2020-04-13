import os, sys
import numpy as np
import pandas as pd
import cv2 
import mapnik

class TileR():
    def __init__(self,lat, lon, city):
        self.city = city 
        self.lat = lat 
        self.lon = lon

    def renderTile(self, zoom=19, tetha=0, size=224):
        self.yaw = tetha
        self.zoom = zoom

        # Define size of tile
        if type(size) == tuple:
            width = size[0]
            height = size[1]
        else: 
            width, height = size, size

        mapfile = os.path.join( os.environ['carto'], 'no_text_style_{}.xml'.format(self.city) )

        # target projection
        projection = '+proj=aeqd +ellps=WGS84 +lat_0={} +lon_0={}'.format(90, -tetha + self.lon)
        merc = mapnik.Projection(projection)
        # WGS lat/long source projection of centrel 
        longlat = mapnik.Projection('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        
        # make a new Map object for the given mapfile
        m = mapnik.Map(width, height)
        mapnik.load_map(m, mapfile)
        
        # ensure the target map projection is mercator
        m.srs = merc.params()
        
        # transform the centre point into the target coord sys
        centre = mapnik.Coord(self.lon, self.lat)  
        transform = mapnik.ProjTransform(longlat, merc)
        merc_centre = transform.forward(centre)

        
        # 360/(2**zoom) degrees = 256 px
        # so in merc 1px = (20037508.34*2) / (256 * 2**zoom)
        # hence to find the bounds of our rectangle in projected coordinates + and - half the image width worth of projected coord units
        dx = ((20037508.34*2*(width/2)))/(width*(2 ** (zoom)))
        minx = merc_centre.x - dx
        maxx = merc_centre.x + dx
        print("merc_centre", merc_centre)
        print("dx", dx)
        print("minx", minx)
        print("maxx", maxx)


        # grow the height bbox, as we only accurately set the width bbox
        m.aspect_fix_mode = mapnik.aspect_fix_mode.ADJUST_BBOX_HEIGHT

        bounds = mapnik.Box2d(minx, merc_centre.y-10, maxx, merc_centre.y+10) # the y bounds will be fixed by mapnik due to ADJUST_BBOX_HEIGHT
        print(bounds)
        # print(bounds)
        # inProj = Proj(init='epsg:3857')
        # outProj = Proj(init='epsg:4326')
        # x1,y1 = bounds() -11705274.6374,4826473.6922
        # x2,y2 = transform(inProj,outProj,x1,y1)
        # print x2,y2
        
        
        m.zoom_to_box(bounds)

        # render the map image to a file
        # mapnik.render_to_file(m, output)

        #render the map to an image
        im = mapnik.Image(width,height)
        mapnik.render(m, im)
        
        img = im.tostring('png256')
        img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), 1)
        img =np.asarray(img)
        return img

if __name__ == "__main__":
    tile = TileR(51.4509111, -2.6001727, 'Bristol') 
    img = tile.renderTile(zoom=19, tetha=77, size=224)
    cv2.imshow("tile", img)
    cv2.waitKey(0)



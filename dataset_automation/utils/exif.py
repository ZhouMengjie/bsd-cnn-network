import os,sys
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
from libxmp.utils import file_to_dict
from libxmp import consts


def process_image(filename):
    # Open image file for reading (binary mode)
    f = open(filename, 'rb')

    # Return Exif tags
    tags = exifread.process_file(f)
    return tags


if __name__ == '__main__':
    name = "00152224 22 ene. 2020 1-00-30 PM GMT+00-00_E.jpg"
    path = os.path.join(os.environ['datasets'],'bsv1', 'panos', str(name))
    # for (k,v) in Image.open(path)._getexif().items():
    #     print(TAGS.get(k), v)

    #     #print("{} = {}".format(TAGS.get(k), v))

    # data = process_image(path)
    # for (k,v) in Image.open(path)._getexif().items():
    #     print("{} : {}".format(k,v))
    #     if type(v) == bytes:
    #         print(v.decode('utf-8'))

    # from PIL import Image
    # with Image.open(path) as im:
    #     for segment, content in im.applist:
    #         print(segment, content)
    #         print()
    #         print()
    #         #marker, body = content.split('\x00', 1)
    #         #if segment == 'APP1' and marker == 'http://ns.adobe.com/xap/1.0/':
    #         #    # parse the XML string with any method you like
    #         #    print(body)

    xmp = file_to_dict( path )
    #dc = xmp[consts.XMP_NS_DC]
    #print(xmp)
    #print(dc[0][0])
    # for item in xmp.items():
    #     print(item)

    #'http://ns.adobe.com/xap/1.0/'
    #'http://ns.google.com/photos/1.0/panorama/'
    #'http://ns.adobe.com/tiff/1.0/'

    properties = xmp['http://ns.google.com/photos/1.0/panorama/']
    print(type(properties))
    for item in properties: # item is a list of lenght 3 ["name", "value","dict"]
        print("{}   :   {}".format(item[0], item[1]))



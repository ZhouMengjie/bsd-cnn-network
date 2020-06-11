# only crop images with multi-thread
import numpy as np
import os, sys
import cv2
import Equirec2Perspec as E2P
from scipy.io import loadmat
import multiprocessing
from subprocess import call
import time

num_threads = 6
counter = 0


class Cropper():
    def __init__(self):
        self.dir = os.getcwd()
    
    def crop_pano(self, pano_id):
        img_path = os.path.join(self.dir, 'images', 'jpegs_manhattan_2019', pano_id+'.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.dir, 'images', 'jpegs_pittsburgh_2019', pano_id+'.jpg')

        pano = cv2.imread(img_path)
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

class RenderThread:
    def __init__(self, save_dir, q, printLock):
        self.save_dir = save_dir
        self.q = q 
        self.printLock = printLock
        self.cropper = Cropper()

    def loop(self):
        global counter
        while True:
            # Fetch a tile from the queue and render it
            r = self.q.get()
            if (r == None):
                self.q.task_done()
                break
            else:
                (pano_id) = r

            msg = ""   
            filename_front = os.path.join(self.save_dir, pano_id + '_front.jpg') 
            filename_left = os.path.join(self.save_dir, pano_id + '_left.jpg') 
            filename_right = os.path.join(self.save_dir, pano_id + '_right.jpg') 
            filename_back = os.path.join(self.save_dir, pano_id + '_back.jpg') 
            if os.path.isfile(filename_front):
                msg = "Exists..."    
            else:
                snaps = self.cropper.crop_pano(pano_id)
                cv2.imwrite(filename_front, snaps[0])
                cv2.imwrite(filename_left, snaps[1])
                cv2.imwrite(filename_right, snaps[2])
                cv2.imwrite(filename_back, snaps[3])
                msg = filename_front + " saved"
            counter += 1
            
            print(pano_id, msg)

            self.printLock.acquire()
            self.printLock.release()
            self.q.task_done()


start = time.clock()
directory = os.getcwd()
snaps_directory = os.path.join(directory, 'images', 'cmu5k') 
if not os.path.isdir(snaps_directory):
    os.mkdir(snaps_directory)

# Open matlab file
# routes_file = os.path.join(os.getcwd(), 'Data', sys.argv[1], 'routes_small.mat')
routes_file = os.path.join(os.getcwd(), 'Data', 'cmu5k.mat')
test = loadmat(routes_file)
routes = test['routes'].squeeze()
# 0 id
# 1 gsv_coords
# 2 gsv_yaw
# 3 neighbor 
# 4 bearing 
# 5 oidx 

# Read all the pano_ids to process
pano_ids = []
for i in range(routes.shape[0]):
    pano_id = routes[i][0].squeeze()
    pano_ids.append(str(pano_id))

# Create threads
queue = multiprocessing.JoinableQueue(32)
printLock = multiprocessing.Lock()
renderers = {}
for i in range(num_threads):
    renderer = RenderThread(snaps_directory, queue, printLock)
    render_thread = multiprocessing.Process(target=renderer.loop)
    render_thread.start()
    renderers[i] = render_thread

for pano_id in pano_ids:
    t = (pano_id)
    queue.put(t)

print("No more locations ...")
# Signal render threads to exit by sending empty request to queue
for i in range(num_threads):
    queue.put(None)
# wait for pending rendering jobs to complete
queue.join()

for i in range(num_threads):
    renderers[i].join()

elapsed = (time.clock() - start)
print(elapsed)
print("All finished")
 

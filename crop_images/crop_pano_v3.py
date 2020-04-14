import numpy as np
import os, sys
import cv2
import pandas as pd
import Equirec2Perspec as E2P
from scipy.io import loadmat
import multiprocessing
from subprocess import call
import time

num_threads = 6
counter = 0

# dev/datasets/streetlearn

class Cropper():
    def __init__(self):
        # self.dir = os.path.join(os.environ['HOME'],'Desktop','dev', 'datasets','streetlearn')
        self.dir = os.path.join(os.environ['HOME'],'dev','datasets','streetlearn')
    
    def crop_pano(self, pano_id):
        img_path = os.path.join(self.dir, 'jpegs_manhattan_2019', pano_id+'.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.dir, 'jpegs_pittsburgh_2019', pano_id+'.jpg')

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
    def __init__(self, save_dir_jc, save_dir_bd, q, printLock):
        self.save_dir_jc = save_dir_jc
        self.save_dir_bd = save_dir_bd
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
                pano_id = r[0]
                front = r[1]
                right = r[2]
                back = r[3]
                left = r[4]

            snaps_directory_jc = os.path.join(self.save_dir_jc, 'junctions') 
            snaps_directory_njc = os.path.join(self.save_dir_jc, 'non_junctions') 
            snaps_directory_bd = os.path.join(self.save_dir_bd, 'gaps') 
            snaps_directory_nbd = os.path.join(self.save_dir_bd, 'non_gaps') 
            if not os.path.isdir(snaps_directory_jc):
                os.mkdir(snaps_directory_jc)

            if not os.path.isdir(snaps_directory_njc):
                os.mkdir(snaps_directory_njc)

            if not os.path.isdir(snaps_directory_bd):
                os.mkdir(snaps_directory_bd)

            if not os.path.isdir(snaps_directory_nbd):
                os.mkdir(snaps_directory_nbd)

            msg = ""   
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
                filename_right = os.path.join(snaps_directory_bd, pano_id + '_right.jpg') 
            else:
                filename_right = os.path.join(snaps_directory_nbd, pano_id + '_right.jpg') 

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
dataset = 'train'
directory = os.getcwd()
snaps_directory_junction = os.path.join(directory, 'network', 'data', 'JUNCTIONS', dataset) 
snaps_directory_gap = os.path.join(directory, 'network', 'data', 'GAPS', dataset) 
if not os.path.isdir(snaps_directory_junction):
    os.makedirs(snaps_directory_junction)
    # os.mkdir(snaps_directory_junction)

if not os.path.isdir(snaps_directory_gap):
    os.makedirs(snaps_directory_gap)
    # os.mkdir(snaps_directory_gap)

# Open matlab file
filename = os.path.join(os.getcwd(), 'data', dataset + '.csv')
names = ["pano_id", "gsv_lat", "gsv_lon", "gsv_yaw", "front", "right", "back", "left", "city"]
routes = pd.read_csv(filename,names=names)

# Read all the pano_ids to process
rows = []
for i in range(routes.shape[0]):
    info = []
    row = routes.loc[i, :]
    pano_id = row['pano_id']
    front = row['front']
    right = row['right']
    back = row['back']
    left = row['left']
    info.append(pano_id)
    info.append(front)
    info.append(right)
    info.append(back)
    info.append(left)
    rows.append(info)

# Create threads
queue = multiprocessing.JoinableQueue(32)
printLock = multiprocessing.Lock()
renderers = {}
for i in range(num_threads):
    renderer = RenderThread(snaps_directory_junction,snaps_directory_gap,queue,printLock)
    render_thread = multiprocessing.Process(target=renderer.loop)
    render_thread.start()
    renderers[i] = render_thread

for row in rows:
    t = (row)
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
 
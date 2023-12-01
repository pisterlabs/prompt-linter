import cv2
import lk_track
import CoherentFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import dropbox
import datetime
import os
from dotenv import load
import settings
import warnings


# Load environment variables from .env
load()

settings.demo_dir = 'demo/frame7%d.jpg'

def USER():
    d = 7   # from t -> t+d
    K = 15  # K Nearest Neighbours
    lamda = 0.6 # Threshold
    result = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + ".jpg"
    settings.result_dir = 'result/' + result
    # Get the tracks from KLT algorithm
    trajectories, numberOfFrames, lastFrame = lk_track.FindTracks()
    vis = lastFrame.copy()
    numberOfTracks = len(trajectories)
    trajectories = np.array(trajectories)
    tracksTime = np.zeros((2,numberOfTracks),dtype=int)
    # Get the first and last frame in every track
    for i in range(numberOfTracks):
        tracksTime[0,i] = trajectories[i][0][2] # the first time when each point is appeared
        tracksTime[1,i] = trajectories[i][-1][2] # the last time when each point is appeared
    for i in range(1,numberOfFrames):
        # get the tracks that this frame 'i' is a part of it or a first frame or last frame of it 
        currentIndexTmp1 = np.asarray(np.where(np.in1d(tracksTime[0], [j for j in tracksTime[0] if i>=j])))
        currentIndexTmp2 = np.asarray(np.where(np.in1d(tracksTime[1], [j for j in tracksTime[1] if j>=i])))
        currentIndexTmp1=list(currentIndexTmp1[0])
        currentIndexTmp2=list(currentIndexTmp2[0])
        currentIndex = np.array(list(set(currentIndexTmp1).intersection(set(currentIndexTmp2))))
        includeSet=[trajectories[j] for j in currentIndex]
        '''coherence filtering clustering'''
        currentAllX, clusterIndex = CoherentFilter.CoherentFilter(includeSet, i , d, K, lamda)
        if clusterIndex!=[]:
            numberOfClusters = max(clusterIndex)
            color = np.array([[0,255,128],[0,0,255],[0,255,0],[255,0,0],[255,255,255],[255,255,0],[255,156,0]])
            counter=0
            if i==numberOfFrames-8:
                for x, y in [[np.int32(currentAllX[0][k]),np.int32(currentAllX[1][k])] for k in range(len(currentAllX[0]))]:
                    cv2.circle(lastFrame, (x,y), 5, color[clusterIndex[counter]].tolist(), -1)
                    counter = counter+1
                cv2.imwrite(result, lastFrame)
    result_dir = settings.result_dir
    cv2.imwrite(result_dir, lastFrame)
    plt.pause(1)
    img = cv2.imread(result_dir)

    ''' uploading the result to Dropbox'''
    im=open(result_dir,'rb')
    f=im.read()
    try:
        dbx = dropbox.Dropbox(os.environ['ACCESS_TOKEN'])
        try:
            dbx.files_delete("/"+result_dir)
        except:
            warnings.warn("WARNING: No files has been deleted")
        try:
            dbx.files_upload(f, '/'+result_dir)

        except Exception as err:
            print("Failed to upload %s\n%s" % (result_dir, err))
        print(dbx.files_get_metadata('/'+result_dir).server_modified)

    cv2.imshow(result_dir,img)
    k = cv2.waitKey(0)
    return result_dir


USER()

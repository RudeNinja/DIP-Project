from skimage.morphology import (disk,dilation, opening, closing, reconstruction,erosion)
import numpy as np
import cv2
from skimage import filters
from skimage.morphology import remove_small_objects
import matlab.engine

import matplotlib.pyplot as plt
def segment_color(image,gd_thresh,se_size,gamma,min_edge):

    filter  = disk(se_size)
    filter2 = disk(se_size-2)
    imgrey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ime = erosion(imgrey,filter)
    opened = reconstruction(ime,imgrey)
    imd = dilation(opened,filter2)
    closed = reconstruction(255-imd,255-opened)
    closed2 = 255-closed
    

    edge = filters.sobel(closed2)
    edge = edge
    edge = np.where(edge>gd_thresh*np.amax(edge),255,0)

    edge = remove_small_objects(edge.astype(bool),min_size=min_edge).astype(int)
    # plt.imshow(edge,cmap = 'gray')
    edge2 = np.where(edge==1,0,255)
    analysis = cv2.connectedComponentsWithStats(edge2.astype(np.uint8),8,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    plt.imshow(label_ids,cmap='gray')
    imageref = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    result = 0.5*imageref*(np.dstack((edge,edge,edge))/255)
    for i in range(totalLabels):

        mask = np.where(label_ids==i,1,0)
        l1 = imageref[:,:,0]*mask
        
        m1 = np.sum(l1)/values[i,4]
        

        a1 = imageref[:,:,1]*mask
        m2 = np.sum(a1)/values[i,4]
        

        b1 = imageref[:,:,2]*mask
        m3 = np.sum(b1)/values[i,4]
        

        M = [m1,m2,m3]
        idx = M.index(max(M))
        # if idx==0:
        #     m1 = m1**gamma
        
        # elif idx==1:
        #     m2 = m2**gamma

        # elif idx==2:
        #     m3 = m3**gamma
        
        

        result = result + np.dstack((m1*mask,m2*mask,m3*mask))
    result = result.astype(np.uint8)
    out = cv2.cvtColor(result,cv2.COLOR_LAB2RGB)

    return out

    


    
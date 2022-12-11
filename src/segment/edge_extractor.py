
import numpy as np
import cv2

def edge_extractor(img,thresh,k):  # be careful, the image intensity and threshold has to be in the range of 0-255
    shape = img.shape

    if len(shape)>2:
        # image is an RGB image

        img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    else:
        img_grey = img

    edge = cv2.Canny(img_grey,0,thresh)

    final_edges  = cv2.dilate(edge,np.ones((k,k)),iterations=1)

    return final_edges
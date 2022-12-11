import cv2
import numpy as np
from matplotlib import pyplot as plt

def noise_image(grey_scale_image,labelled_image,label_frequency,l1,minl1,maxl1,l2,minl2,maxl2):
    
    grey_scale_image = grey_scale_image/np.max(grey_scale_image)
    h,w = grey_scale_image.shape
    
    label_frequency_vals = np.array(list(label_frequency.values()))
    noisy_image = np.empty((h,w),dtype = int)
    number_of_parents = len(label_frequency_vals)    
    grey_parent = np.zeros(number_of_parents)

    for i in range(h):
        for j in range(w):
            grey_parent[labelled_image[i][j]] += grey_scale_image[i][j]

    grey_parent = grey_parent/np.array(label_frequency_vals)

    threshold1 = (l1*(1 - grey_scale_image))**2
    threshold2 = (l2*(1 - grey_scale_image))**2
    rand_img = np.random.random((h,w))
    
    first_part = minl1*(rand_img <= threshold1) + maxl1*(rand_img > threshold1)
    second_part = minl2*(rand_img <= threshold2) + maxl2*(rand_img > threshold2)    
    points_selection = np.zeros((h,w),dtype = int)

    for i in range(h):
        for j in range(w):
            points_selection[i][j] = int(grey_scale_image[i][j] <= grey_parent[labelled_image[i][j]])
    
    noisy_image = first_part*points_selection + second_part*(1 - points_selection)

    return noisy_image
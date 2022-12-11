import numpy as np
import cv2
import matplotlib.pyplot as plt

class union_find():
    def __init__(self,h,w):
        self.parents = np.arange(h*w)
        self.ranking = np.zeros(h*w)
        
    def find(self, k):
        if self.parents[k] == k:
            return k
        return self.find(self.parents[k])

def region_labelling(image,pools):
    h,w = image.shape[0],image.shape[1]
    val = 0
    single_index = np.zeros((h,w),dtype = int)
    for i in range(h):
        for j in range(w):
            single_index[i][j] = val
            val += 1

    cost_two_hor_index = []                             ##### Calculates cost of merging a pixel with its left and right immediate pixels
    diff_hor_img = (image[:,:-1,:] - image[:,1:,:])**2 
    cost_adj_hor = (np.sum(diff_hor_img, axis = 2))
    for i in range(diff_hor_img.shape[0]):
        for j in range(diff_hor_img.shape[1]):
            cost_two_hor_index.append((cost_adj_hor[i][j],single_index[:,:-1][i][j],single_index[:,1:][i][j]))

    cost_two_ver_index = []                             ##### Calculates cost of merging a pixel with its top and bottom immediate pixels
    diff_ver_img = (image[:-1,:,:] - image[1:,:,:])**2
    cost_adj_ver = (np.sum(diff_ver_img, axis = 2))
    for i in range(diff_ver_img.shape[0]):
        for j in range(diff_ver_img.shape[1]):
            cost_two_ver_index.append((cost_adj_ver[i][j],single_index[:-1,:][i][j],single_index[1:,:][i][j]))

    cost_two_leftdiag_index = []                        ##### Calculates cost of merging a pixel with its left top and right bottom immediate pixels
    diff_leftdiag_img = (image[:-1,:-1,:] - image[1:,1:,:])**2
    cost_adj_leftdiag = (np.sum(diff_leftdiag_img, axis = 2))
    for i in range(diff_leftdiag_img.shape[0]):
        for j in range(diff_leftdiag_img.shape[1]):
            cost_two_leftdiag_index.append((cost_adj_leftdiag[i][j],single_index[:-1,:-1][i][j],single_index[1:,1:][i][j]))
    
    cost_two_rightdiag_index = []                       ##### Calculates cost of merging a pixel with its top right and left bottom immediate pixels
    diff_rightdiag_img = (image[1:,:-1,:] - image[:-1,1:,:])**2
    cost_adj_rightdiag = (np.sum(diff_rightdiag_img, axis = 2))
    for i in range(diff_rightdiag_img.shape[0]):
        for j in range(diff_rightdiag_img.shape[1]):
            cost_two_rightdiag_index.append((cost_adj_rightdiag[i][j],single_index[1:,:-1][i][j],single_index[:-1,1:][i][j]))

    total_costs = cost_two_hor_index + cost_two_ver_index + cost_two_leftdiag_index + cost_two_rightdiag_index
    incre_cost = sorted(total_costs)
    class_instant = union_find(h,w)      ##### Union find data structure to find parents and replace pixel intensity with a number which depends on parent.
    total_vals = h*w
    for tup in range(len(incre_cost)):
        if total_vals > pools:
            x,y = incre_cost[tup][1],incre_cost[tup][2]
            px,py = class_instant.find(x),class_instant.find(y)
            
            if px != py:
                if class_instant.ranking[px] > class_instant.ranking[py]:
                    px,py = py,px
                elif class_instant.ranking[px] == class_instant.ranking[py]:
                    class_instant.ranking[py] += 1
                class_instant.parents[px] = py
                total_vals -= 1
        else:
            break
    all_parents = {}
    pixel_value = 0
    label_frequency = {}
    labelled_img = np.empty((h,w),dtype = int)
    for i in range(h):
        for j in range(w):
            par = class_instant.find(i*w + j)
            if all_parents.get(par) is None:
                all_parents[par] = pixel_value
                label_frequency[all_parents[par]] = 0
                pixel_value += 1
            labelled_img[i][j] = all_parents[par]
            label_frequency[all_parents[par]] += 1
    return labelled_img,label_frequency
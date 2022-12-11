import matlab.engine
import numpy as np
import cv2


def segment_color(image,gd_thresh,se_size,gamma,min_edge):
    eng = matlab.engine.start_matlab()
    
    out = eng.segment_color(image,float(gd_thresh),se_size,float(gamma),min_edge)
 
    out = np.array(out)
    out = cv2.cvtColor(out,cv2.COLOR_LAB2RGB)
    out = out/np.amax(out)
    
    
    return out












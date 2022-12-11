import matlab.engine
import numpy as np


def segment_color(image,gd_thresh,se_size,gamma,min_edge):
    eng = matlab.engine.start_matlab()
    
    out = eng.segment_color(image,float(gd_thresh),se_size,float(gamma),min_edge)
 
    out = np.array(out)
    
    
    return out












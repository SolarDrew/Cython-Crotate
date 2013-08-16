# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:24:34 2013

@author: drew
"""

import numpy as np
from aff_tr_stu import affine_transform_kc as aff

def rotate_Map(input_map, angle, scale=1.0, missing=0.0, 
               int_method='bicubic', int_param=-0.5):
    #Calulate the parameters for the affine_transform
    c = np.cos(angle)
    s = np.sin(angle)
    centre = (np.array(input_map.shape)-1)/2.0
    mati = np.array([[c, s],[-s, c]]) / scale   # res->orig
    centre = np.array([centre]).transpose()  # the centre of rotn
    shift = [0, 0]#np.array([shift]).transpose()    # the shift
    kpos = centre - np.dot(mati, (centre + shift))
    rsmat, offs =  mati, np.squeeze((kpos[0,0], kpos[1,0]))
    
    # Call to function that does the actual work.  This one is external. */ 
    output_map = aff(input_map, rsmat, offs, int_method, int_param, missing)
    
    return output_map
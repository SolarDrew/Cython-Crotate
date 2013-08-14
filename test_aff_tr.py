# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:33:34 2013

@author: ajl7
"""
import numpy as np
from aff_tr_stu import affine_transform_kc as aff

in_arr = np.zeros((101,101))
in_arr[40:60, :] = in_arr[: ,40:60] = 1.0

## should these be const?
scale = [1.0, 1.0] # Probably? DL
offset = [0.0, 0.0]
int_method = 'BICUBIC' # default value
int_param = -0.5 # bicubic iterp param
missing = 0.0 # ????? DL # optional arg

# Make a nice numpy array using macro for PyArray_FromAny laid out
#   for C access as floats and doubles.  The alignment is a bit tricky
# - sometimes you *don't* need FORCECAST, but often you do. */
#arr1 = np.array(arg1, dtype=np.float32)
#arr2 = np.array(arg2, dtype=float)

#in_arr = arr1    # that's where the data for C is   
#ndim = arr1.ndim

#for i in range(ndim):
#    dims[i] = int(PyArray_DIM(arr1,i))  # make sure alignment OK

#rotscale = PyArray_DATA(arr2)    # set to the location of the operator matrix

# Call to function that does the actual work.  This one is external. */ 
out_arr = aff(in_arr, scale, offset, int_method, int_param, missing)

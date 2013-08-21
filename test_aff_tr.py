# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:33:34 2013

@author: ajl7
"""
import numpy as np
import matplotlib.pyplot as plt
from aff_tr_stu import affine_transform_kc as aff_py
from affine_transform import affine_transform as aff
from aff_tr.affine_transform import affine_transform as aff_cythonwrap
import sunpy.image.Crotate as Crotate

import timeit

in_arr = np.zeros((501,501))
in_arr[200:300, :] = in_arr[: ,200:300] = 1.0
#plt.imshow(in_arr)
#plt.show()

#Calulate the parameters for the affine_transform
angle = np.pi / 4.0
c = np.cos(angle)
s = np.sin(angle)
scale = 1.0 # Probably? DL
centre = (np.array(in_arr.shape)-1)/2.0
mati = np.array([[c, s],[-s, c]]) / scale   # res->orig
centre = np.array([centre]).transpose()  # the centre of rotn
shift = [0, 0]#np.array([shift]).transpose()    # the shift
kpos = centre - np.dot(mati, (centre + shift))
rsmat, offs =  mati, np.squeeze((kpos[0,0], kpos[1,0]))

## should these be const?
int_method = 'bicubic' # default value
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

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    print "c code"
    print timeit.timeit("out_arr = Crotate.affine_transform(in_arr, rsmat, offset=offs, kernel=Crotate.BICUBIC, cubic=int_param, mode='constant', cval=missing)",
                        setup="from __main__ import *", number=100) /100.
                        
    print "cython wrapper"
    print timeit.timeit("out_arr = aff_cythonwrap(in_arr, rsmat, offs, int_param, missing)",
                        setup="from __main__ import *", number=100) /100.
    print "cython"
    print timeit.timeit("out_arr = aff(in_arr, rsmat, offs, int_param, missing)",
                  setup="from __main__ import *", number=100) /100.
    
#    print "python"
#    print timeit.timeit("out_arr = aff_py(in_arr, rsmat, offs, int_method, int_param, missing)",
#                  setup="from __main__ import *", number=1) /1.
#              


out_arr_C = Crotate.affine_transform(in_arr, rsmat, offset=offs, kernel=Crotate.BICUBIC, cubic=int_param, mode='constant', cval=missing)   
out_arr_wr = aff_cythonwrap(in_arr, rsmat, offs, int_param, missing)
fig, ax = plt.subplots(1,3)
im1 = ax[0].imshow(out_arr_C - out_arr_wr, cmap=plt.get_cmap('Reds'), interpolation='none')
plt.colorbar(im1,ax=ax[0])
im2 = ax[1].imshow(out_arr_C, vmax =1, vmin=0, cmap=plt.get_cmap('Reds'), interpolation='none')
plt.colorbar(im2,ax=ax[1])
im3 = ax[2].imshow(out_arr_wr, vmax =1, vmin=0, cmap=plt.get_cmap('Reds'), interpolation='none')
plt.colorbar(im3,ax=ax[2])
plt.show()
#plt.imshow(out_arr)
#plt.show()

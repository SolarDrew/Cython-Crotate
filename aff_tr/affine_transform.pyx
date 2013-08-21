# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:07:50 2013

@author: stuart
"""
cimport numpy as np
np.import_array()
cimport cython

cdef extern from "aff_tr.h":
    int NEAREST
    int BILINEAR
    int BICUBIC

cdef extern from "aff_tr.c":
    int affine_transform_kc(int *dims, double *out_arr, double *in_arr,
                            double *scale, double *offset, int int_type,
                            double int_param, double miss_val)

@cython.boundscheck(False)
@cython.wraparound(False)
def affine_transform(np.ndarray in_arr, np.ndarray scale, np.ndarray offset,
                     str int_method, double int_param, double missing):
    """
    Perform a kernel convolution affine transform
    
    Parameters
    ----------
    in_arr: np.ndarray
        Input array 2D
    
    scale: [x,y] np.ndarray
        Scale for each axis
    
    offset: [x,y] np.ndarray
        Shift for each axis
    
    int_method: str
        'bicubic' or 'bilinear'
    
    int_param: float
        Parameter to pass to interpolation kernel
    
    missing: float
        value to fill missing elements
    """
    cdef int int_type
    if int_method == "nearest":
        int_type = NEAREST
    elif int_method == "bilinear":
        int_type = BILINEAR
    elif int_method == "bicubic":
        int_type = BICUBIC
    else:
        int_type = -1
    
    cdef double* scale_v = <double *> scale.data
    cdef double* offset_v = <double *> offset.data
    
    cdef int dims[2]
    cdef int kern_size
    cdef double row, col
    cdef int x, y
    
    dims[0] = in_arr.shape[0]
    dims[1] = in_arr.shape[1]
    
    cdef double [:, :] in_arr_v = in_arr
    cdef double [:, :] out_arr_v = in_arr_v.copy()

    affine_transform_kc(dims, &out_arr_v[0,0], &in_arr_v[0,0], scale_v,
                        offset_v, int_type, int_param, missing)
    
    return out_arr_v

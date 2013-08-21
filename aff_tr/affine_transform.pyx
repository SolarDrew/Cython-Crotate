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
    
    Note
    ----
    This function expects a C order contiguious array as input.
    """
    
    #Define varibles
    cdef int dims[2]
    cdef double row, col
    cdef int x, y
    cdef int int_type
    
    #Create a C array from the shape of the array
    dims[0] = in_arr.shape[0]
    dims[1] = in_arr.shape[1]
    
    #Process interpolation type    
    if int_method == "nearest":
        int_type = NEAREST
    elif int_method == "bilinear":
        int_type = BILINEAR
    elif int_method == "bicubic":
        int_type = BICUBIC
    else:
        int_type = -1
    
    #Create array pointers to pass to the C code
    cdef double* scale_v = <double *> scale.data
    cdef double* offset_v = <double *> offset.data
    #Make a cython memory view of the numpy array from which a pointer can be 
    #sent to the C function.
    cdef double [:, :] in_arr_v = in_arr
    #Create a output array the same size as the in_arr by copying it.
    cdef double [:, :] out_arr_v = in_arr_v.copy()

    #Call the C code. The & stands for pointers, pointers for the memoryviews
    #are extracted by indexing the first element.
    #The numpy arrays passed in must be contiguious as the array is addressed
    #in the C code as a 1D flattened C order array.
    affine_transform_kc(dims, &out_arr_v[0,0], &in_arr_v[0,0], scale_v,
                        offset_v, int_type, int_param, missing)
    
    return out_arr_v

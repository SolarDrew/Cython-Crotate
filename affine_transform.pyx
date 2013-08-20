# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:44:42 2013

@author: stuart
"""
import numpy as np
#Import Cython numpy magic
cimport numpy as np

from libc.math  cimport fabs, floor
from cython.view cimport array as cvarray
cimport cython
#Define numpy array DTYPES
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef double (*f_type)(double, double)

# 1D bicubic kernel (for 2D separable)
cdef double k_bicub(double x, double a):
    x = fabs(x)
    if x <= 1.0:
        return (a + 2.0) * x**3.0 - (a + 3.0) * x**2.0 + 1.0
    elif x < 2.0:
        return a * x**3.0 - 5.0 * a*x**2.0 + 8.0 * a * x - 4.0 * a
    else:
        return 0.0

cdef double interpol_kernel(int *dim, double [:] in_arr, double row,
                           double col, f_type kern_func, 
                           double int_param, double missing):

    cdef int kern_size = 4
    cdef int i, j, c0, r0

    cdef int cols[4]
    cdef double colw[4]
    cdef int rows[4]
    cdef double roww[4]
    
    cdef double rsum, csum
    
    # $$ could prob do N-dim
    # Tabulate the values to process for this point
    # Check that astype() rounds in the same way as C's int()
    c0 =  (<int>(floor(col))) - kern_size/2 + 1
    r0 =  (<int>(floor(row))) - kern_size/2 + 1
    for j in range(kern_size):
        cols[j] = c0 + j  # $$ do we want to save this or recalc...
        colw[j] = kern_func((c0 + j) - col, int_param) # Not really sure what this is doing. DL
    for i in range(kern_size):
        rows[i] = r0 + i
        roww[i] = kern_func((r0 + i) - row, int_param) # As above
    # convolve by cols - can be fn $$
    # Each step for separable reduces the dims by one, so NxN->N->scalar
    rsum = 0.0
    for i in range(kern_size):
        csum = 0.0
        for j in range(kern_size):
            if (rows[i] < 0. or rows[i] >= dim[0] or cols[j] < 0. or cols[j] >= dim[1]): # This is right!
                csum = csum + colw[j]*missing
            else:
                csum = csum + colw[j]*in_arr[rows[i]*dim[1] + cols[j]]
        rsum = rsum + roww[i]*csum
    return rsum

@cython.boundscheck(False)
@cython.wraparound(False)
def affine_transform(np.ndarray in_arr, np.ndarray scale, np.ndarray offset,
                     double int_param, double missing):
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
    scale = scale.ravel()
    cdef double [:] scale_v = scale
    cdef double [:] offset_v = offset
    
    cdef int dims[2]
    cdef int kern_size
    cdef double row, col
    cdef int x, y
    
    dims[0] = in_arr.shape[0]
    dims[1] = in_arr.shape[1]
    
    #replicate silly varibles from input
    in_arr = in_arr.ravel()
    cdef double [:] in_arr_v = in_arr
    cdef np.ndarray out_arr = np.zeros((in_arr.shape[0] * in_arr.shape[1]) , dtype=DTYPE)
    cdef double [:] out_arr_v = out_arr
    
    for x in range(dims[0]):    # rows
        for y in range(dims[1]):  # cols
            row = scale_v[0] * x + scale_v[1] * y + offset_v[0]
            col = -scale_v[1] * x + scale_v[0] * y + offset_v[1]
            #print row, col
            out_arr_v[x*dims[1]+y] = interpol_kernel(dims, in_arr_v, row, col, k_bicub, int_param, missing)
    out_arr = out_arr.reshape((in_arr.shape[0], in_arr.shape[1]))
    
    return out_arr
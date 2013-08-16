# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:44:42 2013

@author: stuart
"""
import numpy as np

# 1D bicubic kernel (for 2D separable)
def k_bicub(x, a):
    x = np.abs(x)
    if x <= 1.0:
        return (a + 2.0) * x**3.0 - (a + 3.0) * x**2.0 + 1.0
    elif x < 2.0:
        return a * x**3.0 - 5.0 * a*x**2.0 + 8.0 * a * x - 4.0 * a
    else:
        return 0.0

# 1D bilinear kernel (for 2D separable)
def k_bilin(x, *args):
  x = np.abs(x)
  if (x <= 1.0):
    return 1 - x
  else:
    return 0.0

#TODO: refactor this!
def interpol_kernel(dims, in_arr, row, col, kern_size, kern_func, int_param, missing):
    cols = np.zeros(kern_size)
    colw = np.zeros(kern_size)
    rows = np.zeros(kern_size)
    roww = np.zeros(kern_size)

    # $$ could prob do N-dim
    # Tabulate the values to process for this point
    # Check that astype() rounds in the same way as C's int()
    c0 = col.astype(int) - kern_size/2.0 + 1
    r0 = row.astype(int) - kern_size/2.0 + 1
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
            if (rows[i] < 0 or rows[i] >= dims[0] or cols[j] < 0 or cols[j] >= dims[1]):
                csum = csum + colw[j]*missing
            else:
                csum = csum + colw[j]*in_arr[rows[i]*dims[1] + cols[j]]
                #csum = csum + colw[j]*in_arr[rows[i], cols[j]]
        rsum = rsum + roww[i]*csum
    return rsum
"""
sunpy.image.Crotate.affine_transformation(input, matrix, offset=[0.0, 0.0], kernel=Crotate.BICUBIC, cubic=-0.5, mode='constant', cval=0.0)
Apply an affine transformation to an image array

Parameters
----------
input : ndarray
matrix : ndarray
offset
kernel
cubic
mode
cval
"""
#scale = mat
#offset = tr
#mode is not used
#missing = cval
def affine_transform_kc(in_arr, scale, offset, int_method, int_param, missing):
    """
    Perform a kernel convolution affine transform
    
    Parameters
    ----------
    in_arr: np.ndarray
        Input array 2D
    
    scale: [x,y]
        Scale for each axis
    
    offset: [x,y]
        Shift for each axis
    
    int_method: str
        'bicubic' or 'bilinear'
    
    int_param: float
        Parameter to pass to interpolation kernel
    
    missing: float
        value to fill missing elements
    """
    #Assert input is float
    assert in_arr.dtype == np.float
    
    #replicate silly varibles from input
    dims = in_arr.shape
    in_arr = in_arr.flatten()
    out_arr = np.zeros(in_arr.shape)
    
    # Choose an interpolation kernel function based on int_method argument.
    if int_method == 'bicubic':
        kern_func = k_bicub
    else:
        print "Bilinear interpolation is currently unavailable. You're getting bicubic and you'll like it."
        kern_func = k_bicub
    
    # I think this defines the extent of the sampled data used for the interpolation. DL
    kern_size = 4
    
    #TODO: This needs to be sorted, i.e. modified to work in 2D using numpy??
    #Will this cause issues with the kernel? what is col and row?
    scale = scale.flatten()    
    #print scale
    for x in range(dims[0]):    # rows
        for y in range(dims[1]):  # cols
            row = scale[0] * x + scale[1] * y + offset[0]
            col = -scale[1] * x + scale[0] * y + offset[1]
            #print row, col
            out_arr[x*dims[1]+y] = interpol_kernel(dims, in_arr, row, col, kern_size, kern_func, int_param, missing)
    out_arr = out_arr.reshape(dims)
    
    return out_arr
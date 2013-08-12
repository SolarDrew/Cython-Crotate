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
def interpol_kernel(dims, img, row, col, k_size, k_fun, intparam, missing):
    # Some array declarations that need sorting out. DL
    # Ignore the above. They can probably just be lists.
    #int cols[PATCHLEN];
    #double colw[PATCHLEN];
    #int rows[PATCHLEN];
    #double roww[PATCHLEN];

    assert(k_size <= PATCHLEN)
    # $$ could prob do N-dim
    # Tabulate the values to process for this point
    c0 = int(col) - k_size/2.0 + 1
    r0 = int(row) - k_size/2.0 + 1
    for j in range(0, k_size):
        cols[j] = c0 + j  # $$ do we want to save this or recalc...
        colw[j] = k_fun((c0 + j) - col, intparam) # Not really sure what this is doing. DL
    for i in range(0,k_size):
        rows[i] = r0 + i
        roww[i] = k_fun((r0 + i) - row, intparam) # As above
    # convolve by cols - can be fn $$
    # Each step for separable reduces the dims by one, so NxN->N->scalar
    rsum = 0.0
    for i in range(0, k_size):
        csum = 0.0
        for j in range(0, k_size):
            if (rows[i] < 0 or rows[i] >= dims[0] or cols[j] < 0 or cols[j] >= dims[1]):
                csum = csum + colw[j]*missing
            else:
                csum = csum + colw[j]*img[rows[i]*dims[1] + cols[j]]
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
def affine_transform_kc(in_arr, scale, offset, interpolation_method, int_param, missing):
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
    
    interpolation_method: str
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
    out_arr = np.zeros(in_arr.shape)
    
    #TODO: This needs to be sorted, i.e. modified to work in 2D using numpy??
    #Will this cause issues with the kernel? what is col and row?
    for out1 in range(0, dims[0]):    # rows
        for out2 in range(0, dims[1]):  # cols
              row = scale[0] * out1 + scale[1]  * out2 + offset[0]
              col = -scale[1] * out1 + scale[0]  * out2 + offset[1]
              out_arr[out1*dims[1]+out2] = interpol_kernel(dims, in_arr, row, col, k_size, interpolation_method, int_param, missing)
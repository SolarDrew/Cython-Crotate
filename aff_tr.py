# -*- coding: utf-8 -*-
"""
Copyright (c) 2011 The SunPy developers
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Rotate, scale and shift in plain old unoptimised C
## Cython, bitches. DL

## well OK - 32 bit ints on this box
"""

# Don't know how imports work with Cython, but here's one for now.
import numpy as np
sin = np.sin

"""
Definitions, etc. from aff_tr.h
"""
# Header providing affine transform functions.

if 'Z_AFFTR_H' not in dir(): # Check what this actually is. DL
    Z_AFFTR_H = 1
    
    # Image data types - to get these from include by caller, so can it
    #   cast if required types do not match.
    #define INTYPE float
    #define OUTTYPE float
    
    # define if want sinc() interp option
    #define HAVESINC
    
    # Allowed values for int_param arg
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2
    if 'HAVESINC' in dir():
        SINC = 3
    
    if 'M_PI' not in dir():   # prob the case for c99
        M_PI = 3.14159265358979323846 # Replace this with numpy.pi? DL
    
    # Quoter
    #define QU(A) xQU(A)
    #define xQU(A) #A
    # WTF? DL
    
    # Rotate/scale/shift as seen from output image (i.e. like Scipy fn,
    #   though using kernel convolution).
    affine_transform_kc(dims, out_arr, in_arr, mat, tr, int_type, int_param, mode, miss_val)
    # Is this a call to the function or a declaration? If the latter, it can go. DL


"""
Code translated from aff_tr.c
"""
# Some decl just to catch type errors
## Replace these with assert statements? DL
#static Kernfun k_bicub;
#static Kernfun k_bilin;
#static Intfun interpol_nearest;
#static Intfun interpol_kernel;

# If compile with sinc() code, needs a couple of arrays to be bigger
#   plus the extra code is added.  The extra array space is only used
#   if needed (i.e. by sinc(), but does exist.
## Replace the following with a try statement? Ignore it for now though, since 
## I don't know what it's doing. DL
if 'HAVESINC' not in dir():
    PATCHLEN = 4 # max size for kernel - 4 should be enough for bicub...
else:
    #static Kernfun k_sinc;
    PATCHLEN = 8 # max size for kernel - sinc is quite big


# Basic bicubic as convolution with separable kernel. 

#While you can get exactly five points in a 1D convolution, then the
#end ones are zero, so just do 4 - start arbitrarily at one end.

#Also arb., we'll convolve along rows first.

# 1D bicubic kernel (for 2D separable)
def k_bicub(x, a):
    x = abs(x)
    if x <= 1.0:
        return (a + 2.0) * x**3.0 - (a + 3.0) * x**2.0 + 1.0
    elif x < 2.0:
        return a * x**3.0 - 5.0 * a*x**2.0 + 8.0 * a * x - 4.0 * a
    else:
        return 0.0

if 'HAVESINC' in dir():
# A sinc fn which can be used for test but which is slow and takes up
#   space.  As for others second arg is placeholder.
    def k_sinc(x, a):
        if abs(x) <= 0.00001:
          return 1.0
        else:
          x = x*M_PI #numpy.pi?
        return sin(x)/x


# For a point [row,col] between pixel values in an array img
#calculate from a kernel an interpolated value.  That is, this uses
#fractional indices on the input array.

#k_size is the size actually used for the kernel, so some parts of
#declared arrays might not in fact be used.

#See affine_transform_kc() for more on args.
def interpol_kernel(dims, img, row, col, k_size, k_fun, intparam, mode, missing):
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
        colw[j] = k_fun((double)(c0 + j) - col, intparam) # Not really sure what this is doing. DL
    for i in range(0,k_size):
        rows[i] = r0 + i
        roww[i] = k_fun((double)(r0 + i) - row, intparam) # As above
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

# Run over the *output* image and get the input values.

#Should behave like Python affine_transform, only requiring the top row
#of the rotation matrix divided by mag.  Since by convention
#r11,r12,r21,r22<->r[0],r[1],r[2],r[3] and we pass &r, the matrix
#dimensions do not matter.

#The int_type arg is an integer enum to NEAREST, BILINEAR or BICUBIC,
#with int_param only used if need be.

## mode ignored for now
## not sure what speed implications of function pointers is
## likewise, implications of fixed index array refs
def affine_transform_kc(dims, out_arr, in_arr, mat, tr, int_type, int_param, mode, miss_val):
    if int_type == NEAREST:
        i_fun = interpol_nearest
        k_fun = NULL
        k_size = 0
        break
    elif int_type == BILINEAR:
        i_fun = interpol_kernel
        k_fun = k_bilin
        k_size = 2
        break
    elif int_type == BICUBIC:
        i_fun = interpol_kernel
        k_fun = k_bicub
        k_size = 4
        break
    elif int_type == SINC and 'HAVESINC' in dir():
        i_fun = interpol_kernel
        k_fun = k_sinc
        k_size = 8
        break
    else:
        return -1

    for out1 in range(0, dims[0]):    # rows
        for out2 in range(0, dims[1]):  # cols
              o1 = float(out1) # Is this necessary? I suspect it isn't. DL
              o2 = float(out2)
              in1 = mat[0] * o1 + mat[1]  * o2 + tr[0]
              in2 = -mat[1] * o1 + mat[0]  * o2 + tr[1]
              out_arr[out1*dims[1]+out2] = i_fun(dims, in_arr, in1, in2, k_size, k_fun, int_param, mode, miss_val)
    
    return
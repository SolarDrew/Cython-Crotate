# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:14:14 2013

@author: ajl7
"""
import sunpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io.idl import readsav as read
from pyrot import rotate_map as rotate

"""
==========
Test the Python rotate_Map() function to see if it looks like it's working.
==========
"""
testarr = np.zeros((101, 101))
testarr[40:61, :] = testarr[:81, 40:61] = 1.0
print testarr.sum()
testmap = sunpy.make_map(testarr)

fig = plt.figure()
original = fig.add_subplot(141)
plt.imshow(testarr, cmap=cm.coolwarm, vmin=-0.1, vmax=1.1, interpolation='none')
original.set_title('Original image')
plt.colorbar(orientation='horizontal')

#rads = 90.0 * np.float128(np.pi) / 180.0 #np.radians(20.0)
rads = np.radians(90.0)
#print type(rads)
#print rads, np.radians(90.0)
rotmap_test = rotate(testarr, rads, int_method='bicubic', int_param=-0.5)
print rotmap_test.sum()
rotated_py = fig.add_subplot(142)
plt.imshow(rotmap_test, cmap=cm.coolwarm, vmin=-0.1, vmax=1.1, 
           interpolation='none')
rotated_py.set_title('Image rotated in Python')
plt.colorbar(orientation='horizontal')

"""
==========
Same test for IDL's rot() function.
==========
"""
idl_arrs = read('idl_rot_test')
print idl_arrs['array'].sum()
print idl_arrs['rot'].sum()
rotated_idl = fig.add_subplot(143)
plt.imshow(idl_arrs['rot'], cmap=cm.coolwarm, vmin=-0.1, vmax=1.1, 
           interpolation='none')
rotated_idl.set_title('Image rotated in IDL')
plt.colorbar(orientation='horizontal')

diff = fig.add_subplot(144)
#plt.imshow(testmap.rotate(np.radians(20.0)), cmap=cm.coolwarm)
#diff.set_title('Image rotated in C')
plt.imshow(rotmap_test - idl_arrs['rot'], cmap=cm.coolwarm, 
           interpolation='none')#, vmax=1.1)
diff.set_title('Difference between rotations')
plt.colorbar(orientation='horizontal')

plt.show()
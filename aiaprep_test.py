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
testarr = np.zeros((101,101))
testarr[40:60, :] = testarr[: ,40:60] = 1.0

fig = plt.figure('Python rotate_map() function')
original = fig.add_subplot(231)
plt.imshow(testarr, cmap=cm.coolwarm)
original.set_title('Original image')
plt.colorbar(orientation='horizontal')

rotmap_test = rotate(testarr, np.radians(360.0), int_method='bicubic', int_param=-0.5)
rotated = fig.add_subplot(232)
plt.imshow(rotmap_test, cmap=cm.coolwarm)
rotated.set_title('Image rotated 360 degrees')
plt.colorbar(orientation='horizontal')

diff_im = testarr - rotmap_test
print 'SunPy full rotation difference range: {!s}.'.format(diff_im.max() - diff_im.min())
diff = fig.add_subplot(233)
plt.imshow(diff_im, cmap=cm.coolwarm)
diff.set_title('Difference between images')
plt.colorbar(orientation='horizontal')

original = fig.add_subplot(234)
plt.imshow(testarr, cmap=cm.coolwarm)
original.set_title('Original image')
plt.colorbar(orientation='horizontal')

rotmap_test = rotate(testarr, np.radians(10.0), int_method='bicubic', int_param=-0.5)
rotmap_test = rotate(rotmap_test, np.radians(-10.0), int_method='bicubic', int_param=-0.5)
rotated = fig.add_subplot(235)
plt.imshow(rotmap_test, cmap=cm.coolwarm)
rotated.set_title('Image rotated and derotated 10 degrees')
plt.colorbar(orientation='horizontal')

diff_im = testarr - rotmap_test
print 'SunPy back rotation difference range: {!s}.'.format(diff_im.max() - diff_im.min())
diff = fig.add_subplot(236)
plt.imshow(diff_im, cmap=cm.coolwarm)
diff.set_title('Difference between images')
plt.colorbar(orientation='horizontal')
plt.show()

"""
==========
Same test for IDL's rot() function.
==========
"""
idl_arrs = read('idl_fullrot_test')
fig = plt.figure('Core IDL rot() function')
original = fig.add_subplot(231)
plt.imshow(idl_arrs['array'], cmap=cm.coolwarm)
original.set_title('Original image')
plt.colorbar(orientation='horizontal')

rotated = fig.add_subplot(232)
plt.imshow(idl_arrs['fullrot'], cmap=cm.coolwarm)
rotated.set_title('Image rotated 360 degrees')
plt.colorbar(orientation='horizontal')

diff = fig.add_subplot(233)
diff_im = idl_arrs['fullrot_diff']
print 'IDL rot() full rotation difference range: {!s}.'.format(diff_im.max()-diff_im.min())
plt.imshow(idl_arrs['fullrot_diff'], cmap=cm.coolwarm)
diff.set_title('Difference between images')
plt.colorbar(orientation='horizontal')

idl_arrs2 = read('idl_backrot_test')
original = fig.add_subplot(234)
plt.imshow(idl_arrs['array'], cmap=cm.coolwarm)
original.set_title('Original image')
plt.colorbar(orientation='horizontal')

rotated = fig.add_subplot(235)
plt.imshow(idl_arrs2['backrot'], cmap=cm.coolwarm)
rotated.set_title('Image rotated and derotated 10 degrees')
plt.colorbar(orientation='horizontal')

diff = fig.add_subplot(236)
diff_im = idl_arrs2['backrot_diff']
print 'IDL rot() back rotation difference range: {!s}.'.format(diff_im.max()-diff_im.min())
plt.imshow(idl_arrs2['backrot_diff'], cmap=cm.coolwarm)
diff.set_title('Difference between images')
plt.colorbar(orientation='horizontal')
plt.show()
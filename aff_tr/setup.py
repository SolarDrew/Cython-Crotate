# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


sourcefiles = ['affine_transform.pyx']



setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=[Extension("affine_transform", sourcefiles, include_dirs=['./'])]
)
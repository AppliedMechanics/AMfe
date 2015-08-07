# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:01:13 2015

@author: johannesr
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("test_element", ['test_element.pyx']),
        Extension('element', ['../amfe/element.py']),
        Extension("test_memory_view", ['test_memory_view.pyx']),
                  ],

    # this is important that all np-headers are found of cimport numpy
    include_dirs = [numpy.get_include()],
    )

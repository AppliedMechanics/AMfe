# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:01:13 2015

@author: johannesr
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("test_element", ['test_element.pyx']),]
    )

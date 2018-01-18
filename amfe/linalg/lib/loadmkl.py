# Copyright (c) 2016 David Marchant
#
# Originally distributed under MIT License
#
# Modified by Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# and
# Redistributed under BSD-3-Clause License. See LICENSE-File for more information
#
from ctypes import CDLL
import sys, os

platform = sys.platform

libname = {'linux':'libmkl_rt.so', # works for python3 on linux
           'linux2':'libmkl_rt.so', # works for python2 on linux
           'darwin':'libmkl_rt.dylib',
           'win32':'mkl_rt.dll'}


def _load_mkl():
    
    try:
        # Look for MKL in path
        mkllib = CDLL(libname[platform])
    except:
        try:
            # Look for anaconda mkl
            if 'Anaconda' in sys.version:
                if platform in ['linux', 'linux2', 'darwin']:
                    libpath = ['/']+sys.executable.split('/')[:-2] + \
                              ['lib',libname[platform]]
                elif platform == 'win32':
                    libpath = sys.executable.split(os.sep)[:-1] + \
                              ['Library','bin',libname[platform]]
                mkllib = CDLL(os.path.join(*libpath))
        except Exception as e: 
            raise e

    return mkllib

mkllib = _load_mkl()

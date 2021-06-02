# Copyright (c) 2016 David Marchant
#
# Originally distributed under MIT License
#
# Modified by Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# and
# Redistributed under BSD-3-Clause License.
# See LICENSE-File for more information
#
from ctypes import CDLL
import sys
import os

platform = sys.platform

libname = {'linux': 'libmkl_rt.so',  # works for python3 on linux
           'linux2': 'libmkl_rt.so',  # works for python2 on linux
           'darwin': 'libmkl_rt.dylib',
           'win32': 'mkl_rt.dll'}


def get_mkl_lib():
    """
    Returns a ctypes.CDLL object containing mkl library if available on the
    system

    Returns
    -------
    mkllib: CDLL
        CDLL object containing mkl library
    """
    
    try:
        # Look for MKL in path
        mkllib = CDLL(libname[platform])
        if mkllib is None:
            raise ImportError('MKLlib could not be found on your system')
        return mkllib
    except Exception as e1:
        try:
            # Look for anaconda mkl
            if platform in ['linux', 'linux2', 'darwin']:
                try:
                    libpath = ['/']+sys.executable.split('/')[:-2] + \
                              ['lib', libname[platform]]
                    mkllib = CDLL(os.path.join(*libpath))
                    return mkllib
                except OSError:
                    try:
                        libpath = ['/']+sys.executable.split('/')[:-2] + \
                                  ['lib', libname[platform]+'.1']
                        mkllib = CDLL(os.path.join(*libpath))
                        return mkllib
                    except OSError:
                        raise e1

            elif platform == 'win32':
                try:
                    libpath = sys.executable.split(os.sep)[:-1] + \
                          ['Library', 'bin', libname[platform]]
                    mkllib = CDLL(os.path.join(*libpath))
                    return mkllib
                except OSError:
                    raise e1
            else:
                raise TypeError('platform {} unknown'.format(platform))

        except Exception:
            raise e1

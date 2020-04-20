# Copyright (c) 2016 David Marchant
#
# Originally distributed under MIT License
# Redistributed under BSD-3-Clause License. See LICENSE-File for more information
#
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# from future import standard_library
# standard_library.install_aliases()


from ctypes import POINTER, c_int, c_longlong
from .loadmkl import get_mkl_lib

# Two MKL PARDISO Routines are loaded from mkllib:
# pardisoinit: Initializes Intel MKL PARDISO with default parameters depending on the matrix type
# pardiso: Calculates the solution of a set of sparse linear equations with single or multiple rhs

try:
    # Load MKL lib
    mkllib = get_mkl_lib()

    # 1. pardisoinit
    pardisoinit = mkllib.pardisoinit
    pardisoinit.argtypes = [POINTER(c_longlong),
                            POINTER(c_int),
                            POINTER(c_int)]
    pardisoinit.restype = None

    # 2. pardiso
    pardiso = mkllib.pardiso
    pardiso.argtypes = [POINTER(c_longlong),  # pt
                        POINTER(c_int),       # maxfct
                        POINTER(c_int),       # mnum
                        POINTER(c_int),       # mtype
                        POINTER(c_int),       # phase
                        POINTER(c_int),       # n
                        POINTER(None),        # a
                        POINTER(c_int),       # ia
                        POINTER(c_int),       # ja
                        POINTER(c_int),       # perm
                        POINTER(c_int),       # nrhs
                        POINTER(c_int),       # iparm
                        POINTER(c_int),       # msglvl
                        POINTER(None),        # b
                        POINTER(None),        # x
                        POINTER(c_int)]       # error)
    pardiso.restype = None
except ImportError as e:
    raise e

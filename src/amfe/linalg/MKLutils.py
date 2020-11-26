# Copyright (c) 2016 David Marchant
#
# Originally distributed under MIT License
# Redistributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
MKLutils provides some functions for information about the MKL C-Library in general. Wrappers are implemented in other.

"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# from future import standard_library
# standard_library.install_aliases()

from ctypes import Structure, POINTER, c_int, c_char_p

try:
    from .lib.loadmkl import get_mkl_lib

    mkllib = get_mkl_lib()

    class PyMKLVersion(Structure):
        _fields_ = [('MajorVersion', c_int),
                    ('MinorVersion', c_int),
                    ('UpdateVersion', c_int),
                    ('ProductStatus', c_char_p),
                    ('Build', c_char_p),
                    ('Processor', c_char_p),
                    ('Platform', c_char_p)]
    _mkl_get_version = mkllib.mkl_get_version
    _mkl_get_version.argtypes = [POINTER(PyMKLVersion)]
    _mkl_get_version.restype = None

    def mkl_get_version():
        """
        mkl_get_version returns a string with version information about MKL C-Library

        Returns
        -------
        version : String
            MKL C-Library Version Number
        """
        MKLVersion = PyMKLVersion()
        _mkl_get_version(MKLVersion)
        version = {'MajorVersion': MKLVersion.MajorVersion,
                   'MinorVersion': MKLVersion.MinorVersion,
                   'UpdateVersion': MKLVersion.UpdateVersion,
                   'ProductStatus': MKLVersion.ProductStatus,
                   'Build': MKLVersion.Build,
                   'Platform': MKLVersion.Platform}

        versionString = 'Intel(R) Math Kernel Library Version {MajorVersion}.' \
                        '{MinorVersion}.{UpdateVersion} {ProductStatus} Build {Build} ' \
                        'for {Platform} applications'.format(**version)

        return versionString


    _mkl_get_max_threads = mkllib.mkl_get_max_threads
    _mkl_get_max_threads.argtypes = None
    _mkl_get_max_threads.restype = c_int

    def mkl_get_max_threads():
        """
        mkl_get_max_threads() returns the number of threads that can be used for running the MKL C-Library functions

        Returns
        -------
        max_threads : int
            Maximum number of threads that can be used for running the MKL C-Library functions
        """
        max_threads = _mkl_get_max_threads()
        return max_threads


    _mkl_set_num_threads = mkllib.mkl_set_num_threads
    _mkl_set_num_threads.argtypes = [POINTER(c_int)]
    _mkl_set_num_threads.restype = None

    def mkl_set_num_threads(num_threads):
        """
        mkl_set_num_threads(num_treads) sets the number of threads with which the library shall run

        Parameters
        ----------
        num_threads : int
            Number of threads that shall be used by the library

        Returns
        -------
        None

        """
        _mkl_set_num_threads(c_int(num_threads))

except ImportError as e:
    def mkl_get_version():
        raise ImportError('MKL lib could not be found on your system')

    def mkl_get_max_threads():
        raise ImportError('MKL lib could not be found on your system')

    def mkl_set_num_threads(num_threads):
        raise ImportError('MKL lib could not be found on your system')

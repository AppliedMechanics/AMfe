#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Tools for I/O module.
"""

from os import path, makedirs

__all__ = [
    'check_dir'
    ]


def check_dir(*filenames):
    """
    Check if path(s) exist; if not, given path(s) will be created.

    Parameters
    ----------
    *filenames : str or list of str
        String or list of strings containing path(s).

    Returns
    -------
    None
    """

    for filename in filenames:  # loop over files
        dir_name = path.dirname(filename)
        # check whether directory does exist
        if not path.exists(dir_name) or dir_name == '':
            makedirs(path.dirname(filename))  # if not, then create directory
            print('Created directory \'' + path.dirname(filename) + '\'.')
    return

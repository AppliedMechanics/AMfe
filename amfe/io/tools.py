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
from os.path import splitext, isfile, join, dirname

__all__ = [
    'amfe_dir',
    'check_dir',
    'insert_line_breaks_in_xml',
    'check_filename_or_filepointer'
    ]


def amfe_dir(filename=''):
    '''
    Return the absolute path of the filename given relative to the amfe directory.

    Parameters
    ----------
    filename : string, optional
        relative path to something inside the amfe directory.

    Returns
    -------
    dir : string
        string of the filename inside the AMFE-directory. Default value is '', so the AMFE-directory is returned.
    '''

    amfe_abs_path = dirname(dirname(dirname(__file__)))
    return join(amfe_abs_path, filename.lstrip('/'))


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


def insert_line_breaks_in_xml(root, level=0):
    """
    Prettify the XML Output before an xml tree is written to a file

    It inserts line breaks to the tree such that the tree will be better readable for humans

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root element of an xml.etree.ElementTree where the linebreak insertions shall start
    level : int
        If level = 0 (default) It starts with just line breaks and adds some blanks for children
        If level = 1 it is assumed thath the root element shall also get some blanks

    Returns
    -------
    None
    """
    i = "\n" + level*"  "
    if len(root):
        if not root.text or not root.text.strip():
            root.text = i + "  "
        if not root.tail or not root.tail.strip():
            root.tail = i
        for root in root:
            insert_line_breaks_in_xml(root, level + 1)
        if not root.tail or not root.tail.strip():
            root.tail = i
    else:
        if level and (not root.tail or not root.tail.strip()):
            root.tail = i


def check_filename_or_filepointer(ftype, open_func, argindex=0, writeable=False):
    """
    Decorator for functions that assume to get a filepointer as an argument. It makes this function also accepting
    filename strings instead of filepointers

    Parameters
    ----------
    ftype : class
        class of a FileObject that is accepted, e.g. tables.File or just File (for Pythons File implementation).
    open_func : function
        Handle to a function that can open the file and returns the right filepointer for the function that is decorated
        e.g. open, or tables.open_file
    argindex : int
        integer that describes at which position the filepointer argument is assumed in the decorated function
    writeable : bool (default: False)
        It sets a flag if the file should be open in just readable mode or write mode.
        If the file exists, the write mode is 'a' otherwise the writemode is 'w'

    Returns
    -------

    """
    def fp_decorator(fun):
        def func_wrapper(*args, **kwargs):
            filename_or_pointer = args[argindex]
            if isinstance(filename_or_pointer, str):
                if isfile(filename_or_pointer) and writeable:
                    mode = 'a'
                else:
                    if not splitext(filename_or_pointer)[1]:
                        raise ValueError('The given filename {} has no fileextension'.format(filename_or_pointer))
                    check_dir(filename_or_pointer)
                    if writeable:
                        mode = 'w'
                    else:
                        mode = 'r'
                with open_func(filename_or_pointer, mode=mode) as fp:
                    new_args = list()
                    new_args.extend(args[:argindex])
                    new_args.extend([fp])
                    new_args.extend(args[argindex+1:])
                    return fun(*new_args, **kwargs)
            elif isinstance(filename_or_pointer, ftype):
                new_args = list()
                new_args.extend(args[:argindex])
                new_args.extend([filename_or_pointer])
                new_args.extend(args[argindex + 1:])
                return fun(*new_args, **kwargs)
            else:
                raise ValueError('Filename must be valid path string or tables.File object')
        return func_wrapper
    return fp_decorator

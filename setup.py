# -*- coding: utf-8 -*-
"""
Setup file for automatic installation and distribution of AMfe. 
"""

import sys

try:
    from setuptools import setup
    from numpy.distutils.core import Extension, setup
except ImportError:
    from distutils.core import setup


ext_assembly = Extension(name='amfe.f90_assembly',
                         sources=['amfe/fortran/assembly.f90'],
                         language='f90',)
ext_element = Extension(name='amfe.f90_element',
                        sources=['amfe/fortran/element.pyf', 
                                 'amfe/fortran/element.f90'],
                        language='f90',)
ext_material = Extension(name='amfe.f90_material',
                         sources=['amfe/fortran/material.f90'],
                         language='f90',)
                         
ext_modules = [ext_assembly, ext_element, ext_material]


config = {
    'name': 'amfe',
    'version': '0.2',
    'description': 'Nonlinear Finite Element Code with simplicity in mind.',
    'author': 'Johannes Rutzmoser',
    'url': 'No URL provided yet',
    'download_url': 'Where to download it.',
    'author_email': 'johannes.rutzmoser@tum.de',
    'install_requires': ['numpy>=1.10', 'scipy>=0.17', 'pandas', 'h5py'],
    'tests_require': ['nose', 'sphinx>=1.3.0'],
    'packages': ['amfe'],
    'scripts': [],
    'entry_points': {},
    # 'ext_modules' : [ext_assembly, ext_element, ext_material],
}


no_fortran_str = '''

###############################################################################
############### Compilation of Fortran sources is disabled!  ##################
###############################################################################
'''

if 'no_fortran' in sys.argv:
    sys.argv.remove('no_fortran')
    print(no_fortran_str)
    setup(**config)
else:
    setup(ext_modules=ext_modules, **config)

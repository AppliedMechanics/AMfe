# -*- coding: utf-8 -*-
"""
"""

try:
    from setuptools import setup
    from numpy.distutils.core import Extension, setup
except ImportError:
    from distutils.core import setup


ext_assembly = Extension(name='amfe.f90_assembly',
                         sources=['amfe/fortran/assembly.f90'],
                         runtime_library_dirs=['amfe',], 
                         language='fortran')
ext_element = Extension(name='amfe.f90_element',
                        sources=['amfe/fortran/element.pyf', 
                                 'amfe/fortran/element.f90'])
ext_material = Extension(name='amfe.f90_material',
                         sources=['amfe/fortran/material.f90'])

config = {
    'name': 'amfe',
    'version': '0.2',
    'description': 'Nonlinear Finite Element Code with simplicity in mind',
    'author': 'Johannes Rutzmoser',
    'url': 'No URL provided yet',
    'download_url': 'Where to download it.',
    'author_email': 'johannes.rutzmoser@tum.de',
    'install_requires': ['numpy', 'scipy', 'pandas'],
    'tests_require': ['nose', 'sphinx>=1.3.0'],
    'packages': ['amfe'],
    'scripts': [],
    'entry_points': {},
    'ext_modules' : [ext_assembly, ext_element, ext_material],
}

setup(**config)

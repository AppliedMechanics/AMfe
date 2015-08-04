# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:06:20 2015

@author: johannesr
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Finite Element Code for simple cases',
    'author': 'Johannes Rutzmoser',
    'url': 'No URL provided yet',
    'download_url': 'Where to download it.',
    'author_email': 'johannes.rutzmoser@tum.de',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': [],
    'scripts': [],
    'name': 'amfe'
}

setup(**config)
# -*- coding: utf-8 -*-
"""
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    'name': 'amfe',
    'version': '0.2',
    'description': 'Nonlinear Finite Element Code with simplicity in mind',
    'author': 'Johannes Rutzmoser',
    'url': 'No URL provided yet',
    'download_url': 'Where to download it.',
    'author_email': 'johannes.rutzmoser@tum.de',
    'install_requires': ['numpy', 'scipy', 'pandas', 'sphinx>=1.3.0'],
    'tests_require': ['nose'],
    'packages': find_packages(where='amfe'),
    'scripts': [],
    'entry_points': {},
}

setup(**config)
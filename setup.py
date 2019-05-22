# -*- coding: utf-8 -*-
"""
Setup file for automatic installation and distribution of AMfe.
Run: 'python setup.py sdist' for Source Distribution
Run: 'python setup.py install' for Installation
Run: 'python setup.py bdist_wheel' for Building Binary-Wheel
    (recommended for windows-distributions)
    Attention: For every python-minor-version an extra wheel has to be built
    Use environments and install different python versions by using
    conda create -n python34 python=3.4 anaconda
Run: 'pip install wheelfile.whl' for Installing Binary-Wheel
Run: 'python setup.py bdist --format=<format> für Binary-Distribution:
    <format>=gztar|ztar|tar|zip|rpm|pgktool|sdux|wininst|msi
    Recommended: tar|zip|wininst (evtl. msi)
Run: 'python setup.py bdist --help-formats' to find out which distribution
    formats are available
"""

# Uncomment next line for debugging
# DISTUTILS_DEBUG='DEBUG'

import sys

def query_yes_no(question, default="yes"):
    '''
    Ask a yes/no question and return their answer.

    Parameters
    ----------
    question: String
        The question to be asked

    default: String "yes" or "no"
        The default answer

    Returns:
    --------
    answer: Boolean
        Answer: True if yes, False if no.
    '''

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'.\n")


config = {
    'name': 'amfe',
    'version': '1.0',
    'description': 'Nonlinear Finite Element Code with simplicity in mind.',
    'long_description': 'Nonlinear Finite Element Code for \
    Structural Dynamics. \
    This code includes Model Order Reduction Techniques for nonlinear Systems \
    such as DEIM or ECSW.',
    'author': 'Johannes Rutzmoser',
    'url': 'https://github.com/tum-am/amfe',
    'download_url': 'Where to download it.',
    'author_email': 'johannes.rutzmoser@tum.de',
    'maintainer': 'Christian Meyer',
    'maintainer_email': 'christian.meyer@tum.de',
    'python_requires': '>=3.7',
    'install_requires': ['numpy>=1.10', 'scipy>=0.17', 'pandas>=0.23', 'theano',
                         'h5py', 'matplotlib', 'vtk==8.1.2', 'tables'],  # carefull: condapackage is pytables not tables
    'tests_require': ['nose', 'sphinx==1.3.1', 'sphinx_rtd_theme', 'numpydoc'],
    'packages': ['amfe'],
    'scripts': [],
    'entry_points': {},
    'provides': 'amfe',
    'platforms': 'Linux, Windows',
    'license': 'BSD3-License'
}

no_fortran_str = '''

###############################################################################
############### Compilation of Fortran sources is disabled!  ##################
###############################################################################
'''

no_extension_str = '''

###############################################################################
############### Fortran-Extensions cannot be installed!      ##################
############### Install Numpy before installing AMfe         ##################
###############################################################################
'''

no_feti_str = '''

###############################################################################
###############    Import of PYFETI-library is disabled!     ##################
###############################################################################
'''

if __name__ == '__main__':
    if 'no_feti' in sys.argv:
        sys.argv.remove('no_feti')
        print(no_feti_str)
    else:
        pyfeti_ver = 'pyfeti>=0.12'
        pyfeti_repo = 'git+ssh://git@gitlab.lrz.de/AM/pyfeti.git#egg=pyfeti'

        if 'install_requires' in config:
            config['install_requires'] = config['install_requires'] + [pyfeti_ver]
        else:
            config['install_requires'] = [pyfeti_ver]
        if 'dependency_links' in config:
            config['dependency_links'] = config['dependency_links'] + [pyfeti_repo]
        else:
            config['dependency_links'] = [pyfeti_repo]

    if 'no_fortran' in sys.argv:
        sys.argv.remove('no_fortran')
        print(no_fortran_str)
        from setuptools import setup
        setup(**config)
    else:
        try:
            from setuptools import setup
            from numpy.distutils.core import Extension, setup
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

            setup(ext_modules=ext_modules, **config)

        except ImportError:
            # from distutils.core import setup
            from setuptools import setup
            print(no_extension_str)
            answer = query_yes_no('Fortran files cannot be installed. \
                                  It is recommended to abort installation and \
                                  first install numpy. Then retry \
                                  installation of AMfe. \
                                  Do you want to continue installation?', 'no')
            if answer:
                setup(**config)
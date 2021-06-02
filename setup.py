# coding=utf-8
import sys

"""
Setup file for automatic installation of AMfe in development mode.
Run: 'python setup.py sdist' for Source Distribution
Run: 'python setup.py install' for Installation
Run: 'python setup.py bdist_wheel' for Building Binary-Wheel
    (recommended for windows-distributions)
    Attention: For every python-minor-version an extra wheel has to be built
    Use environments and install different python versions by using
    conda create -n python34 python=3.4 anaconda
Run: 'pip install wheelfile.whl' for Installing Binary-Wheel
Run: 'python setup_develop.py bdist --format=<format> für Binary-Distribution:
    <format>=gztar|ztar|tar|zip|rpm|pgktool|sdux|wininst|msi
    Recommended: tar|zip|wininst (evtl. msi)
Run: 'python setup_develop.py bdist --help-formats' to find out which distribution
    formats are available
"""

# Uncomment next line for debugging
# DISTUTILS_DEBUG='DEBUG'


def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question and return their answer.

    Parameters
    ----------
    question: String
        The question to be asked

    default: String "yes" or "no"
        The default answer

    Returns
    -------
    answer: Boolean
        Answer: True if yes, False if no.
    """

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

if __name__ == '__main__':
    from setuptools.config import read_configuration
    from setuptools import find_packages
    config_raw = read_configuration('./meta.cfg')
    config = dict()
    config.update(config_raw['metadata'])
    config.update(config_raw['options'])
    packages = find_packages('src')
    package_dir = {'': 'src'}
    config['packages'] = packages
    config['package_dir'] = package_dir
    config['include_package_data'] = True
    config['zip_safe'] = False
    ext_modules = []

    if 'no_fortran' in sys.argv:
        sys.argv.remove('no_fortran')
        from setuptools import setup
        setup(**config)
    else:
        try:
            from numpy.distutils.core import Extension, setup
            ext_assembly = Extension(name='amfe.f90_assembly',
                                     sources=['src/amfe/fortran/assembly.f90'],
                                     language='f90',)
            ext_element = Extension(name='amfe.f90_element',
                                    sources=['src/amfe/fortran/element.pyf',
                                             'src/amfe/fortran/element.f90'],
                                    language='f90',)
            ext_material = Extension(name='amfe.f90_material',
                                     sources=['src/amfe/fortran/material.f90'],
                                     language='f90',)

            ext_modules = [ext_assembly, ext_element, ext_material]
            config.update({'ext_modules': ext_modules})
            setup(**config)

        except ModuleNotFoundError:
            # from distutils.core import setup
            print(no_extension_str)
            print('Fortran files cannot be installed. \
                                  It is recommended to abort installation and \
                                  first install numpy. Then retry \
                                  installation of AMfe.')
            raise ImportError('Numpy not found')


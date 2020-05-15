.. _installation_sdist:

Non-Developer Source-Code
^^^^^^^^^^^^^^^^^^^^^^^^^

Installation
""""""""""""

1. Download Source-Release on https://github.com/AppliedMechanics/AMfe
2. Unzip Archive
3. Change into the directory with setup.py-file
4. Run::

    python setup.py install
    

This installs AMfe in your python environment.
The extracted Archive can be deleted after installation.


Fortran-Extensions
""""""""""""""""""

The Amfe package includes some fortran routines which has to be built in the development version.
Thus, you will need a fortran compiler, such as gfortran to compile the fortran-sources.
If you do not want to compile the fortran routines you may also use equivalent python routines
which do not need to be compiled.
However, this is not recommended if you want to run large finite element models because the
performance of fortran routines is much better.

Use::

  python setup.py develop no_fortran
 
in step 2 with no_fortran flag, if you do not want to use fortran-routines.



Additional hints for windows users
""""""""""""""""""""""""""""""""""

The compilation of fortran extensions may cause problems on windows-platforms.

.. _installation_git:

Git clone/fork
^^^^^^^^^^^^^^

Installation
""""""""""""

For developers it is recommended to clone or fork the github repository of AMfe.
Git is a version control system (VCS) which helps to coordinate work of many developers.
But it is also helpful if you work on your own.
For those who are not familiar with git, visit https://youtu.be/Qthor07loHM
for a good video tutorial.

Before you can use git, you must have installed git on your computer.
Git is available for windows and for linux/mac.

To install AMfe via git as developer follow these instructions:

    1.Clone or Fork the AMfe Project via::
    
        git clone https://github.com/AppliedMechanics/AMfe.git
    
    2.Change into the directory of your local AMfe repository
    
    3.For installing the package type

    :: 

        python setup.py develop

    This installs AMfe as developer version on your computer.
    This means that all changes you make to your files in the repository will directly affect your run.

    4. If you also want to install a documentation, change to the docs/ directory
    in your local repository and run

    ::

        make html

    The documentation will be built in the docs/_build folder. Just open docs/_build/index.html in a browser.


Fortran-Extensions
""""""""""""""""""

The Amfe package includes some fortran routines which has to be built in the development version.
Thus, you will need a fortran compiler, such as gfortran to compile the fortran-sources.
If you do not want to compile the fortran routines you may also use equivalent python routines
which do not need to be compiled.
However this is not recommended if you want to run large finite element models because the
performance of fortran routines is much better.

Use::

  python setup.py develop no_fortran
 
in step 2 with no_fortran flag, if you do not want to use fortran-routines.



Additional hints for windows users
""""""""""""""""""""""""""""""""""

The compilation of fortran extensions may cause problems on windows-platforms.
Here are some hints on how to install the routines on windows:


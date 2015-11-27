AMfe - Finite Element Research Code at the Chair of Applied Mechanics
---------------------------------------------------------------------

(c) 2015 Lehrstuhl für Angewandte Mechanik, Technische Universität München

This Finite Element Research code is developed, maintained and used by a part of the numerics group of AM. 


Overview: 
----------
1.  [Documentation](#1-documentation)
2.  [Fortran-Routines](#2-fortran-routines)
3.  [Hints](#3-hints)


1. Documentation
----------------
Further documentation to this code is in the folder `docs/`.
For building the documentation, the following software packages have to be installed:

   - Python version 3.4 or higher
   - Python package sphinx 1.3 oder higher (potentially has to be installed using pip3). Version 1.2 does not work!
   - Python-package numpydoc

This documentation can be built with entering 
```bash
make html
```
to the console in folder `docs/`.
   

2. Fortran-Routines
-------------------
In order to use the fast Fortran routines, which are used within the assembly process,
the scirpt `install_fortran_routines.sh` in folder `f2py/` has to be executed.
A working Fortan compiler (e.g. `gfortran`, `gfortran-4.8`) has to be installed. 

   
3. Hints
-----------

### Sphinx:

`sphinx` has to be installed for `pyhton3`. Maybe, `sphinx` was automatically intalled for `python2`. 
Using `python3`, one can test which `sphinx`-version is installed:
```python
python3
>>> import sphinx
>>> sphinx.__version__
```
The version shuld be at least `'1.3.1'`.


### Spyder:

For use within the code, the Integrated Development Environment `spyder3` for `python3` is recommended.
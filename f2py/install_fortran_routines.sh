#!/bin/bash

echo 'Install the following fortran routines into AMfe:\nAssembly\nElement'

# Compile the fortran routines and install them into the amfe folder
f2py3 -c  --fcompiler=gnu95 -m f90_assembly assembly.f90 && cp f90_assembly*.so ../amfe/
f2py3 -c  --fcompiler=gnu95 -m f90_element element.f90 && cp f90_element*.so ../amfe/
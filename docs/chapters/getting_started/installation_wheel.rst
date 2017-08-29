.. _installation_wheel:

Already built code
^^^^^^^^^^^^^^^^^^

If you are not familiar with compilers or if you want an installation "out of the box",
installing AMfe by using a python-wheel is best for you.
After installation of a python distribution of your choice (e.g. Anaconda), you
first have to download a python-wheel-file (.whl) on https://github.com/AppliedMechanics/AMfe
which is suitable to your platform and python version.
Then you can easy install AMfe by running::

    pip install wheel_file.whl
    
where "wheel_file" is the name of your downloaded wheel-file.

You do not need to compile any sources.
But you should also install pyMKL for using the MKL-speed-up for your solver
(see pyMKL-section for details on how to install).

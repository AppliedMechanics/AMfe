# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

"""
Forces module of AMfe.

In this module you can find some examples for force-functions. You can apply them to your model as Neumann-conditions.
"""

import numpy as np

__all__ = ['constant_force',
           'linearly_increasing_force',
           'triangular_force',
           'step_force']


def constant_force(value):
    """
    Returns a constant force function

    Parameters
    ----------
    value: float
        value of constant force

    Returns
    -------
    f: callable
        function f(t)

    """
    def f(t):
        return value
    return f


def linearly_increasing_force(t_start, t_end, f_max):
    """
    Returns a linearly increasing force:

            f_max at t_end
            -------------
           /
          / |
         /  |
        /   |
    ----    |__ t_end
        |__ t_start

    Parameters
    ----------
    t_start: float
        time where linear force starts to raise
    t_end: float
        time where force reaches maximum
    f_max: float
        maximum value of force

    Returns
    -------
    f: callable
        function f(t)
    """
    def f(t):
        if t <= t_start:
            return 0.0
        elif t_start < t <= t_end:
            return f_max * (t-t_start) / (t_end-t_start)
        else:
            return f_max
    return f


def triangular_force(t_start, t_max, t_end, f_max):
    """
    Returns a triangular force:

            f_max at t_max
           /\
          /  \
         /    \
        /      \
    ----        -----
        |       |
        |       |__ t_end
        |__ t_start


    Parameters
    ----------
    t_start: float
        time where triangular force starts to raise
    t_max: float
        time where force reaches maximum
    t_end: float
        time where force is gone to zero again
    f_max: float
        maximum value of force

    Returns
    -------
    f: callable
        function f(t)
    """
    def f(t):
        if t_start < t <= t_max:
            return f_max * (t-t_start) / (t_max-t_start)
        elif t_max < t < t_end:
            return f_max * (1.0 - (t - t_max) / (t_end - t_max))
        else:
            return 0.0
    return f


def step_force(t_start=0.0, f_max=1.0, t_end=np.inf):
    """

    Parameters
    ----------
    t_start: float
        start where step starts
    f_max: float
        maximum value of the step
    t_end: float or np.inf
        time where step goes to zero again (default: np.inf)

    Returns
    -------
    f: callable
        function f(t)
    """
    def f(t):
        if t_start <= t <= t_end:
            return f_max
        else:
            return 0.0
    return f

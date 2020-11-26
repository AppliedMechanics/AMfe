#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

__all__ = ['NullAccelerationInitializer',
           'LinearAccelerationInitializer'
           ]


class NullAccelerationInitializer:
    def __init__(self):
        return

    def get_acceleration(self, t0, q0, dq0):
        return np.zeros_like(q0)


class LinearAccelerationInitializer:
    def __init__(self, M, f_int, f_ext, K, D, solve_func, solve_function_kwargs):
        self.M = M
        self.f_int = f_int
        self.f_ext = f_ext
        self.K = K
        self.D = D
        self.solve_function = solve_func
        self.solve_function_kwargs = solve_function_kwargs

    def get_acceleration(self, t0, q0, dq0):
        A = self.M(q0, dq0, t0)
        b = self.f_ext(q0, dq0, t0) - self.D(q0, dq0, t0) @ dq0 - self.f_int(q0, dq0, t0)
        return self.solve_function(A, b, **self.solve_function_kwargs)

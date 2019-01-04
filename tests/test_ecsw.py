# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import numpy as np
from numpy.testing import assert_
from numpy.linalg import norm

from amfe.hyper_red.ecsw import sparse_nnls


class TestNnls(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nnls(self):
        # Copyright Notice:
        #   The Nnls testcase is a modified version of the nnls test case of the Scipy-package.
        #   This was distributed under BSD-3 License
        # Copyright(c) 2001, 2002 Enthought, Inc.
        # All rights reserved.
        #
        # Copyright (c) 2003-2019 SciPy Developers.
        # All rights reserved.
        #
        # Author: Uwe Schmitt
        # Sep 2008
        #

        # Build a matrix a
        a = np.arange(25.0).reshape(-1, 5)
        # Build a vector x
        x = np.arange(5.0)
        # Calculate the correct right hand side
        y = np.dot(a, x)
        # Calculate tau from residual tolerance tol = tau * norm(y)
        tol = 1e-7
        tau = tol/norm(y)

        # run algorithm
        x, stats = sparse_nnls(a, y, tau)
        # get last residual before return
        res = stats[-1][1]
        # test if residual is smaller than desired tolerance
        assert_(res <= tol)
        assert_(norm(np.dot(a, x.toarray()).reshape(-1)-y) <= 1e-7)
        # test if all x are greater equal zero
        np.all(np.greater_equal(x.toarray(), 0.0))

        # make a second test with random a and x >= 0
        a = np.random.rand(5, 5)
        x = np.random.rand(5)
        # Calculate the correct right hand side
        y = np.dot(a, x)
        # Calculate tau from residual tolerance tol = tau * norm(y)
        if norm(y) != 0:
            tau = 1e-1/norm(y)
        else:
            tau = 1e-8
        # run algorithm
        x, stats = sparse_nnls(a, y, tau)
        # get last residual before return
        res = stats[-1][1]
        # test if residual is smaller than desired tolerance
        assert_(res < 1e-1)
        assert_(norm(np.dot(a, x.toarray()).reshape(-1)-y) < 1e-1)
        # test if all x are greater equal zero
        np.all(np.greater_equal(x.toarray(), 0.0))

        # run algorithm
        x, stats = sparse_nnls(a, y, tau)
        # get last residual before return
        res = stats[-1][1]
        # test if residual is smaller than desired tolerance
        assert_(res <= tol)
        assert_(norm(np.dot(a, x.toarray()).reshape(-1) - y) <= 1e-7)
        # test if all x are greater equal zero
        np.all(np.greater_equal(x.toarray(), 0.0))

        # make a fourth test that does not converge
        a = np.array([[0.21235441, 0.32701625, 0.67680346, 0.72724123, 0.51983536],
                      [0.82603172, 0.76654767, 0.69746447, 0.58220156, 0.2564705 ],
                      [0.04594648, 0.78409449, 0.85036132, 0.4888821 , 0.92390904],
                      [0.10404788, 0.37767343, 0.30689839, 0.77633873, 0.42464905],
                      [0.66897911, 0.59824198, 0.60212744, 0.02402656, 0.75641132]])
        y = np.array([0., 0., 0., 0., 0.19731525])
        tau = 1e-1/norm(y)

        # this should not converge test if RuntimeError is raised
        with self.assertRaises(RuntimeError):
            sparse_nnls(a, y, tau)

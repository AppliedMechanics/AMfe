# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import numpy as np
from scipy.linalg import subspace_angles, solve, toeplitz, eigh
from numpy.testing import assert_allclose

from amfe.linalg.tools import arnoldi
from amfe.mor.reduction_basis import krylov_basis, pod, modal_derivatives,\
    static_derivatives, shifted_static_derivatives, modal_derivatives_cruz


class TestArnoldi(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_arnoldi(self):
        dim = 10
        A = np.arange(dim**2).reshape(dim, dim)
        b = np.zeros(dim, dtype=float)
        b[0] = 1.0
        n = 3
        V = arnoldi(A, b, n)
        # check orthogonality
        assert_allclose(V.T.dot(V), np.identity(n), atol=1e-12)
        # check subspace angles
        V_sub = np.zeros((dim, n), dtype=float)
        V_sub[:, 0] = b
        V_sub[:, 1] = A.dot(b)
        V_sub[:, 2] = A.dot(V_sub[:, 1])
        angles = subspace_angles(V, V_sub)
        assert_allclose(angles, np.zeros(n, dtype=float), atol=1e-12)

        # Test without orthogonalization
        V = arnoldi(A, b, n, orthogonal=False)
        # check normals
        for v in V.T:
            assert_allclose(np.linalg.norm(v), 1.0)
        # check subspace angles
        V_sub = np.zeros((dim, n), dtype=float)
        V_sub[:, 0] = b
        V_sub[:, 1] = A.dot(b)
        V_sub[:, 2] = A.dot(V_sub[:, 1])
        angles = subspace_angles(V, V_sub)
        assert_allclose(angles, np.zeros(n, dtype=float), atol=1e-12)


class TestKrylovBasis(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_krylov_basis(self):
        dim = 10
        m_toeplitz_arr = np.zeros(dim)
        m_toeplitz_arr[0] = 2.0
        m_toeplitz_arr[1] = -1.0
        M = toeplitz(m_toeplitz_arr)

        k_toeplitz_arr = np.zeros(dim)
        k_toeplitz_arr[0] = 5.0
        k_toeplitz_arr[1] = -2.0
        k_toeplitz_arr[1] = -0.5
        K = toeplitz(k_toeplitz_arr)

        b = np.zeros(dim, dtype=float)
        b[0] = 1.0
        n = 3

        V = krylov_basis(M, K, b, n, mass_orth=False)
        # Check if K^{-1} b is in basis
        U = K.dot(V)
        angle = subspace_angles(U, b.reshape(-1, 1))
        assert_allclose(angle, 0.0, atol=1e-10)

        # Check if K^{-1} M K^{-1} b is in basis
        u = solve(K, M.dot(solve(K, b)))
        angle = subspace_angles(u.reshape(-1, 1), V)
        assert_allclose(angle, 0.0, atol=1e-10)

        # check orthogonality and shape
        assert_allclose(V.T.dot(V), np.identity(n), atol=1e-12)
        rows, cols = V.shape
        self.assertEqual(rows, dim)
        self.assertEqual(cols, n)

        # check m-orthogonality and shape
        V = krylov_basis(M, K, b, n, mass_orth=True, n_iter_orth=2)
        assert_allclose(V.T.dot(M).dot(V), np.identity(n), atol=1e-12)
        rows, cols = V.shape
        self.assertEqual(rows, dim)
        self.assertEqual(cols, n)

        # check different number of moments
        n = 5
        V = krylov_basis(M, K, b, n, mass_orth=False)
        assert_allclose(V.T.dot(V), np.identity(n), atol=1e-12)
        rows, cols = V.shape
        self.assertEqual(rows, dim)
        self.assertEqual(cols, n)

        # check if it also works with matrix inputs
        no_of_inputs = 2
        B = np.zeros((dim, no_of_inputs))
        B[0, 0] = 1.0
        B[3, 1] = 1.0
        # check orthogonality and shape
        V = krylov_basis(M, K, B, n, mass_orth=False)
        assert_allclose(V.T.dot(V), np.identity(n*no_of_inputs), atol=1e-12)
        rows, cols = V.shape
        self.assertEqual(rows, dim)
        self.assertEqual(cols, n*no_of_inputs)
        # check m-orthogonality and shape
        V = krylov_basis(M, K, B, n, mass_orth=True)
        assert_allclose(V.T.dot(M).dot(V), np.identity(n * no_of_inputs), atol=1e-12)
        rows, cols = V.shape
        self.assertEqual(rows, dim)
        self.assertEqual(cols, n * no_of_inputs)


class TestPOD(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pod(self):
        S = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                      [2.5, 1.5, 0.0, 0.0, 0.0, 0.5],
                      [4.0, 0.0, 0.0, 4.00000000001, 0.0, 0.0]]).T

        sigma, V = pod(S, 2)
        angles = subspace_angles(V, S)
        assert_allclose(angles, 0.0, atol=1e-14)
        self.assertEqual(V.shape[1], 2)

        sigma, V = pod(S, 4)
        angles = subspace_angles(V, S)
        assert_allclose(angles, 0.0, atol=1e-14)
        self.assertEqual(V.shape[1], 4)
        print(sigma)

        sigma, V = pod(S, tol=1e-10)
        self.assertEqual(V.shape[1], 3)


class TestModalDerivatives(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_modal_derivatives(self):
        n = 3

        def K_func(u):
            K = np.array(
                [[50.0 - u[0], -20.0 + u[0] + u[1], 0.0], [-20.0 + u[0] + u[1], 40.0 - u[1], -20.0 + u[1] + u[2]],
                 [0.0, -20.0 + u[1] + u[2], 30.0 - u[2]]], dtype=float)
            return K

        M = np.diag([2.0, 2.0, 2.0])

        u0 = np.zeros(n, dtype=float)
        K0 = K_func(u0)
        lambda_, V0 = eigh(K0, M)
        for i, v0 in enumerate(V0.T):
            V0[:, i] = v0 / np.sqrt(v0.dot(M).dot(v0))
        omega = np.sqrt(lambda_)
        delta = 1e-6

        Theta = modal_derivatives(V0, omega, K_func, M, h=delta, symmetric=False)
        results = np.zeros((n, n), dtype=float)
        for j in range(n):  # Iterate over Directions
            _, Vplus = eigh(K_func(u0 + delta * V0[:, j]), M)
            _, Vminus = eigh(K_func(u0 - delta * V0[:, j]), M)
            _, Vminus = eigh(K_func(u0), M)
            for k, (vp, vm) in enumerate(zip(Vplus.T, Vminus.T)):
                Vplus[:, k] = vp / np.sqrt(vp.dot(M).dot(vp))
                Vminus[:, k] = vm / np.sqrt(vm.dot(M).dot(vm))
                Vminus[:, k] = Vminus[:, k] * np.sign(np.dot(Vminus[:, k], Vplus[:, k]))
            Vdiff = (Vplus - Vminus) / (2 * delta)
            for i in range(n):  # Iterate over Eigenvectors
                dotproduct = Vdiff[:, i].dot(Theta[:, i, j])
                cosine = np.clip(dotproduct / np.linalg.norm(Vdiff[:, i]) / np.linalg.norm(Theta[:, i, j]), -1.0,
                                 1.0)
                degrees = np.rad2deg(np.arccos(cosine))
                results[i, j] = degrees

        assert_allclose(results, np.zeros_like(results), atol=1e-5)


class TestStaticDerivatives(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_static_derivatives(self):
        n = 3

        def K_func(u):
            K = np.array(
                [[50.0 - u[0], -20.0 + u[0] + u[1], 0.0], [-20.0 + u[0] + u[1], 40.0 - u[1], -20.0 + u[1] + u[2]],
                 [0.0, -20.0 + u[1] + u[2], 30.0 - u[2]]], dtype=float)
            return K

        M = np.diag([2.0, 2.0, 2.0])

        u0 = np.zeros(n, dtype=float)
        K0 = K_func(u0)
        lambda_, V0 = eigh(K0, M)
        for i, v0 in enumerate(V0.T):
            V0[:, i] = v0 / np.sqrt(v0.dot(M).dot(v0))

        delta = 1e-6

        Theta = static_derivatives(V0, K_func, h=delta, symmetric=False)
        residuals = np.zeros((n, n), dtype=float)
        for j in range(n):  # Iterate over Directions
            Kplus = K_func(u0 + delta * V0[:, j])
            Kminus = K_func(u0 - delta * V0[:, j])
            Kdiff = (Kplus - Kminus) / (2 * delta)
            for i in range(n):  # Iterate over Eigenvectors
                # K Theta = - dK/dv * v    => K*Theta + dK/dv = 0.0
                residuals[i, j] = np.linalg.norm(K0.dot(Theta[:, i, j]) + Kdiff.dot(V0[:, i]))

        assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)


class TestShiftedStaticDerivatives(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_shifted_static_derivatives_and_cruz(self):
        n = 3

        def K_func(u):
            K = np.array(
                [[80.0 - u[0], -25.0 + u[0] + u[1], 0.0], [-25.0 + u[0] + u[1], 40.0 - u[1], -20.0 + u[1] + u[2]],
                 [0.0, -20.0 + u[1] + u[2], 30.0 - u[2]]], dtype=float)
            return K

        M = np.diag([2.0, 2.0, 2.0])

        u0 = np.zeros(n, dtype=float)
        K0 = K_func(u0)
        lambda_, V0 = eigh(K0, M)
        for i, v0 in enumerate(V0.T):
            V0[:, i] = v0 / np.sqrt(v0.dot(M).dot(v0))

        omega = np.array([np.sqrt(om) for om in lambda_])

        Shifts_a = np.zeros((len(omega), len(omega)), dtype=float)
        Shifts_b = np.zeros((len(omega), len(omega)), dtype=float)
        for i in range(len(omega)):
            for j in range(len(omega)):
                Shifts_a[i, j] = omega[i] + omega[j]
                Shifts_b[i, j] = omega[i] - omega[j]

        delta = 1e-9

        Theta_a = shifted_static_derivatives(V0, K_func, M, Shifts_a, h=delta, symmetric=False)
        residuals = np.zeros((n, n), dtype=float)
        for j in range(n):  # Iterate over Directions
            Kplus = K_func(u0 + delta * V0[:, j])
            Kminus = K_func(u0 - delta * V0[:, j])
            Kdiff = (Kplus - Kminus) / (2 * delta)
            for i in range(n):  # Iterate over Eigenvectors
                # (K - s_ij**2 M) Theta = - dK/dv * v    => K*Theta + dK/dv = 0.0
                residuals[i, j] = np.linalg.norm((K0 - Shifts_a[i, j]**2 * M).dot(Theta_a[:, i, j]) + Kdiff.dot(V0[:, i]))

        assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)

        # test special derivatives by cruz:
        Theta_actual, Theta_tilde_actual = modal_derivatives_cruz(V0, K_func, M, omega, h=delta, symmetric=False)

        Theta_tilde_desired = shifted_static_derivatives(V0, K_func, M, Shifts_b, h=delta, symmetric=False)

        for t_actual, t_desired in zip(Theta_actual.reshape(-1, 1), Theta_a.reshape(-1, 1)):
            assert_allclose(np.linalg.norm(t_actual-t_desired), 0.0, atol=1e-9)
        for t_actual, t_desired in zip(Theta_tilde_actual.reshape(-1, 1), Theta_tilde_desired.reshape(-1, 1)):
            assert_allclose(np.linalg.norm(t_actual-t_desired), 0.0, atol=1e-9)


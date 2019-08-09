from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from amfe.solver.integrator import *
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from amfe.linalg.linearsolvers import ScipySparseLinearSolver


def M(q, dq, t):
    return 2 * csr_matrix(np.eye(4))


def K(q, dq, t):
    return csr_matrix(np.array([[2, -0.3, 0, 0], [-0.3, 4, -0.3, 0], [0, -0.3, 4, -0.3], [0, 0, -0.3, 2]])) + np.diag(q)


def D(q, dq, t):
    return csr_matrix(np.array([[0.2, 0.1, 0, 0], [0.1, 0.2, 0.1, 0], [0, 0.1, 0.2, 0.1], [0, 0, 0.1, 0.2]]))


def f_int(q, dq, t):
    return K(q, dq, t) @ q + D(q, dq, t) @ dq


def f_ext(q, dq, t):
    f = 2 * t
    return np.array([0, 0, 0, f])


class GeneralizedAlphaTest(TestCase):
    def setUp(self):
        self.integrator = GeneralizedAlpha(M, f_int, f_ext, K, D)
        self.integrator.dt = 0.1

    def tearDown(self):
        pass

    def test_set_integration_parameters(self):
        alpha_m = 0.3
        alpha_f = 0.4
        beta = 0.5
        gamma = 0.6

        integrator = GeneralizedAlpha(M, f_int, f_ext, K, D, alpha_m, alpha_f, beta, gamma)

        self.assertEqual(integrator.alpha_m, alpha_m)
        self.assertEqual(integrator.alpha_f, alpha_f)
        self.assertEqual(integrator.beta, beta)
        self.assertEqual(integrator.gamma, gamma)

    def test_get_midstep(self):
        q_0 = np.array([0, 0.2, 0, 0.1])
        q_1 = np.array([0.1, 0, -0.25, 0.4])
        alpha = 0.4
        q_mid = self.integrator._get_midstep(alpha, q_0, q_1)

        assert_allclose(q_mid, np.array([0.06, 0.08, -0.15, 0.28]))

    def test_set_prediction(self):
        t_0 = 0.0
        q_0 = np.array([0, 0, 0, 0], dtype=float)
        dq_0 = np.array([0, 0, 0, 0], dtype=float)
        ddq_0 = np.array([0, 0, 0, 1], dtype=float)

        self.integrator.set_prediction(q_0, dq_0, ddq_0, t_0)

        self.assertEqual(self.integrator.t_p, 0.1)
        assert_allclose(self.integrator.q_p, np.array([0, 0, 0, 0]))
        assert_allclose(self.integrator.dq_p, np.array([0, 0, 0, 0.00025]))
        assert_allclose(self.integrator.ddq_p, np.array([0, 0, 0, -0.805], dtype=float))

    def test_residual(self):
        t_0 = 0.0
        q_0 = np.array([0, 0, 0, 0], dtype=float)
        dq_0 = np.array([0, 0, 0, 0], dtype=float)
        ddq_0 = np.array([0, 0, 0, 1], dtype=float)

        self.integrator.set_prediction(q_0, dq_0, ddq_0, t_0)

        q_n = np.array([0, 0.1, 0.2, 0.3])

        res = self.integrator.residual(q_n)

        assert_allclose(res, np.array([[-0.015789, 0.181717, 0.368988, 0.113904]]), 1e-06, 1e-06)

    def test_jacobian(self):
        t_0 = 0.0
        q_0 = np.array([0, 0, 0, 0], dtype=float)
        dq_0 = np.array([0, 0, 0, 0], dtype=float)
        ddq_0 = np.array([0, 0, 0, 1], dtype=float)

        self.integrator.set_prediction(q_0, dq_0, ddq_0, t_0)

        q_n = np.array([0, 0.1, 0.2, 0.3])

        jac = self.integrator.jacobian(q_n)

        assert_allclose(jac, np.array([[421.152632, 0.892105, 0, 0], [0.892105, 422.232964, 0.892105, 0],
                                       [0, 0.892105, 422.260665, 0.892105], [0, 0, 0.892105, 421.235734]]), 1e-06)

    def test_set_correction(self):
        t_0 = 0.0
        q_0 = np.array([0, 0, 0, 0], dtype=float)
        dq_0 = np.array([0, 0, 0, 0], dtype=float)
        ddq_0 = np.array([0, 0, 0, 1], dtype=float)

        self.integrator.set_prediction(q_0, dq_0, ddq_0, t_0)

        q_p = np.array([0, 0.1, -0.2, 0.3])

        self.integrator.set_correction(q_p)

        self.assertEqual(self.integrator.t_p, 0.1)
        assert_allclose(self.integrator.q_p, np.array([0, 0.1, -0.2, 0.3]))
        assert_allclose(self.integrator.dq_p, np.array([0, 1.995, -3.99, 5.98525]))
        assert_allclose(self.integrator.ddq_p, np.array([0, 36.1, -72.2, 107.495]))


class NewmarkBetaTest(TestCase):
    def setUp(self):
        self.integrator = NewmarkBeta(M, f_int, f_ext, K, D)
        self.integrator.dt = 0.1

    def tearDown(self):
        pass

    def test_set_integration_parameters(self):
        beta = 0.5
        gamma = 0.6

        integrator = NewmarkBeta(M, f_int, f_ext, K, D, beta, gamma)

        self.assertEqual(integrator.beta, beta)
        self.assertEqual(integrator.gamma, gamma)


class WBZAlphaTest(TestCase):
    def setUp(self):
        self.integrator = WBZAlpha(M, f_int, f_ext, K, D)
        self.integrator.dt = 0.1

    def tearDown(self):
        pass

    def test_set_integration_parameters(self):
        self.assertAlmostEqual(self.integrator.alpha_m, -0.05263157894736841)
        self.assertAlmostEqual(self.integrator.alpha_f, 0.0)
        self.assertAlmostEqual(self.integrator.beta, 0.27700831)
        self.assertAlmostEqual(self.integrator.gamma, 0.55263157894736841)


class HHTAlphaTest(TestCase):
    def setUp(self):
        self.integrator = HHTAlpha(M, f_int, f_ext, K, D)
        self.integrator.dt = 0.1

    def tearDown(self):
        pass

    def test_set_integration_parameters(self):
        self.assertAlmostEqual(self.integrator.alpha_m, 0.0)
        self.assertAlmostEqual(self.integrator.alpha_f, 0.052631579)
        self.assertAlmostEqual(self.integrator.beta, 0.27700831024930744)
        self.assertAlmostEqual(self.integrator.gamma, 0.55263157894736841)


class LinearOneMassOscillatorTest(TestCase):
    r"""
               _____
    /|     k  |     |
    /|-/\/\/\-|  m  |
    /|        |_____|

    """
    def setUp(self):
        self.m = 0.1
        self.k = 0.5

        def M(q, dq, t):
            return np.array([self.m])

        def K(q, dq, t):
            return np.array([self.k])

        def f_int(q, dq, t):
            return np.array([self.k]).dot(q)

        def f_ext(q, dq, t):
            return np.array([0])

        def D(q, dq, t):
            return np.array([0])

        self.integrator = GeneralizedAlpha(M, f_int, f_ext, K, D)
        self.integrator.dt = 0.001

        self.integration_stepper = LinearIntegrationStepper(self.integrator)
        self.linear_solver = ScipySparseLinearSolver()
        self.integration_stepper.linear_solver_func = self.linear_solver.solve

    def test_linear_oscillator(self):
        t_end = 2
        t0 = 0.0
        q0 = 0.1
        q = np.array([q0])
        dq = np.array([0.0])
        ddq = np.array([0.0])
        N_dt = int((t_end-t0)/self.integrator.dt)

        t = t0
        q_numerical = np.zeros(N_dt)
        q_numerical[0] = q
        q_analytical = q_numerical.copy()
        for i in range(1, N_dt):
            t, q, dq, ddq = self.integration_stepper.step(t, q, dq, ddq)
            q_numerical[i] = q[0]
            q_analytical[i] = q0*np.cos(np.sqrt(self.k/self.m)*t)

        for num, ana in zip(q_numerical, q_analytical):
            assert_allclose(num, ana, atol=1e-5)

        def plot_oscillator_path(u_plot, label_name):
            plt.plot(range(0, N_dt), u_plot, label=label_name)
            plt.title('Linear oscillator-test')
            return

        # UNCOMMENT THESE LINES IF YOU LIKE TO SEE A TRAJECTORY (THIS CAN NOT BE DONE FOR GITLAB-RUNNER
        plot_oscillator_path(q_analytical, 'Analytisch')
        plot_oscillator_path(q_numerical, 'Numerisch')
        plt.legend()
        # plt.show()





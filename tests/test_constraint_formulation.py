"""Test routines that compares nullspace and lagrange formulations on simple pendulum test
It is also an test if the constraint managerm, constraints and constraint formulations can work together"""

from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt

from amfe.constraint.constraint_formulation_lagrange_multiplier import SparseLagrangeMultiplierConstraintFormulation
from amfe.constraint.constraint_formulation_nullspace_elimination import NullspaceConstraintFormulation
from amfe.constraint.constraint_formulation import ConstraintFormulationBase
from amfe.constraint.constraint_manager import ConstraintManager
from amfe.solver.nonlinear_solver import NewtonRaphson
from amfe.solver import GeneralizedAlpha
from amfe.solver import AmfeSolution


class PendulumConstraintTest(TestCase):
    """
    Simple Pendulum test:

              y ^ u_2
                |
                |   u_1
    Fixed:      o----->
                \     x
                 \
                  \
                 L \ ^ u_4
                    \|
    Mass m:          O --> u_3
                    m

    """

    def setUp(self):
        # Length of the pendulum
        self.L = 2.0
        # Gravitation
        self.g = 10.0
        # Mass of the pendulum
        self.m = 1.0

        # Mass matrix
        def M_raw(u, du, t):
            return np.array([[self.m, 0, 0, 0], [0, self.m, 0, 0], [0, 0, self.m, 0], [0, 0, 0, self.m]],
                            dtype=float)

        # External force (gravitation)
        def F_ext(u, du, t):
            return np.array([0.0, 0.0, 0.0, -self.m * self.g])

        def jac_u(u, du, t):
            return csr_matrix((4, 4), dtype=float)

        def jac_du(u, du, t):
            return csr_matrix((4, 4), dtype=float)

        self.M_raw_func = M_raw
        self.h_func = F_ext
        self.dh_dq = jac_u
        self.dh_ddq = jac_du

        # X coordinates in reference configuration
        self.X = np.array([0.0, 0.0, 0.0, -self.L], dtype=float)
        # initial condition u_0: 90deg to the right
        self.u_0 = np.array([0.0, 0.0, self.L, self.L])
        # other initial conditions:
        self.du_0 = np.zeros(4)
        self.ddu_0 = np.zeros(4)
        # Create ConstraintManager
        self.cm = ConstraintManager(4)
        # create constraints (fixation and fixed distance as pendulum)
        self.constraint_fixation = self.cm.create_dirichlet_constraint(U=lambda t: 0.0, dU=lambda t: 0.0,
                                                                       ddU=lambda t: 0.0)
        self.constraint_fixed_distance = self.cm.create_fixed_distance_constraint()

        # Add constraints to manager
        self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([0], dtype=int), ())
        self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([1], dtype=int), ())
        self.cm.add_constraint('Pendulum', self.constraint_fixed_distance, np.array([0, 1, 2, 3], dtype=int),
                               np.array([0, 1, 2, 3], dtype=int))

        def g_func(u, t):
            return self.cm.g(self.X, u, t)

        self.g_func = g_func

        def B_func(u, t):
            return self.cm.B(self.X, u, t)

        self.B_func = B_func

        def a_func(u, du, t):
            return self.cm.a(self.X, u, du, t)

        self.a_func = a_func

    def tearDown(self):
        self.cm = None

    def test_base_formulation(self):
        formulation = ConstraintFormulationBase(self.cm.no_of_dofs_unconstrained, self.M_raw_func,
                                                self.h_func, self.B_func, self.dh_dq, self.dh_ddq,
                                                self.g_func)
        x = np.array([0.0, 0.0, self.L, self.L, 0.0, 0.0, 0.0])
        dx = np.zeros_like(x)
        t = 0.0

        with self.assertRaises(NotImplementedError):
            formulation.M(x, dx, t)

        with self.assertRaises(NotImplementedError):
            formulation.D(x, dx, t)

        with self.assertRaises(NotImplementedError):
            formulation.K(x, dx, t)

        with self.assertRaises(NotImplementedError):
            formulation.F(x, dx, t)

        with self.assertRaises(NotImplementedError):
            formulation.u(x, t)

        with self.assertRaises(NotImplementedError):
            formulation.du(x, dx, t)

        with self.assertRaises(NotImplementedError):
            formulation.ddu(x, dx, dx, t)

        with self.assertRaises(NotImplementedError):
            formulation.lagrange_multiplier(x, t)

        with self.assertRaises(NotImplementedError):
            _ = formulation.dimension

    def test_pendulum_pure_lagrange(self):
        """
        Test that tests a pure lagrange formulation.
        """
        formulation = SparseLagrangeMultiplierConstraintFormulation(self.cm.no_of_dofs_unconstrained, self.M_raw_func,
                                                                    self.h_func, self.B_func, self.dh_dq, self.dh_ddq,
                                                                    self.g_func)
        # Define state: 90deg to the right
        x = np.array([0.0, 0.0, self.L, self.L, 0.0, 0.0, 0.0])
        unconstrained_u_desired = np.array([0.0, 0.0, self.L, self.L])
        unconstrained_u_actual = formulation.u(x, 0.0)

        # test if unconstrained u is correctly calculated
        assert_array_equal(unconstrained_u_actual, unconstrained_u_desired)

        # Check if g function is zero for different positions
        # -90 deg
        x = np.array([0.0, 0.0, -self.L, self.L, 0.0, 0.0, 0.0])
        F1 = formulation.F(x, np.zeros_like(x), 0.0)
        F2 = formulation.F(x, np.zeros_like(x), 2.0)
        F3 = formulation.F(x, x, 2.0)

        assert_array_equal(F1[-3:], np.array([0.0, 0.0, 0.0]))
        assert_array_equal(F2[-3:], np.array([0.0, 0.0, 0.0]))
        assert_array_equal(F3[-3:], np.array([0.0, 0.0, 0.0]))

        x = np.array([0.0, 0.0, 0.0, -0.1 * self.L, 0.0, 0.0, 0.0])

        F1 = formulation.F(x, np.zeros_like(x), 0.0)
        # Violated constraint, thus g > 0.0
        self.assertGreater(np.linalg.norm(F1[-3:]), 0.0)

        # Test jacobians of constraint formulation by comparing with finite difference approximation
        delta = 0.00000001
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        K_actual = formulation.K(x, x, 0.0).todense()

        K_finite_difference = np.zeros_like(K_actual)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] = x_plus[i] + delta
            x_minus = x.copy()
            x_minus[i] = x_minus[i] - delta

            F_plus = formulation.F(x_plus, x, 0.0).copy()
            F_minus = formulation.F(x_minus, x, 0.0)
            K_finite_difference[:, i] = -(F_plus - F_minus).reshape(-1, 1) / (2 * delta)

        assert_allclose(K_actual, K_finite_difference, rtol=1e-7)

    def test_pendulum_nullspace_elimination(self):
        """
        Test the nullspace elimination for the pendulum.
        """
        formulation = NullspaceConstraintFormulation(self.cm.no_of_dofs_unconstrained, self.M_raw_func, self.h_func,
                                                     self.B_func, self.dh_dq, self.dh_ddq, self.g_func,
                                                     a_func=self.a_func)

        # Define state: 90deg to the right
        x = np.array([0.0, 0.0, self.L, self.L])
        unconstrained_u_desired = np.array([0.0, 0.0, self.L, self.L])
        unconstrained_u_actual = formulation.u(x, 0.0)

        # test if unconstrained u is correctly calculated
        assert_array_equal(unconstrained_u_actual, unconstrained_u_desired)

        # Check if g function is zero for different positions
        # -90 deg
        x = np.array([0.0, 0.0, -self.L, self.L])
        F1 = formulation.F(x, np.zeros_like(x), 0.0)
        F2 = formulation.F(x, np.zeros_like(x), 2.0)

        assert_array_equal(F1[-3:], np.array([0.0, 0.0, 0.0]))
        assert_array_equal(F2[-3:], np.array([0.0, 0.0, 0.0]))

        # Test jacobians of constraint formulation by comparing with finite difference approximation
        delta = 0.00000001
        x = np.array([0.0, 0.0, 0.0, 0.0])
        K_actual = formulation.K(x, x, 0.0).todense()

        K_finite_difference = np.zeros_like(K_actual)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] = x_plus[i] + delta
            x_minus = x.copy()
            x_minus[i] = x_minus[i] - delta

            F_plus = formulation.F(x_plus, x, 0.0).copy()
            F_minus = formulation.F(x_minus, x, 0.0)
            K_finite_difference[:, i] = -(F_plus - F_minus).reshape(-1, 1) / (2 * delta)

        assert_allclose(K_actual, K_finite_difference, rtol=1e-7)

    def test_pendulum_time_integration(self):
        """
        Test that computes the time integration of the pendulum
        """
        formulation_lagrange = SparseLagrangeMultiplierConstraintFormulation(self.cm.no_of_dofs_unconstrained,
                                                                             self.M_raw_func, self.h_func, self.B_func,
                                                                             self.dh_dq, self.dh_ddq, self.g_func)

        formulation_nullspace = NullspaceConstraintFormulation(self.cm.no_of_dofs_unconstrained, self.M_raw_func,
                                                               self.h_func, self.B_func, self.dh_dq, self.dh_ddq,
                                                               self.g_func, a_func=self.a_func)

        sol_lagrange = AmfeSolution()
        sol_nullspace = AmfeSolution()

        formulations = [(formulation_lagrange, sol_lagrange), (formulation_nullspace, sol_nullspace)]
        # Define state: 90deg to the right
        q0 = np.array([0.0, 0.0, self.L, self.L])
        dq0 = np.array([0.0, 0.0, 0.0, 0.0])

        nonlinear_solver = NewtonRaphson()

        for (formulation, sol) in formulations:
            def f_int(u, du, t):
                return np.zeros(formulation.dimension)

            def write_callback(t, x, dx, ddx):
                u = formulation.u(x, t)
                du = formulation.du(x, dx, t)
                ddu = formulation.ddu(x, dx, ddx, t)
                sol.write_timestep(t, u, du, ddu)

            integrator = GeneralizedAlpha(formulation.M, f_int, formulation.F, formulation.K, formulation.D,
                                          alpha_m=0.0)
            integrator.dt = 0.025
            integrator.nonlinear_solver_func = nonlinear_solver.solve
            integrator.nonlinear_solver_options = {'rtol': 1e-7, 'atol': 1e-6}

            # Initialize first timestep
            t = 0.0
            t_end = 0.5
            x = np.zeros(formulation.dimension)
            dx = np.zeros(formulation.dimension)
            ddx = np.zeros(formulation.dimension)
            x[:4] = q0
            dx[:4] = dq0

            # Call write timestep for initial conditions
            write_callback(t, x, dx, ddx)

            # Run Loop
            while t < t_end:
                t, x, dx, ddx = integrator.step(t, x, dx, ddx)
                write_callback(t, x, dx, ddx)

        def plot_pendulum_path(u_plot):
            plt.scatter(self.X[1, 0]+[u[2] for u in u_plot], self.X[1, 1]+[u[3] for u in u_plot])
            plt.title('Pendulum-test: Positional curve of rigid pendulum under gravitation')
            return

        # UNCOMMENT THESE LINES IF YOU LIKE TO SEE A TRAJECTORY (THIS CAN NOT BE DONE FOR GITLAB-RUNNER
        # plot_pendulum_path(sol_lagrange.q)
        # plot_pendulum_path(sol_nullspace.q)
        # plt.show()

        # test if nullspace formulation is almost equal to lagrangian
        # and test if constraint is not violated
        for q_lagrange, q_nullspace in zip(sol_lagrange.q, sol_nullspace.q):
            assert_allclose(q_nullspace, q_lagrange, atol=1e-1)
            x_lagrange = self.X.reshape(-1) + q_lagrange
            x_nullspace = self.X.reshape(-1) + q_nullspace
            assert_allclose(np.array([np.linalg.norm(x_lagrange[2:] - x_lagrange[:2])]), np.array([self.L]), atol=1e-3)
            assert_allclose(np.array([np.linalg.norm(x_nullspace[2:] - x_nullspace[:2])]), np.array([self.L]),
                            atol=1e-1)

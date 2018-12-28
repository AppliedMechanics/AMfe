"""Test Routine for constraint manager"""

from unittest import TestCase
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_raises, assert_allclose
from pandas.testing import assert_frame_equal

from amfe.constraint.constraint_manager import ConstraintManager
from lib2to3.pgen2.token import N_TOKENS


class ConstraintManagerTest(TestCase):
    def setUp(self):
        M = np.array([[1, -1, 0], [-1, 1.2, -1.5], [0, -1.5, 2]], dtype=float)
        K = np.array([[2, -1, 0], [-1, 2, -1.5], [0, -1.5, 3]], dtype=float)
        D = 0.2 * M + 0.1 * K

        self.M_unconstr = csr_matrix(M)
        self.D_unconstr = csr_matrix(D)
        self.K_unconstr = csr_matrix(K)
        self.f_int_unconstr = np.array([1, 2, 3], dtype=float)
        self.f_ext_unconstr = np.array([3, 4, 5], dtype=float)
        self.cm = ConstraintManager(3)

        class DummyDirichletConstraint:
            def __init__(self, u_constr):
                self.no_of_constraints = 1
                self.u_constr = u_constr
                
            def constraint_func(self, X_local, u_local, du_local, ddu_local, t):
                return u_local - self.u_constr
            
            def jacobian(self, X_local, u_local, du_local, ddu_local, t, primary_type):
                return np.array([1])

        self.diric_constraint = DummyDirichletConstraint(0)
        self.diric_constraint2 = DummyDirichletConstraint(0.5)

    def tearDown(self):
        self.cm = None

    def testaddconstraint(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        constraint_df_desired = pd.DataFrame({'name': 'Dirich0', 'constraint_obj': self.diric_constraint, 'dofids': [[2]], 'strategy': 'elim'})
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired)
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], 'lagrmult')
        constraint_df_desired = constraint_df_desired.append(pd.DataFrame({'name': 'Dirich1', 'constraint_obj': self.diric_constraint, 'dofids': [[1]], 'strategy': 'lagrmult'}), ignore_index=True)
        print(self.cm._constraints_df)
        print(constraint_df_desired)
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired)
        
    def test_remove_constraint_by_name(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], 'elim')
        self.cm.remove_constraint_by_name('Dirich0')
        constraint_df_desired = pd.DataFrame({'name': 'Dirich1', 'constraint_obj': self.diric_constraint, 'dofids': [[1]], 'strategy': 'elim'})
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired)
        
    def test_remove_constraint_by_dofids(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], 'elim')
        self.cm.remove_constraint_by_dofids([2])
        constraint_df_desired = pd.DataFrame({'name': 'Dirich1', 'constraint_obj': self.diric_constraint, 'dofids': [[1]], 'strategy': 'elim'})
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired)
        
    def test_update_constraints(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], 'lagrmult')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        C_elim_desired = np.array([[0.0, 0.0, 1.0]])
        g_elim_desired = np.array([0.0])
        C_lagr_desired = np.array([[0.0, 1.0]])
        g_lagr_desired = np.array([0.0])
        assert_array_equal(C_elim_desired,self.cm._C_elim.todense())
        assert_array_equal(C_lagr_desired,self.cm._C_lagr.todense())
        assert_array_equal(g_elim_desired,self.cm._g_elim)
        assert_array_equal(g_lagr_desired,self.cm._g_lagr)
        

    def test_constrain_matrix(self):
        # Constrain third dof
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        M_constr = self.cm.constrain_matrix(self.M_unconstr)
        M_constr_desired = self.M_unconstr[0:2, 0:2]
        assert_array_equal(M_constr.todense(), M_constr_desired.todense())

        K_constr = self.cm.constrain_matrix(self.K_unconstr)
        K_constr_desired = self.K_unconstr[0:2, 0:2]
        assert_array_equal(K_constr.todense(), K_constr_desired.todense())

        D_constr = self.cm.constrain_matrix(self.D_unconstr)
        D_constr_desired = self.D_unconstr[0:2, 0:2]
        assert_array_equal(D_constr.todense(), D_constr_desired.todense())

    def test_constrain_vector(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        f_int_desired = self.f_int_unconstr[0:2]
        f_int = self.cm.constrain_vector(self.f_int_unconstr)
        assert_array_equal(f_int, f_int_desired)

        f_ext_desired = self.f_ext_unconstr[0:2]
        f_ext = self.cm.constrain_vector(self.f_ext_unconstr)
        assert_array_equal(f_ext, f_ext_desired)
        
        u_actual = self.cm.constrain_vector(np.array([1, 2, 3], dtype=float))
        u_desired = np.array([1, 2], dtype=float)
        assert_array_equal(u_actual, u_desired)
        
    def test_get_constrained_coupling_quantities(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint2, [2], 'elim')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        K_coupling = self.cm.get_constrained_coupling_quantities(self.K_unconstr)
        K_coupling_desired = np.array([[0],[-0.75]])
        assert_array_equal(K_coupling.todense(),K_coupling_desired)

    def test_unconstrain_vector(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint2, [2], 'elim')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        u_actual = self.cm.unconstrain_vector(np.array([0, 0], dtype=float))
        u_desired = np.array([0, 0, 0.5], dtype=float)
        assert_array_equal(u_actual, u_desired)

    def test_L(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        L_desired = csr_matrix(np.array([[1, 0], [0, 1], [0, 0]]))
        L_actual = self.cm.L
        assert_array_equal(np.dot(self.cm.C_elim.todense(),L_actual.todense()), np.dot(self.cm.C_elim.todense(),L_desired.todense()))

    def test_no_of_constrained_dofs(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], 'elim')
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], 'lagrmult')
        self.cm.update_constraints(np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]))
        no_of_constrained_dofs_actual = self.cm.no_of_constrained_dofs
        no_of_constrained_dofs_desired = 2
        self.assertEqual(no_of_constrained_dofs_actual, no_of_constrained_dofs_desired)

    def test_no_of_constrained_dofs_2(self):
        # Set no_of_unconstrained_dofs to 10
        unconstr_dofs = 10
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1, 2, 0])
        self.cm.update_no_of_unconstrained_dofs(unconstr_dofs)
        self.cm.add_constraint('FixedDistance', self.cm.create_fixed_distance_constraint(), [3, 4, 5, 6], 'elim')
        u = np.zeros(unconstr_dofs)
        self.cm.update_constraints(X, u, u, u, 0.0)
        no_of_constrained_dofs_actual = self.cm.no_of_constrained_dofs
        no_of_constrained_dofs_desired = unconstr_dofs - 1
        self.assertEqual(no_of_constrained_dofs_actual, no_of_constrained_dofs_desired)

        # Add a second fixed distance constraint
        self.cm.add_constraint('FixedDistance', self.cm.create_fixed_distance_constraint(), [3, 4, 7, 8], 'elim')
        self.cm.update_constraints(X, u, u, u, 0.0)
        no_of_constrained_dofs_actual = self.cm.no_of_constrained_dofs
        no_of_constrained_dofs_desired = unconstr_dofs - 2
        self.assertEqual(no_of_constrained_dofs_actual, no_of_constrained_dofs_desired)


class PendulumConstraintManagerTest(TestCase):
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
            self.M_raw = np.array([[self.m, 0, 0, 0], [0, self.m, 0, 0], [0, 0, self.m, 0], [0, 0, 0, self.m]],
                                  dtype=float)
            # External force (gravitation)
            self.F_ext = np.array([0.0, 0.0, 0.0, -self.m * self.g])
            # X coordinates in reference configuration
            self.X = np.array([0.0, 0.0, 0.0, -self.L], dtype=float)
            # initial condition u_0: 90deg to the right
            self.u_0 = np.array([0.0, 0.0, self.L, self.L])
            # other initial conditions:
            self.du_0 = np.zeros(4)
            self.ddu_0 = np.zeros(4)
            # Create ConstraintManager
            self.cm = ConstraintManager(4)
            # Number of dofs before constraint: 4  (x and y of mounting point and pendulum mass)
            self.cm.update_no_of_unconstrained_dofs(4)
            # create constraints (fixation and fixed distance as pendulum)
            self.constraint_fixation = self.cm.create_dirichlet_constraint(2, U=lambda t: 0.0, dU=lambda t: 0.0,
                                                                      ddU=lambda t: 0.0)
            self.constraint_pendulum = self.cm.create_fixed_distance_constraint()

        def tearDown(self):
            self.cm = None

        def get_dae(self, X, u, du, ddu, t, lambda_):
            self.cm.update_constraints(X, u, du, ddu, t)
            M_constr = self.cm.constrain_matrix(self.M_raw)
            C_constr = self.cm.C_lagr
            if C_constr is None:
                M_for_solver = M_constr
                F_ext_for_solver = self.cm.constrain_vector(self.F_ext)
                f_int_for_solver = np.zeros((M_for_solver.shape[0], 1))
                K_for_solver = np.zeros(M_for_solver.shape)
            else:
                M1_for_solver = np.concatenate((M_constr, np.zeros((M_constr.shape[0], C_constr.shape[0]))), axis=1)
                M_for_solver = np.concatenate(
                    (M1_for_solver, np.zeros((C_constr.shape[0], M_constr.shape[1] + C_constr.shape[0]))), axis=0)
                F_ext_for_solver = self.cm.constrain_vector(self.F_ext)
                F_ext_for_solver = np.concatenate((F_ext_for_solver, np.zeros(C_constr.shape[0])))
                g = self.cm._g_lagr
                # These lines are heavy to code. This is due to wrong ndims in arrays
                f_int_for_solver = np.concatenate(((C_constr.T @ lambda_), np.array(g, ndmin=2).T), axis=0)
                K_for_solver = np.concatenate((np.concatenate((np.zeros(M_constr.shape), C_constr.T.toarray()), axis=1),
                                               np.concatenate((C_constr.toarray(), np.zeros((C_constr.shape[0],
                                                                                         C_constr.shape[0]))), axis=1)),
                                              axis=0)

            return M_for_solver, K_for_solver, f_int_for_solver, F_ext_for_solver

        def test_pendulum_mixed(self):
            """
            Test that tests a mixed formulation. The fixation is eliminated and the fixed distance has a Lagrange
            Multiplier
            """
            # Add constraints to manager
            self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([0, 1], dtype=int), 'elim')
            self.cm.add_constraint('Pendulum', self.constraint_pendulum, np.array([0, 1, 2, 3], dtype=int), 'lagrmult')

            # Initialize constraint with initial condition
            self.cm.update_constraints(self.X, self.u_0, self.du_0, self.ddu_0, 0.0)

            # Define state: 90deg to the right
            constrained_u = np.array([self.L, self.L])
            unconstrained_u_desired = np.array([0.0, 0.0, self.L, self.L])
            unconstrained_u_actual = self.cm.unconstrain_vector(constrained_u)
            # test if unconstrained u is correctly calculated
            assert_array_equal(unconstrained_u_actual, unconstrained_u_desired)

            M_1, K_1, f_1, F_1 = self.get_dae(self.X, self.u_0, self.du_0, self.ddu_0, 0.0, np.array([0.0], ndmin=2))

            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0, 0.0)
            M_2, K_2, f_2, F_2 = self.get_dae(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                              0.0, np.array([0.0], ndmin=2))
            # Check if the same entities appear if constrained is updated and unconstrain_vector function is used
            assert_array_equal(M_1, M_2)
            assert_array_equal(K_1, K_2)
            assert_array_equal(f_1, f_2)
            assert_array_equal(F_1, F_2)

            # Check if g function is zero for different positions
            # -90 deg
            constrained_u = [-self.L, self.L]
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            M_2, K_2, f_2, F_2 = self.get_dae(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                              0.0, np.array([0.0], ndmin=2))
            self.assertEqual(f_2[-1], 0.0)

            constrained_u = np.array([0.0, -0.1*self.L])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            M_2, K_2, f_2, F_2 = self.get_dae(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                              0.0, np.array([0.0], ndmin=2))
            # Violated constrait, thus g > 0.0
            self.assertGreater(f_2[-1], 0.0)
            # updated with lambda_ = 0.0, thus f_2[:-1] must be zero
            assert_array_equal(f_2[:-1], np.zeros_like(f_2[:-1]))

        def test_pendulum_mixed_C(self):
            # Add constraints to manager
            self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([0, 1], dtype=int), 'elim')
            self.cm.add_constraint('Pendulum', self.constraint_pendulum, np.array([0, 1, 2, 3], dtype=int), 'lagrmult')

            # Initialize constraint with initial condition
            self.cm.update_constraints(self.X, self.u_0, self.du_0, self.ddu_0, 0.0)

            delta = 0.001
            constrained_u = np.array([0.0, 0.0])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            C = self.cm.C_lagr
            constrained_u = np.array([delta, 0.0])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            g_after_x1_plus = self.cm._g_lagr

            constrained_u = np.array([0.0, delta])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            g_after_x2_plus = self.cm._g_lagr

            constrained_u = np.array([-delta, 0.0])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            g_after_x1_minus = self.cm._g_lagr

            constrained_u = np.array([0.0, -delta])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            g_after_x2_minus = self.cm._g_lagr

            finite_difference = np.array([(g_after_x1_plus-g_after_x1_minus)/(2*delta),
                                          (g_after_x2_plus-g_after_x2_minus)/(2*delta)], ndmin=2).T
            assert_allclose(C.toarray(), finite_difference, rtol=1e-7)

        def test_pendulum_pure_lagrange(self):
            """
            Test that tests a pure lagrange formulation.
            """
            # Add constraints to manager
            self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([0, 1], dtype=int), 'lagrmult')
            self.cm.add_constraint('Pendulum', self.constraint_pendulum, np.array([0, 1, 2, 3], dtype=int), 'lagrmult')

            # Initialize constraint with initial condition
            self.cm.update_constraints(self.X, self.u_0, self.du_0, self.ddu_0, 0.0)

            # Define state: 90deg to the right
            constrained_u = np.array([0.0, 0.0, self.L, self.L])
            unconstrained_u_desired = np.array([0.0, 0.0, self.L, self.L])
            unconstrained_u_actual = self.cm.unconstrain_vector(constrained_u)
            # test if unconstrained u is correctly calculated
            assert_array_equal(unconstrained_u_actual, unconstrained_u_desired)

            # define lambda_ := 0 vector
            lambda_ = np.array([0.0, 0.0, 0.0], ndmin=2).T

            M_1, K_1, f_1, F_1 = self.get_dae(self.X, self.u_0, self.du_0, self.ddu_0, 0.0, lambda_)

            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0, 0.0)
            M_2, K_2, f_2, F_2 = self.get_dae(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                              0.0, lambda_)
            # Check if the same entities appear if constrained is updated and unconstrain_vector function is used
            assert_array_equal(M_1, M_2)
            assert_array_equal(K_1, K_2)
            assert_array_equal(f_1, f_2)
            assert_array_equal(F_1, F_2)

            # Check if g function is zero for different positions
            # -90 deg
            constrained_u = np.array([0.0, 0.0, -self.L, self.L])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0, 0.0)
            M_2, K_2, f_2, F_2 = self.get_dae(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                              0.0, lambda_)
            self.assertEqual(f_2[-1], 0.0)

            constrained_u = np.array([0.0, 0.0, 0.0, -0.1*self.L])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0, 0.0)
            M_2, K_2, f_2, F_2 = self.get_dae(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                              0.0, lambda_)
            # Violated constrait, thus g > 0.0
            self.assertGreater(f_2[-1], 0.0)
            # updated with lambda_ = 0.0, thus f_2[:-1] must be zero
            assert_array_equal(f_2[:-1], np.zeros_like(f_2[:-1]))

        def test_pendulum_pure_lagrange_C(self):
            # Add constraints to manager
            self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([0, 1], dtype=int), 'lagrmult')
            self.cm.add_constraint('Pendulum', self.constraint_pendulum, np.array([0, 1, 2, 3], dtype=int), 'lagrmult')

            # Initialize constraint with initial condition
            self.cm.update_constraints(self.X, self.u_0, self.du_0, self.ddu_0, 0.0)

            delta = 0.00000001
            constrained_u = np.array([0.0, 0.0, 0.0, 0.0])
            self.cm.update_constraints(self.X, self.cm.unconstrain_vector(constrained_u), self.du_0, self.ddu_0,
                                       np.array([0.0], ndmin=2))
            C = self.cm.C_lagr.toarray()

            C_finite_difference = np.zeros_like(C)

            for i in range(len(constrained_u)):
                delta_u = constrained_u
                delta_u[i] = delta
                self.cm.update_constraints(self.X, self.cm.unconstrain_vector(delta_u), self.du_0, self.ddu_0,
                                           np.array([0.0], ndmin=2))
                g_after_plus = self.cm._g_lagr
                delta_u = constrained_u
                delta_u[i] = -delta
                self.cm.update_constraints(self.X, self.cm.unconstrain_vector(delta_u), self.du_0, self.ddu_0,
                                           np.array([0.0], ndmin=2))
                g_after_minus = self.cm._g_lagr
                C_finite_difference[:, i] = (g_after_plus-g_after_minus)/(2*delta)

            assert_allclose(C, C_finite_difference, rtol=1e-7)

        def test_pendulum_pure_elimination(self):
            """
            Test that tests a pure elimination formulation.
            """
            # Add constraints to manager
            self.cm.add_constraint('Fixation', self.constraint_fixation, np.array([0, 1], dtype=int), 'elim')
            self.cm.add_constraint('Pendulum', self.constraint_pendulum, np.array([0, 1, 2, 3], dtype=int), 'elim')

            # Initialize constraint with initial condition
            self.cm.update_constraints(self.X, self.u_0, self.du_0, self.ddu_0, 0.0)

            # Define state: 90deg to the right
            unconstrained_u_desired = np.array([0.0, 0.0, self.L, self.L])
            
            ################################################
            ##                                            ##
            ##    Solve pendulum with pure elimination    ##
            ##                                            ##
            ################################################
            def plot_pendulum_path(u_plot, N_t):
                #plt.plot(self.X[2]+u_plot[2,:],self.X[3]+u_plot[3,:])
                #scat = plt.scatter(self.X[2]+u_plot[2,:],self.X[3]+u_plot[3,:], c = range(0, N_t))
                #plt.title('Penulum-test: Positional curve of rigid pendulum under gravitation')
                #plt.show()
                pass
            
            u_t = self.u_0
            du_t = self.du_0
            #du_t[3] = -0.3
            ddu_t = self.ddu_0
            T = 5
            n_max = 10
            N_t = 100
            delta_t = T/N_t
            
            t_plot = np.arange(0,delta_t,T)
            u_plot = np.zeros((u_t.shape[0],N_t))
            beta = 0.3
            gamma = 0.6
            
            for t in range(0, N_t):
                u_plot[:,t] = u_t
                u_t1 = u_t + delta_t*du_t + (0.5-beta)*delta_t**2*ddu_t
                du_t1 = du_t + (1-gamma)*delta_t*ddu_t
                ddu_t1 = ddu_t*0
                self.cm.update_constraints(self.X, u_t1, du_t1, ddu_t1, 0.0)
                norm_u_t1 = np.empty([])
                
                n=0
                #fig = plt.figure()
                #ax = fig.add_subplot(111, projection='3d')
                while n <= n_max:
                    constrained_m = self.cm.constrain_matrix(self.M_raw * 1/(beta*delta_t**2))
                    constrained_rhs = self.cm.constrain_vector(self.M_raw @ (1/(beta*delta_t**2)*(u_t1 - u_t) - 1/(beta*delta_t)*du_t - (0.5-beta)/beta*ddu_t) - self.F_ext)                 
                    delta_u_constrained = np.linalg.solve(constrained_m, -constrained_rhs)
                    delta_u = self.cm.unconstrain_vector(delta_u_constrained)

                    if n==0 or n==n_max:
                        u3_plot = []
                        u4_plot = []
                        constr = []
                        for u3 in np.arange(-1,1, 0.1):
                            for u4 in np.arange(-1,1, 0.1):
                                u_test = np.array([0,0,u3,u4])
                                constr = np.append(constr, (self.cm.C_elim@u_test + self.cm._g_elim)[2])
                                u3_plot = np.append(u3_plot, u3)
                                u4_plot = np.append(u4_plot, u4)
                        #ax.plot_trisurf(u3_plot, u4_plot, constr)
                    
                    u_t1 += delta_u
                    #ax.scatter(delta_u[2],delta_u[3],(self.cm.C_elim@delta_u + self.cm._g_elim)[2], s=2)
                    du_t1 += gamma/(beta*delta_t)*delta_u
                    ddu_t1 += 1/(beta*delta_t**2) * delta_u
                    self.cm.update_constraints(self.X, u_t1, du_t1, ddu_t1, 0.0)
                    residual = np.max(np.abs(self.cm._g_elim))
                    print('t: ', t, ' residual: ', residual)
                    norm_u_t1 = np.append(norm_u_t1, np.linalg.norm(delta_u))
                    if residual <= 1e-8:
                        break
                    n += 1
                #plt.plot(np.arange(0,n+1),norm_u_t1)
                #plt.show()
                u_t = u_t1
                du_t = du_t1
                ddu_t = ddu_t1
                
                # test if unconstrained u is correctly calculated
                assert_allclose(np.abs(u_t[0:2]), np.zeros(2), atol=1e-7)
                assert_allclose(np.linalg.norm(self.X+u_t),self.L, atol=1e-7)
                
            u_plot[:,t] = u_t1
            plot_pendulum_path(u_plot, N_t)
    
                
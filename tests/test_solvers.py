# -*- coding: utf-8 -*-
'''
Test for testing the solvers
'''

import unittest
import numpy as np
import re
from os.path import join, dirname, abspath
import pytest

from numpy.testing import assert_allclose
import amfe


def read_grf(file):
    times = list()
    displacements = list()
    with open(file, 'r') as fp:
        for line in fp.readlines():
            if not line.startswith('#'):
                matchobject = re.search(r'([0-9eE.\-]*)\s([0-9eE.\-]*)', line)
                time = float(matchobject.group(1))
                displacement = float(matchobject.group(2))
                times.append(time)
                displacements.append(displacement)
    # First dimension (rows): 0: time, 1: displacement
    # Second dimension (columns): different timesteps
    data = np.array([times, displacements])
    return data


@pytest.mark.skip("temporarily disabled")
class SolversTest(unittest.TestCase):
    def setUp(self):
        # define input-file and prefix for output
        here = dirname(abspath(__file__))
        self.input_file = join('..', 'meshes', 'gmsh', 'beam', 'Beam10x1Quad8.msh')
        self.output_file_prefix = join(here, '.results', 'beam', 'Beam10x1Quad8')
        # setup mechanical system
        self.material = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
        self.system = amfe.MechanicalSystem()
        self.system.load_mesh_from_gmsh(self.input_file, 1, self.material)
        self.system.apply_dirichlet_boundaries(5, 'xy')
        ndof = self.system.dirichlet_class.no_of_constrained_dofs
        self.system.apply_rayleigh_damping(1e0, 1e-5)  # set damping and ...
        self.system.apply_no_damping()  # ... reset damping for testing
        self.options = {
            'number_of_load_steps': 10,
            'newton_damping': 1.0,
            'simplified_newton_iterations': 1,
            't': 1.0,
            't0': 0.0,
            't_end': 0.4,
            'dt': 5e-4,
            'dt_output': 5e-4,
            'rho_inf': 0.95,
            'initial_conditions': {
                'x0': np.zeros(2 * ndof),
                'q0': np.zeros(ndof),
                'dq0': np.zeros(ndof)},
            'relative_tolerance': 1.0E-6,
            'absolute_tolerance': 1.0E-9,
            'verbose': True,
            'max_number_of_iterations': 99,
            'convergence_abort': True,
            'write_iterations': False,
            'track_number_of_iterations': False,
            'save_solution': True}
        rho_inf = 0.95
        alpha = 0.0005

    def tearDown(self):
        self.system = None

    def test_nonlinear_dynamics_solver(self):
        pass

    def test_linear_dynamics_solver(self):
        pass

    def test_generalized_alpha_nonlinear_dynamics_solver(self):
        self.system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: 1)
        self.solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=self.system, **self.options)
        self.solver.solve()
        x = np.array([self.system.T_output[:], [displacement[2] for displacement in self.system.u_output]])
        y = np.array([self.system.T_output[:], [displacement[3] for displacement in self.system.u_output]])
        here = dirname(abspath(__file__))
        fnx = join(here, 'kratos', 'Kratos_beam10x1Quad8_nonlinear_dynamics_x_wbzalpha_rhoinf095_dt1e-6.grf')
        fny = join(here, 'kratos', 'Kratos_beam10x1Quad8_nonlinear_dynamics_y_wbzalpha_rhoinf095_dt1e-6.grf')
        reference_x = read_grf(fnx)
        reference_y = read_grf(fny)
        assert_allclose(x[1, 1:], reference_x[1, 499::500], rtol=1e-12, atol=0.06)
        assert_allclose(y[1, 1:], reference_y[1, 499::500], rtol=1e-12, atol=0.06)

    def test_jwh_alpha_nonlinear_dynamics_solver(self):
        pass

    def test_generalized_alpha_linear_dynamics_solver(self):
        pass

    def test_jwh_alpha_linear_dynamics_solver(self):
        pass


if __name__ == '__main__':
    st = SolversTest()
    st.setUp()
    st.test_generalized_alpha_nonlinear_dynamics_solver()

#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import asyncio
import h5py
import numpy as np
from unittest import TestCase
from numpy.testing import assert_array_equal

from amfe.io.tools import amfe_dir, check_dir
from amfe.solver import AmfeSolution, AmfeSolutionHdf5, solve_async


class DummySolver:
    def __init__(self):
        self.ndof = 100

    def solve(self, callback, callbackargs=(), t0=0.0, tend=1.0, dt=0.01):
        t, q, dq, ddq = self._initialize(t0, tend, dt, self.ndof)

        t = np.arange(t0, tend, dt)

        for i, t_current in enumerate(t):
            callback(t_current, q[i, :], dq[i, :], ddq[i, :], *callbackargs)

    async def solve_async(self, callback, callbackargs=(), t0=0.0, tend=1, dt=0.01):
        t, q, dq, ddq = self._initialize(t0, tend, dt, self.ndof)

        t = np.arange(t0, tend, dt)

        for i, t_current in enumerate(t):
            await callback(t_current, q[i, :], dq[i, :], ddq[i, :], *callbackargs)

    @staticmethod
    def _initialize(t0, tend, dt, ndof):
        t = np.arange(t0, tend, dt)
        q = np.array([np.arange(0, ndof)*scale for scale in t])
        dq = q.copy()
        ddq = q.copy()
        return t, q, dq, ddq


class AmfeSolutionTest(TestCase):
    def setUp(self):
        self.solver = DummySolver()
        return

    def tearDown(self):
        return

    def test_amfe_solution(self):
        # Only q
        solution = AmfeSolution()
        q1 = np.arange(0, 60, dtype=float)
        q2 = np.arange(10, 70, dtype=float)
        t1 = 0.1
        t2 = 0.5

        solution.write_timestep(t1, q1)
        solution.write_timestep(t2, q2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertEqual(len(solution.t), 2)

        # q and dq
        solution = AmfeSolution()
        dq1 = np.arange(20, 80, dtype=float)
        dq2 = np.arange(30, 90, dtype=float)

        solution.write_timestep(t1, q1, dq1)
        solution.write_timestep(t2, q2, dq2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.dq[0], dq1)
        assert_array_equal(solution.dq[1], dq2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertTrue(len(solution.t) == len(solution.dq))
        self.assertEqual(len(solution.t), 2)

        # q, dq and ddq
        solution = AmfeSolution()
        ddq1 = np.arange(40, 100, dtype=float)
        ddq2 = np.arange(50, 110, dtype=float)

        solution.write_timestep(t1, q1, dq1, ddq1)
        solution.write_timestep(t2, q2, dq2, ddq2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.dq[0], dq1)
        assert_array_equal(solution.dq[1], dq2)
        assert_array_equal(solution.ddq[0], ddq1)
        assert_array_equal(solution.ddq[1], ddq2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertTrue(len(solution.t) == len(solution.dq))
        self.assertTrue(len(solution.t) == len(solution.ddq))
        self.assertEqual(len(solution.t), 2)

        # q and ddq
        solution = AmfeSolution()

        solution.write_timestep(t1, q1, ddq=ddq1)
        solution.write_timestep(t2, q2, ddq=ddq2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.ddq[0], ddq1)
        assert_array_equal(solution.ddq[1], ddq2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertTrue(len(solution.t) == len(solution.ddq))
        self.assertEqual(len(solution.t), 2)
        self.assertIsNone(solution.dq[0])
        self.assertIsNone(solution.dq[1])
        return


class AmfeSolutionHdf5Test(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_write_hdf5_timesteps(self):
        filename = amfe_dir('results/tests/amfe_solution_hdf5.h5')
        check_dir(filename)

        q1 = np.arange(0, 60, dtype=float)
        q2 = np.arange(10, 70, dtype=float)
        dq1 = np.arange(20, 80, dtype=float)
        dq2 = np.arange(30, 90, dtype=float)
        ddq1 = np.arange(40, 100, dtype=float)
        ddq2 = np.arange(50, 110, dtype=float)
        t1 = 0.1
        t2 = 0.5

        # Only q
        h5amfe = AmfeSolutionHdf5(filename, 'Sim1', tablename='testcase')
        with h5amfe as writer:
            # Only q
            writer.write_timestep(t1, q1)
            writer.write_timestep(t2, q2)

        with h5py.File(filename, mode='r') as fp:
            dataset = fp['Sim1/testcase']

            assert_array_equal(dataset[0]['q'], q1)
            assert_array_equal(dataset[1]['q'], q2)
            self.assertEqual(dataset[0]['t'], t1)
            self.assertEqual(dataset[1]['t'], t2)
            self.assertEqual(len(dataset), 2)

        # Only q and dq
        h5amfe = AmfeSolutionHdf5(filename, 'Sim1', tablename='testcase')
        with h5amfe as writer:
            # Only q and dq
            writer.write_timestep(t1, q1, dq1)
            writer.write_timestep(t2, q2, dq2)

        with h5py.File(filename, mode='r') as fp:
            dataset = fp['Sim1/testcase']

            assert_array_equal(dataset[0]['q'], q1)
            assert_array_equal(dataset[1]['q'], q2)
            assert_array_equal(dataset[0]['dq'], dq1)
            assert_array_equal(dataset[1]['dq'], dq2)
            self.assertEqual(dataset[0]['t'], t1)
            self.assertEqual(dataset[1]['t'], t2)
            self.assertEqual(len(dataset), 2)

        # q, dq and ddq
        h5amfe = AmfeSolutionHdf5(filename, 'Sim1', tablename='testcase')
        with h5amfe as writer:
            # q, dq and ddq
            writer.write_timestep(t1, q1, dq1, ddq1)
            writer.write_timestep(t2, q2, dq2, ddq2)

        with h5py.File(filename, mode='r') as fp:
            dataset = fp['Sim1/testcase']

            assert_array_equal(dataset[0]['q'], q1)
            assert_array_equal(dataset[1]['q'], q2)
            assert_array_equal(dataset[0]['dq'], dq1)
            assert_array_equal(dataset[1]['dq'], dq2)
            assert_array_equal(dataset[0]['ddq'], ddq1)
            assert_array_equal(dataset[1]['ddq'], ddq2)
            self.assertEqual(dataset[0]['t'], t1)
            self.assertEqual(dataset[1]['t'], t2)
            self.assertEqual(len(dataset), 2)

        # only q and ddq
        h5amfe = AmfeSolutionHdf5(filename, 'Sim1', tablename='testcase')
        with h5amfe as writer:
            # only q and ddq
            writer.write_timestep(t1, q1, ddq=ddq1)
            writer.write_timestep(t2, q2, ddq=ddq2)

        with h5py.File(filename, mode='r') as fp:
            dataset = fp['Sim1/testcase']

            assert_array_equal(dataset[0]['q'], q1)
            assert_array_equal(dataset[1]['q'], q2)
            assert_array_equal(dataset[0]['ddq'], ddq1)
            assert_array_equal(dataset[1]['ddq'], ddq2)
            self.assertEqual(dataset[0]['t'], t1)
            self.assertEqual(dataset[1]['t'], t2)
            self.assertEqual(len(dataset), 2)


class AsyncSolutionHdf5Test(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_async_solve(self):
        # Test Asnychronous solution
        filename = amfe_dir('results/tests/amfe_solution_async_hdf5.h5')
        check_dir(filename)

        mysolver = DummySolver()
        t0 = 0.0
        tend = 1.0
        dt = 0.1

        # define task for asyncio
        async def task(tfilename, tsolver, tt0, ttend, tdt):
            no_of_buffer_slots = 3
            print("Run Asynchronous HDF5 Version")
            h5amfe = AmfeSolutionHdf5(tfilename, 'Sim1', 'asyncsolution')
            with h5amfe as writer:
                solverkwargs = {'t0': tt0, 'tend': ttend, 'dt': tdt}
                await solve_async(no_of_buffer_slots, writer, tsolver, **solverkwargs)
            return h5amfe

        # Run asynchronous
        _ = asyncio.run(task(filename, mysolver, t0, tend, dt))

        # Test if results have been written into hdf5 correctly
        with h5py.File(filename, mode='r') as fp:
            # Get the desired results from solver
            t, q, dq, ddq = mysolver._initialize(t0, tend, dt, mysolver.ndof)

            # Get dataset
            dataset = fp['Sim1/asyncsolution']

            # Test if the first two entries are correct
            assert_array_equal(dataset[0]['q'], q[0, :])
            assert_array_equal(dataset[1]['q'], q[1, :])
            assert_array_equal(dataset[0]['dq'], dq[0, :])
            assert_array_equal(dataset[1]['dq'], dq[1, :])
            assert_array_equal(dataset[0]['ddq'], ddq[0, :])
            assert_array_equal(dataset[1]['ddq'], ddq[1, :])
            self.assertEqual(dataset[0]['t'], t[0])
            self.assertEqual(dataset[1]['t'], t[1])
            # test if all entries have been written
            self.assertEqual(len(dataset), len(t))

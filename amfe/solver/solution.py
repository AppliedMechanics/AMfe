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
from tables import *

from amfe.io.tools import check_dir

__all__ = ['AmfeSolution',
           'AmfeSolutionHdf5',
           'solve_async']


class AmfeSolutionBase:
    """
    AmfeSolutionBase Base class for container of Solutions for Amfe simulations

    This Base class provides an API for all subclasses

    The subclasses must implement write_timestep and if the results shall be written into a file,
    the __enter__ and __exit__ methods must be implemented that ensure to close the file if an error occurs
    """
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def write_timestep(self, t, q, dq=None, ddq=None, strain=None, stress=None):
        """
        This function is called to write a timestep into the solution container

        Parameters
        ----------
        t : float
            time
        q : numpy.array
            solution vector at time t
        dq : numpy.array (optional)
            first time derivative of solution vector at time t
        ddq : numpy.array (optional)
            second time derivative of solution vector at time t
        stress : ndarray
            nodal stresses
        strain : ndarray
            nodal strains

        Returns
        -------
        None
        """
        pass


class AmfeSolution(AmfeSolutionBase):
    """
    Simple Container for solutions of AMfe simulations

    It is based on lists of numpy.arrays

    Attributes
    ----------
    t : list
        list of timesteps the solution has been computed
    q : list
        list of ndarrays containing the solution vectors for each timestep
    dq: list
        list of ndarrays containing the first time derivative of the solution vectors if available
    ddq : list
        list of ndarrays containing the second time derivative of the solution vectors if available
    stress : ndarray
        nodal stresses
    strain : ndarray
        nodal strains
    """
    def __init__(self):
        """
        Constructor of AmfeSolution

        """
        super().__init__()
        # Initialize t, q, dq and ddq as empty lists
        self.t = []
        self.q = []
        self.dq = []
        self.ddq = []
        self.stress = []
        self.strain = []

    def write_timestep(self, t, q, dq=None, ddq=None, strain=None, stress=None):
        """
        This function is called to write a timestep into the solution container

        Parameters
        ----------
        t : float
            time
        q : numpy.array
            solution vector at time t
        dq : numpy.array (optional)
            first time derivative of solution vector at time t
        ddq : numpy.array (optional)
            second time derivative of solution vector at time t
        stress : ndarray
            nodal stresses
        strain : ndarray
            nodal strains

        Returns
        -------
        None
        """
        # Append t, q, dq, ddq to the lists
        self.t.append(t)
        self.q.append(q)
        self.dq.append(dq)
        self.ddq.append(ddq)
        self.stress.append(stress)
        self.strain.append(strain)


class AmfeSolutionAsync(AmfeSolutionBase):
    """
    This is a container for AmfeSolutions that provide asynchronous access.

    Usually you do not need to instantiate an object of this class
    It is automatially constructed when asynchronous option is used in the solve_async function

    Attributes
    ----------
    queue : asyncio.Queue
        Queue that contains solution vectors that have already been computed by the solver but
        not yet written into the solution AmfeSolution container of choice
    """
    def __init__(self, size):
        """
        Constructor of AmfeSolutionAsync

        Parameters
        ----------
        size : int
            Size of Queue
        """
        super().__init__()
        # Initialize a Queue with given size.
        # Maximum <size> solution timesteps can be contained in the Queue.
        # If Queue is full, the solver must wait until a timestep is popped through the get() method
        self.queue = asyncio.Queue(maxsize=size)

    async def write_timestep(self, t, q, dq=None, ddq=None, strain=None, stress=None):
        """
        Coroutine that waits for writing the passed timestep into the Queue

        Parameters
        ----------
        t : float
            time
        q : numpy.array
            solution vector at time t
        dq : numpy.array (optional)
            first time derivative of solution vector at time t
        ddq : numpy.array (optional)
            second time derivative of solution vectot at time t
        stress : ndarray
            nodal stresses
        strain : ndarray
            nodal strains

        Returns
        -------
        None
        """
        # Make a dict that can be put into a Queue slot
        sol = {'t': t,
               'q': q,
               'dq': dq,
               'ddq': ddq,
               'strain': strain,
               'stress': stress
               }
        # Wait until the timestep has been put into the Queue
        print("Put timestep {} into Queue".format(t))
        await self.queue.put(sol)

    async def get(self):
        """
        Coroutine that pops a timestep from the Queue

        Returns
        -------
        sol : dict
            dict providing the popped timestep
            sol['t'] : float (timestep)
            sol['q'] : ndarray (solution vector at timestep)
            sol['dq'] : ndarray (first derivative if available otherwise None)
            sol['ddq'] : ndarray (second derivative if available otherwise None)
        """
        # Wait until a timestep can be popped from the Queue
        sol = await self.queue.get()
        # Inform Queue that this slot is now free for new data
        self.queue.task_done()
        # Return the popped timestep
        return sol


class AmfeSolutionHdf5(AmfeSolutionBase):
    """
    AmfeSolutionHdf5 is a AmfeSolution Container that directly writes into a Hdf5 file.

    This process is controlled by the pytables library. This provides an efficient handling of when to
    write data from RAM to disk

    Attributes
    ----------
    _filename : str
        str containing the path to a file
    _path_to_table : str
        Hdf5 path to the table where the results shall be written to
        e.g. Sim1/Component1
    _fp : file object
        contains the file object of the file filename
    _is_allocated : bool
        Boolean that contains status if the table is already allocated in the hdf5 file
    _wp : function
        pointer to function where to write new lines into the HDF5 Table
    _table : tables.Table
        Table that shall contain the solution data
    _buffer_size : int
        buffer_size that contains the number of slots that can be filled until flush() is called
    _iteration : int
        counts iterations until buffer_size is reached
    """
    def __init__(self, filename, path_to_table, tablename='odesolution', buffer_size=10):
        """
        Constructor for HDF5 Result

        Parameters
        ----------
        filename : str
            Path to file where the HDF5-File shall be stored
        path_to_table : str
            internal hdf5 path to the table where the results shall be stored inside the HDF5 structure
        tablename : str
            name of the table where the results shall be stored inside the HDF5 structure (default: odesolution)
        buffer_size : int
            number of slots in the RAM buffer until the writer is forced to write from disk to RAM (default: 10)
            Attention: It is possible that the buffer is written to disk before this boundary is reached.
            But it is never written later.
        """
        super().__init__()
        self._filename = filename
        self._path_to_table = path_to_table
        self._tablename = tablename
        check_dir(filename)
        self._fp = None
        self._is_allocated = False
        self._tabledict = {'t': Float64Col()}
        self._wp = None
        self._table = None
        self._buffer_size = buffer_size
        self._iteration = 0

    @property
    def filename(self):
        """

        Returns
        -------
        filename : str
            The filename where the HDF5 is stored
        """
        return self._filename

    @filename.setter
    def filename(self, filename):
        """

        Parameters
        ----------
        filename : str
            Set a new filename
        Returns
        -------
        None
        """
        self._filename = filename
        check_dir(filename)

    def __enter__(self):
        """
        Implement the with statement for the class

        opens a file and returns itself
        """
        self._fp = open_file(self._filename, 'w')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Implements the exit of the with statement

        Parameters
        ----------
        exc_type
            Type of Exception
        exc_value
            Value of Exception
        traceback
            Traceback of Exception

        Returns
        -------
        None
        """
        # Flush table to ensure that all timesteps are saved in file
        if self._table is not None:
            self._table.flush()
        # Ensure that file is closed
        self._fp.close()

    def _allocate(self, n_dim, is_dq, is_ddq, is_strain, is_stress):
        """
        Private method that allocates the table

        Parameters
        ----------
        n_dim : int
            dimension of the q vectors that shall be stored
        is_dq : bool
            Flag if dq will be saved
        is_ddq : bool
            Flag if ddq will be saved
        is_strain : bool
            Flag if strain will be saved
        is_stress : bool
            Flag if stress will be saved

        Returns
        -------
        None
        """
        new_key_values = {'q': Float64Col(shape=n_dim)}
        if is_dq:
            new_key_values.update({'dq': Float64Col(shape=n_dim)})
        if is_ddq:
            new_key_values.update({'ddq': Float64Col(shape=n_dim)})
        if is_strain:
            new_key_values.update({'strain': Float64Col(shape=(n_dim, 6))})
        if is_stress:
            new_key_values.update({'stress': Float64Col(shape=(n_dim, 6))})
        self._tabledict.update(new_key_values)
        root = self._fp.root
        sim1 = self._fp.create_group(root, self._path_to_table)
        self._table = self._fp.create_table(sim1, self._tablename, self._tabledict, 'Time Integration Full')
        self._wp = self._table.row
        self._is_allocated = True

    def write_timestep(self, t, q, dq=None, ddq=None, strain=None, stress=None):
        """
        This function is called to write a timestep into the Hdf solution container

        Parameters
        ----------
        t : float
            time
        q : numpy.array
            solution vector at time t
        dq : numpy.array (optional)
            first time derivative of solution vector at time t
        ddq : numpy.array (optional)
            second time derivative of solution vectot at time t
        stress : ndarray
            nodal stresses
        strain : ndarray
            nodal strains

        Returns
        -------
        None
        """
        # Check if table has already been created/allocated
        if not self._is_allocated:
            # if not allocated do:
            # get dimension of q vectors
            n_dim = q.shape[0]
            # Check if dq and ddq will be saved and created boolean flags for allocate function
            if dq is None:
                is_dq = False
            else:
                is_dq = True
            if ddq is None:
                is_ddq = False
            else:
                is_ddq = True
            if strain is None:
                is_strain = False
            else:
                is_strain = True
            if stress is None:
                is_stress = False
            else:
                is_stress = True
            # call allocate function to create table
            self._allocate(n_dim, is_dq, is_ddq, is_strain, is_stress)
        # Write t and q
        self._wp['t'] = t
        self._wp['q'] = q
        # check if dq and ddq are available and write if possible
        if dq is not None:
            self._wp['dq'] = dq
        if ddq is not None:
            self._wp['ddq'] = ddq
        if strain is not None:
            self._wp['strain'] = strain
        if stress is not None:
            self._wp['stress'] = stress
        self._wp.append()
        # If buffer is full, call flush() to write current buffer to disk
        # (Note: Possibly pytables have already called the flush function by itself it it thinks that it is time
        # to flush)
        if self._iteration >= self._buffer_size:
            self._table.flush()
            self._iteration = 0
            return
        self._iteration += 1
        print("Timestep {} written to HDF5".format(t))


async def solve_async(queue_size, writer, solver, **solverkwargs):
    """
    Solve a problem and write results asynchronous

    Parameters
    ----------
    queue_size : int
        Number of slots available in the Queue
        The Queue is used by the solver to put results into and
        it is used by the writer to get results (pop) to write
    writer : AmfeSolutionBase
        AmfeSolution container to write the results into in parallel
    solver : Solver
        Solver that has an solve_async method
    solverkwargs : dict
        Keyword arguments that shall be passed to the solve_async function

    Returns
    -------
    None
    """
    # Create a container for the Queue
    container = AmfeSolutionAsync(queue_size)

    # Define a solver function for the asynchornous tasks
    # (Adapt solver to a solver function for the _solver_async_fun API)
    def _solver_func(write_callback):
        # Check if callback is already set in solverkwargs.
        # If set, then add the Queue as a new callback
        return solver.solve_async(write_callback, **solverkwargs)

    # Define a writer function for the solve_async_fun
    writer_func = writer.write_timestep

    # Call _solve_async_fun
    await _solve_async_fun(container, writer_func, _solver_func)


async def _solve_async_fun(container, writer_func, solver_func):
    """
    Run solve and write process asynchronous

    Parameters
    ----------
    container : AmfeSolutionAsync
        Container for storing the solutions for communication between solver and writer
    writer_func : function
        function for writing the solution asynchronously with signature fun(t, q, dq=None, ddq=None)
    solver_func : Coroutine
        Pointer to a coroutine that runs the solution process with signature async fun(write_callback)
        where write callback is a function with signature fun(t, q, dq=None, ddq=None)

    Returns
    -------
    None
    """

    # Define the solution coroutine
    async def solve():
        await solver_func(container.write_timestep)

    # Define the Write coroutine
    async def write():
        # Ask for new solutions until it is cancelled
        while True:
            # Ask for a timestep from the queue
            sol = await container.get()
            # Call writer callback
            writer_func(sol['t'], sol['q'], sol['dq'], sol['ddq'], sol['strain'], sol['stress'])

    # Create tasks for writing and solving
    print("Create Tasks for Writing and Solving... ", end="")
    task_write = asyncio.create_task(write())
    task_solve = asyncio.create_task(solve())
    print("Done.")

    # Wait until solver is done.
    await task_solve
    print("Task Solve done.")

    # Wait until all timesteps in queue has been written and then cancel write task
    await container.queue.join()
    print("Queues for writing results are now empty, cancel writing task... ")
    task_write.cancel()
    print("Done.")
    # Finish asynchronous run.
    await asyncio.gather(task_write, return_exceptions=True)
    print("Asynchronous Solve/Write process has been finished.")

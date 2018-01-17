# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module contains time integration
"""

from scipy.sparse import issparse
import scipy as sp
from scipy.sparse.linalg import spsolve

try:
    from amfe.linalg.lib import PardisoWrapper
    use_pardiso = True
except:
    use_pardiso = False


class LinearSolver:

    def __init__(self, A=None, options=None):
        pass

    def set_A(self, A):
        pass

    def clear(self):
        '''
        Method for clearing memory space
        '''
        pass

    def solve(self, b):
        '''
        Method for solving for a rhs
        :param b: 
        :return: x
        '''
        pass

    def get_options(self):
        '''
        Return a sting that shows current options set
        :return: 
        '''
        pass

    def __str__(self):
        '''
        Return a string that gives information about the solver state
        :return: 
        '''


class PardisoSolver(LinearSolver):

    mtypes = {'sym': 1,
              'spd': 2,
              'sid': -2,
              'nonsym': 11,
              }

    available_options = {'mtype': 'Matrix Type, can be sym, spd, sid or nonsym',
                         'verbose': 'Verbose',
                         'saddle_point': 'Set options for saddlepoint problem',
                         'refinement_steps': 'Number of Refinement steps',
                         'pivoting_perturbation': 'Small Pivots are perturbed with eps=10^-(this_value)',
                         'scaling': 'Use maximum weight matching algorithm to permute large elements to diagonal and'
                                    'scale them to one',
                         'transposed': 'use transposed matrix A instead',
                         'maximum_weighted_matching': 'Use maximum weighted matching to permute large elements close'
                                                      'diagonal',
                         'indefinite_pivoting': 'Pivoting for indefinite matrices',
                         'partial_solve': '',
                         'storage_mode': '',
    }

    # info:
    # For changing iparms that are not listed here, just add a name for the iparm parameter
    # Then you can pass an options dictionary to change the iparms
    iparm_dict = {'refinement_steps': '7',
               'pivoting_perturbation': '9',
               'scaling': '10',
               'transposed': '11',
               'maximum_weighted_matching': '12',
               'indefinite_pivoting': '20',
               'partial_solve': '30',
               'storage_mode': '59',
                  }

    def __init__(self, A=None, options=None):
        # call options initialization from superclass
        super().__init__(A, options)
        # Set some default values
        self.iparm = {}
        self.verbose = False
        self.mtype = 'nonsym'
        if A:
            # Check if A is sparse
            if not issparse(A):
                raise ValueError('Error in PardisoSolver: Matrix A is not sparse!')
            # First check if saddle_point problem option is set (this can be overwritten by other options)
            if options is not None:
                if 'saddle_point' in options:
                    # Check if saddle_point option is None or False
                    if options['saddle_point']:
                        # Update two important parameters for saddle_point Problems
                        self.iparm.update({'scaling': 1, 'maximum_weighted_matching': 1})
                # Write other option parameters
                for key in options:
                    if key not in self.available_options:
                        raise ValueError('Error in PardisoSolver: Options Value {} not valid'.format(key))
                    else:
                        # Check if verbose option is activated
                        if key == 'verbose':
                            self.verbose = options['verbose']
                        # Check if mtype is in options
                        if key == 'mtype':
                            # Check if mtype is valid
                            if options[key] in self.mtypes:
                                self.mtype = self.mtypes[options['mtype']]
                            # Otherwise raise value error
                            else:
                                raise ValueError('Error in PardisoSolver mtype {} not available'.format(options[key]))
                        # Check if key belongs to iparm parmeters
                        if key in self.iparm_dict:
                            self.iparm.update({self.iparm_dict[key]: options[key]})

            self.wrapper_class = PardisoWrapper(A, mtype=self.mtype, iparm=self.iparm, verbose=self.verbose)
        # Otherwise (if A is not sparse)
        else:
            raise NotImplementedError('Please implement preallocation for A and updating of A')

    def set_A(self, A):
        if isinstance(A, sp.sparse.csr_matrix):
            if len(A.data) == len(self.wrapper_class.a) & len(A.indptr) == len(self.wrapper_class.ia) & \
                    len(A.indices) == len(self.wrapper_class.ja):
                self.wrapper_class.a = A.data
                self.wrapper_class.ia = A.indptr
                self.wrapper_class.ja = A.indices
            else:
                self.wrapper_class = PardisoWrapper(A, mtype=self.mtype, verbose=self.verbose)
        else:
            try:
                Acsr = sp.sparse.csr_matrix(A)
                self.wrapper_class = PardisoWrapper(Acsr, mtype=self.mtype, verbose=self.verbose)
            except:
                raise ValueError('A must be A csr_matrix or at least a csr convertible matrix')

    def clear(self):
        self.wrapper_class.clear()

    def factorize(self):
        self.wrapper_class.factor()

    def solve(self, b):
        self.wrapper_class.solve(b)

    def get_options(self, prefix=''):
        print('Verbose: {}, Matrix-Type (mtype): {}, iparms: {}'.format(self.verbose, self.mtype, self.iparm))



# Shortcut for compatibility
def solve_sparse(A, b, matrix_type='symm', verbose=False):
    '''
    Abstraction of the solution of the sparse system Ax=b using the fastest
    solver available for sparse and non-sparse matrices.

    Parameters
    ----------
    A : sp.sparse.CSR
        sparse matrix in CSR-format
    b : ndarray
        right hand side of equation
    matrix_type : {'spd', 'symm', 'unsymm'}, optional
        Specifier for the matrix type:

        - 'spd' : symmetric positive definite
        - 'symm' : symmetric indefinite, default.
        - 'unsymm' : generally unsymmetric

    Returns
    -------
    x : ndarray
        solution of system Ax=b

    Notes
    -----
    This tool uses the Intel MKL library provided by Anaconda. If the Intel MKL
    is not installed, especially for large systems the computation time can go
    crazy. To adjust the number of threads used for the computation, it is
    recommended to use the mkl-service module provided by Anaconda:

    >>> import mkl
    >>> mkl.get_max_threads()
    2
    >>> mkl.set_num_threads(1)
    >>> mkl.get_max_threads()
    1

    '''
    print('The function solve_sparse is deprecated and will be removed in next release')
    if sp.sparse.issparse(A):
        if use_pardiso:
            mtype = PardisoSolver.mtypes[matrix_type]
            pSolve = PardisoWrapper(A, mtype=mtype, verbose=verbose)
            x = pSolve.run_pardiso(13, b)
            pSolve.clear()
        else:
            x = spsolve(A, b)
    else:
        x = sp.linalg.solve(A, b)
    return x

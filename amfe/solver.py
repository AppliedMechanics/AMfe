# -*- coding: utf-8 -*-
# pylint: disable=trailing-whitespace, C0103, E1101

'''
Module for solving static and dynamic problems.

'''

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def norm_of_vector(array):
    '''
    Compute the (2-)norm of a vector.
    
    Parameters
    ----------
    array : ndarray
        one dimensional array
        
    Returns
    -------
    abs : float
        2-norm of the given array. 
    '''
    return np.sqrt(array.T.dot(array))

class NewmarkIntegrator():
    '''
    Newmark-integration scheme using generalized alpha routine for nonlinear 
    second order systems

    Notes
    ------
    This integration scheme is an unconditionally stable implicit nonlinear 
    integration scheme. The unconditional stabiltiy refers to the stability of 
    the solution itself, not on the solutin procedure. With a too tight 
    tolerance (eps > 1E8) the solution might not converge, as the solution 
    procedure can not lower the residual below a threshold that's higher than 
    the eps. In general the solution should converge in less than ten iteration 
    steps.

    Examples
    --------
    TODO

    References
    ----------
    M. Géradin and D. J. Rixen. Mechanical vibrations: theory and application
    to structural dynamics. John Wiley & Sons, 2014. pp. 564.

    '''

    def __init__(self, alpha=0, verbose=False, n_iter_max=40):
        '''
        Parameters
        -----------
        alpha : float, optional
            damping factor of the generalized alpha routine for numerical damping
        verbose : bool, optional
            flag for making the integration process verbose, i.e. printing the 
            residual for every correction in the newton iteration
        n_iter_max : int, optional
            number of maximum iteration in the newton correction process

        Returns
        -------
        None
        
        '''
        self.beta = 1/4*(1 + alpha)**2
        self.gamma = 1/2 + alpha
        self.delta_t = 1E-3
        self.eps = 1E-8
        self.newton_damping = 1.0
        self.residual_threshold = 1E6
        self.mechanical_system = None
        self.verbose = verbose
        self.n_iter_max = n_iter_max
        
        # variables to be overloaded in the setting functions
        self.K_and_f_non = None
        self.f_non = None
        self.M = None
        self.f_ext = None

    def set_nonlinear_model(self, f_non, K, M, f_ext=None):
        '''
        Sets the nonlinear model with explicit function metioning.

        Parameters
        -----------
        f_non : function
            function of nonlinear force being called with f_non(q) and 
            returning the nonlinear force as ndarray
        K : function
            function of tangential stiffness matrix being called with K(q) and 
            returning the nonlinear stiffness matrix as ndarray
        M : ndarray
            mass matrix
        f_ext : function, optional
            function of external force being called with f_ext(q, dq, t) and 
            returning the external force as ndarray

        Examples
        ---------

        Notes
        ------
        This function serves basically as a test function. For elaborate finite
        element work the set_mechanical_system interface is more convenient and
        does everything automatically including the recording of displacements 
        etc.

        See Also
        --------
        set_mechanical_system

        '''
        # decorator for the efficient computation of the tangential stiffness
        # matrix and force in one step; Basic intention is to make assembly 
        # process only once.
        def K_and_f_non(q, t):
            '''
            Decorating function for wrapping K() and f() in one function. 
            '''
            return K(q, t), f_non(q, t)
        self.K_and_f_non = K_and_f_non
        self.f_non = f_non
        self.M = M
        self.f_ext = f_ext

    def set_mechanical_system(self, mechanical_system):
        '''
        hands over the mechanical system as a whole to the integrator.
        The matrices for the integration routine are then taken right from the 
        mechanical system

        Parameters
        -----------
        mechanical_system : MechanicalSystem
            instance of MechanicalSystem which should be integrated

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        self.mechanical_system = mechanical_system
        # assuming constant mass matrix! 
        self.M = mechanical_system.M()
#        self.K = mechanical_system.K
        self.f_non = mechanical_system.f_int
        self.f_ext = mechanical_system.f_ext
        self.K_and_f_non = mechanical_system.K_and_f


    def _residual(self, f_non, q, dq, ddq, t):
        '''computes the residual of the system with the given variables'''
        if self.f_ext is not None:
            res = self.M.dot(ddq) + f_non - self.f_ext(q, dq, t)
        else:
            res = self.M.dot(ddq) + f_non
        return res


    def integrate_nonlinear_system(self, q_start, dq_start, time_range):
        '''
        Integrates the system using generalized alpha method.

        Parameters
        -----------
        q_start : ndarray
            initial displacement of the constrained system in voigt notation
        dq_start : ndarray
            initial velocity of the constrained system in voigt notation
        time_range : ndarray
            vector containing the time points at which the state is written to the output

        Returns
        --------
        q : ndarray
            displacements of the system with q[:, i] being the
            displacement of the i-th timestep in voigt notation
        dq : ndarray
            velocities of the system with dq[:, i] being the velocity of the
            i-th timestep in voigt notation

        Examples
        ---------
        TODO

        '''
        # initialize starting variables
        q = q_start.copy()
        dq = dq_start.copy()
        ddq = np.zeros(len(q_start))

        q_global = []
        dq_global = []
        t = 0
        time_index = 0 # index of the timestep in the time_range array
        write_flag = False
        # catch start value 0:
        if time_range[0] < 1E-12:
            q_global.append(q)
            dq_global.append(dq)
            time_index = 1
        else:
            time_index = 0

        # predict start values for ddq:
        ddq = linalg.spsolve(self.M, self.f_non(q, t))
        no_newton_convergence_flag = False
        while time_index < len(time_range):
            # time tolerance fitting...
            if t + self.delta_t + 1E-8 >= time_range[time_index]:
                dt = time_range[time_index] - t
                if dt < 1E-8:
                    dt = 1E-7
                write_flag = True
                time_index += 1
            else:
                dt = self.delta_t
                if no_newton_convergence_flag:
                    dt /= 2
                    no_newton_convergence_flag = False
            # Handling if no convergence is gained:
            t_old = t
            q_old = q.copy()
            dq_old = dq.copy()

            t += dt
            # Prediction
            q += dt*dq + (1/2-self.beta)*dt**2*ddq
            dq += (1-self.gamma)*dt*ddq
            ddq *= 0

            # checking residual and convergence
            K, f_non = self.K_and_f_non(q, t)
            res = self._residual(f_non, q, dq, ddq, t)
            res_abs = norm_of_vector(res)

            # Newcton-Correction-loop
            n_iter = 0
            while res_abs > self.eps*norm_of_vector(f_non):
                S = K + 1/(self.beta*dt**2)*self.M
                delta_q = - linalg.spsolve(S, res)
                if res_abs > self.residual_threshold:
                    delta_q *= self.newton_damping
                q += delta_q
                dq += self.gamma/(self.beta*dt)*delta_q
                ddq += 1/(self.beta*dt**2)*delta_q
                K, f_non = self.K_and_f_non(q, t)
                res = self._residual(f_non, q, dq, ddq, t)
                res_abs = norm_of_vector(res)
                n_iter += 1
                if self.verbose:
                    print('Iteration', n_iter, 'Residuum:', res_abs)
                # catch when the newton loop doesn't converge
                if n_iter > self.n_iter_max:
                    t = t_old
                    q = q_old.copy()
                    dq = dq_old.copy()
                    no_newton_convergence_flag = True
                    break

            print('Zeit:', t, 'Anzahl an Iterationen:', n_iter, 'Residuum:', res_abs)
            # Writing if necessary:
            if write_flag:
                # writing to the mechanical system, if possible
                if self.mechanical_system:
                    self.mechanical_system.write_timestep(t, q)
                else:
                    q_global.append(q.copy())
                    dq_global.append(dq.copy())
                write_flag = False
                
        # end of time loop

        return np.array(q_global), np.array(dq_global)



def solve_linear_displacement(mechanical_system, t=1, verbose=True):
    '''
    Solve the linear static problem of the mechanical system and print
    the results directly to the mechanical system. 

    Parameters
    ----------
    mechanical_system : Instance of the class MechanicalSystem
        Mechanical system to be solved. 
    t : float
        time for the external force call in MechanicalSystem.
    verbose : bool
        Flag for verbose output. 
        
    Returns
    -------
    None

    '''
    if verbose: 
        print('Assembling force and stiffness')
    K, f_int = mechanical_system.K_and_f(t=t)
    f_ext = mechanical_system.f_ext(None, None, t)
    mechanical_system.write_timestep(0, f_ext*0) # write zeros
    if verbose: 
        print('Start solving linear static problem')
    u = linalg.spsolve(K, f_ext - f_int)
    mechanical_system.write_timestep(1, u)
    if verbose: 
        print('Static problem solved')


def solve_nonlinear_displacement(mechanical_system, no_of_load_steps=10,
                                 t=0, eps=1E-12, newton_damping=1,
                                 n_max_iter=1000, smplfd_nwtn_itr=1, 
                                 wrt_iter=False, verbose=True):
    '''
    Solver for the nonlinear system applied directly on the mechanical system.

    Prints the results directly to the mechanical system

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem
    no_of_load_steps : int
        Number of equally spaced load steps which are applied in order to 
        receive the solution
    t : float, optional
        time for the external force call in mechanical_system
    eps : float, optional
        Epsilon for assessment, when a loadstep has converged
    newton_damping : float, optional
        Newton-Damping factor applied in the solution routine; 1 means no damping,
        0 < newton_damping < 1 means damping
    n_max_iter : int, optional
        Maximum number of interations in the Newton-Loop
    smplfd_nwtn_itr : int, optional
          Number at which the jacobian is updated; if 1, then a full newton 
          scheme is applied; if very large, it's a fixpoint iteration with 
          constant jacobian
    wrt_iter : bool, optional
        export every iteration step to ParaView. 
    verbose : bool, optional
        print messages if necessary

    Returns
    -------
    None

    Examples
    ---------
    TODO

    '''
    stepwidth = 1/no_of_load_steps
    f_ext = mechanical_system.f_ext(None, None, t=1)
    ndof = f_ext.shape[0]
    u = np.zeros(ndof)
    mechanical_system.write_timestep(0, u) # initial write

    K, f_int= mechanical_system.K_and_f(u, t=1)
    abs_f_ext = np.sqrt(f_int @ f_int)
    for t in np.arange(stepwidth, 1+stepwidth, stepwidth):
        # prediction
        K, f_int= mechanical_system.K_and_f(u, t)
        res = f_int
        abs_res = norm_of_vector(res)

        # Newton-Loop
        n_iter = 0
        while (abs_res > eps*abs_f_ext) and (n_max_iter > n_iter):
            corr = linalg.spsolve(K, res)
            u -= corr*newton_damping
            if (n_iter % smplfd_nwtn_itr) is 0:
                K, f_int = mechanical_system.K_and_f(u, t)
            res = f_int
            abs_res = norm_of_vector(res)
            n_iter += 1
            if verbose: 
                print('Stufe', t, 'Iteration Nr.', n_iter, 'Residuum:', abs_res)
            if wrt_iter: 
                mechanical_system.write_timestep(n_iter, u)
        mechanical_system.write_timestep(t, u)


def give_mass_and_stiffness(mechanical_system):
    '''
    Determine mass and stiffness matrix of a mechanical system.

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem

    Returns
    -------
    M : ndarray
        Mass matrix of the mechanical system
    K : ndarray
        Stiffness matrix of the mechanical system
    '''

    K = mechanical_system.K()
    M = mechanical_system.M()
    return M, K


class HHTConstrained():
    '''
    Generalized-alpha integration scheme for constrained mechanical systems.

    The integrator solves the DAE on the direct given index, i.e. index 3 DAEs 
    are solved directly without index reduction technique. In order to keep the 
    algorithm stable, scaling of the equations is used to avoid instabilities 
    due to small time steps. Secondly, an Augmented Lagrange Term is used to 
    stabilize the algorithm.
    '''

    def __init__(self, delta_t=1E-3, alpha=0, verbose=True, n_iter_max=40):
        '''
        Initialization of the integration scheme.

        Parameters
        ----------
        delta_t : float, optional
            time step size of the time marching procedure. Default value 1E-3.
        alpha : float, optional
            damping parameter of the integration scheme.
            alpha has to be in range 0 < alpha < 0.3. Default value 0.
        verbose : bool, optional
            flag for verbose output. Default value True.
        n_iter_max : int, optional
            number of maximal iterations in the newton correction loop. If 
            maximum number of iterations is reached, the iteration is aborted. 
            If limit is reached, usually either the time step size is to large 
            or the jacobian of the system is wrong.  Default value 40.

        Returns
        -------
        None

        References
        ----------
        Bauchau, Olivier Andre: Flexible multibody dynamics, volume 176. 
        Springer Science & Business Media, 2010.
        '''
        self.beta = 1/4*(1 + alpha)**2
        self.gamma = 1/2 + alpha
        self.delta_t = delta_t
        self.eps = 1E-8
        self.atol = 1E-11
        self.verbose = verbose
        self.n_iter_max = n_iter_max

        # Stuff which has to be overloaded later on: 
        self.constrained_system = None
        self.s = None
        self.f_non = None
        

    def set_constrained_system(self, constrained_system):
        '''
        Set a constrained system for the integrator.

        Parameters
        ----------
        constrained_system : instance of class ConstrainedSystem
            Constrained system

        Returns
        -------
        None

        '''
        self.constrained_system = constrained_system
        # compute the spectral radius of the system matrices in order to receive
        # the scaling factor s:
        ndof = self.constrained_system.ndof
        q0 = sp.zeros(ndof)
        dq0 = sp.zeros(ndof)
        mr = sp.linalg.norm(self.constrained_system.M(q0, dq0), sp.inf)
        dr = sp.linalg.norm(self.constrained_system.D(q0, dq0), sp.inf)
        kr = sp.linalg.norm(self.constrained_system.K(q0, dq0), sp.inf)
        self.s = mr + self.delta_t*dr + self.delta_t**2*kr
        self.f_non = self.constrained_system.f_non


    def integrate_nonlinear_system(self, q_start, dq_start, time_range):
        '''
        Integrates the nonlinear constrained system unsing the HHT-scheme.

        Parameters
        ----------
        q_start : ndarray
            initial generalized position of the system.
        dq_start : ndarray
            initial generalized velocity of the system.
        time_range : ndarray
            Array of the time steps to be exported by the integration scheme.

        Returns
        -------
        q : ndarray
            Matrix of the generalized positions of the system corresponding 
            to the time points given in the time_range array. q[i] is the 
            generalized position of the i-th time step in time_range, 
            i.e. time_range[i].
        dq : ndarray
            Matrix of the generalized velocities of the system corresponding 
            to the time points given in time_range array.
        lambda : ndarray
            Matrix of the Lagrange multipliers associated with the constraints 
            of the system. The columns of lambda correspond to the vectors of 
            the time steps given in time_range.

        Examples
        --------
        TODO

        '''
        # some internal variables defined for better readabiltiy of code
        beta = self.beta
        gamma = self.gamma
        s = self.s
        const_sys = self.constrained_system
        
        ndof = const_sys.ndof
        ndof_const = const_sys.ndof_const
        q = q_start.copy()
        dq = dq_start.copy()
        ddq = np.zeros(ndof)
        lambda_ = np.zeros(ndof_const)
        res = np.zeros(ndof + ndof_const)
        S = sp.zeros((ndof+ndof_const, ndof+ndof_const))

        q_global = []
        dq_global = []
        lambda_global = []
        t = 0
        time_index = 0 # index of the timestep in the time_range array
        write_flag = False

        # catch start value 0:
        if time_range[0] < 1E-12:
            q_global.append(q)
            dq_global.append(dq)
            time_index = 1
        else:
            time_index = 0

        # predict start values for ddq:
#        ddq = linalg.spsolve(const_sys.M(q, dq), self.f_non(q))

        while time_index < len(time_range):
            # time tolerance fitting...
            if t + self.delta_t + 1E-8 >= time_range[time_index]:
                dt = time_range[time_index] - t
                if dt < 1E-8:
                    dt = 1E-7
                write_flag = True
                time_index += 1
            else:
                dt = self.delta_t

            t += dt
            # Prediction
            q += dt*dq + (1/2 - beta)*dt**2*ddq
            dq += (1-gamma)*dt*ddq
            ddq *= 0
            lambda_ *= 1 # leave it the way it was...

            # checking residual and convergence
            B = const_sys.B(q, dq, t)
            K = const_sys.K(q, dq)
            D = const_sys.D(q, dq)
            M = const_sys.M(q, dq)
            K += B.T.dot(B) * s # take care of the augmentation term
            S[:ndof, :ndof] = K + gamma/(dt*beta)*D + 1/(beta*dt**2)*M
            S[:ndof, ndof:] = s / dt**2 * B.T
            S[ndof:, :ndof] = s / dt**2 * B
            C = const_sys.C(q, dq, t)
            f_non= const_sys.f_non(q, dq)
            f_ext = const_sys.f_ext(q, dq, t)
            res[:ndof] = M.dot(ddq) + f_non - f_ext
            res[ndof:] = C * s/dt**2
            res_abs = norm_of_vector(res)

            # Newcton-Correction-loop
            n_iter = 0
            while res_abs > self.eps*norm_of_vector(f_non+ f_ext):

                # solve the system
                delta_x = - linalg.spsolve(S, res)
                delta_q = delta_x[:ndof]
                delta_lambda = delta_x[ndof:]

                # update of system state
                q   += delta_q
                dq  += gamma/(beta*dt)*delta_q
                ddq += 1/(beta*dt**2)*delta_q
                lambda_ += delta_lambda

                # make the new calculations
                B = const_sys.B(q, dq, t)
                K = const_sys.K(q, dq)
                D = const_sys.D(q, dq)
                M = const_sys.M(q, dq)
                K += B.T.dot(B) * s/dt**2 # take care of the augmentation term
                S[:ndof, :ndof] = K + gamma/(beta*dt)*D + 1/(beta*dt**2)*M
                S[:ndof, ndof:] = B.T * s/dt**2
                S[ndof:, :ndof] = B * s/dt**2
                C = const_sys.C(q, dq, t)
                f_non= const_sys.f_non(q, dq)
                f_ext = const_sys.f_ext(q, dq, t)
                res[:ndof] = M.dot(ddq) + f_non + s/dt**2*B.T.dot(lambda_) - f_ext
                res[ndof:] = C * s/dt**2
                res_abs = norm_of_vector(res)

                n_iter += 1
                if self.verbose:
                    print('Iteration', n_iter, 'Residual:', res_abs, 
                          'Cond-Nr of S:', np.linalg.cond(S))
                if n_iter > self.n_iter_max:
                    raise Exception('Maximum number of iterations reached.'
                                    'The process will be aborted. Current time step:', t)

            print('Time:', t, 'No of iterations:', n_iter, 'residual:', res_abs)


            # Writing if necessary:
            if write_flag:
                # writing to the constrained system, if possible
                # TODO: the functionality for the constrained system to read 
                # TODO: and write timesteps is not implemented yet.
                if False:# self.mechanical_system:
                    self.constrained_system.write_timestep(t, q)
                else:
                    q_global.append(q.copy())
                    dq_global.append(dq.copy())
                    lambda_global.append(lambda_.copy())
                write_flag = False
        # end of time loop


        return np.array(q_global), np.array(dq_global), np.array(lambda_global)



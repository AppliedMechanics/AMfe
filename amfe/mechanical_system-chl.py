class MechanicalSystemStateSpace(MechanicalSystem):
    '''
    TBD
    '''

    def __init__(self,regular_matrix=None, **kwargs):
        '''
        TBD
        '''
        MechanicalSystem.__init__(self, **kwargs)
        if regular_matrix is None:
            self.R_constr = self.K()
        else:
            self.R_constr = regular_matrix
        self.x_red_output = []
        self.R_constr = regular_matrix
        self.E_constr = None


    def M(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            self.M_constr = MechanicalSystem.M(self, x[0:int(x.size/2)], t)
        else:
            self.M_constr = MechanicalSystem.M(self, None, t)
        return self.M_constr


    def E(self, x=None, t=0):
        '''
        TBD
        '''
        if self.M_constr is None:
            self.M(x, t)
        self.E_constr = bmat([[self.R_constr, None],
                              [None, self.M_constr]])
        return self.E_constr


    def D(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            self.D_constr = MechanicalSystem.D(self, x[0:int(x.size/2)], t)
        else:
            self.D_constr = MechanicalSystem.D(self, None, t)
        return self.D_constr

    def K(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            K = MechanicalSystem.K(self, x[0:int(x.size/2)], t)
        else:
            K = MechanicalSystem.K(self, None, t)
        return K


    def A(self, x=None, t=0):
        '''
        TBD
        '''
        if self.D_constr is None:
            A = bmat([[None, self.R_constr], [-self.K(x, t), None]])
        else:
            A = bmat([[None, self.R_constr], [-self.K(x, t), -self.D_constr]])
        return A


    def f_int(self, x=None, t=0):
        '''
        TBD
        '''
        if x is None:
            x = np.zeros(2*self.dirichlet_class.no_of_constrained_dofs)
        f_int = MechanicalSystem.f_int(self, x[0:int(x.size/2)], t)
        return f_int


    def F_int(self, x=None, t=0):
        '''
        TBD
        '''
        if x is None:
            x = np.zeros(2*self.dirichlet_class.no_of_constrained_dofs)
        if self.D_constr is None:
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):],
                                    -self.f_int(x, t)), axis=0)
        else:
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):],
                                    -self.D_constr@x[int(x.size/2):]
                                     - self.f_int(x, t)), axis=0)
        return F_int


    def f_ext(self, x, t):
        '''
        TBD
        '''
        if x is None:
            f_ext = MechanicalSystem.f_ext(self, None, None, t)
        else:
            f_ext = MechanicalSystem.f_ext(self, x[0:int(x.size/2)],
                                           x[int(x.size/2):], t)
        return f_ext


    def F_ext(self, x, t):
        '''
        TBD
        '''
        F_ext = np.concatenate((np.zeros(self.dirichlet_class.no_of_constrained_dofs),
                                self.f_ext(x, t)), axis=0)
        return F_ext


    def K_and_f(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            K, f_int = MechanicalSystem.K_and_f(self, x[0:int(x.size/2)], t)
        else:
            K, f_int = MechanicalSystem.K_and_f(self, None, t)
        return K, f_int


    def A_and_F(self, x=None, t=0):
        '''
        TBD
        '''
        if x is None:
            x = np.zeros(2*self.dirichlet_class.no_of_constrained_dofs)
        K, f_int = self.K_and_f(x, t)
        if self.D_constr is None:
            A = bmat([[None, self.R_constr], [-K, None]])
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):], -f_int), axis=0)
        else:
            A = bmat([[None, self.R_constr], [-K, -self.D_constr]])
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):],
                                    -self.D_constr@x[int(x.size/2):] - f_int), axis=0)
        return A, F_int


    def set_parameters_jwh_alpha(self, dt, rho_inf):
        '''
        Set parameters for JWH-alpha time integration scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        self.dt = dt
        self.rho_inf = rho_inf
        self.alpha_m = (3 - rho_inf)/(2*(1 + rho_inf))
        self.alpha_f = 1/(1 + rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return


    def predict_jwh_alpha(self, x, dx):
        '''
        Predict displacement, velocity and acceleration  for non-linear
        JWH-alpha time integration scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        x += self.dt*(1 - self.gamma)*dx
        dx *= 0
        return


    def newton_raphson_jwh_alpha(self, x, dx, x_old, dx_old, t, t_old):
        '''
        Return actual Jacobian and residuum for non-linear JWH-alpha time
        integration scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        if self.E_constr is None:
            self.E(x, t)

        dx_m = self.alpha_m*dx + (1 - self.alpha_m)*dx_old
        x_f = self.alpha_f*x + (1 - self.alpha_f)*x_old
        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        A_f, F_f = self.A_and_F(x_f, t_f)

        F_ext_f = self.F_ext(x_f, t_f)

        Jac = self.alpha_f*A_f - self.alpha_m/(self.gamma*self.dt)*self.E_constr
        res = F_f + F_ext_f - self.E_constr@dx_m

        return Jac, res, F_ext_f


    def update_jwh_alpha(self, x, dx, delta_x):
        '''
        Update displacement, velocity and acceleration for non-linear JWH-alpha
        time integration scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        x += delta_x
        dx += 1/(self.gamma*self.dt)*delta_x
        return


    def effective_stiffness_jwh_alpha(self):
        '''
        Return effective stiffness matrix for linear JWH-alpha time integration
        scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        self.E()
        self.A_constr = self.A()

        K_eff = self.alpha_m/(self.gamma*self.dt)*self.E_constr \
                - self.alpha_f*self.A_constr
        return K_eff


    def effective_force_jwh_alpha(self, x_old, dx_old, t, t_old):
        '''
        Return actual effective force for linear JWH-alpha time integration
        scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        F_ext_f = self.F_ext(None, t_f)

        F_eff = (self.alpha_m/(self.gamma*self.dt)*self.E_constr \
                 + (1 - self.alpha_f)*self.A_constr)@x_old \
                + ((self.alpha_m - self.gamma)/self.gamma*self.E_constr)@dx_old \
                + F_ext_f
        return F_eff


    def post_process_jwh_alpha(self, x, x_old, dx_old):
        '''
        Return actual velocity and acceleration for linear JWH-alpha time
        integration scheme.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.

        '''
        dx = 1/(self.gamma*self.dt)*(x - x_old) + (self.gamma - 1)/self.gamma*dx_old
        return dx


    def write_timestep(self, t, x):
        '''
        TBD
        '''
        MechanicalSystem.write_timestep(self, t, x[0:int(x.size/2)])
        self.x_output.append(x.copy())
        return


    def export_paraview(self, filename, field_list=None):
        '''
        TBD
        '''
        x_export = np.array(self.x_output).T
        x_dict = {'ParaView':'False', 'Name':'x'}
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()
        new_field_list.append((x_export, x_dict))
        MechanicalSystem.export_paraview(self, filename, new_field_list)
        return


    def clear_timesteps(self):
        MechanicalSystem.clear_timesteps(self)
        self.x_output = []
        return


class ReducedSystem(MechanicalSystem):
    '''
    Class for reduced systems.
    It is directly inherited from MechanicalSystem.
    Provides the interface for an integration scheme and so on where a basis
    vector is to be chosen...

    Notes
    -----
    The Basis V is a Matrix with x = V*q mapping the reduced set of coordinates
    q onto the physical coordinates x. The coordinates x are constrained, i.e.
    the x denotes the full system in the sense of the problem set and not of
    the pure finite element set.

    The system runs without providing a V_basis when constructing the method
    only for the unreduced routines.


    Attributes
    ----------
    V : ?
        Set of basis vectors the system has been reduced with u_constr = V * q
    V_unconstr : ?
        Extended reduction basis that is extended by the displacement coordinates of the constrained degrees of
        freedom
    u_red_output : ?
        Stores the timeseries of the generalized coordinates (similar to u_output)
    assembly_type : {'indirect', 'direct'}
        Stores the type of assembly method how the reduced system is computed

    Examples
    --------

    my_system = amfe.MechanicalSystem()
    V = vibration_modes(my_system, n=20)
    my_reduced_system = amfe.reduce_mechanical_system(my_system, V)


    '''

    def __init__(self, V_basis=None, assembly='indirect', **kwargs):
        '''
        Parameters
        ----------
        V_basis : ndarray, optional
            Basis onto which the problem will be projected with an
            Galerkin-Projection.
        assembly : str {'direct', 'indirect'}
            flag setting, if direct or indirect assembly is done. For larger
            reduction bases, the indirect method is much faster.
        **kwargs : dict, optional
            Keyword arguments to be passed to the mother class MechanicalSystem.

        Returns
        -------
        None
        '''
        MechanicalSystem.__init__(self, **kwargs)
        self.V = V_basis
        self.u_red_output = []
        self.V_unconstr = self.dirichlet_class.unconstrain_vec(V_basis)
        self.assembly_type = assembly

    def K_and_f(self, u=None, t=0):
        if u is None:
            u = np.zeros(self.V.shape[1])
        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr,
                                                                u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u,
                                                                t)
            K = self.V_unconstr.T @ K_raw @ self.V_unconstr
            f_int = self.V_unconstr.T @ f_raw
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')
        return K, f_int

    def K(self, u=None, t=0):
        if u is None:
            u = np.zeros(self.V.shape[1])

        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr,
                                                                u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u,
                                                                t)
            K = self.V_unconstr.T @ K_raw @ self.V_unconstr
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')
        return K

    def f_ext(self, u, du, t):
        return self.V.T @ MechanicalSystem.f_ext(self, self.V @ u, du, t)

    def f_int(self, u, t=0):

        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr,
                                                                u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u,
                                                                t)
            f_int = self.V_unconstr.T @ f_raw
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')

        return f_int

    def D(self, u=None, t=0):

        if self.assembly_type == 'direct':
            raise NotImplementedError('The direct method is note implemented yet for damping matrices')
        elif self.assembly_type == 'indirect':
            self.D_constr = self.V.T @ MechanicalSystem.D(self, self.V @ u, t) @ self.V
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')

        return self.D_constr

    def M(self, u=None, t=0):
        # Just a plain projection
        # not so well but works...
        self.M_constr = self.V.T @ MechanicalSystem.M(self, u, t) @ self.V
        return self.M_constr

    def write_timestep(self, t, u):
        MechanicalSystem.write_timestep(self, t, self.V @ u)
        self.u_red_output.append(u.copy())

    def K_unreduced(self, u=None, t=0):
        '''
        Unreduced Stiffness Matrix.

        Parameters
        ----------
        u : ndarray, optional
            Displacement of constrained system. Default is zero vector.
        t : float, optionial
            Time. Default is 0.

        Returns
        -------
        K : sparse csr matrix
            Stiffness matrix

        '''
        return MechanicalSystem.K(self, u, t)

    def f_int_unreduced(self, u, t=0):
        '''
        Internal nonlinear force of the unreduced system.

        Parameters
        ----------
        u : ndarray
            displacement of unreduces system.
        t : float, optional
            time, default value: 0.

        Returns
        -------
        f_nl : ndarray
            nonlinear force of unreduced system.

        '''
        return MechanicalSystem.f_int(self, u, t)

    def M_unreduced(self):
        '''
        Unreduced mass matrix.
        '''
        return MechanicalSystem.M(self)

    def export_paraview(self, filename, field_list=None):
        '''
        Export the produced results to ParaView via XDMF format.
        '''
        u_red_export = np.array(self.u_red_output).T
        u_red_dict = {'ParaView':'False', 'Name':'q_red'}

        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()

        new_field_list.append((u_red_export, u_red_dict))

        MechanicalSystem.export_paraview(self, filename, new_field_list)

        # add V and Theta to the hdf5 file
        filename_no_ext, _ = os.path.splitext(filename)
        with h5py.File(filename_no_ext + '.hdf5', 'r+') as f:
            f.create_dataset('reduction/V', data=self.V)

        return

    def clear_timesteps(self):
        MechanicalSystem.clear_timesteps(self)
        self.u_red_output = []
        return


class ReducedSystemStateSpace(MechanicalSystemStateSpace):
    '''
    TBD
    '''

    def __init__(self, right_basis=None, left_basis=None, **kwargs):
        '''
        TBD
        '''
        MechanicalSystemStateSpace.__init__(self, **kwargs)
        self.V = right_basis
        self.W = left_basis
        self.x_red_output = []


    def E(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            self.E_constr = self.W.T@MechanicalSystemStateSpace.E(self, self.V@x, \
                                                                  t)@self.V
        else:
            self.E_constr = self.W.T@MechanicalSystemStateSpace.E(self, None, t)@self.V
        return self.E_constr


    def E_unreduced(self, x_unreduced=None, t=0):
        '''
        TBD
        '''
        return MechanicalSystemStateSpace.E(self, x_unreduced, t)


    def A(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            A = self.W.T@MechanicalSystemStateSpace.A(self, self.V@x, t)@self.V
        else:
            A = self.W.T@MechanicalSystemStateSpace.A(self, None, t)@self.V
        return A


    def A_unreduced(self, x_unreduced=None, t=0):
        '''
        TBD
        '''
        return MechanicalSystemStateSpace.A(self, x_unreduced, t)


    def F_int(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            F_int = self.W.T@MechanicalSystemStateSpace.F_int(self, self.V@x, t)
        else:
            F_int = self.W.T@MechanicalSystemStateSpace.F_int(self, None, t)
        return F_int


    def F_int_unreduced(self, x_unreduced=None, t=0):
        '''
        TBD
        '''
        return MechanicalSystemStateSpace.F_int(self, x_unreduced, t)


    def F_ext(self, x, t):
        '''
        TBD
        '''
        if x is not None:
            F_ext = self.W.T@MechanicalSystemStateSpace.F_ext(self, self.V@x, t)
        else:
            F_ext = self.W.T@MechanicalSystemStateSpace.F_ext(self, None, t)
        return F_ext


    def F_ext_unreduced(self, x_unreduced, t):
        '''
        TBD
        '''
        return MechanicalSystemStateSpace.F_ext(self, x_unreduced, t)


    def A_and_F(self, x=None, t=0):
        '''
        TBD
        '''
        if x is not None:
            A_, F_int_ = MechanicalSystemStateSpace.A_and_F(self, self.V@x, t)
        else:
            A_, F_int_ = MechanicalSystemStateSpace.A_and_F(self, None, t)
        A = self.W.T@A_@self.V
        F_int = self.W.T@F_int_
        return A, F_int


    def A_and_F_unreduced(self, x_unreduced=None, t=0):
        '''
        TBD
        '''
        return MechanicalSystemStateSpace.A_and_F(self, x_unreduced, t)


    def write_timestep(self, t, x):
        '''
        TBD
        '''
        MechanicalSystemStateSpace.write_timestep(self, t, self.V@x)
        self.x_red_output.append(x.copy())
        return


    def export_paraview(self, filename, field_list=None):
        '''
        TBD
        '''
        x_red_export = np.array(self.x_red_output).T
        x_red_dict = {'ParaView':'False', 'Name':'x_red'}
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()
        new_field_list.append((x_red_export, x_red_dict))
        MechanicalSystemStateSpace.export_paraview(self, filename, new_field_list)
        return


    def clear_timesteps(self):
        MechanicalSystemStateSpace.clear_timesteps(self)
        self.x_red_output = []
        return



def reduce_mechanical_system(mechanical_system, V, overwrite=False,
                             assembly='indirect'):
    '''
    Reduce the given mechanical system with the linear basis V.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem.
    V : ndarray
        Reduction Basis for the reduced system
    overwrite : bool, optional
        switch, if mechanical system should be overwritten (is less memory
        intensive for large systems) or not.
    assembly : str {'direct', 'indirect'}
            flag setting, if direct or indirect assembly is done. For larger
            reduction bases, the indirect method is much faster.

    Returns
    -------
    reduced_system : instance of ReducedSystem
        Reduced system with same properties of the mechanical system and
        reduction basis V

    Example
    -------

    '''

    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)
    reduced_sys.__class__ = ReducedSystem
    reduced_sys.V = V.copy()
    reduced_sys.V_unconstr = reduced_sys.dirichlet_class.unconstrain_vec(V)
    reduced_sys.u_red_output = []
    reduced_sys.M_constr = None
    # reduce Rayleigh damping matrix
    if reduced_sys.D_constr is not None:
        reduced_sys.D_constr = V.T @ reduced_sys.D_constr @ V
    reduced_sys.assembly_type = assembly
    return reduced_sys


def convert_mechanical_system_to_state_space(
        mechanical_system,
        regular_matrix=None,
        overwrite=False):
    '''
    TBD
    '''

    if overwrite:
        sys = mechanical_system
    else:
        sys = copy.deepcopy(mechanical_system)
    sys.__class__ = MechanicalSystemStateSpace
    sys.x_output = []
    if regular_matrix is None:
        sys.R_constr = sys.K()
    else:
        sys.R_constr = regular_matrix
    sys.E()
    return sys


def reduce_mechanical_system_state_space(
        mechanical_system_state_space, right_basis,
        left_basis=None,
        overwrite=False):
    '''
    TBD
    '''

    if overwrite:
        red_sys = mechanical_system_state_space
    else:
        red_sys = copy.deepcopy(mechanical_system_state_space)
    red_sys.__class__ = ReducedSystemStateSpace
    red_sys.V = right_basis.copy()
    if left_basis is None:
        red_sys.W = right_basis.copy()
    else:
        red_sys.W = left_basis.sopy()
    red_sys.x_red_output = []
    red_sys.E_constr = None
    return red_sys
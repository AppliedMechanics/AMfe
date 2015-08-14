# -*- coding: utf-8 -*-
"""
Element Module in which the Finite Elements are described on Element level.

This Module is arbitrarily extensible. The idea is to use the basis class Element which provides the functionality for an efficient solution of a time integration by only once calling the internal tensor computation and then extracting the tangential stiffness matrix and the internal force vector in one run.

Some remarks resulting in the observations of the profiler:
Most of the time is spent with pyhton-functions, when they are used. For instance the kron-function in order to build the scattered geometric stiffness matrix or the trace function are very inefficient. They can be done better when using direct functions.

"""


import numpy as np



def scatter_geometric_matrix(Mat, ndim):
    '''
    Scatter the symmetric geometric stiffness matrix to all dofs.

    What is basically done is to perform the kron(Mat, eye(ndof))

    Parameters
    ----------
    Mat : ndarray
        Matrix that should be scattered
    ndim : int
        number of dimensions of finite element. If it's a 2D element, ndim=2,
        if 3D, ndim=3

    Returns
    -------
    Mat_scattered : ndarray
        scattered matrix

    '''
    dof_small_row = Mat.shape[0]
    dof_small_col = Mat.shape[1]
    Mat_scattered = np.zeros((dof_small_row*ndim, dof_small_col*ndim))

    for i in range(dof_small_row):
        for j in range(dof_small_col):
            for k in range(ndim):
                Mat_scattered[ndim*i+k,ndim*j+k] = Mat[i,j]
    return Mat_scattered


def compute_B_matrix(B_tilde, F):
    '''
    Compute the B-matrix used in Total Lagrangian Finite Elements.

    Parameters
    ----------
    F : ndarray
        deformation gradient
    B_tilde : ndarray
        Matrix of the spatial derivative of the shape functions

    Returns
    -------
    B : ndarray
        B matrix such that {dE} = B @ {u^e}

    Notes
    -----
    When the Voigt notation is used in this Reference, the variables are denoted with curly brackets.
    '''

    no_of_nodes = B_tilde.shape[1]
    no_of_dims = B_tilde.shape[0] # spatial dofs per node, i.e. 2 for 2D or 3 for 3D
    b = B_tilde
    B = np.zeros((no_of_dims*(no_of_dims+1)/2, no_of_nodes*no_of_dims))
    F11 = F[0,0]
    F12 = F[0,1]
    F21 = F[1,0]
    F22 = F[1,1]
    if no_of_dims == 3:

        F13 = F[0,2]
        F31 = F[2,0]
        F23 = F[1,2]
        F32 = F[2,1]
        F33 = F[2,2]
    for i in range(no_of_nodes):
        if no_of_dims == 2:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
            [F11*b[0,i], F21*b[0,i]],
            [F12*b[1,i], F22*b[1,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i]]]
        else:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
            [F11*b[0,i], F21*b[0,i], F31*b[0,i]],
            [F12*b[1,i], F22*b[1,i], F32*b[1,i]],
            [F13*b[2,i], F23*b[2,i], F33*b[2,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i]], F31*b[1,i]+F32*b[0,i],
            [F12*b[2,i] + F13*b[1,i], F22*b[2,i] + F23*b[1,i]], F32*b[2,i]+F33*b[1,i],
            [F13*b[0,i] + F11*b[2,i], F23*b[0,i] + F21*b[2,i]], F33*b[0,i]+F31*b[2,i]]
    return B


class Element():
    '''
    this is the baseclass for all elements. It contains the methods needed
    for the computation of the element stuff...
    '''

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, density=1E4):
        pass

    def _compute_tensors(self, X, u):
        '''
        Virtual function for the element specific implementation of a tensor
        computation routine which will be called before _k_int and _f_int
        will be called. For many computations the tensors need to be computed
        the same way.
        '''
        pass

    def _k_int(self, X, u):
        pass

    def _f_int(self, X, u):
        pass

    def _m_int(self, X, u):
        '''
        Virtual function for the element specific implementation of the mass
        matrix;
        '''
        print('The function is not implemented yet...')
        pass

    def k_and_f_int(self, X, u):
        '''
        Returns the tangential stiffness matrix and the internal nodal force
        of the Element.

        Parameters
        -----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array
            of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation

        Returns
        --------
        k_int : ndarray
            The tangential stiffness matrix (ndarray of dimension (ndim, ndim))
        f_int : ndarray
            The nodal force vector (ndarray of dimension (ndim,))

        Examples
        ---------
        TODO

        '''
        self._compute_tensors(X, u)
        return self._k_int(X, u), self._f_int(X, u)

    def k_int(self, X, u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        k_int :     The tangential stiffness matrix (numpy.ndarray of
                    type ndim x ndim)

        '''
        self._compute_tensors(X, u)
        return self._k_int(X, u)

    def f_int(self, X, u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        f_int :     The nodal force vector (numpy.ndarray of dimension (ndim,))

        '''
        self._compute_tensors(X, u)
        return self._f_int(X, u)

    def m_int(self, X, u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        m_int :     The consistent mass matrix of the element
                    (numpy.ndarray of dimension (ndim,ndim))

        '''
        return self._m_int(X, u)


    def k_and_m_int(self, X, u):
        '''
        Returns the stiffness and mass matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        m_int :     The consistent mass matrix of the element
                    (numpy.ndarray of dimension (ndim,ndim))

        '''
        return self._k_and_m_int(X, u)

class Tri3(Element):
    '''
    Element class for a plane triangle element in Total Lagrangian formulation.
    The displacements are given in x- and y-coordinates;

    Element-properties:
    -------------------
    The Element assumes constant strain and stress over the whole element.
    Thus the approximation quality is very moderate.


    References:
    -----------
    Basis for this implementation is the Monograph of Ted Belytschko:
    Nonlinear Finite Elements for Continua and Structures.
    pp. 201 and 207.

    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4):
        '''
        Definition of material properties and thickness as they are 2D-Elements.
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul       = E_modul
        self.lame_mu       = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda   = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        # ATTENTION: here the switch between plane stress and plane strain makes sense.
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])
        self.t     = element_thickness
        self.rho   = density
        self.I     = np.eye(2)
        self.S     = np.zeros((2,2))
        self.K_geo = np.zeros((6,6))
        pass

    def _compute_tensors(self, X, u):
        '''
        Compute the tensors B0_tilde, B0, F, E and S at the Gauss Points.

        The tensors are:
            B0_tilde:   Die Ableitung der Ansatzfunktionen nach den x- und y-Koordinaten (2x3-Matrix)
                        In den Zeilein stehen die Koordinatenrichtungen, in den Spalten die Ansatzfunktionen
            B0:         The mapping matrix of delta E = B0 * u^e
            F:          Deformation gradient (2x2-Matrix)
            E:          Der Green-Lagrange strain tensor (2x2-Matrix)
            S:          2. Piola-Kirchhoff stress tensor, using Kirchhoff material (2x2-Matrix)

        The thickness information is used later for the internal forces, the mass and the stiffness matrix.
        '''
        X1, Y1, X2, Y2, X3, Y3 = X
        self.u        = u.reshape((-1,2))
        self.A0       = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.B0_tilde = 1/(2*self.A0)*np.array([[Y2-Y3, X3-X2], [Y3-Y1, X1-X3], [Y1-Y2, X2-X1]]).T
        self.H        = self.u.T.dot(self.B0_tilde.T)
        self.F        = self.H + self.I
        self.E        = 1./2.*(self.H + self.H.T + self.H.T.dot(self.H))

        self.S_voigt = self.C_SE.dot([self.E[0,0], self.E[1,1], 2*self.E[0,1]])
        self.S[0,0] , self.S[1,1], self.S[1,0], self.S[0,1] = self.S_voigt[0], self.S_voigt[1], self.S_voigt[2], self.S_voigt[2]

        # Building B0 with the product of the deformation gradient
        self.B0 = np.zeros((3, 6))
        for i in range(3):
            self.B0[:,2*i:2*i+2] = np.array([[self.B0_tilde[0,i], 0], [0, self.B0_tilde[1,i]], [self.B0_tilde[1,i], self.B0_tilde[0,i]]]).dot(self.F.T)


    def _f_int(self, X, u):
        '''
        Private method for the computation of the internal nodal forces without computation of the relevant tensors
        '''
        f = self.B0.T.dot(self.S_voigt)*self.A0*self.t
        return f

    def _k_int(self, X, u):
        '''
        Private method for computation of internal tangential stiffness matrix without an update of the internal tensors

        '''
        self.K_geo_small = self.B0_tilde.T.dot(self.S.dot(self.B0_tilde))*self.A0*self.t
        self.K_geo = scatter_geometric_matrix(self.K_geo_small, 2)
        self.K_mat = self.B0.T.dot(self.C_SE.dot(self.B0))*self.A0*self.t
        return self.K_mat + self.K_geo

    def _m_int(self, X, u):
        '''
        Bestimmt die Massenmatrix. Erstellt die Massenmatrix durch die fest einprogrammierte Darstellung aus dem Lehrbuch.
        '''
        X1, Y1, X2, Y2, X3, Y3 = X
        self.A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.M = np.array([[2, 0, 1, 0, 1, 0],
                           [0, 2, 0, 1, 0, 1],
                           [1, 0, 2, 0, 1, 0],
                           [0, 1, 0, 2, 0, 1],
                           [1, 0, 1, 0, 2, 0],
                           [0, 1, 0, 1, 0, 2]])*self.A0/12*self.t*self.rho
        return self.M


class Tri6(Element):
    '''
    6 node second order triangle
    Triangle Element with 6 dofs; 3 dofs at the corner, 3 dofs in the intermediate point of every face.
    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4):
        '''
        Definition der Materialgrößen und Dicke, da es sich um 2D-Elemente handelt
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.lame_mu = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        # Achtung: hier gibt's ebene Dehnung
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])
        self.t = element_thickness
        self.rho = density
        self.I = np.eye(2)
        self.B0_tilde = [[],[],[]]
        self.H = [[],[],[]]
        self.F = [[],[],[]]
        self.E = [[],[],[]]
        self.S_voigt = [[],[],[]]
        self.S = [np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2))]
        self.B0 = [[],[],[]]

    def _B0_tilde_func(self, X_vec, X, Y):
        '''
        compute the B0_tilde matrix for a given X and Y

        Parameters
        ----------
        X_vec : ndarray
            Array giving the positions in the reference configuration
        X : float
            x-position of the quadrature point given in the reference coordinate system
        Y : float
            y-position of the quadrature point given in the reference coordinate system
        Returns
        -------
        '''
        # maybe this is in the scope here
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X_vec
        det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
        # linear coordinates
        L1 = (X*(Y2 - Y3) + X2*Y3 - X3*Y2 + Y*(-X2 + X3))/det
        L2 = (X*(-Y1 + Y3) - X1*Y3 + X3*Y1 + Y*(X1 - X3))/det
        L3 = (X*(Y1 - Y2) + X1*Y2 - X2*Y1 + Y*(-X1 + X2))/det
        # diff of coordinates with respect to X and Y
        L1_X = (Y2-Y3)/det
        L1_Y = (X3-X2)/det
        L2_X = (-Y1+Y3)/det
        L2_Y = (X1-X3)/det
        L3_X = (Y1-Y2)/det
        L3_Y = (-X1+X2)/det
        # Full diff of coordinates
        N1_X = (4*L1 - 1)*L1_X
        N1_Y = (4*L1 - 1)*L1_Y
        N2_X = (4*L2 - 1)*L2_X
        N2_Y = (4*L2 - 1)*L2_Y
        N3_X = (4*L3 - 1)*L3_X
        N3_Y = (4*L3 - 1)*L3_Y

        N4_X = 4*(L1*L2_X + L1_X*L2)
        N4_Y = 4*(L1*L2_Y + L1_Y*L2)
        N5_X = 4*(L2*L3_X + L2_X*L3)
        N5_Y = 4*(L2*L3_Y + L2_Y*L3)
        N6_X = 4*(L1*L3_X + L1_X*L3)
        N6_Y = 4*(L1*L3_Y + L1_Y*L3)

        B0_tilde = np.array([[N1_X, N2_X, N3_X, N4_X, N5_X, N6_X], [N1_Y, N2_Y, N3_Y, N4_Y, N5_Y, N6_Y]])
        return B0_tilde

    def _compute_tensors(self, X, u):
        '''
        Tensor computation the same way as in the Tri3 element
        '''
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        self.u = u.reshape((-1,2))
        det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
        self.A0 = det/2
        quadrature_points = np.array([[X4, Y4], [X5, Y5], [X6, Y6]])
        for i, quadrature_coord in enumerate(quadrature_points):
            self.B0_tilde[i] = self._B0_tilde_func(X, *quadrature_coord)
            self.H[i] = self.u.T.dot(self.B0_tilde[i].T)
            self.F[i] = self.H[i] + self.I
            self.E[i] = 1/2*(self.H[i] + self.H[i].T + self.H[i].T.dot(self.H[i]))
            self.S_voigt[i] = self.C_SE.dot([self.E[i][0,0], self.E[i][1,1], 2*self.E[i][0,1]])
            self.S[i][0,0] , self.S[i][1,1], self.S[i][1,0], self.S[i][0,1] = \
                self.S_voigt[i][0], self.S_voigt[i][1], self.S_voigt[i][2], self.S_voigt[i][2]
            self.B0[i] = np.zeros((3, 12))
            for j in range(6):
                self.B0[i][:,2*j:2*j+2] = np.array([[self.B0_tilde[i][0,j], 0],
                        [0, self.B0_tilde[i][1,j]], [self.B0_tilde[i][1,j],
                         self.B0_tilde[i][0,j]]]).dot(self.F[i].T)

    def _f_int(self, X, u):
        f = np.zeros(12)
        for i in range(3):
            f += self.B0[i].T.dot(self.S_voigt[i])*self.A0*self.t*1/3
        return f

    def _k_int(self, X, u):
        self.K_geo = np.zeros((12, 12))
        self.K_mat = np.zeros((12, 12))
        for i in range(3):
            self.K_geo_small = self.B0_tilde[i].T.dot(self.S[i].dot(self.B0_tilde[i]))*self.A0*self.t*1/3
            self.K_geo += scatter_geometric_matrix(self.K_geo_small, 2)
            self.K_mat += self.B0[i].T.dot(self.C_SE.dot(self.B0[i]))*self.A0*self.t*1/3
        return self.K_mat + self.K_geo

    def _m_int(self, X, u):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
        self.A0 = det/2
        self.M = self.A0 / 180 * self.t * self.rho * np.array([
        [  6.,   0.,  -1.,  -0.,  -1.,  -0.,   0.,   0.,  -4.,  -0.,   0.,  0.],
        [  0.,   6.,  -0.,  -1.,  -0.,  -1.,   0.,   0.,  -0.,  -4.,   0.,  0.],
        [ -1.,  -0.,   6.,   0.,  -1.,  -0.,   0.,   0.,   0.,   0.,  -4., -0.],
        [ -0.,  -1.,   0.,   6.,  -0.,  -1.,   0.,   0.,   0.,   0.,  -0., -4.],
        [ -1.,  -0.,  -1.,  -0.,   6.,   0.,  -4.,  -0.,   0.,   0.,   0.,  0.],
        [ -0.,  -1.,  -0.,  -1.,   0.,   6.,  -0.,  -4.,   0.,   0.,   0.,  0.],
        [  0.,   0.,   0.,   0.,  -4.,  -0.,  32.,   0.,  16.,   0.,  16.,  0.],
        [  0.,   0.,   0.,   0.,  -0.,  -4.,   0.,  32.,   0.,  16.,   0., 16.],
        [ -4.,  -0.,   0.,   0.,   0.,   0.,  16.,   0.,  32.,   0.,  16.,  0.],
        [ -0.,  -4.,   0.,   0.,   0.,   0.,   0.,  16.,   0.,  32.,   0., 16.],
        [  0.,   0.,  -4.,  -0.,   0.,   0.,  16.,   0.,  16.,   0.,  32.,  0.],
        [  0.,   0.,  -0.,  -4.,   0.,   0.,   0.,  16.,   0.,  16.,   0., 32.]])
        return self.M


class Quad4(Element):
    '''
    Elementklasse für viereckiges ebenes Element mit linearen Ansatzfunktionen.
    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4):
        '''
        Definition of material properties and thickness as they are 2D-Elements.
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul       = E_modul
        self.lame_mu       = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda   = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.t             = element_thickness
        self.rho           = density
        # ATTENTION: here the switch between plane stress and plane strain makes sense.
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])

        self.K = np.zeros((8,8))
        self.f = np.zeros(8)

        # Gauss-Point-Handling:
        g1 = 0.577350269189626

        self.gauss_points = ((-g1, -g1, 1.), (g1, -g1, 1.), (-g1, g1, 1.), (g1, g1, 1.))


    def _compute_tensors(self, X, u):
        '''
        Compute the tensors.
        '''
        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X
        X_mat = X.reshape(-1, 2)
        u_e = u.reshape(-1, 2)

        self.K *= 0
        self.f *= 0

        for xi, eta, w in self.gauss_points:

            dN_dxi = np.array([ [ eta/4 - 1/4,  xi/4 - 1/4],
                                [-eta/4 + 1/4, -xi/4 - 1/4],
                                [ eta/4 + 1/4,  xi/4 + 1/4],
                                [-eta/4 - 1/4, -xi/4 + 1/4]])

            dX_dxi = X_mat.T.dot(dN_dxi)
            det = dX_dxi[0,0]*dX_dxi[1,1] - dX_dxi[1,0]*dX_dxi[0,1]
            dxi_dX = 1/det*np.array([[ dX_dxi[1,1], -dX_dxi[0,1]],
                                     [-dX_dxi[1,0],  dX_dxi[0,0]]])

            B0_tilde = np.transpose(dN_dxi.dot(dxi_dX))
            H = u_e.T.dot(B0_tilde.T)
            F = H + np.eye(2)
            E = 1/2*(H + H.T + H.T.dot(H))
            E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
            S_v = self.C_SE.dot(E_v)
            S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde))*det*self.t
            K_geo = scatter_geometric_matrix(K_geo_small, 2)
            K_mat = B0.T.dot(self.C_SE.dot(B0))*det*self.t
            self.K += (K_geo + K_mat)*w
            self.f += B0.T.dot(S_v)*det*self.t*w

    def _f_int(self, X, u):
        return self.f.copy()

    def _k_int(self, X, u):
        return self.K.copy()

    def _m_int(self, X, u):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X
        det = 1/8*(X1*Y2 - X1*Y4 - X2*Y1 + X2*Y3 - X3*Y2 + X3*Y4 + X4*Y1 - X4*Y3)
        self.M = det / 9 * self.rho * self.t * np.array([
                 [ 4.,  0.,  2.,  0.,  1.,  0.,  2.,  0.],
                 [ 0.,  4.,  0.,  2.,  0.,  1.,  0.,  2.],
                 [ 2.,  0.,  4.,  0.,  2.,  0.,  1.,  0.],
                 [ 0.,  2.,  0.,  4.,  0.,  2.,  0.,  1.],
                 [ 1.,  0.,  2.,  0.,  4.,  0.,  2.,  0.],
                 [ 0.,  1.,  0.,  2.,  0.,  4.,  0.,  2.],
                 [ 2.,  0.,  1.,  0.,  2.,  0.,  4.,  0.],
                 [ 0.,  2.,  0.,  1.,  0.,  2.,  0.,  4.]])
        return self.M.copy()


class Quad8(Element):
    '''
    Elementklasse für viereckiges ebenes Element mit quadratischen Ansatzfunktionen.
    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4):
        '''
        Definition of material properties and thickness as they are 2D-Elements.
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul       = E_modul
        self.lame_mu       = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda   = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.t             = element_thickness
        self.rho           = density

        # ATTENTION: here the switch between plane stress and plane strain makes sense.
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])

        self.K = np.zeros((16,16))
        self.f = np.zeros(16)

        # Gauss-Point-Handling
        g3 = 0.861136311594053
        w3 = 0.347854845137454
        g4 = 0.339981043584856
        w4 = 0.652145154862546
        self.gauss_points = (
                 (-g3, -g3, w3*w3), (-g4, -g3, w4*w3), ( g3,-g3, w3*w3), ( g4,-g3, w4*w3),
                 (-g3, -g4, w3*w4), (-g4, -g4, w4*w4), ( g3,-g4, w3*w4), ( g4,-g4, w4*w4),
                 (-g3,  g3, w3*w3), (-g4,  g3, w4*w3), ( g3, g3, w3*w3), ( g4, g3, w4*w3),
                 (-g3,  g4, w3*w4), (-g4,  g4, w4*w4), ( g3, g4, w3*w4), ( g4, g4, w4*w4))


    def _compute_tensors(self, X, u):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6, X7, Y7, X8, Y8 = X
        X_mat = X.reshape(-1, 2)
        u_e = u.reshape(-1, 2)

        self.K *= 0
        self.f *= 0

        for xi, eta, w in self.gauss_points:
            # this is now the standard procedure for Total Lagrangian behavior
            dN_dxi = np.array([
                            [-(eta - 1)*(eta + 2*xi)/4, -(2*eta + xi)*(xi - 1)/4],
                            [ (eta - 1)*(eta - 2*xi)/4,  (2*eta - xi)*(xi + 1)/4],
                            [ (eta + 1)*(eta + 2*xi)/4,  (2*eta + xi)*(xi + 1)/4],
                            [-(eta + 1)*(eta - 2*xi)/4, -(2*eta - xi)*(xi - 1)/4],
                            [             xi*(eta - 1),            xi**2/2 - 1/2],
                            [          -eta**2/2 + 1/2,            -eta*(xi + 1)],
                            [            -xi*(eta + 1),           -xi**2/2 + 1/2],
                            [           eta**2/2 - 1/2,             eta*(xi - 1)]])
            dX_dxi = X_mat.T.dot(dN_dxi)
            det = dX_dxi[0,0]*dX_dxi[1,1] - dX_dxi[1,0]*dX_dxi[0,1]
            dxi_dX = 1/det*np.array([[ dX_dxi[1,1], -dX_dxi[0,1]],
                                     [-dX_dxi[1,0],  dX_dxi[0,0]]])

            B0_tilde = np.transpose(dN_dxi.dot(dxi_dX))
            H = u_e.T.dot(B0_tilde.T)
            F = H + np.eye(2)
            E = 1/2*(H + H.T + H.T.dot(H))
            E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
            S_v = self.C_SE.dot(E_v)
            S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde))*det*self.t
            K_geo = scatter_geometric_matrix(K_geo_small, 2)
            K_mat = B0.T.dot(self.C_SE.dot(B0))*det*self.t
            self.K += w*(K_geo + K_mat)
            self.f += B0.T.dot(S_v)*det*self.t*w

    def _f_int(self, X, u):
        return self.f.copy()

    def _k_int(self, X, u):
        return self.K.copy()

    def _m_int(self, X, u):
        '''
        Mass matrix using CAS-System
        '''
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6, X7, Y7, X8, Y8 = X

        det = ( -X1*Y2 + X1*Y4 + 4*X1*Y5 - 4*X1*Y8 + X2*Y1 - X2*Y3 - 4*X2*Y5
                + 4*X2*Y6 + X3*Y2 - X3*Y4 - 4*X3*Y6 + 4*X3*Y7 - X4*Y1 + X4*Y3
                - 4*X4*Y7 + 4*X4*Y8 - 4*X5*Y1 + 4*X5*Y2 - 4*X6*Y2 + 4*X6*Y3
                - 4*X7*Y3 + 4*X7*Y4 + 4*X8*Y1 - 4*X8*Y4)/24

        self.M = det/45 * self.rho * self.t * np.array([
        [  6.,  0.,  2.,  0.,  3.,  0.,  2.,  0., -6.,  0., -8.,  0., -8.,  0., -6.,  0.],
        [  0.,  6.,  0.,  2.,  0.,  3.,  0.,  2.,  0., -6.,  0., -8.,  0., -8.,  0., -6.],
        [  2.,  0.,  6.,  0.,  2.,  0.,  3.,  0., -6.,  0., -6.,  0., -8.,  0., -8.,  0.],
        [  0.,  2.,  0.,  6.,  0.,  2.,  0.,  3.,  0., -6.,  0., -6.,  0., -8.,  0., -8.],
        [  3.,  0.,  2.,  0.,  6.,  0.,  2.,  0., -8.,  0., -6.,  0., -6.,  0., -8.,  0.],
        [  0.,  3.,  0.,  2.,  0.,  6.,  0.,  2.,  0., -8.,  0., -6.,  0., -6.,  0., -8.],
        [  2.,  0.,  3.,  0.,  2.,  0.,  6.,  0., -8.,  0., -8.,  0., -6.,  0., -6.,  0.],
        [  0.,  2.,  0.,  3.,  0.,  2.,  0.,  6.,  0., -8.,  0., -8.,  0., -6.,  0., -6.],
        [ -6.,  0., -6.,  0., -8.,  0., -8.,  0., 32.,  0., 20.,  0., 16.,  0., 20.,  0.],
        [  0., -6.,  0., -6.,  0., -8.,  0., -8.,  0., 32.,  0., 20.,  0., 16.,  0., 20.],
        [ -8.,  0., -6.,  0., -6.,  0., -8.,  0., 20.,  0., 32.,  0., 20.,  0., 16.,  0.],
        [  0., -8.,  0., -6.,  0., -6.,  0., -8.,  0., 20.,  0., 32.,  0., 20.,  0., 16.],
        [ -8.,  0., -8.,  0., -6.,  0., -6.,  0., 16.,  0., 20.,  0., 32.,  0., 20.,  0.],
        [  0., -8.,  0., -8.,  0., -6.,  0., -6.,  0., 16.,  0., 20.,  0., 32.,  0., 20.],
        [ -6.,  0., -8.,  0., -8.,  0., -6.,  0., 20.,  0., 16.,  0., 20.,  0., 32.,  0.],
        [  0., -6.,  0., -8.,  0., -8.,  0., -6.,  0., 20.,  0., 16.,  0., 20.,  0., 32.]])
        return self.M.copy()


#
#class Tetra4(Element):
#    pass
#
#class Tetra10(Element):
#    pass
#

class Quad4_FG(Element):

    '''
    Element Klasse fuer ebenes, viereckiges Element (Quad4)
    Verschiebungen in x- und y-Richtungen.
    '''
    plane_stress = True

    def __init__(self, E_modul=1.0, poisson_ratio=0., element_thickness=1.0,
                 density=1.0, plane_stress = True):
        '''
        Definition der Materialgrößen und Dicke, da es sich um 2D-Elemente handelt
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.t = element_thickness
        self.rho = density
        self.plane_stress = plane_stress

        # Ebene Spannung
        if self.plane_stress:
            self.C = E_modul/(1-poisson_ratio**2)*np.array(
                             [[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio)/2]])
        else: # Ebene Dehnung
            print('Ebene Dehnung noch nicht implementiert')

    def _compute_tensors(self, X, u):
        pass

    def _k_and_m_int(self, X, u):

        def gauss_quadrature(option):
            '''
            Gauss quadrature for Q4 elements
            option 'complete' (2x2)
            option 'reduced'  (1x1)
            locations: Gauss point locations
            weights: Gauss point weights
            '''
            def complete():
                locations = np.array(
                    [[-0.577350269189626, -0.577350269189626],
                    [0.577350269189626, -0.577350269189626],
                    [0.577350269189626,  0.577350269189626],
                    [-0.577350269189626,  0.577350269189626]])
                weights = np.array([1,1,1,1])
                return locations, weights
            def reduced():
                locations = np.array([0, 0])
                weights = np.array(4)
                return locations, weights
            integration = {'reduced': reduced,
                           'complete': complete}
            locations, weights = integration[option]()
            return weights, locations

        def f_shape_Q4(xi, eta):
            '''
            shape function and derivatives for Q4 elements
            shape : Shape functions
            d_shape: derivatives w.r.t. xi and eta
            xi, eta: natural coordinates (-1 ... +1)
            '''
            shape = 1/4*np.array([(1-xi)*(1-eta),     # N1
                                  (1+xi)*(1-eta),     # N2
                                  (1+xi)*(1+eta),     # N3
                                  (1-xi)*(1+eta)])    # N4
            d_shape=1/4*np.array([[-(1-eta), -(1-xi)],      # dN1/dxi, dN1/deta
                                  [1-eta, -(1+xi)],         # dN2/dxi, dN2/deta
                                  [1+eta, 1+xi],            # dN3/dxi, dN3/deta
                                  [-(1+eta), 1-xi]])        # dN4/dxi, dN4/deta
            return shape, d_shape

        def jacobi(X,d_shape):
            '''
            jac: Jacobian matrix
            invjac: inverse of Jacobian Matrix
            d_shape_XY: derivatives w.r.t. x and y
            d_shape: derivatives w.r.t. xi and eta
            X: nodal coordinates at element level
            '''
            jac = X.T.dot(d_shape)
            invjac = np.linalg.inv(jac)
            d_shape_XY = d_shape.dot(invjac)
            return jac, d_shape_XY


        self.k_el = np.zeros((8, 8))
        self.m_el = np.zeros((8, 8))
        gauss_weights, gauss_loc = gauss_quadrature('complete')
        no_gp = len(gauss_weights)

        # Loop over Gauss points
        for i_gp in range(no_gp):
            # Get Gauss locations
            xi, eta = gauss_loc[i_gp,:]
            # Get shape functions and derivatives with respect to xi, eta
            shape, d_shape = f_shape_Q4(xi,eta)
            # Get Jacobi and derivatives with respect to x,y
            jac, d_shape_XY = jacobi(X.reshape(4,2),d_shape)

            # Build B-matrix
            B = np.zeros((3, 8))
            B[0, [0, 2, 4, 6]] = d_shape_XY[:,0]
            B[1, [1, 3, 5, 7]] = d_shape_XY[:,1]
            B[2, [0, 2, 4, 6]] = d_shape_XY[:,1]
            B[2, [1, 3, 5, 7]] = d_shape_XY[:,0]

            # Build N-matrix
            N = np.zeros((2, 8))
            N[0, [0, 2, 4, 6]]  = shape
            N[1, [1, 3, 5, 7]]  = shape

            # Add stiffness part from Gauss point
            self.k_el = self.k_el + (self.t*B.T.dot(self.C.dot(B))*
                                    gauss_weights[i_gp]*np.linalg.det(jac))
            self.m_el = self.m_el + (self.t*self.rho*N.T.dot(N)*
                                    gauss_weights[i_gp]*np.linalg.det(jac))

        # Make symmetric (because of round-off errors)
        self.k_el = 1/2*(self.k_el+self.k_el.T)
        self.m_el = 1/2*(self.m_el+self.m_el.T)
        return self.k_el, self.m_el

    def _k_int(self, X, u):
        k_el, m_el = self._k_and_m_int(X, u)
        return k_el

    def _m_int(self, X, u):
        k_el, m_el = self._k_and_m_int(X, u)
        return m_el



    def _f_int(self, X, u):
        print('The function is not implemented yet...')
        pass









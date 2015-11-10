# -*- coding: utf-8 -*-
"""
Element Module in which the Finite Elements are described on Element level.

This Module is arbitrarily extensible. The idea is to use the basis class Element which provides the functionality for an efficient solution of a time integration by only once calling the internal tensor computation and then extracting the tangential stiffness matrix and the internal force vector in one run.

Some remarks resulting in the observations of the profiler:
Most of the time is spent with pyhton-functions, when they are used. For instance the kron-function in order to build the scattered geometric stiffness matrix or the trace function are very inefficient. They can be done better when using direct functions.

"""


import numpy as np

fortran_use = False

try:
    import amfe.f90_element
    fortran_use = True
except:
    print('''
Python was not able to load the fast fortran element routines.
run the script 

f2py/install_fortran_routines.sh 

in order to get the full speed! 
''')

fortran_use = False


def scatter_matrix(Mat, ndim):
    '''
    Scatter the symmetric (geometric stiffness) matrix to all dofs.

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
            [F12*b[2,i] + F13*b[1,i], F22*b[2,i] + F23*b[1,i], F32*b[2,i] + F33*b[1,i]],
            [F13*b[0,i] + F11*b[2,i], F23*b[0,i] + F21*b[2,i], F33*b[0,i] + F31*b[2,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i], F31*b[1,i] + F32*b[0,i]]]
    return B


if fortran_use:
    compute_B_matrix = amfe.f90_element.compute_b_matrix
    scatter_matrix = amfe.f90_element.scatter_matrix


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
        return self.K, self.f

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
        return self.K

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
        return self.f

    def m_and_vec_int(self, X, u):
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
        return self._m_int(X, u), np.zeros_like(X)

    def m_int(self, X, u):
        return self._m_int(X, u)



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

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4, **kwargs):
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
        u_mat = u.reshape((-1,2))
        det = (X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2)
        A0       = 0.5*det
        B0_tilde = 1/det*np.array([[Y2-Y3, X3-X2], [Y3-Y1, X1-X3], [Y1-Y2, X2-X1]]).T
        H        = u_mat.T.dot(B0_tilde.T)
        F = H + np.eye(2)
        E = 1/2*(H + H.T + H.T.dot(H))
        E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
        S_v = self.C_SE.dot(E_v)
        S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        B0 = compute_B_matrix(B0_tilde, F)
        K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde))*det/2*self.t
        K_geo = scatter_matrix(K_geo_small, 2)
        K_mat = B0.T.dot(self.C_SE.dot(B0))*det/2*self.t
        self.K = (K_geo + K_mat)
        self.f = B0.T.dot(S_v)*det/2*self.t


    def _m_int(self, X, u):
        '''
        Compute the mass matrix.

        Parameters
        ----------

        X : ndarray
            Position of the nodal coordinates in undeformed configuration
            using voigt notation X = (X1, Y1, X2, Y2, X3, Y3)
        u : ndarray
            Displacement of the element using same voigt notation as for X

        Returns
        -------

        M : ndarray
            Mass matrix of the given element
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

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4, **kwargs):
        '''
        Definition der Materialgrößen und Dicke, da es sich um 2D-Elemente handelt
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.lame_mu = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.t = element_thickness
        self.rho = density
        self.M_small = np.zeros((6,6))

        self.gauss_points2 = ((1/6, 1/6, 2/3, 1/3),
                             (1/6, 2/3, 1/6, 1/3),
                             (2/3, 1/6, 1/6, 1/3))

        self.gauss_points3 = ((1/3, 1/3, 1/3, -27/48),
                             (0.6, 0.2, 0.2, 25/48),
                             (0.2, 0.6, 0.2, 25/48),
                             (0.2, 0.2, 0.6, 25/48))

        alpha1 = 0.0597158717
        beta1 = 0.4701420641 # 1/(np.sqrt(15)-6)
        w1 = 0.1323941527

        alpha2 = 0.7974269853 #
        beta2 = 0.1012865073 # 1/(np.sqrt(15)+6)
        w2 = 0.1259391805

        self.gauss_points5 = ((1/3, 1/3, 1/3, 0.225),
              (alpha1, beta1, beta1, w1), (beta1, alpha1, beta1, w1), (beta1, beta1, alpha1, w1),
              (alpha2, beta2, beta2, w2), (beta2, alpha2, beta2, w2), (beta2, beta2, alpha2, w2))



        # Achtung: hier gibt's ebene Dehnung
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])



    def _compute_tensors(self, X, u):
        '''
        Tensor computation the same way as in the Tri3 element
        '''
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        u_mat = u.reshape((-1,2))
        self.K = np.zeros((12, 12))
        self.f = np.zeros(12)

        for L1, L2, L3, w in self.gauss_points2:

            dN_dL = np.array([  [4*L1 - 1,        0,        0],
                                [       0, 4*L2 - 1,        0],
                                [       0,        0, 4*L3 - 1],
                                [    4*L2,     4*L1,        0],
                                [       0,     4*L3,     4*L2],
                                [    4*L3,        0,     4*L1]])

            # the entries in the jacobian dX_dL
            Jx1 = 4*L2*X4 + 4*L3*X6 + X1*(4*L1 - 1)
            Jx2 = 4*L1*X4 + 4*L3*X5 + X2*(4*L2 - 1)
            Jx3 = 4*L1*X6 + 4*L2*X5 + X3*(4*L3 - 1)
            Jy1 = 4*L2*Y4 + 4*L3*Y6 + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y4 + 4*L3*Y5 + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y6 + 4*L2*Y5 + Y3*(4*L3 - 1)

            det = Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2


            dL_dX = 1/det*np.array([[ Jy2 - Jy3, -Jx2 + Jx3],
                                    [-Jy1 + Jy3,  Jx1 - Jx3],
                                    [ Jy1 - Jy2, -Jx1 + Jx2]])

            B0_tilde = np.transpose(dN_dL.dot(dL_dX))

            H = u_mat.T.dot(B0_tilde.T)
            F = H + np.eye(2)
            E = 1/2*(H + H.T + H.T.dot(H))
            E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
            S_v = self.C_SE.dot(E_v)
            S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde))*det/2*self.t
            K_geo = scatter_matrix(K_geo_small, 2)
            K_mat = B0.T.dot(self.C_SE.dot(B0))*det/2*self.t
            self.K += (K_geo + K_mat)*w
            self.f += B0.T.dot(S_v)*det/2*self.t*w
        pass

    def _f_int(self, X, u):
        return self.f

    def _k_int(self, X, u):
        return self.K

    def _m_int(self, X, u):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X

        self.M_small *= 0
        for L1, L2, L3, w in self.gauss_points2:

            # the entries in the jacobian dX_dL
            Jx1 = 4*L2*X4 + 4*L3*X6 + X1*(4*L1 - 1)
            Jx2 = 4*L1*X4 + 4*L3*X5 + X2*(4*L2 - 1)
            Jx3 = 4*L1*X6 + 4*L2*X5 + X3*(4*L3 - 1)
            Jy1 = 4*L2*Y4 + 4*L3*Y6 + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y4 + 4*L3*Y5 + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y6 + 4*L2*Y5 + Y3*(4*L3 - 1)

            det = Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2

            N = np.array([  [L1*(2*L1 - 1)],
                            [L2*(2*L2 - 1)],
                            [L3*(2*L3 - 1)],
                            [      4*L1*L2],
                            [      4*L2*L3],
                            [      4*L1*L3]])

            self.M_small += N.dot(N.T) * det/2 * self.rho * self.t * w

        self.M = scatter_matrix(self.M_small, 2)
        return self.M


class Quad4(Element):
    '''
    Elementklasse für viereckiges ebenes Element mit linearen Ansatzfunktionen.
    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4, **kwargs):
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
        self.M_small = np.zeros((4,4))
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
            K_geo = scatter_matrix(K_geo_small, 2)
            K_mat = B0.T.dot(self.C_SE.dot(B0))*det*self.t
            self.K += (K_geo + K_mat)*w
            self.f += B0.T.dot(S_v)*det*self.t*w

    def _f_int(self, X, u):
        return self.f

    def _k_int(self, X, u):
        return self.K

    def _m_int(self, X, u):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X
        self.M_small *= 0

        for xi, eta, w in self.gauss_points:
            det = 1/8*(-X1*Y2*eta + X1*Y2 + X1*Y3*eta - X1*Y3*xi + X1*Y4*xi
                        - X1*Y4 + X2*Y1*eta - X2*Y1 + X2*Y3*xi + X2*Y3
                        - X2*Y4*eta - X2*Y4*xi - X3*Y1*eta + X3*Y1*xi
                        - X3*Y2*xi - X3*Y2 + X3*Y4*eta + X3*Y4 - X4*Y1*xi
                        + X4*Y1 + X4*Y2*eta + X4*Y2*xi - X4*Y3*eta - X4*Y3)
            N = np.array([  [(-eta + 1)*(-xi + 1)/4],
                            [ (-eta + 1)*(xi + 1)/4],
                            [  (eta + 1)*(xi + 1)/4],
                            [ (eta + 1)*(-xi + 1)/4]])
            self.M_small += N.dot(N.T) * det * self.rho * self.t * w

        self.M = scatter_matrix(self.M_small, 2)
        return self.M


class Quad8(Element):
    '''
    Plane Quadrangle with quadratic shape functions and 8 nodes. 4 nodes are
    at every corner, 4 nodes on every face.
    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4, **kwargs):
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
        self.M_small = np.zeros((8,8))
        self.M = np.zeros((16,16))

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
                 
        g2 = 0.577350269189626
        w2 = 1.
        self.gauss_points = ((-g2, -g2, w2), (-g2, g2, w2),
                             ( g2, -g2, w2), ( g2, g2, w2))

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
            K_geo = scatter_matrix(K_geo_small, 2)
            K_mat = B0.T.dot(self.C_SE.dot(B0))*det*self.t
            self.K += w*(K_geo + K_mat)
            self.f += B0.T.dot(S_v)*det*self.t*w

    def _f_int(self, X, u):
        return self.f

    def _k_int(self, X, u):
        return self.K

    def _m_int(self, X, u):
        '''
        Mass matrix using CAS-System
        '''
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6, X7, Y7, X8, Y8 = X
        X_mat = X.reshape(-1, 2)
        self.M_small *= 0

        for xi, eta, w in self.gauss_points:
            N = np.array([  [(-eta + 1)*(-xi + 1)*(-eta - xi - 1)/4],
                            [ (-eta + 1)*(xi + 1)*(-eta + xi - 1)/4],
                            [   (eta + 1)*(xi + 1)*(eta + xi - 1)/4],
                            [  (eta + 1)*(-xi + 1)*(eta - xi - 1)/4],
                            [             (-eta + 1)*(-xi**2 + 1)/2],
                            [              (-eta**2 + 1)*(xi + 1)/2],
                            [              (eta + 1)*(-xi**2 + 1)/2],
                            [             (-eta**2 + 1)*(-xi + 1)/2]])

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
            self.M_small += N.dot(N.T) * det * self.rho * self.t * w

        self.M = scatter_matrix(self.M_small, 2)
        return self.M


class Tet4(Element):
    '''
    Tetraeder-Element with 4 nodes
    '''
    def __init__(self,  E_modul=210E9, poisson_ratio=0.3, density=1E4, **kwargs):

        self.poisson_ratio = poisson_ratio
        self.e_modul       = E_modul
        self.lame_mu       = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda   = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.rho           = density

        self.C_SE = np.array([
                [self.lame_lambda + 2*self.lame_mu, self.lame_lambda, self.lame_lambda, 0, 0, 0],
                [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0, 0, 0],
                [self.lame_lambda , self.lame_lambda, self.lame_lambda + 2*self.lame_mu, 0, 0, 0],
                [0, 0, 0, self.lame_mu, 0, 0],
                [0, 0, 0, 0, self.lame_mu, 0],
                [0, 0, 0, 0, 0, self.lame_mu] ])

        self.K = np.zeros((12,12))
        self.f = np.zeros(12)

    def _compute_tensors(self, X, u):
        X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4 = X
        u_e = u.reshape(-1, 3)
        # not sure yet if the determinant is correct when doing the integration
        det = -X1*Y2*Z3 + X1*Y2*Z4 + X1*Y3*Z2 - X1*Y3*Z4 - X1*Y4*Z2 + X1*Y4*Z3 \
             + X2*Y1*Z3 - X2*Y1*Z4 - X2*Y3*Z1 + X2*Y3*Z4 + X2*Y4*Z1 - X2*Y4*Z3 \
             - X3*Y1*Z2 + X3*Y1*Z4 + X3*Y2*Z1 - X3*Y2*Z4 - X3*Y4*Z1 + X3*Y4*Z2 \
             + X4*Y1*Z2 - X4*Y1*Z3 - X4*Y2*Z1 + X4*Y2*Z3 + X4*Y3*Z1 - X4*Y3*Z2

        B0_tilde = 1/det*np.array([
                [-Y2*Z3 + Y2*Z4 + Y3*Z2 - Y3*Z4 - Y4*Z2 + Y4*Z3,
                  X2*Z3 - X2*Z4 - X3*Z2 + X3*Z4 + X4*Z2 - X4*Z3,
                 -X2*Y3 + X2*Y4 + X3*Y2 - X3*Y4 - X4*Y2 + X4*Y3],
                [ Y1*Z3 - Y1*Z4 - Y3*Z1 + Y3*Z4 + Y4*Z1 - Y4*Z3,
                 -X1*Z3 + X1*Z4 + X3*Z1 - X3*Z4 - X4*Z1 + X4*Z3,
                  X1*Y3 - X1*Y4 - X3*Y1 + X3*Y4 + X4*Y1 - X4*Y3],
                [-Y1*Z2 + Y1*Z4 + Y2*Z1 - Y2*Z4 - Y4*Z1 + Y4*Z2,
                  X1*Z2 - X1*Z4 - X2*Z1 + X2*Z4 + X4*Z1 - X4*Z2,
                 -X1*Y2 + X1*Y4 + X2*Y1 - X2*Y4 - X4*Y1 + X4*Y2],
                [ Y1*Z2 - Y1*Z3 - Y2*Z1 + Y2*Z3 + Y3*Z1 - Y3*Z2,
                 -X1*Z2 + X1*Z3 + X2*Z1 - X2*Z3 - X3*Z1 + X3*Z2,
                  X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2]]).T

        H = u_e.T.dot(B0_tilde.T)
        F = H + np.eye(3)
        E = 1/2*(H + H.T + H.T.dot(H))
        E_v = np.array([  E[0,0],   E[1,1],   E[2,2],
                        2*E[1,2], 2*E[0,2], 2*E[0,1]])

        S_v = self.C_SE.dot(E_v)
        S = np.array([[S_v[0], S_v[5], S_v[4]],
                      [S_v[5], S_v[1], S_v[3]],
                      [S_v[4], S_v[3], S_v[2]]])

        B0 = compute_B_matrix(B0_tilde, F)
        K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde))*det/6
        K_geo = scatter_matrix(K_geo_small, 3)
        K_mat = B0.T.dot(self.C_SE.dot(B0))*det/6
        self.K = K_geo + K_mat
        self.f = B0.T.dot(S_v)*det/6

    def _f_int(self, X, u):
        return self.f

    def _k_int(self, X, u):
        return self.K

    def _m_int(self, X, u):
        '''
        Mass matrix using CAS-System
        '''
        X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4 = X
        det =   X1*Y2*Z3 - X1*Y2*Z4 - X1*Y3*Z2 + X1*Y3*Z4 + X1*Y4*Z2 - X1*Y4*Z3 \
              - X2*Y1*Z3 + X2*Y1*Z4 + X2*Y3*Z1 - X2*Y3*Z4 - X2*Y4*Z1 + X2*Y4*Z3 \
              + X3*Y1*Z2 - X3*Y1*Z4 - X3*Y2*Z1 + X3*Y2*Z4 + X3*Y4*Z1 - X3*Y4*Z2 \
              - X4*Y1*Z2 + X4*Y1*Z3 + X4*Y2*Z1 - X4*Y2*Z3 - X4*Y3*Z1 + X4*Y3*Z2
        det *= -1 # same thing as above - it's not clear yet how the node numbering is done.
        self.V = det/6
        self.M = self.V / 20 * self.rho * np.array([
            [ 2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.]])
        return self.M



class Tet10(Element):
    '''
    Tet10 solid element 
    
    '''
    
    def __init__(self,  E_modul=210E9, poisson_ratio=0.3, density=1E4, **kwargs):

        self.poisson_ratio = poisson_ratio
        self.e_modul       = E_modul
        self.lame_mu       = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda   = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.rho           = density

        self.M = np.zeros((30,30))
        self.K = np.zeros((30,30))
        self.f = np.zeros(30)

        self.C_SE = np.array([
                [self.lame_lambda + 2*self.lame_mu, self.lame_lambda, self.lame_lambda, 0, 0, 0],
                [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0, 0, 0],
                [self.lame_lambda , self.lame_lambda, self.lame_lambda + 2*self.lame_mu, 0, 0, 0],
                [0, 0, 0, self.lame_mu, 0, 0],
                [0, 0, 0, 0, self.lame_mu, 0],
                [0, 0, 0, 0, 0, self.lame_mu] ])

        self.gauss_points = ((1/4, 1/4, 1/4, 1/4, 1), )
        
        w1 = 0.030283678097089*6
        w2 = 0.006026785714286*6
        w3 = 0.011645249086029*6
        w4 = 0.010949141561386*6

        a1 = 1/4
        a2 = 0.
        b2 = 1/3
        a3 = 72/99
        b3 =  9/99
        a4 = 0.066550153573664
        c4 = 0.433449846426336

        self.gauss_points = ((a1, a1, a1, a1, w1),
                              
                             (a2, b2, b2, b2, w2),
                             (b2, a2, b2, b2, w2),
                             (b2, b2, a2, b2, w2),
                             (b2, b2, b2, a2, w2),

                             (a3, b3, b3, b3, w3),
                             (b3, a3, b3, b3, w3),
                             (b3, b3, a3, b3, w3),
                             (b3, b3, b3, a3, w3),

                             (a4, a4, c4, c4, w4),
                             (a4, c4, a4, c4, w4),
                             (a4, c4, c4, a4, w4),
                             (c4, c4, a4, a4, w4),
                             (c4, a4, c4, a4, w4),
                             (c4, a4, a4, c4, w4))

    def _compute_tensors(self, X, u):
        X1, Y1, Z1, \
        X2, Y2, Z2, \
        X3, Y3, Z3, \
        X4, Y4, Z4, \
        X5, Y5, Z5, \
        X6, Y6, Z6, \
        X7, Y7, Z7, \
        X8, Y8, Z8, \
        X9, Y9, Z9, \
        X10, Y10, Z10 = X
        
        u_mat = u.reshape((10,3))
        self.K *= 0
        self.f *= 0
        
        for L1, L2, L3, L4, w in self.gauss_points:
            
            Jx1 = 4*L2*X5 + 4*L3*X7 + 4*L4*X8  + X1*(4*L1 - 1)
            Jx2 = 4*L1*X5 + 4*L3*X6 + 4*L4*X9  + X2*(4*L2 - 1)
            Jx3 = 4*L1*X7 + 4*L2*X6 + 4*L4*X10 + X3*(4*L3 - 1)
            Jx4 = 4*L1*X8 + 4*L2*X9 + 4*L3*X10 + X4*(4*L4 - 1)
            Jy1 = 4*L2*Y5 + 4*L3*Y7 + 4*L4*Y8  + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y5 + 4*L3*Y6 + 4*L4*Y9  + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y7 + 4*L2*Y6 + 4*L4*Y10 + Y3*(4*L3 - 1)
            Jy4 = 4*L1*Y8 + 4*L2*Y9 + 4*L3*Y10 + Y4*(4*L4 - 1)
            Jz1 = 4*L2*Z5 + 4*L3*Z7 + 4*L4*Z8  + Z1*(4*L1 - 1)
            Jz2 = 4*L1*Z5 + 4*L3*Z6 + 4*L4*Z9  + Z2*(4*L2 - 1)
            Jz3 = 4*L1*Z7 + 4*L2*Z6 + 4*L4*Z10 + Z3*(4*L3 - 1)
            Jz4 = 4*L1*Z8 + 4*L2*Z9 + 4*L3*Z10 + Z4*(4*L4 - 1)
            
            det = -Jx1*Jy2*Jz3 + Jx1*Jy2*Jz4 + Jx1*Jy3*Jz2 - Jx1*Jy3*Jz4 \
                 - Jx1*Jy4*Jz2 + Jx1*Jy4*Jz3 + Jx2*Jy1*Jz3 - Jx2*Jy1*Jz4 \
                 - Jx2*Jy3*Jz1 + Jx2*Jy3*Jz4 + Jx2*Jy4*Jz1 - Jx2*Jy4*Jz3 \
                 - Jx3*Jy1*Jz2 + Jx3*Jy1*Jz4 + Jx3*Jy2*Jz1 - Jx3*Jy2*Jz4 \
                 - Jx3*Jy4*Jz1 + Jx3*Jy4*Jz2 + Jx4*Jy1*Jz2 - Jx4*Jy1*Jz3 \
                 - Jx4*Jy2*Jz1 + Jx4*Jy2*Jz3 + Jx4*Jy3*Jz1 - Jx4*Jy3*Jz2
            
            a1 = -Jy2*Jz3 + Jy2*Jz4 + Jy3*Jz2 - Jy3*Jz4 - Jy4*Jz2 + Jy4*Jz3
            a2 =  Jy1*Jz3 - Jy1*Jz4 - Jy3*Jz1 + Jy3*Jz4 + Jy4*Jz1 - Jy4*Jz3
            a3 = -Jy1*Jz2 + Jy1*Jz4 + Jy2*Jz1 - Jy2*Jz4 - Jy4*Jz1 + Jy4*Jz2
            a4 =  Jy1*Jz2 - Jy1*Jz3 - Jy2*Jz1 + Jy2*Jz3 + Jy3*Jz1 - Jy3*Jz2
            b1 =  Jx2*Jz3 - Jx2*Jz4 - Jx3*Jz2 + Jx3*Jz4 + Jx4*Jz2 - Jx4*Jz3
            b2 = -Jx1*Jz3 + Jx1*Jz4 + Jx3*Jz1 - Jx3*Jz4 - Jx4*Jz1 + Jx4*Jz3
            b3 =  Jx1*Jz2 - Jx1*Jz4 - Jx2*Jz1 + Jx2*Jz4 + Jx4*Jz1 - Jx4*Jz2
            b4 = -Jx1*Jz2 + Jx1*Jz3 + Jx2*Jz1 - Jx2*Jz3 - Jx3*Jz1 + Jx3*Jz2
            c1 = -Jx2*Jy3 + Jx2*Jy4 + Jx3*Jy2 - Jx3*Jy4 - Jx4*Jy2 + Jx4*Jy3
            c2 =  Jx1*Jy3 - Jx1*Jy4 - Jx3*Jy1 + Jx3*Jy4 + Jx4*Jy1 - Jx4*Jy3
            c3 = -Jx1*Jy2 + Jx1*Jy4 + Jx2*Jy1 - Jx2*Jy4 - Jx4*Jy1 + Jx4*Jy2
            c4 =  Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2
            
            B0_tilde = 1/det*np.array([
                [    a1*(4*L1 - 1),     b1*(4*L1 - 1),     c1*(4*L1 - 1)],
                [    a2*(4*L2 - 1),     b2*(4*L2 - 1),     c2*(4*L2 - 1)],
                [    a3*(4*L3 - 1),     b3*(4*L3 - 1),     c3*(4*L3 - 1)],
                [    a4*(4*L4 - 1),     b4*(4*L4 - 1),     c4*(4*L4 - 1)],
                [4*L1*a2 + 4*L2*a1, 4*L1*b2 + 4*L2*b1, 4*L1*c2 + 4*L2*c1],
                [4*L2*a3 + 4*L3*a2, 4*L2*b3 + 4*L3*b2, 4*L2*c3 + 4*L3*c2],
                [4*L1*a3 + 4*L3*a1, 4*L1*b3 + 4*L3*b1, 4*L1*c3 + 4*L3*c1],
                [4*L1*a4 + 4*L4*a1, 4*L1*b4 + 4*L4*b1, 4*L1*c4 + 4*L4*c1],
                [4*L2*a4 + 4*L4*a2, 4*L2*b4 + 4*L4*b2, 4*L2*c4 + 4*L4*c2],
                [4*L3*a4 + 4*L4*a3, 4*L3*b4 + 4*L4*b3, 4*L3*c4 + 4*L4*c3]]).T
            
            H = u_mat.T.dot(B0_tilde.T)
            F = H + np.eye(3)
            E = 1/2*(H + H.T + H.T.dot(H))
            E_v = np.array([  E[0,0],   E[1,1],   E[2,2],
                            2*E[1,2], 2*E[0,2], 2*E[0,1]])
    
            S_v = self.C_SE.dot(E_v)
            S = np.array([[S_v[0], S_v[5], S_v[4]],
                          [S_v[5], S_v[1], S_v[3]],
                          [S_v[4], S_v[3], S_v[2]]])
    
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde)) * det/6
            K_geo = scatter_matrix(K_geo_small, 3)
            K_mat = B0.T.dot(self.C_SE.dot(B0)) * det/6
            
            self.K += (K_geo + K_mat) * w
            self.f += B0.T.dot(S_v)*det/6 * w

            
        pass
    
    def _f_int(self, X, u):
        return self.f

    def _k_int(self, X, u):
        return self.K

    def _m_int(self, X, u):
        '''
        Mass matrix using CAS-System
        '''
        X1, Y1, Z1, \
        X2, Y2, Z2, \
        X3, Y3, Z3, \
        X4, Y4, Z4, \
        X5, Y5, Z5, \
        X6, Y6, Z6, \
        X7, Y7, Z7, \
        X8, Y8, Z8, \
        X9, Y9, Z9, \
        X10, Y10, Z10 = X
        
        self.M *= 0
        
        for L1, L2, L3, L4, w in self.gauss_points:
            
            Jx1 = 4*L2*X5 + 4*L3*X7 + 4*L4*X8  + X1*(4*L1 - 1)
            Jx2 = 4*L1*X5 + 4*L3*X6 + 4*L4*X9  + X2*(4*L2 - 1)
            Jx3 = 4*L1*X7 + 4*L2*X6 + 4*L4*X10 + X3*(4*L3 - 1)
            Jx4 = 4*L1*X8 + 4*L2*X9 + 4*L3*X10 + X4*(4*L4 - 1)
            Jy1 = 4*L2*Y5 + 4*L3*Y7 + 4*L4*Y8  + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y5 + 4*L3*Y6 + 4*L4*Y9  + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y7 + 4*L2*Y6 + 4*L4*Y10 + Y3*(4*L3 - 1)
            Jy4 = 4*L1*Y8 + 4*L2*Y9 + 4*L3*Y10 + Y4*(4*L4 - 1)
            Jz1 = 4*L2*Z5 + 4*L3*Z7 + 4*L4*Z8  + Z1*(4*L1 - 1)
            Jz2 = 4*L1*Z5 + 4*L3*Z6 + 4*L4*Z9  + Z2*(4*L2 - 1)
            Jz3 = 4*L1*Z7 + 4*L2*Z6 + 4*L4*Z10 + Z3*(4*L3 - 1)
            Jz4 = 4*L1*Z8 + 4*L2*Z9 + 4*L3*Z10 + Z4*(4*L4 - 1)
            
            det = -Jx1*Jy2*Jz3 + Jx1*Jy2*Jz4 + Jx1*Jy3*Jz2 - Jx1*Jy3*Jz4 \
                 - Jx1*Jy4*Jz2 + Jx1*Jy4*Jz3 + Jx2*Jy1*Jz3 - Jx2*Jy1*Jz4 \
                 - Jx2*Jy3*Jz1 + Jx2*Jy3*Jz4 + Jx2*Jy4*Jz1 - Jx2*Jy4*Jz3 \
                 - Jx3*Jy1*Jz2 + Jx3*Jy1*Jz4 + Jx3*Jy2*Jz1 - Jx3*Jy2*Jz4 \
                 - Jx3*Jy4*Jz1 + Jx3*Jy4*Jz2 + Jx4*Jy1*Jz2 - Jx4*Jy1*Jz3 \
                 - Jx4*Jy2*Jz1 + Jx4*Jy2*Jz3 + Jx4*Jy3*Jz1 - Jx4*Jy3*Jz2

            N = np.array([  [L1*(2*L1 - 1)],
                            [L2*(2*L2 - 1)],
                            [L3*(2*L3 - 1)],
                            [L4*(2*L4 - 1)],
                            [      4*L1*L2],
                            [      4*L2*L3],
                            [      4*L1*L3],
                            [      4*L1*L4],
                            [      4*L2*L4],
                            [      4*L3*L4]])
            
            M_small = N.dot(N.T) * det/6 * self.rho * w
            self.M += scatter_matrix(M_small, 3)
        return self.M

class Quad4_FG(Element):

    '''
    Element Klasse fuer ebenes, viereckiges Element (Quad4)
    Verschiebungen in x- und y-Richtungen.
    '''

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
        self._k_and_m_int(X, u)
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
        self.K = self.k_el
        self.M = self.m_el
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


if fortran_use:
    def compute_tri3_tensors(self, X, u):
        self.K, self.f = amfe.f90_element.tri3_k_and_f(X, u, self.C_SE, self.t)

    def compute_tri6_tensors(self, X, u):
        self.K, self.f = amfe.f90_element.tri6_k_and_f(X, u, self.C_SE, self.t)

    def compute_tet4_tensors(self, X, u):
        self.K, self.f = amfe.f90_element.tet4_k_and_f(X, u, self.C_SE)


    def compute_tri6_mass(self, X, u):
        self.M = amfe.f90_element.tri6_m(X, self.rho, self.t)
        return self.M

    # overloading the routines with fortran routines
    Tri3._compute_tensors = compute_tri3_tensors
    Tri6._compute_tensors = compute_tri6_tensors
    Tet4._compute_tensors = compute_tet4_tensors
    Tri6._m_int = compute_tri6_mass







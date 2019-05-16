#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class for all elements.
"""

__all__ = [
    'Element'
]

import numpy as np

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Element:
    """
    Anonymous baseclass for all elements. It contains the methods needed
    for the computation of the element stuff.

    Attributes
    ----------
    material : instance of amfe.HyperelasticMaterial
        Class containing the material behavior.
    name : str
        Name for the postprocessing tool to identify the characteristics of the
        element
    """
    name = None

    def __init__(self, material=None):
        """
        Parameters
        ----------
        material : amfe.HyperelasticMaterial - object
            Object handling the material
        """
        self.material = material
        self.K = None
        self.f = None
        self.S = None
        self.E = None

    @staticmethod
    def fields():
        """
        Returns unique physical fields for elements

        Returns
        -------
        fields: tuple[str]
            unique physical fields in element
        """
        return ()

    def dofs(self):
        """
        Method that returns a tuple that contains local dof information for the element

        ((<type>, <nr>, <physics>), (<type>, <nr>, <physics>), ... )

        with: <type> =  'N': nodal dof
                        'E': elemental dof

              <nr>   =  number of nodal or elemental dof, respectively (starts at zero)
              <physics> = string that describe the physic of the dof (e.g. 'ux', 'uy', 'T', 'visc', ...)


        Examples
        --------
        2D three node element for displacements returns:
            (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 1, 'ux'), ('N', 1, 'ux'), ('N', 2, 'uy'))

        2D bar element for displacements, including one viscoelastic information returns:
            (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 1, 'ux'), ('N', 1, 'uy'), ('E', 0, 'visc'))

        Returns
        -------
        dofs : tuple
        """
        return ()

    def _compute_tensors(self, X, u, t):
        '''
        Virtual function for the element specific implementation of a tensor
        computation routine which will be called before _k_int and _f_int
        will be called. For many computations the tensors need to be computed
        the same way.
        '''
        pass

    def _m_int(self, X, u, t=0):
        '''
        Virtual function for the element specific implementation of the mass
        matrix;
        '''
        pass

    def k_and_f_int(self, X, u, t=0):
        '''
        Returns the tangential stiffness matrix and the internal nodal force
        of the Element.

        Parameters
        ----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array
            of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation
        t : float
            time

        Returns
        -------
        k_int : ndarray
            The tangential stiffness matrix (ndarray of dimension (ndim, ndim))
        f_int : ndarray
            The nodal force vector (ndarray of dimension (ndim,))

        Examples
        --------
        TODO

        '''
        self._compute_tensors(X, u, t)
        return self.K, self.f

    def k_int(self, X, u, t=0):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        ----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array of
            type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation
        t : float
            time

        Returns
        -------
        k_int : ndarray
            The tangential stiffness matrix (numpy.ndarray of type ndim x ndim)

        '''
        self._compute_tensors(X, u, t)
        return self.K

    def f_int(self, X, u, t=0):
        '''
        Returns the internal element restoring force f_int

        Parameters
        ----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array of
            type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation
        t : float, optional
            time, default value: 0.

        Returns
        -------
        f_int : ndarray
            The nodal force vector (numpy.ndarray of dimension (ndim,))

        '''
        self._compute_tensors(X, u, t)
        return self.f

    def m_and_vec_int(self, X, u, t=0):
        '''
        Returns mass matrix of the Element and zero vector of size X.


        Parameters
        ----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array of
            type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation
        t : float, optional
            time, default value: 0.

        Returns
        -------
        m_int : ndarray
            The consistent mass matrix of the element (numpy.ndarray of
            dimension (ndim,ndim))
        vec : ndarray
            vector (containing zeros) of dimension (ndim,)

        '''
        return self._m_int(X, u, t), np.zeros_like(X)

    def m_int(self, X, u, t=0):
        '''
        Returns the mass matrix of the element.

        Parameters
        ----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array of
            type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation
        t : float, optional
            time, default value: 0.

        Returns
        -------
        m_int : ndarray
            The consistent mass matrix of the element (numpy.ndarray of
            dimension (ndim,ndim))

        '''
        return self._m_int(X, u, t)

    def k_f_S_E_int(self, X, u, t=0):
        '''
        Returns the tangential stiffness matrix, the internal nodal force,
        the strain and the stress tensor (voigt-notation) of the Element.

        Parameters
        ----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array
            of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation
        t : float
            time

        Returns
        -------
        K : ndarray
            The tangential stiffness matrix (ndarray of dimension (ndim, ndim))
        f : ndarray
            The nodal force vector (ndarray of dimension (ndim,))
        S : ndarray
            The stress tensor (ndarray of dimension (no_of_nodes, 6))
        E : ndarray
            The strain tensor (ndarray of dimension (no_of_nodes, 6))

        Examples
        --------
        TODO

        '''
        self._compute_tensors(X, u, t)
        return self.K, self.f, self.S, self.E

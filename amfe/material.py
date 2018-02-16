# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module for material handling withing the FE context.

Up to now, only Hyperelastic materials are implemented. Hyperelastic Materials
are Materials, where the constitutive law can be expressed such, that the
second Piola-Kirchhoff stress tensor S is a function of the Green-Lagrange
strain tensor E. This computation is carried out in this module.

"""

import numpy as np
import abc


__all__ = ['Material',
           'HyperelasticMaterial',
           'KirchhoffMaterial',
           'LinearMaterial',
           'NeoHookean',
           'MooneyRivlin',
           ]


use_fortran = False

try:
    import amfe.f90_material as f90_material
    use_fortran = True
except Exception:
    print('''
Python was not able to load the fast fortran material routines.
''')

# use_fortran = False


class Material:
    def __init__(self):
        self._observers = list()

    def add_observer(self, observer, verbose=True):
        self._observers.append(observer)
        if verbose:
            print('Added observer to material')

    def remove_observer(self, observer, verbose=True):
        self._observers.remove(observer)
        if verbose:
            print('Removed observer from material')

    def notify(self):
        for observer in self._observers:
            observer.update()


class HyperelasticMaterial(Material):
    '''
    Base class for hyperelastic material.
    '''
    def __init__(self):
        super().__init__()

    def S_Sv_and_C(self, E):
        '''
        Compute 2nd Piola Kirchhoff stress tensor in matrix form and voigt
        notation as well as material tangent modulus.

        Parameters
        ----------
        E : ndarray
            Green-Lagrange strain tensor, shape: (3,3)

        Returns
        -------
        S : ndarray
            2nd Piola Kirchhoff stress tensor in matrix representation,
            shape: (3,3)
        Sv : ndarray
            2nd Piola Kirchhoff stress tensor in voigt notation, shape: (6,)
        C_SE : ndarray
            tangent moduli between Green-Lagrange strain tensor and 2nd Piola
            Kirchhoff stress tensor, shape (6,6)

        '''
        pass

    def S_Sv_and_C_2d(self, E):
        '''
        Compute 2nd Piola Kirchhoff stress tensor in matrix form and voigt
        notation as well as material tangent modulus for 2D-Problems.

        Parameters
        ----------
        E : ndarray
            Green-Lagrange strain tensor, shape: (2,2)

        Returns
        -------
        S : ndarray
            2nd Piola Kirchhoff stress tensor in matrix representation,
            shape: (2,2)
        Sv : ndarray
            2nd Piola Kirchhoff stress tensor in voigt notation, shape: (3,)
        C_SE : ndarray
            tangent moduli between Green-Lagrange strain tensor and 2nd Piola
            Kirchhoff stress tensor, shape (3,3)

        Notes
        -----
        The result is dependent on the the option plane stress or plane strain.
        Take care to choose the right option!
        '''
        pass


class KirchhoffMaterial(HyperelasticMaterial):
    r'''
    Kirchhoff-Material that mimicks the linear elastic behavior.

    The strain energy potential is

    .. math::
        W(E) = \frac{\lambda}{2} \mathrm{tr}\,(\mathbf{E})^2 + \
        \mu \cdot \mathrm{tr}\,(\mathbf{E}^2)

    with:
        :math:`W` = strain energy potential

        :math:`\lambda` = first Lamé constant:
        :math:`\lambda = \frac{\nu E}{(1+\nu)(1-2\nu)}`

        :math:`\mu` = second Lamé constant: :math:`\mu = \frac{E}{2(1+\nu)}`

        :math:`\mathbf{E}` = Green-Lagrange strain tensor

    '''
    def __init__(self, E=210E9, nu=0.3, rho=1E4, plane_stress=True, thickness=1.):
        '''

        Parameters
        ----------
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        rho : float
            Density of the material.
        plane_stress : bool, optional
            flat if plane stress or plane strain is chosen, if a 2D-problem is
            considered
        thickness : float
            Thickness of the material, if 2D-prolbem is considered

        Returns
        -------
        None
        '''
        super().__init__()
        self._E_modulus = E
        self._nu = nu
        self._rho = rho
        self._plane_stress = plane_stress
        self._thickness = thickness
        self._update_variables()

    def __repr__(self):
        '''
        repr(obj) function for smart representing for debugging
        '''
        return 'amfe.material.KirchhoffMaterial(%f,%f,%f,%s,%f)'\
            % (self.E_modulus, self.nu, self.rho,
               str(self.plane_stress), self.thickness)

    def _update_variables(self):
        '''
        Update internal variables...
        '''
        E = self.E_modulus
        nu = self.nu
        # The two lame constants:
        lam = nu*E / ((1 + nu) * (1 - 2*nu))
        mu = E / (2*(1 + nu))
        self.C_SE = np.array([[lam + 2*mu, lam, lam, 0, 0, 0],
                              [lam, lam + 2*mu, lam, 0, 0, 0],
                              [lam, lam, lam + 2*mu, 0, 0, 0],
                              [0, 0, 0, mu, 0, 0],
                              [0, 0, 0, 0, mu, 0],
                              [0, 0, 0, 0, 0, mu]])

        if self.plane_stress:
            self.C_SE_2d = E/(1-nu**2)*np.array([[1, nu, 0],
                                                 [nu, 1, 0],
                                                 [0, 0, (1-nu)/2]])
        else:
            self.C_SE_2d = np.array([[lam + 2*mu, lam, 0],
                                     [lam, lam + 2*mu, 0],
                                     [0, 0, mu]])

    @property
    def E_modulus(self):
        return self._E_modulus

    @E_modulus.setter
    def E_modulus(self, E):
        self._E_modulus = E
        self._update_variables()
        self.notify()
    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, nu):
        self._nu = nu
        self._update_variables()
        self.notify()

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self._update_variables()
        self.notify()

    @property
    def plane_stress(self):
        return self._plane_stress

    @plane_stress.setter
    def plane_stress(self, plane_stress):
        self._plane_stress = plane_stress
        self._update_variables()
        self.notify()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self._update_variables()
        self.notify()

    def S_Sv_and_C(self, E):
        '''
        '''
        E_v = np.array([  E[0,0],   E[1,1],   E[2,2],
                        2*E[1,2], 2*E[0,2], 2*E[0,1]])
        S_v = self.C_SE.dot(E_v)
        S = np.array([[S_v[0], S_v[5], S_v[4]],
                      [S_v[5], S_v[1], S_v[3]],
                      [S_v[4], S_v[3], S_v[2]]])
        return S, S_v, self.C_SE

    def S_Sv_and_C_2d(self, E):
        '''
        '''
        E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
        S_v = self.C_SE_2d.dot(E_v)
        S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        return S, S_v, self.C_SE_2d

# For simplicity: rename KirchhoffMaterial
LinearMaterial = KirchhoffMaterial


#%%

class NeoHookean(HyperelasticMaterial):
    r'''
    Neo-Hookean hyperelastic material. It is the same material as the Mooney-
    Rivlin material with constant A01=0.

    The Neo-Hookean material has the strain energy potential:

    .. math::
        W(J_1, J_3) = \frac{\mu}{2}(J_1-3) + \frac{\kappa}{2}(J_3 - 1)^2

    with:
        :math:`W` = strain energy potential

        :math:`J_1` = first deviatoric strain invariant

        :math:`J_3` = third deviatoric strain invariant (determinant of elastic
        deformation gradient :math:`\mathbf{F}`)

        :math:`\mu` = initial shear modulus of the material

        :math:`\kappa` = bulk modulus (material incompressibility parameter)

    '''
    def __init__(self, mu, kappa, rho, plane_stress=False, thickness=1.):
        '''

        '''
        super().__init__()
        self._mu = mu
        self._kappa = kappa
        self._rho = rho
        self._thickness = thickness
        self._plane_stress = plane_stress
        if plane_stress:
            raise ValueError('Attention! plane stress is not supported yet \
within the NeoHookean material!')

    def __repr__(self):
        '''
        repr(obj) function for smart representing for debugging
        '''
        return 'amfe.material.NeoHookeanMaterial(%f,%f,%f,%s,%f)'\
            % (self.mu, self.kappa, self.rho,
               str(self.plane_stress), self.thickness)

    @property
    def mu(self):
        return self._mu

    @property
    def kappa(self):
        return self._kappa

    @property
    def rho(self):
        return self._rho

    @property
    def thickness(self):
        return self._thickness

    @property
    def plane_stress(self):
        return self._plane_stress

    @mu.setter
    def mu(self, mu):
        self._mu = mu
        self.notify()

    @kappa.setter
    def kappa(self, kappa):
        self._kappa = kappa
        self.notify()

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self.notify()

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self.notify()

    @plane_stress.setter
    def plane_stress(self, plane_stress):
        if plane_stress:
            raise ValueError('Plane_stress not supportet yet for NeoHookean material')
        else:
            self._plane_stress = plane_stress
            self.notify()


    def S_Sv_and_C(self, E):
        ''' '''

        mu = self.mu
        kappa = self.kappa
        C = 2*E + np.eye(3)
        C11 = C[0,0]
        C22 = C[1,1]
        C33 = C[2,2]
        C23 = C[1,2]
        C13 = C[0,2]
        C12 = C[0,1]
        # invariants and reduced invariants
        I1  = C11 + C22 + C33
        I3  = C11*C22*C33 - C11*C23**2 - C12**2*C33 + 2*C12*C13*C23 - C13**2*C22

        J3  = np.sqrt(I3)
        # derivatives
        J1I1 = I3**(-1/3)
        J1I3 = -I1/(3*I3**(4/3))
        J3I3 = 1/(2*np.sqrt(I3))

        I1E = 2*np.array([1, 1, 1, 0, 0, 0])
        I3E = 2*np.array([C22*C33 - C23**2,
                          C11*C33 - C13**2,
                          C11*C22 - C12**2,
                          -C11*C23 + C12*C13,
                          C12*C23 - C13*C22,
                          -C12*C33 + C13*C23])

        J1E = J1I1*I1E + J1I3*I3E
        J3E = J3I3*I3E
        # stresses
        S_v = mu/2*J1E + kappa*(J3 - 1)*J3E
        S = np.array([[S_v[0], S_v[5], S_v[4],],
                      [S_v[5], S_v[1], S_v[3],],
                      [S_v[4], S_v[3], S_v[2],]])

        I3EE = np.array([   [     0,  4*C33,  4*C22, -4*C23,      0,      0],
                            [ 4*C33,      0,  4*C11,      0, -4*C13,      0],
                            [ 4*C22,  4*C11,      0,      0,      0, -4*C12],
                            [-4*C23,      0,      0, -2*C11,  2*C12,  2*C13],
                            [     0, -4*C13,      0,  2*C12, -2*C22,  2*C23],
                            [     0,      0, -4*C12,  2*C13,  2*C23, -2*C33]])

        # second derivatives
        J1I1I3 = -1/(3*I3**(4/3))
        J1I3I3 = 4*I1/(9*I3**(7/3))
        J3I3I3 = -1/(4*I3**(3/2))

        J1EE = J1I1I3*(np.outer(I1E, I3E) + np.outer(I3E, I1E)) \
                 + J1I3I3*np.outer(I3E, I3E) + J1I3*I3EE
        J3EE = J3I3I3*(np.outer(I3E, I3E)) + J3I3*I3EE


        C_SE = mu/2*J1EE + kappa*(np.outer(J3E, J3E)) + kappa*(J3-1)*J3EE
        return S, S_v, C_SE


    def S_Sv_and_C_2d(self, E):
        '''
        '''

        mu = self.mu
        kappa = self.kappa
        C = 2*E + np.eye(2)
        C11 = C[0,0]
        C22 = C[1,1]
        C12 = C[0,1]
        C33 = 1
        # invatiants and reduced invariants
        I1  = C11 + C22 + C33
        I3  = C11*C22 - C12**2
        J3  = np.sqrt(I3)

        # derivatives
        J1I1 = I3**(-1/3)
        J1I3 = -I1/(3*I3**(4/3))
        J3I3 = 1/(2*np.sqrt(I3))

        I1E = 2*np.array([1, 1, 0])
        I3E = 2*np.array([C22*C33, C11*C33, -C12*C33 ])

        J1E = J1I1*I1E + J1I3*I3E
        J3E = J3I3*I3E
        # stresses
        S_v = mu/2*J1E + kappa*(J3 - 1)*J3E
        S = np.array([[S_v[0], S_v[2]],
                      [S_v[2], S_v[1]]])

        I3EE = np.array([   [ 0,  4, 0],
                            [ 4,  0, 0],
                            [ 0,  0,-2]])

        # second derivatives
        J1I1I3 = -1/(3*I3**(4/3))
        J1I3I3 = 4*I1/(9*I3**(7/3))
        J3I3I3 = -1/(4*I3**(3/2))

        J1EE = J1I1I3*(np.outer(I1E, I3E) + np.outer(I3E, I1E)) \
                 + J1I3I3*np.outer(I3E, I3E) + J1I3*I3EE
        J3EE = J3I3I3*(np.outer(I3E, I3E)) + J3I3*I3EE

        C_SE = mu/2*J1EE + kappa*(np.outer(J3E, J3E)) + kappa*(J3-1)*J3EE
        return S, S_v, C_SE


class MooneyRivlin(HyperelasticMaterial):
    r'''
    Mooney-Rivlin hyperelastic material

    The Mooney-Rivlin material has the strain energy potential:

    .. math::
        W(J_1, J_2, J_3) = A_{10}(J_1-3) + A_{01}(J_2 - 3) + \frac{\kappa}{2}(J_3 - 1)^2

    with:
        :math:`W` = strain energy potential

        :math:`J_1` = first deviatoric strain invariant

        :math:`J_2` = second deviatoric strain invariant

        :math:`J_3` = third deviatoric strain invariant (determinant of elastic
        deformation gradient :math:`\mathbf{F}`)

        :math:`A_{10}, A_{01}` = material constants characterizing the
        deviatoric deformation of the material

        :math:`\kappa` = bulk modulus (material incompressibility parameter)

    '''
    def __init__(self, A10, A01, kappa, rho, plane_stress=False, thickness=1.):
        '''
        Parameters
        ----------
        A10 : float
            first material constant for deviatoric deformation of material.
        A01 : float
            second material constant for deviatoric deformation of material.
        kappa : float
            bulk modulus of material.
        rho : float
            density of the material.
        plane_stress : bool, optional
            flag for plane stress or plane strain, preset = False
        thickness : float
            Thickness of the material, if 2D-prolbem is considered.

        Returns
        -------
        None

        '''
        super().__init__()
        self._A10 = A10
        self._A01 = A01
        self._kappa = kappa
        self._rho = rho
        self._thickness = thickness
        if plane_stress:
            raise ValueError('Attention! plane stress is not supported yet \
            within the MooneyRivlin material!')
        else:
            self._plane_stress = plane_stress

    @property
    def A10(self):
        return self._A10

    @property
    def A01(self):
        return self._A01

    @property
    def kappa(self):
        return self._kappa

    @property
    def rho(self):
        return self._rho

    @property
    def thickness(self):
        return self._thickness

    @property
    def plane_stress(self):
        return self._plane_stress

    @A10.setter
    def A10(self, A10):
        self._A10 = A10
        self.notify()

    @A01.setter
    def A01(self, A01):
        self._A01 = A01
        self.notify()

    @kappa.setter
    def kappa(self, kappa):
        self._kappa = kappa
        self.notify()

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self.notify()

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self.notify()

    @plane_stress.setter
    def plane_stress(self, plane_stress):
        if plane_stress:
            raise ValueError('Plane stress is not supported for Mooney Rivlin')
        else:
            self._plane_stress = plane_stress
            self.notify()


    def __repr__(self):
        '''
        repr(obj) function for smart representing for debugging
        '''
        return 'amfe.material.MooneyRivlin(%f,%f,%f,%f,%s,%f)'\
            % (self.A10, self.A01, self.kappa, self.rho,
               str(self.plane_stress), self.thickness)

    def S_Sv_and_C(self, E):
        '''
        '''
        A10 = self.A10
        A01 = self.A01
        kappa = self.kappa
        C = 2*E + np.eye(3)
        C11 = C[0,0]
        C22 = C[1,1]
        C33 = C[2,2]
        C23 = C[1,2]
        C13 = C[0,2]
        C12 = C[0,1]
        # invariants and reduced invariants
        I1  = C11 + C22 + C33
        I2  = C11*C22 + C11*C33 - C12**2 - C13**2 + C22*C33 - C23**2
        I3  = C11*C22*C33 - C11*C23**2 - C12**2*C33 + 2*C12*C13*C23 - C13**2*C22
        # J1  = I1*I3**(-1/3)
        # J2  = I2*I3**(-2/3)
        J3  = np.sqrt(I3)
        # derivatives
        J1I1 = I3**(-1/3)
        J1I3 = -I1/(3*I3**(4/3))
        J2I2 = I3**(-2/3)
        J2I3 = -2*I2/(3*I3**(5/3))
        J3I3 = 1/(2*np.sqrt(I3))

        I1E = 2*np.array([1, 1, 1, 0, 0, 0])
        I2E = 2*np.array([C22 + C33, C11 + C33, C11 + C22, -C23, -C13, -C12])
        I3E = 2*np.array([C22*C33 - C23**2,
                          C11*C33 - C13**2,
                          C11*C22 - C12**2,
                          -C11*C23 + C12*C13,
                          C12*C23 - C13*C22,
                          -C12*C33 + C13*C23])

        J1E = J1I1*I1E + J1I3*I3E
        J2E = J2I2*I2E + J2I3*I3E
        J3E = J3I3*I3E
        # stresses
        S_v = A10*J1E + A01*J2E + kappa*(J3 - 1)*J3E
        S = np.array([[S_v[0], S_v[5], S_v[4],],
                      [S_v[5], S_v[1], S_v[3],],
                      [S_v[4], S_v[3], S_v[2],]])

        I2EE = np.array([   [0, 4, 4,  0,  0,  0],
                            [4, 0, 4,  0,  0,  0],
                            [4, 4, 0,  0,  0,  0],
                            [0, 0, 0, -2,  0,  0],
                            [0, 0, 0,  0, -2,  0],
                            [0, 0, 0,  0,  0, -2]])

        I3EE = np.array([   [     0,  4*C33,  4*C22, -4*C23,      0,      0],
                            [ 4*C33,      0,  4*C11,      0, -4*C13,      0],
                            [ 4*C22,  4*C11,      0,      0,      0, -4*C12],
                            [-4*C23,      0,      0, -2*C11,  2*C12,  2*C13],
                            [     0, -4*C13,      0,  2*C12, -2*C22,  2*C23],
                            [     0,      0, -4*C12,  2*C13,  2*C23, -2*C33]])

        # second derivatives
        J1I1I3 = -1/(3*I3**(4/3))
        J1I3I3 = 4*I1/(9*I3**(7/3))
        J2I2I3 = -2/(3*I3**(5/3))
        J2I3I3 = 10*I2/(9*I3**(8/3))
        J3I3I3 = -1/(4*I3**(3/2))

        J1EE = J1I1I3*(np.outer(I1E, I3E) + np.outer(I3E, I1E)) \
                 + J1I3I3*np.outer(I3E, I3E) + J1I3*I3EE
        J2EE = J2I2I3*(np.outer(I2E, I3E) + np.outer(I3E, I2E)) \
                 + J2I3I3*np.outer(I3E, I3E) + J2I2*I2EE + J2I3*I3EE
        J3EE = J3I3I3*(np.outer(I3E, I3E)) + J3I3*I3EE

        # alternative formulation, using JEs
#        J3J3J3 = -1/np.sqrt(I3)
#        J2J3J3 = 8*I2/(9*I3**(5/3))
#        J2J2J3 = - 4/(3*np.sqrt(I3))
#        J1J3J3 = 8*I1/(9*I3**(4/3))
#        J1J1J3 = - 2/(3*np.sqrt(I3))
#
#        J1EE = J1J1J3*(np.outer(J1E, J3E) + np.outer(J3E, J1E)) \
#                     + J1J3J3*np.outer(J3E, J3E) + J1I3*I3EE
#        J2EE = J2J2J3*(np.outer(J2E, J3E) + np.outer(J3E, J2E)) \
#                      + J2J3J3*np.outer(J3E, J3E) + J2I2*I2EE + J2I3*I3EE
#        J3EE = J3J3J3*(np.outer(J3E,J3E)) + J3I3*I3EE

        C_SE = A10*J1EE + A01*J2EE + kappa*(np.outer(J3E, J3E)) + kappa*(J3-1)*J3EE
        return S, S_v, C_SE

    def S_Sv_and_C_2d(self, E):
        '''
        '''

        A10 = self.A10
        A01 = self.A01
        kappa = self.kappa
        C = 2*E + np.eye(2)
        C11 = C[0,0]
        C22 = C[1,1]
        C12 = C[0,1]
        C33 = 1
        # invatiants and reduced invariants
        I1  = C11 + C22 + C33
        I2  = C11*C22 + C11*C33 - C12**2 + C22*C33
        I3  = C11*C22 - C12**2
        # J1  = I1*I3**(-1/3)
        # J2  = I2*I3**(-2/3)
        J3  = np.sqrt(I3)

        # derivatives
        J1I1 = I3**(-1/3)
        J1I3 = -I1/(3*I3**(4/3))
        J2I2 = I3**(-2/3)
        J2I3 = -2*I2/(3*I3**(5/3))
        J3I3 = 1/(2*np.sqrt(I3))

        I1E = 2*np.array([1, 1, 0])
        I2E = 2*np.array([C22 + C33, C11 + C33, -C12])
        I3E = 2*np.array([C22*C33, C11*C33, -C12*C33 ])

        J1E = J1I1*I1E + J1I3*I3E
        J2E = J2I2*I2E + J2I3*I3E
        J3E = J3I3*I3E
        # stresses
        S_v = A10*J1E + A01*J2E + kappa*(J3 - 1)*J3E
        S = np.array([[S_v[0], S_v[2]],
                      [S_v[2], S_v[1]]])

        I2EE = np.array([   [0, 4, 0],
                            [4, 0, 0],
                            [0, 0,-2]])

        I3EE = np.array([   [ 0,  4, 0],
                            [ 4,  0, 0],
                            [ 0,  0,-2]])

        # second derivatives
        J1I1I3 = -1/(3*I3**(4/3))
        J1I3I3 = 4*I1/(9*I3**(7/3))
        J2I2I3 = -2/(3*I3**(5/3))
        J2I3I3 = 10*I2/(9*I3**(8/3))
        J3I3I3 = -1/(4*I3**(3/2))

        J1EE = J1I1I3*(np.outer(I1E, I3E) + np.outer(I3E, I1E)) \
                 + J1I3I3*np.outer(I3E, I3E) + J1I3*I3EE
        J2EE = J2I2I3*(np.outer(I2E, I3E) + np.outer(I3E, I2E)) \
                 + J2I3I3*np.outer(I3E, I3E) + J2I2*I2EE + J2I3*I3EE
        J3EE = J3I3I3*(np.outer(I3E, I3E)) + J3I3*I3EE

        C_SE = A10*J1EE + A01*J2EE + kappa*(np.outer(J3E, J3E)) + kappa*(J3-1)*J3EE
        return S, S_v, C_SE


# overloading of the python functions in case FORTRAN should be used
if use_fortran:
    def kirchhoff_S_Sv_and_C(self, E):
        return f90_material.kirchhoff_s_sv_and_c(E, self.E_modulus, self.nu)

    def kirchhoff_S_Sv_and_C_2d(self, E):
        return f90_material.kirchhoff_s_sv_and_c_2d(E, self.E_modulus, self.nu,
                                                    self.plane_stress)

    def mooney_rivlin_S_Sv_and_C(self, E):
        return f90_material.mooney_rivlin_s_sv_and_c(E, self.A10, self.A01, self.kappa)

    def neo_hookean_S_Sv_and_C(self, E):
        return f90_material.neo_hookean_s_sv_and_c(E, self.mu, self.kappa)

    def neo_hookean_S_Sv_and_C_2d(self, E):
        return f90_material.neo_hookean_s_sv_and_c_2d(E, self.mu, self.kappa)

    # overloading the functions
    KirchhoffMaterial.S_Sv_and_C = kirchhoff_S_Sv_and_C
    KirchhoffMaterial.S_Sv_and_C_2d = kirchhoff_S_Sv_and_C_2d
    MooneyRivlin.S_Sv_and_C = mooney_rivlin_S_Sv_and_C
    NeoHookean.S_Sv_and_C = neo_hookean_S_Sv_and_C
    NeoHookean.S_Sv_and_C_2d = neo_hookean_S_Sv_and_C_2d

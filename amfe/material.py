# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:30:03 2015

@author: johannesr
"""

import numpy as np


class HyperelasticMaterial():
    
    def __init__(self):
        pass
    
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
            
        Note
        ----
        The result is dependent on the the option plane stress or plane strain. 
        Take care to choose the right option! 
        '''
        pass

class KirchhoffMaterial(HyperelasticMaterial):
    r'''
    Kirchhoff-Material that mimicks the linear elastic behavior. 
    
    The strain energy potential is
    
    .. math::
        W(E) = \frac{\lambda}{2}trace(\mathbf{E})^2 + \mu*trace(\mathbf{E}^2)
    
    with:
        :math:`W` = strain energy potential
        
        :math:`\lambda` = first Lamé constant: :math:`\lambda = \frac{\nu E}{(1+\nu)(1-2\nu)}`
        
        :math:`\mu` = second Lamé constant: :math:`\mu = \frac{E}{2(1+\nu)}`
        
        :math:`\mathbf{E}` = Green-Lagrange strain tensor
        
    '''
    def __init__(self, E=210E9, nu=0.3, rho=1E4, plane_stress=True):
        '''
        
        Parameters
        ----------
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        rho : flot
            Density
        plane_stress : bool, optional
            flat if plane stress or plane strain is chosen, if a 2D-problem is 
            considered
        
        Returns
        -------
        None
        '''
        self.E_modulus = E
        self.nu = nu
        self.rho = rho
        self.plane_stress = plane_stress
        self._update_variables()
   
    def _update_variables(self):
        '''
        Update internal variables... 
        '''
        E = self.E_modulus
        nu = self.nu
        # The two lame constants:
        lam = nu*E / ((1 + nu) * (1 - 2*nu))
        mu  = E / (2*(1 + nu))
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

    
    def S_Sv_and_C(self, E):
        # copy docstring
#        self.S_Sv_and_C.__doc__ = HyperelasticMaterial.S_Sv_and_C.__doc__

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
        # copy docstring
#        self.S_Sv_and_C_2d.__doc__ = HyperelasticMaterial.S_Sv_and_C_2d.__doc__

        E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
        S_v = self.C_SE_2d.dot(E_v)
        S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        return S, S_v, self.C_SE_2d


#%%
# @inherit_docs(HyperelasticMaterial)
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
    def __init__(self, mu, kappa, plane_stress=False):
        self.mu = mu
        self.kappa = kappa
        self.plane_stress = plane_stress
        if plane_stress:
            raise ValueError('Attention! plane stress is not supported yet \
within the MooneyRivlin material!')


    def S_Sv_and_C(self, E):
        ''' '''
        # copy docstring
#        self.S_Sv_and_C.__doc__ = HyperelasticMaterial.S_Sv_and_C.__doc__

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
        # copy docstring
#        self.S_Sv_and_C_2d.__doc__ = HyperelasticMaterial.S_Sv_and_C_2d.__doc__
        
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
        
        :math:`J_3` = third deviatoric strain invariant (determinant of elastic deformation gradient :math:`\mathbf{F}`)
                
        :math:`A_{10}, A_{01}` = material constants characterizing the deviatoric deformation of the material
        
        :math:`\kappa` = bulk modulus (material incompressibility parameter)
            
    '''
    def __init__(self, A10, A01, kappa, plane_stress=False):
        '''
        Parameters
        ----------
        A10 : float
            first material constant for deviatoric deformation of material
        A01 : float
            second material constant for deviatoric deformation of material
        kappa : float
            bulk modulus of material
        plane_stress : bool, optional
            flag for plane stress or plane strain, preset = False
            
        Returns
        -------
        None
        
        '''
        self.A10 = A10
        self.A01 = A01
        self.kappa = kappa
        self.plane_stress = plane_stress
        if plane_stress:
            raise ValueError('Attention! plane stress is not supported yet \
            within the MooneyRivlin material!')
            
    
    def S_Sv_and_C(self, E):
        '''
        '''
        # copy docstring
#        self.S_Sv_and_C.__doc__ = HyperelasticMaterial.S_Sv_and_C.__doc__
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
        # copy docstring
#        self.S_Sv_and_C_2d.__doc__ = HyperelasticMaterial.S_Sv_and_C_2d.__doc__
        
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

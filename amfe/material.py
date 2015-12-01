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
        pass
    
    def S_Sv_and_C_2d(self, E):
        pass


class KirchhoffMaterial(HyperelasticMaterial):
    '''
    
    '''
    def __init__(self, E=210E9, nu=0.3, rho=1E4, plane_stress=True):
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
                                                 [0, 0, (1-nu)/2],])
        else:
            self.C_SE_2d = np.array([[lam + 2*mu, lam, 0],
                                     [lam, lam + 2*mu, 0],
                                     [0, 0, mu]])

    
    def S_Sv_and_C(self, E):
        '''
        Compute 2nd Piola Kirchhoff stress and tangential stress-strain relationship. 
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
        Compute 2nd Piola Kirchhoff stress and tangential stress-strain relationship for 2D-Problems. 
        '''
        E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
        S_v = self.C_SE_2d.dot(E_v)
        S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        return S, S_v, self.C_SE_2d


#%%

class NeoHookean(HyperelasticMaterial):
    pass


class MooneyRivlin(HyperelasticMaterial):
    '''
    
    
    '''
    def __init__(self, A10, A01, kappa, plane_stress=True):
        self.A10 = A10
        self.A01 = A01
        self.kappa = kappa
        self.plane_stress = plane_stress
    
    def S_Sv_and_C(self, E):
        '''
        Compute S, S in Voigt notation and material stiffness 
        '''
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
        J1  = I1*I3**(-1/3)
        J2  = I2*I3**(-2/3)
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
        S_v = self.A10*J1E + self.A01*J2E + self.kappa*(J3 - 1)*J3E
        S = np.array([[S_v[0], S_v[5], S_v[4],],
                      [S_v[5], S_v[1], S_v[3],],
                      [S_v[4], S_v[3], S_v[2],]])
        
        # second derivatives
        J1I1I3 = -1/(3*I3**(4/3))
        J1I3I3 = 4*I1/(9*I3**(7/3))
        J2I2I3 = -2/(3*I3**(5/3))
        J2I3I3 = 10*I2/(9*I3**(8/3))
        J3I3I3 = -1/(4*I3**(3/2))

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

        J1EE = J1I1I3*(I1E.T.dot(I3E) + I3E.T.dot(I1E)) + J1I3I3*I3E.T.dot(I3E) + J1I3*I3EE        
        J2EE = J2I2I3*(I2E.T.dot(I3E) + I3E.T.dot(I2E)) + J2I3I3*I3E.T.dot(I3E) + J2I2*I2EE + J2I3*I3EE
        J3EE = J3I3I3*(I3E.T.dot(I3E)) + J3I3*I3EE
        C_SE = self.A10*J1EE + self.A01*J2EE + self.kappa*(J3E.T.dot(J3E)) + self.kappa*(J3-1)*J3EE
        return S, S_v, C_SE

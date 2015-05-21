# -*- coding: utf-8 -*-

'''
Integratoren-Klasse, mit der dynamische Probleme berechnet werden können
'''

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def norm_of_vector(array):
    return np.sqrt(array.T.dot(array))

class NewmarkIntegrator():
    '''
    Newmark-Integrator-Schema zur Bestimmung der dynamischen Antwort...
    '''
    verbose = False

    def __init__(self, alpha=0):
        self.beta = 1/4*(1 + alpha)**2
        self.gamma = 1/2 + alpha
        self.delta_t = 1E-3
        self.eps = 1E-10
        self.newton_damping = 0.8
        self.residual_threshold = 1E6
        self.dynamical_system = None
        pass

    def set_nonlinear_model(self, f_non, K, M, f_ext=None):
        '''
        Funktion zum Übergeben des nichtlinearen Modells an die Integrationsklasse

        Idee: K, M, f_non kommen direkt aus der Assembly-Klasse und müssen nicht verändert werden;
        '''
        self.f_non = f_non
        self.K = K
        self.M = M
        self.f_ext = f_ext

    def set_dynamical_system(self, dynamical_system):
        '''
        hands over the dynamical system as a whole to the integrator. The matrices for the integration routine are then taken right from the dynamical system

        '''
        self.dynamical_system = dynamical_system
        self.M = dynamical_system.M_global()
        self.K = dynamical_system.K_global
        self.f_non = dynamical_system.f_int_global
        self.f_ext = dynamical_system.f_ext_global


    def residual(self, q, dq, ddq, t):
        if self.f_ext is not None:
            res = self.M.dot(ddq) + self.f_non(q) - self.f_ext(q, dq, t)
        else:
            res = self.M.dot(ddq) + self.f_non(q)
        return res


    def integrate_nonlinear_system(self, q_start, dq_start, time_range):
        '''
        Funktion, die das System nach der generalized-alpha-Methode integriert
        '''
        # Initialisieren der Startvariablen
        q = q_start.copy()
        dq = dq_start.copy()
        ddq = np.zeros(len(q_start))

        q_global = []
        dq_global = []
        t = 0
        time_index = 0
        write_flag = False
        # Abfangen, wenn das Ding mit 0 beginnt:
        if time_range[0] < 1E-12:
            q_global.append(q)
            dq_global.append(dq)
            time_index = 1
        else:
            time_index = 0

        # Korrekte Startbedingungen ddq:
        ddq = linalg.spsolve(self.M, self.f_non(q))
        while time_index < len(time_range):
            if t + self.delta_t >= time_range[time_index]:
                dt = time_range[time_index] - t
                if dt < 1E-8:
                    dt = 1E-7
                write_flag = True
                time_index += 1
            else:
                dt = self.delta_t
            t += dt
            # Prediction
            q += dt*dq + (1/2-self.beta)*dt**2*ddq
            dq += (1-self.gamma)*dt*ddq
            ddq *= 0

            # checking residual and convergence
            f_non = self.f_non(q)
            res = self.residual(q, dq, ddq, t)
            # Newcton-Correction-loop

            n_iter = 0
            while norm_of_vector(res) > self.eps*norm_of_vector(f_non):
                S = self.K(q) + 1/(self.beta*dt**2)*self.M
                delta_q = - linalg.spsolve(S, res)
                if norm_of_vector(res) > self.residual_threshold:
                    delta_q *= self.newton_damping
                q   += delta_q
                dq  += self.gamma/(self.beta*dt)*delta_q
                ddq += 1/(self.beta*dt**2)*delta_q
                f_non = self.f_non(q)
                res = self.residual(q, dq, ddq, t)
                n_iter += 1
                pass

            if self.verbose:
                print('Zeit:', t, 'Anzahl an Iterationen:', n_iter, 'Residuum:', norm_of_vector(res))
            # Writing if necessary:
            if write_flag:
                # writint in the dynamical system, if possible
                if self.dynamical_system:
                    self.dynamical_system.write_timestep(t, q)
                else:
                    q_global.append(q.copy())
                    dq_global.append(dq.copy())
                write_flag = False
                if self.verbose:
                    print('Zeit:', t, 'Anzahl an Iterationen:', n_iter)
                pass

        return np.array(q_global), np.array(dq_global)

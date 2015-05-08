# -*- coding: utf-8 -*-

'''
Integratoren-Klasse, mit der dynamische Probleme berechnet werden können
'''

import numpy as np


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

    def residual(self, q, dq, ddq, t):
        if self.f_ext is not None:
            res = self.M.dot(ddq) + self.f_non(q) - self.f_ext(q, dq, t)
        else:
            res = self.M.dot(ddq) + self.f_non(q)
        return res


    def integrate(self, q_start, dq_start, time_range):
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
        ddq = np.linalg.solve(M, self.f_non(q))
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
                delta_q = - np.linalg.solve(S, res)
                q   += delta_q
                dq  += self.gamma/(self.beta*dt)*delta_q
                ddq += 1/(self.beta*dt**2)*delta_q
                f_non = self.f_non(q)
                res = self.residual(q, dq, ddq, t)
                n_iter += 1
                pass

            # Writing if necessary:
            if write_flag:
                q_global.append(q.copy())
                dq_global.append(dq.copy())
                write_flag = False
                if self.verbose:
                    print('Zeit:', t, 'Anzahl an Iterationen:', n_iter)
                pass

        return np.array(q_global), np.array(dq_global)



#%%

if __name__ == "__main__":
    c1 = 10
    c2 = 20
    c3 = 10
    c4 = 0
    K = np.array([[c1 + c2,-c2,0],
                  [-c2 , c2 + c3, -c3],
                  [0, -c3, c3 + c4]])

    def my_k(q):
        return K


    M = np.diag([3,1,2])

    def f_non(q):
        return K.dot(q) # + np.array([c1*q[0]**3, 0, 0])

    omega = 2*np.pi*1
    amplitude = 5
    def f_ext(q, dq, t):
        return np.array([0, 0., amplitude*np.cos(omega*t)])




    q_start = np.array([1, 0, 2.])*0
    dq_start = q_start*0

    T = np.arange(0,30,0.01)

    my_integrator = NewmarkIntegrator()
    my_integrator.set_nonlinear_model(f_non, my_k, M, f_ext)
    #my_integrator.verbose = True
    q, dq = my_integrator.integrate(q_start, dq_start, T)

    from matplotlib import pyplot
    pyplot.plot(T, q)
    #pyplot.plot(T, dq)


 #%%
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Comparison of AMfe's pinciple angles and SciPy.linalg's subspace angles.
"""


import amfe
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


V1 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [0, 0, 0]])


V2 = np.array([[0, 0],
               [0, 0],
               [1, 0],
               [0, 1]])


angles_amfe, F1, F2 = amfe.principal_angles(V1=V1, V2=V2, unit='deg', method=None, principal_vectors=True)
angles_scipy = np.sort(sp.linalg.subspace_angles(A=V1, B=V2))/np.pi*180


print(F1)
print(F2)


print(angles_amfe)
print(angles_scipy)


fig, ax = plt.subplots()
ax.plot(angles_amfe, 'ro-', label='AMfe\'s principle_angles(...)')
ax.plot(angles_scipy, 'b+-', label='SciPy.linalg\'s subspace_angles(...)')
ax.grid(True)
ax.legend()
plt.xlabel('number - 1')
plt.ylabel('angle (deg)')
plt.show()


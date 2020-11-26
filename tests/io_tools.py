"""
io test-tools
"""

import numpy as np
import pandas as pd
import pickle
import os
from amfe.mesh import Mesh


def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def create_amfe_obj():
    # Define input file path
    meshobj = Mesh(dimension=2)

    nodes = np.array([[1.345600000e-02, 3.561675700e-02],
                      [5.206839561e-01, 3.740820950e-02],
                      [3.851982918e-02, 5.460016703e-01],
                      [5.457667372e-01, 5.477935420e-01],
                      [1.027911912e+00, 3.919966200e-02],
                      [6.358365836e-02, 1.056386584e+00],
                      [1.040469476e+00, 5.445628213e-01],
                      [5.582746582e-01, 1.053154002e+00],
                      [1.052965658e+00, 1.049921420e+00],
                      [1.535139868e+00, 4.099111450e-02],
                      [1.547697432e+00, 5.463542738e-01],
                      [1.547656658e+00, 1.046688838e+00],
                      [2.042367825e+00, 4.278256700e-02],
                      [2.042357741e+00, 5.431194119e-01],
                      [2.042347658e+00, 1.043456257e+00]], dtype=float)

    connectivity = [np.array([13, 15, 9, 14, 12, 11], dtype=int),
                    np.array([9, 6, 5, 8, 4, 7], dtype=int),
                    np.array([9, 5, 13, 7, 10, 11], dtype=int),
                    np.array([1, 5, 6, 2, 4, 3], dtype=int),
                    np.array([5, 13, 10], dtype=int),
                    np.array([1, 5, 2], dtype=int),
                    np.array([6, 1, 3], dtype=int),
                    np.array([9, 6, 8], dtype=int),
                    np.array([13, 15, 14], dtype=int),
                    np.array([15, 9, 12], dtype=int)]

    data = {'shape': ['Tri6', 'Tri6', 'Tri6', 'Tri6', 'quadratic_line',
                      'quadratic_line', 'quadratic_line', 'quadratic_line',
                      'quadratic_line', 'quadratic_line'],
            'connectivity': connectivity,
            'is_boundary': [False, False, False, False, True, True, True, True, True, True],
            'domain': [2, 1, 2, 1, 0, 0, 0, 0, 0, 0],
            'weight': [0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    indices = list(np.arange(1, 11))

    meshobj.el_df = pd.DataFrame(data, index=indices)
    meshobj.el_df['domain'] = meshobj.el_df['domain'].astype(pd.Int64Dtype())
    meshobj.el_df['weight'] = meshobj.el_df['weight'].astype(float)

    meshobj.groups = {'left': {'nodes': [], 'elements': [2, 4]},
                      'right': {'nodes': [], 'elements': [1, 3]},
                      'left_boundary': {'nodes': [], 'elements': [7]},
                      'right_boundary': {'nodes': [], 'elements': [9]},
                      'top_boundary': {'nodes': [], 'elements': [8, 10]},
                      'left_dirichlet': {'nodes': [1, 3, 6], 'elements': []}}

    nodeids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    meshobj.nodes_df = pd.DataFrame({'x': nodes[:, 0], 'y': nodes[:, 1]}, index=nodeids)
    return meshobj


def clean_test_outputs(directory):
    for f in os.listdir(directory):
        if f == 'hdf5_dummy.hdf5' or f == 'hdf5postprocessing.hdf5':
            return
        filename = directory + f
        if os.path.isfile(filename):
            os.remove(filename)
    return

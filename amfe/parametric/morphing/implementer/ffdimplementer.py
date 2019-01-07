#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
import scipy as sp
from scipy.special import binom

from amfe.linalg.tools import coordinate_transform
from amfe.parametric.morphing.implementer import MorpherImplementer


class FfdMorpherImplementer(MorpherImplementer):
    '''
    Implements morphing with FFD technique
    '''
    def __init__(self, origin=np.array([[0],[0],[0]]), csys=np.eye(3), mu_shape=(3,3,3)):
        super().__init__()
        self._origin_box = np.array(origin).reshape((3, 1))
        self._csys = np.array(csys).reshape((3,3))
        # save transformations
        physical_frame = self._csys
        reference_frame = np.eye(3)
        (self._transformation, self._inverse_transformation) = coordinate_transform(reference_frame, physical_frame)
        (dim_n_mu, dim_m_mu, dim_t_mu) = mu_shape
        self._dim_n_mu = dim_n_mu
        self._dim_m_mu = dim_m_mu
        self._dim_t_mu = dim_t_mu
        self._bernstein_x = None
        self._bernstein_y = None
        self._bernstein_z = None
        self._shift_mesh_points = None
        self._no_of_dim = 3
        self._no_of_mesh_points = 0

    @property
    def mu_shape(self):
        return self._dim_n_mu, self._dim_m_mu, self._dim_t_mu

    def offline(self, nodes_reference):
        # apply transformation to original mesh points
        reference_frame_mesh_points = self._transformation(nodes_reference.T - self._origin_box).T

        # TODO: Raise error if not in bounding box

        mesh_points = reference_frame_mesh_points
        (n_rows_mesh, n_cols_mesh) = mesh_points.shape
        self._no_of_mesh_points = n_rows_mesh
        self._no_of_dim = n_cols_mesh

        # Initialization. In order to exploit the contiguity in memory the
        # following are transposed

        self._bernstein_x = np.zeros((self._dim_n_mu, n_rows_mesh))
        self._bernstein_y = np.zeros((self._dim_m_mu, n_rows_mesh))
        self._bernstein_z = np.zeros((self._dim_t_mu, n_rows_mesh))

        for i in range(0, self._dim_n_mu):
            aux1 = np.power((1 - mesh_points[:, 0]), self._dim_n_mu - 1 - i)
            aux2 = np.power(mesh_points[:, 0], i)
            self._bernstein_x[i, :] = binom(self._dim_n_mu - 1, i) * np.multiply(
                aux1, aux2)

        for i in range(0, self._dim_m_mu):
            aux1 = np.power((1 - mesh_points[:, 1]), self._dim_m_mu - 1 - i)
            aux2 = np.power(mesh_points[:, 1], i)
            self._bernstein_y[i, :] = binom(self._dim_m_mu - 1, i) * np.multiply(
                aux1, aux2)

        for i in range(0, self._dim_t_mu):
            aux1 = np.power((1 - mesh_points[:, 2]), self._dim_t_mu - 1 - i)
            aux2 = np.power(mesh_points[:, 2], i)
            self._bernstein_z[i, :] = binom(self._dim_t_mu - 1, i) * np.multiply(
                aux1, aux2)

    def morph(self, nodes_reference, mu_x, mu_y, mu_z):
        shifted_mesh_points = np.zeros((self._no_of_dim, self._no_of_mesh_points))
        aux_x = 0.
        aux_y = 0.
        aux_z = 0.

        for j in range(0, self._dim_m_mu):
            for k in range(0, self._dim_t_mu):
                bernstein_yz = np.multiply(self._bernstein_y[j, :], self._bernstein_z[k, :])
                for i in range(0, self._dim_n_mu):
                    aux = np.multiply(self._bernstein_x[i, :], bernstein_yz)
                    aux_x += aux * mu_x[i, j, k]
                    aux_y += aux * mu_y[i, j, k]
                    aux_z += aux * mu_z[i, j, k]
        shifted_mesh_points[0, :] += aux_x
        shifted_mesh_points[1, :] += aux_y
        shifted_mesh_points[2, :] += aux_z

        # shift_mesh_points needs to be transposed to be summed with mesh_points
        # apply inverse transformation to shifted mesh points
        new_mesh_points = self._inverse_transformation(shifted_mesh_points).T + nodes_reference

        return new_mesh_points


class FfdMorpherImplementer2D(MorpherImplementer):
    '''
    Implements morphing with FFD technique
    '''
    def __init__(self, origin=np.eye(2), csys=np.eye(2), mu_shape=(3,3)):
        super().__init__()
        self._origin_box = np.array(origin).reshape((2, 1))
        self._csys = np.array(csys).reshape((2,2))
        # save transformations
        physical_frame = self._csys
        reference_frame = sp.eye(2)
        (self._transformation, self._inverse_transformation) = coordinate_transform(reference_frame, physical_frame)
        (dim_n_mu, dim_m_mu) = mu_shape
        self._dim_n_mu = dim_n_mu
        self._dim_m_mu = dim_m_mu
        self._bernstein_x = None
        self._bernstein_y = None
        self._shift_mesh_points = None
        self._no_of_dim = 2
        self._no_of_mesh_points = 0

    @property
    def mu_shape(self):
        return self._dim_n_mu, self._dim_m_mu

    def offline(self, nodes_reference):
        # apply transformation to original mesh points
        reference_frame_mesh_points = self._transformation(nodes_reference.T - self._origin_box).T

        # select mesh points inside bounding box
        # not necessary?:
        # TODO: Raise error if not in bounding box
        # mesh_points = reference_frame_mesh_points[
#            (reference_frame_mesh_points[:, 0] >= 0.)
#            & (reference_frame_mesh_points[:, 0] <= 1.) &
#            (reference_frame_mesh_points[:, 1] >= 0.) &
#            (reference_frame_mesh_points[:, 1] <= 1.) &
#            (reference_frame_mesh_points[:, 2] >= 0.) &
#            (reference_frame_mesh_points[:, 2] <= 1.)]
        mesh_points = reference_frame_mesh_points
        (n_rows_mesh, n_cols_mesh) = mesh_points.shape
        self._no_of_mesh_points = n_rows_mesh
        self._no_of_dim = n_cols_mesh

        # Initialization. In order to exploit the contiguity in memory the
        # following are transposed

        self._bernstein_x = np.zeros((self._dim_n_mu, n_rows_mesh))
        self._bernstein_y = np.zeros((self._dim_m_mu, n_rows_mesh))

        for i in range(0, self._dim_n_mu):
            aux1 = np.power((1 - mesh_points[:, 0]), self._dim_n_mu - 1 - i)
            aux2 = np.power(mesh_points[:, 0], i)
            self._bernstein_x[i, :] = binom(self._dim_n_mu - 1, i) * np.multiply(
                aux1, aux2)

        for i in range(0, self._dim_m_mu):
            aux1 = np.power((1 - mesh_points[:, 1]), self._dim_m_mu - 1 - i)
            aux2 = np.power(mesh_points[:, 1], i)
            self._bernstein_y[i, :] = binom(self._dim_m_mu - 1, i) * np.multiply(
                aux1, aux2)

    def morph(self, nodes_reference, mu_x, mu_y):
        shifted_mesh_points = np.zeros((self._no_of_dim, self._no_of_mesh_points))
        aux_x = 0.
        aux_y = 0.

        for j in range(0, self._dim_m_mu):
            for i in range(0, self._dim_n_mu):
                aux = np.multiply(self._bernstein_x[i, :], self._bernstein_y[j,:])
                aux_x += aux * mu_x[i, j]
                aux_y += aux * mu_y[i, j]
        shifted_mesh_points[0, :] += aux_x
        shifted_mesh_points[1, :] += aux_y

        # shift_mesh_points needs to be transposed to be summed with mesh_points
        # apply inverse transformation to shifted mesh points
        new_mesh_points = self._inverse_transformation(shifted_mesh_points).T + nodes_reference

        # NOT NECESSARY:
        # merge non-shifted mesh points with shifted ones
        #modified_mesh_points = np.copy(self.original_mesh_points)

        # The next is commented out and replaced by Meyer:
        #self.modified_mesh_points[(reference_frame_mesh_points[:, 0] >= 0.)
#                                  & (reference_frame_mesh_points[:, 0] <= 1.) &
#                                  (reference_frame_mesh_points[:, 1] >= 0.) &
#                                  (reference_frame_mesh_points[:, 1] <= 1.) &
#                                  (reference_frame_mesh_points[:, 2] >= 0.) &
#                                  (reference_frame_mesh_points[:, 2] <=
#                                   1.)] = new_mesh_points
        return new_mesh_points

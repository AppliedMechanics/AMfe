"""
Constraint Module in which the Constraints are described on Element level.

This Module is arbitrarily extensible. The idea is to use the basis class
Constraint.

"""

__all__ = ['Constraint',
           'DirichletConstraint',
           'FixedDistanceConstraint',
           'FixedDistanceToLineConstraint',
           'NodesCollinear2DConstraint',
           'EqualDisplacementConstraint',
           'FixedDistanceToPlaneConstraint',
           'NodesCoplanarConstraint',
           ]

import numpy as np


class Constraint():
    '''
    Anonymous baseclass for all constraints.
    It contains the methods c_equation and vector_b that have to be
    implemented for each (new) kind of constraint.
    '''
    name = None

    def c_equation(self, X_local, u_local, t=0):
        return np.array([0])

    def vector_b(self, X_local, u_local, t=0):
        return np.array([0])


class DirichletConstraint(Constraint):
    '''
    DirichletConstraint
    
    Class to define a Dirichlet Constraint of one node
    
    This class has to be extended for Dirichlet Constraints with
    u not equal to zero!
    '''
    name = 'Dirichlet'

    def c_equation(self, X_local, u_local, t=0):
        '''
        Returns residual of c_equation for a Dirichlet constraint with u = 0
         
        Parameters
        ----------
        X_local : numpy.array
            X-Coords in reference Domain of the node that shall be constrained
        u_local : numpy.array
            current displacement of the node that shall be constrained
        t : float
            time, default: 0
            
        Returns
        -------
        res : numpy.array
            residual of c_equation
            
        '''
        return u_local

    def vector_b(self, X_local, u_local, t=0):
        '''
        Returns residual of b vector (line of B matrix) for a Dirichlet constraint with u = 0

        Parameters
        ----------
        X_local : numpy.array
            X-Coords in reference Domain of the node that shall be constrained
        u_local : numpy.array
            current displacement of the node that shall be constrained
        t : float
            time, default: 0

        Returns
        -------
        b : numpy.array
            b_vector that can be used to assemble constraint matrix B

        '''
        return np.array([1.0])


class FixedDistanceConstraint(Constraint):
    '''
    Fixed Distance Constraint
    
    Class to define a fixed distance between two nodes
    '''
    name = 'FixedDistance'

    def c_equation(self, X_local, u_local, t=0):
        '''
        Returns residual of c_equation for a fixed distance constraint between two nodes
        
        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for both points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2], [x1 y1 x2 y2] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time
            
        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector
            
        '''

        dofs_per_node = len(u_local)//2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        scaling = np.linalg.norm(X2-X1)

        return 10*(np.sqrt(np.sum(((x2-x1)**2))) - \
                   np.sqrt(np.sum(((X2-X1)**2))))/scaling

    def vector_b(self, X_local, u_local, t=0):
        '''
        Returns the b vector (for assembly in B matrix) for a fixed distance constraint between two nodes

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for both points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2], [x1 y1 x2 y2] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time

        Returns
        -------
        b : numpy.array
            b vector for assembly in B matrix [bx1 by1 bz1 bx2 by2 bz2]

        '''
        dofs_per_node = len(u_local)//2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        scaling = np.linalg.norm(X2-X1)
        l_current = np.sqrt(np.sum(((x2-x1)**2)))


        return 10*np.concatenate((-(x2-x1)/(l_current*scaling),
                               (x2-x1)/(l_current*scaling)))


class FixedDistanceToLineConstraint(Constraint):
    '''
    Fixed distance to line constraint

    Class to define a fixed distance to line constraint
    '''
    name = 'FixedDistanceToLine'

    def c_equation(self, X_local, u_local, t=0):
        """
        Returns residual of c_equation for a fixed distance to line constraint
        
        This function calculates the residuum of the constraints for a Fixed
        Distance To Line Constraint. The idea is that I will have a lot of
        nodes, forming a Line (not necessarily a straight line), and a point x3.
        This point is then constrained to keep a fixed distance from this line,
        based on linear approximations of the line by its nodes.
        
        
        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for 2 points forming a line
            and a third point that shall keep the same distance to this line
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3], [x1 y1 x2 y2 x3 y3] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time
            
        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector
        """

        dofs_per_node = len(u_local)//3

        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:2*dofs_per_node]
        X3 = X_local[-dofs_per_node:]

        x = X_local + u_local
        x1 = x[:dofs_per_node]
        x2 = x[dofs_per_node:2*dofs_per_node]
        x3 = x[-dofs_per_node:]

        # The vector of direction of the line is
        line_dir = x2 - x1
        # x3_dir is the vector from x1 to x3, so that we can find the distance
        x3_dir = x3 - x1

        # The norm of the rejection of x3_dir relative to line_dir gives us the
        # distance from x3 to the small line we have
        # rejection current is the perpendicular line from line to x3
        # therefore the norm of rejection_current is the distance of x3 to the line
        rejection_current = x3_dir - ((x3_dir.dot(line_dir)) \
                                      /(np.linalg.norm(line_dir))**2)*line_dir

        # Calculate the initial rejection vector
        initial_dir = X2 - X1
        X3_dir_initial = X3 - X1

        rejection_initial = X3_dir_initial - \
                             ((X3_dir_initial.dot(initial_dir)) \
                             /(np.linalg.norm(initial_dir))**2)*initial_dir

        # the squared current distance must be equal to the squared initial distance
        return rejection_current.dot(rejection_current) \
               - rejection_initial.dot(rejection_initial)

    def vector_b(self, X_local, u_local, t=0):
        """
        Returns residual of c_equation for a fixed distance to line constraint
        
        This function calculates the residuum of the constraints for a Fixed
        Distance To Line Constraint. The idea is that I will have a lot of
        nodes, forming a Line (not necessarily a straight line), and a point x3.
        This point is then constrained to keep a fixed distance from this line,
        based on linear approximations of the line by its nodes.
        
        
        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for 2 points forming a line
            and a third point that shall keep the same distance to this line
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3], [x1 y1 x2 y2 x3 y3] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time
            
        Returns
        -------
        b : numpy.array
            b vector [bx1 by1 bz1 bx2 by2 bz2 bx3 by3 bz3] or [bx1 by1 bx2 by2 bx3 by3] for 3D and 2D respectively
        
        """

        dofs_per_node = len(u_local)//3

        if dofs_per_node > 2:

            x = X_local + u_local
            x1 = x[:dofs_per_node]
            x2 = x[dofs_per_node:2*dofs_per_node]
            x3 = x[-dofs_per_node:]


            r_12 = x2 - x1
            r_13 = x3 - x1

            a1 = x1[0]
            b1 = x1[1]
            c1 = x1[2]
            a2 = x2[0]
            b2 = x2[1]
            c2 = x2[2]
            # a3 = x3[0]
            # b3 = x3[1]
            # c3 = x3[2]

            # r_13   = [a3-a1, b3-b1, c3-c1]
            # r_12 = [a2-a1, b2-b1, c2-c1]

            # coef is s in documentation
            coef = ((r_13.dot(r_12))/(r_12.dot(r_12)))

            v = r_13 - coef*r_12

            # dcoefdP1 = ( r_12 @ (-I) ) / (r_12.dot(r_12))
            # + ( r_13 @  ( (-I) / (r_12.dot(r_12))
            # + (2*r_12.T @ r_13)/(r_12.dot(r_12)**2))
            # drejdP1 = -I - (dcoefdP1.T @ r_12) + coef * I
            # bP1 = 2 * v @ drejdP1.T

            # -------------------------------------------------------------------
            # Derivatives of coeff with respect to...
            # ... x1
            dcoefda1 = np.array([-1,0,0]).dot(r_12)/(r_12.dot(r_12)) \
                       + r_13.dot(np.array([-1,0,0])/(r_12.dot(r_12)) \
                       +r_12*(2*(a2-a1)/(r_12.dot(r_12)**2)))

            # ... y1
            dcoefdb1 = np.array([0, -1, 0]).dot(r_12) / (r_12.dot(r_12)) \
                       + r_13.dot(np.array([0, -1, 0]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (b2 - b1) / (r_12.dot(r_12) ** 2)))

            # ... z1
            dcoefdc1 = np.array([0, 0, -1]).dot(r_12) / (r_12.dot(r_12)) \
                       + r_13.dot(np.array([0, 0, -1]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (c2 - c1) / (r_12.dot(r_12) ** 2)))

            # ... x2
            dcoefda2 = r_13.dot(np.array([1, 0, 0]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (a2 - a1) / (r_12.dot(r_12) ** 2)))

            # ... y2
            dcoefdb2 = r_13.dot(np.array([0, 1, 0]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (b2 - b1) / (r_12.dot(r_12) ** 2)))

            # ... z2
            dcoefdc2 = r_13.dot(np.array([0, 0, 1]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (c2 - c1) / (r_12.dot(r_12) ** 2)))

            # ... x3
            dcoefda3 = np.array([1, 0, 0]).dot(r_12) / (r_12.dot(r_12))

            # ... y3
            dcoefdb3 = np.array([0, 1, 0]).dot(r_12) / (r_12.dot(r_12))

            # ... z3
            dcoefdc3 = np.array([0, 0, 1]).dot(r_12) / (r_12.dot(r_12))

            # END of derivatives of coeff
            # All formulas checked by Meyer
            # ----------------------------------------------------------------

            # ----------------------------------------------------------------
            # Comment by Meyer: THIS SECTION IS PROBABLY WRONG!
            #
            drejda1 = np.array([-1, 0, 0]) - dcoefda1*r_12 \
                      + np.array([coef, 0, 0])

            drejdb1 = np.array([0, -1, 0]) - dcoefdb1*r_12 \
                      + np.array([0, coef, 0])


            drejdc1 = np.array([0, 0, -1]) - dcoefdc1*r_12 \
                      + np.array([0, 0, coef])

            drejda2 = - dcoefda2*r_12 - np.array([coef, 0, 0])

            drejdb2 = - dcoefdb2*r_12 - np.array([0, coef, 0])

            drejdc2 = - dcoefdc2*r_12 - np.array([0, 0, coef])

            drejda3 = np.array([1,0,0]) - dcoefda3*r_12

            drejdb3 = np.array([0,1,0]) - dcoefdb3*r_12

            drejdc3 = np.array([0,0,1]) - dcoefdc3*r_12

            bx1 = np.array([
                            2*v.dot(drejda1),
                            2*v.dot(drejdb1),
                            2*v.dot(drejdc1)
                            ])
            bx2 = np.array([
                            2*v.dot(drejda2),
                            2*v.dot(drejdb2),
                            2*v.dot(drejdc2)
                            ])
            bx3 = np.array([
                            2*v.dot(drejda3),
                            2*v.dot(drejdb3),
                            2*v.dot(drejdc3)
                            ])

            b = np.concatenate((bx1,bx2,bx3))

        else:

            x = X_local + u_local
            x1 = x[:dofs_per_node]
            x2 = x[dofs_per_node:2*dofs_per_node]
            x3 = x[-dofs_per_node:]

            r_12 = x2 - x1
            r_13 = x3 - x1

            a1 = x1[0]
            b1 = x1[1]
            a2 = x2[0]
            b2 = x2[1]

            # coef is s in documentation
            coef = ((r_13.dot(r_12))/(r_12.dot(r_12)))

            v = r_13 - coef*r_12

            # -------------------------------------------------------------------
            # Derivatives of coeff with respect to...
            # ... x1
            dcoefda1 = np.array([-1,0]).dot(r_12)/(r_12.dot(r_12)) \
                       + r_13.dot(np.array([-1,0])/(r_12.dot(r_12)) \
                       +r_12*(2*(a2-a1)/(r_12.dot(r_12)**2)))

            # ... y1
            dcoefdb1 = np.array([0, -1]).dot(r_12) / (r_12.dot(r_12)) \
                       + r_13.dot(np.array([0, -1]) / (r_12.dot(r_12)) \
                       + r_12 * (2 * (b2 - b1) / (r_12.dot(r_12) ** 2)))

            # ... x2
            dcoefda2 = r_13.dot(np.array([1, 0]) / (r_12.dot(r_12)) \
                       + r_12 * (2 * (a2 - a1) / (r_12.dot(r_12) ** 2)))

            # ... y2
            dcoefdb2 = np.array([0,0]).dot(r_12)/(r_12.dot(r_12)) \
                       + r_13.dot(np.array([0,1])/(r_12.dot(r_12)) \
                       +r_12*(2*(b2-b1)/(r_12.dot(r_12)**2)))

            # ... x3
            dcoefda3 = np.array([1, 0]).dot(r_12) / (r_12.dot(r_12))

            # ... y3
            dcoefdb3 = np.array([0, 1]).dot(r_12) / (r_12.dot(r_12))

            # END of derivatives of coeff
            # All formulas checked by Meyer
            # -----------------------------------------------------------------------

            # Comment by Meyer: CAUTION: The following formulas seem to be wrong!
            # The formulas in the thesis of Gruber seem to be correct but are not the same as here
            drejda1 = np.array([-1, 0]) - dcoefda1*r_12 + np.array([coef, 0])

            drejdb1 = np.array([0, -1]) - dcoefdb1*r_12 + np.array([0, coef])

            drejda2 = - dcoefda2*r_12 - np.array([coef, 0])

            drejdb2 = - dcoefdb2*r_12 - np.array([0, coef])

            drejda3 = np.array([1,0]) - dcoefda3*r_12

            drejdb3 = np.array([0,1]) - dcoefdb3*r_12

            bx1 = np.array([
                            2*v.dot(drejda1),
                            2*v.dot(drejdb1)
                            ])
            bx2 = np.array([
                            2*v.dot(drejda2),
                            2*v.dot(drejdb2)
                            ])
            bx3 = np.array([
                            2*v.dot(drejda3),
                            2*v.dot(drejdb3)
                            ])

            b = np.concatenate((bx1,bx2,bx3))

        return b


class NodesCollinear2DConstraint(Constraint):
    """
    Class to define collinearity (three points on a line).
    
    This function works with two given coordinates of the nodes and makes
    them collinear. If you want a 3D effect, you have to make two constraints,
    one for (X and Y) and the other for (X and Z or Y and Z).
    This is made in this manner because each constraint can only remove one
    degree of freedom of the system, not two, at a time.
    """
    name = 'NodesCollinear2D'

    def c_equation(self, X_local, u_local, t=0):
        '''
        Returns residual of c_equation for a collinearity constraint between three nodes
        
        Caution: Only three points are allowed to be passed to the function
        Otherwise the results will be wrong.
        
        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for three points just concatenated but only for 2 dimensions
            e.g. x and y [x1 y1 x2 y2 x3 y3] or y and z [y1 z1 y2 z2 y3 z3]
        u_local: numpy.array
            current displacements for three points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary
            
        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector
            
        '''
        dofs_per_node = len(u_local)//3

        x = X_local + u_local
        x1 = x[:dofs_per_node]
        x2 = x[dofs_per_node:2*dofs_per_node]
        x3 = x[-dofs_per_node:]

        # Three points are collinear if and only if the determinant of the matrix
        # A here is zero.
        A = np.hstack((np.vstack((x1,x2,x3)), np.ones((3,1))))

        return np.linalg.det(A) #**2

    def vector_b(self, X_local, u_local, t=0):
        '''
        Returns b vector for a collinearity constraint between three nodes

        Caution: Only three points are allowed to be passed to the function
        Otherwise the results will be wrong.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for three points just concatenated but only for 2 dimensions
            e.g. x and y [x1 y1 x2 y2 x3 y3] or y and z [y1 z1 y2 z2 y3 z3]
        u_local: numpy.array
            current displacements for three points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time

        Returns
        -------
        b : numpy.array
            b vector for assembly in B matrix

        '''
        dofs_per_node = len(u_local)//3

        x = X_local + u_local
        x1 = x[:dofs_per_node]
        x2 = x[dofs_per_node:2*dofs_per_node]
        x3 = x[-dofs_per_node:]
#        A = np.hstack((np.vstack((x1,x2,x3)), np.ones((3,1))))
#        det_A = np.linalg.det(A)
        a1 = x1[0]
        b1 = x1[1]
        a2 = x2[0]
        b2 = x2[1]
        a3 = x3[0]
        b3 = x3[1]

        b = np.array([
                       b2-b3, #2*(b2 - b3)*(det_A),
                      -a2+a3, #-2*(a2 - a3)*(det_A), #
                      -b1+b3, #-2*(b1 - b3)*(det_A), #
                       a1-a3, #2*(a1 - a3)*(det_A), #
                       b1-b2, #2*(b1 - b2)*(det_A), #
                      -a1+a2, #-2*(a1 - a2)*(det_A), #
                      ])
        return b


class EqualDisplacementConstraint(Constraint):
    '''
    Class to define an equal displacements constraint.
    
    Note: This constraint can also return a matrix as b vector!
    This is not consistent to the other functions!
    
    '''
    name = 'EqualDisplacement'

    def c_equation(self, X_local, u_local, t=0):
        '''
        Returns residual of c_equation for an equal displacement constraint between two nodes

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for both points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2], [x1 y1 x2 y2] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary

        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector

        '''
        dofs_per_node = len(u_local)//2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]

        return u2-u1

    def vector_b(self, X_local, u_local, t=0):
        '''
        Returns the b vector (for assembly in B matrix) for an equal displacements constraint between two nodes

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for both points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2], [x1 y1 x2 y2] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary

        Returns
        -------
        b : numpy.array
            b vector for assembly in B matrix [bx1 by1 bz1 bx2 by2 bz2]

        '''
        dofs_per_node = len(u_local)//2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]

        return np.concatenate((-np.ones_like(u1), np.ones_like(u2)))


class FixedDistanceToPlaneConstraint(Constraint):
    '''
    Fixed distance to plane constraint

    Class to define a fixed distance to plane constraint
    '''
    name = 'FixedDistanceToPlane'

    def c_equation(self, X_local, u_local, t=0):
        '''
        Returns residual of c_equation for a fixed distance to plane constraint
        
        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            The first three points define the plane, the fourth point shall have fixed distance
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for three points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary
            
        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector
            
        '''

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2*3]
        x3 = x[2*3:3*3]
        x4 = x[-3:]

        plane_vector_1 = x2 - x1
        plane_vector_2 = x3 - x1
        plane_normal = np.cross(plane_vector_1, plane_vector_2)
        plane_normal = plane_normal/(np.linalg.norm(plane_normal))
        x4_vector = x4 - x1

        X1 = X_local[:3]
        X2 = X_local[3:2*3]
        X3 = X_local[2*3:3*3]
        X4 = X_local[-3:]

        initial_vector_1 = X2 - X1
        initial_vector_2 = X3 - X1
        ini_plane_normal = np.cross(initial_vector_1, initial_vector_2)
        ini_plane_normal = ini_plane_normal/(np.linalg.norm(ini_plane_normal))
        X4_vector = X4-X1

        return np.dot(x4_vector, plane_normal)**2 - \
               np.dot(X4_vector, ini_plane_normal)**2

    def vector_b(self, X_local, u_local, t=0):
        '''
        Returns the b vector (for assembly in B matrix) for a fixed distance to plane constraint

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because

        Returns
        -------
        b : numpy.array
            b vector for assembly in B matrix [bx1 by1 bz1 bx2 by2 bz2 bx3 by3 bz3 bx4 by4 bz4]

        '''

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2*3]
        x3 = x[2*3:3*3]
        x4 = x[-3:]
        a1, b1, c1 = x1
        a2, b2, c2 = x2
        a3, b3, c3 = x3
        a4, b4, c4 = x4

        plane_vector_1 = x2 - x1
        plane_vector_2 = x3 - x1
        plane_normal = np.cross(plane_vector_1, plane_vector_2)
        plane_normal = plane_normal/(np.linalg.norm(plane_normal))
        x4_vector = x4 - x1

        c_dep = np.dot(x4_vector, plane_normal)**2
#        plane_normal = np.cross(x2 - x1, x3 - x1)
#        norm2_plane = np.linalg.norm(plane_normal)**2

        b = np.array([
                       (-(2*b2 - 2*b3)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)) - \
                       (2*c2 - 2*c3)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)))*(c_dep)**2 + \
                       ((a1 - a4)*((b1 - b2)*(c1 - c3) - (b1 - b3)*(c1 - c2)) - \
                       (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))*(2*(b1 - b2)*(c1 - c3) - \
                       2*(b1 - b3)*(c1 - c2) - 2*(b1 - b4)*(c2 - c3) + \
                       2*(b2 - b3)*(c1 - c4))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # a1

                       (-(-2*a2 + 2*a3)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)) - \
                       (2*c2 - 2*c3)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)))*(c_dep)**2 + \
                       ((a1 - a4)*((b1 - b2)*(c1 - c3) - (b1 - b3)*(c1 - c2)) - \
                       (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))*(-2*(-a1 + a2)*(-c1 + c3) + \
                       2*(-a1 + a3)*(-c1 + c2) + 2*(a1 - a4)*(c2 - c3) + \
                       2*(-a2 + a3)*(c1 - c4))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # b1

                       (-(-2*a2 + 2*a3)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) - \
                       (-2*b2 + 2*b3)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)))*(c_dep)**2 + \
                       ((a1 - a4)*((b1 - b2)*(c1 - c3) - (b1 - b3)*(c1 - c2)) - \
                       (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))*(2*(a1 - a2)*(b1 - b3) - \
                       2*(a1 - a3)*(b1 - b2) + 2*(a1 - a4)*(-b2 + b3) - \
                       2*(-a2 + a3)*(b1 - b4))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # c1

                       (-(-2*b1 + 2*b3)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)) - \
                       (-2*c1 + 2*c3)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)))*(c_dep)**2 + \
                       (2*(-b1 + b3)*(c1 - c4) - \
                       2*(b1 - b4)*(-c1 + c3))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # a2

                       (2*(a1 - a3)*(c1 - c4) + \
                       2*(a1 - a4)*(-c1 + c3))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2) + \
                       (-(2*a1 - 2*a3)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)) - \
                       (-2*c1 + 2*c3)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)))*(c_dep)**2, # b2

                       (-2*(a1 - a3)*(b1 - b4) + \
                       2*(a1 - a4)*(b1 - b3))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2) + \
                       (-(2*a1 - 2*a3)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) - \
                       (2*b1 - 2*b3)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)))*(c_dep)**2, # c2

                       (2*(b1 - b2)*(c1 - c4) - \
                       2*(b1 - b4)*(c1 - c2))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2) + \
                       (-(2*b1 - 2*b2)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)) - \
                       (2*c1 - 2*c2)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)))*(c_dep)**2, # a3

                       (-(-2*a1 + 2*a2)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)) - \
                       (2*c1 - 2*c2)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)))*(c_dep)**2 + \
                       (2*(-a1 + a2)*(c1 - c4) + \
                       2*(a1 - a4)*(c1 - c2))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # b3

                       (-(-2*a1 + 2*a2)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) - \
                       (-2*b1 + 2*b2)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)))*(c_dep)**2 + \
                       (-2*(-a1 + a2)*(b1 - b4) + \
                       2*(a1 - a4)*(-b1 + b2))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # c3

                       (-2*(b1 - b2)*(c1 - c3) + \
                       2*(b1 - b3)*(c1 - c2))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # a4

                       (2*(-a1 + a2)*(-c1 + c3) - \
                       2*(-a1 + a3)*(-c1 + c2))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + ((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2))**2), # b4

                       (-2*(a1 - a2)*(b1 - b3) + \
                       2*(a1 - a3)*(b1 - b2))*((a1 - a4)*((b1 - b2)*(c1 - c3) - \
                       (b1 - b3)*(c1 - c2)) - (b1 - b4)*((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2)) + (c1 - c4)*((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2)))/(((-a1 + a2)*(-c1 + c3) - \
                       (-a1 + a3)*(-c1 + c2))**2 + ((a1 - a2)*(b1 - b3) - \
                       (a1 - a3)*(b1 - b2))**2 + \
                       ((b1 - b2)*(c1 - c3) - (b1 - b3)*(c1 - c2))**2), # c4
                      ])

        # NOTE: This has not been checked by Meyer yet!
        return b


class NodesCoplanarConstraint(Constraint):
    '''
    Nodes Coplanar Constraint

    Class to define a coplanary constraint between nodes such that 4 nodes belong to one plane
    '''
    name = 'NodesCoplanar'

    def c_equation(self, X_local, u_local, t=0):
        '''
        Returns residual of c_equation for a colplanarity constraint between four nodes

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for three points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary

        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector

        '''

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2*3]
        x3 = x[2*3:3*3]
        x4 = x[-3:]

        # x1, x2, x3 and x4 are coplanar if the determinant of A is 0
        A = np.hstack((x1.T-x4.T, x2.T-x4.T, x3.T-x4.T)).reshape((3,3))
        return np.linalg.det(A)

    def vector_b(self, X_local, u_local, t=0):
        '''
        Returns the b vector (for assembly in B matrix) for a coplanarity constraint

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because

        Returns
        -------
        b : numpy.array
            b vector for assembly in B matrix [bx1 by1 bz1 bx2 by2 bz2 bx3 by3 bz3 bx4 by4 bz4]

        '''

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2*3]
        x3 = x[2*3:3*3]
        x4 = x[-3:]
        a1, b1, c1 = x1
        a2, b2, c2 = x2
        a3, b3, c3 = x3
        a4, b4, c4 = x4

        b = np.array([
                        b2*c3 - b2*c4 - b3*c2 + b3*c4 + b4*c2 - b4*c3, # a1
                       -a2*c3 + a2*c4 + a3*c2 - a3*c4 - a4*c2 + a4*c3, # b1
                        a2*b3 - a2*b4 - a3*b2 + a3*b4 + a4*b2 - a4*b3, # c1
                       -b1*c3 + b1*c4 + b3*c1 - b3*c4 - b4*c1 + b4*c3, # a2
                        a1*c3 - a1*c4 - a3*c1 + a3*c4 + a4*c1 - a4*c3, # b2
                       -a1*b3 + a1*b4 + a3*b1 - a3*b4 - a4*b1 + a4*b3, # c2
                        b1*c2 - b1*c4 - b2*c1 + b2*c4 + b4*c1 - b4*c2, # a3
                       -a1*c2 + a1*c4 + a2*c1 - a2*c4 - a4*c1 + a4*c2, # b3
                        a1*b2 - a1*b4 - a2*b1 + a2*b4 + a4*b1 - a4*b2, # c3
                       -b1*c2 + b1*c3 + b2*c1 - b2*c3 - b3*c1 + b3*c2, # a4
                        a1*c2 - a1*c3 - a2*c1 + a2*c3 + a3*c1 - a3*c2, # b4
                       -a1*b2 + a1*b3 + a2*b1 - a2*b3 - a3*b1 + a3*b2, # c4
                      ])
        # Checked that this is the same as in the thesis by Gruber. But only first two rows have been checked by Meyer
        return b
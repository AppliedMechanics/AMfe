#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
GiD ascii mesh reader for I/O module.
"""

from re import search

from amfe.io.mesh.base import MeshReader

__all__ = [
    'GidAsciiMeshReader'
    ]


class GidAsciiMeshReader(MeshReader):
    """
    Reader for GiD ascii files.
    """

    eletypes = {
        ('Linear', 2): 'straight_line',
        ('Linear', 3): 'quadratic_line',
        ('Triangle', 3): 'Tri3',
        ('Triangle', 6): 'Tri6',
        ('Triangle', 10): 'Tri10',
        ('Quadrilateral', 4): 'Quad4',
        ('Quadrilateral', 8): 'Quad8',
        ('Tetrahedra', 4): 'Tet4',
        ('Tetrahedra', 10): 'Tet10',
        ('Hexahedra', 8): 'Hexa8',
        ('Hexahedra', 20): 'Hexa20',
        ('Prism', 6): 'Prism6',
        ('Pyramid', 6): None,
        ('Point', 1): 'point',
        ('Sphere', -1): None,
        ('Circle', -1): None,
    }

    def __init__(self, filename=None):
        super().__init__()
        self._filename = filename
        return

    def parse(self, builder, verbose=False):
        """

        Parameters
        ----------
        builder : MeshConverter
            MeshConverter that is called for building the converted mesh
        verbose : bool
            If True, verbose mode is activated
        Returns
        -------

        """
        with open(self._filename, 'r') as infile:
            line = next(infile)
            pattern = 'dimension (\d) ElemType\s([A-Za-z0-9]*)\sNnode\s(\d)'
            match = search(pattern, line)
            dimension = int(match.group(1))  # dimension (nodes have 2 or 3 coordinates)
            eleshape = match.group(2)  # elementtype
            nnodes = int(match.group(3))  # number of nodes per element
            eletype = None

            builder.build_mesh_dimension(dimension)
            try:
                eletype = self.eletypes[(eleshape, nnodes)]
            except Exception:
                print('Eletype ({},{}) cannot be found in eletypes dictionary, it is not implemented in AMfe.'
                      .format(eletype, nnodes))
            if eletype is None:
                raise ValueError('Element ({},{}) is not implemented in AMfe.'.format(eletype, nnodes))
            if verbose:
                print('Eletype {} identified.'.format(eletype))

            # Coordinates
            for line in infile:
                if line.strip() == 'Coordinates':
                    for line in infile:
                        try:
                            nodeid = int(line[0:5])
                            x = float(line[5:21])
                            y = float(line[21:37])
                            z = float(line[37:53])
                        except ValueError:
                            if line.strip() == "End Coordinates":
                                break
                            else:
                                raise
                        builder.build_node(nodeid, x, y, z)

                elif line.strip() == 'Elements':
                    for line in infile:
                        try:
                            element = [int(e) for e in line.split()]
                            eleid = element[0]
                            nodes = element[1:]
                        except ValueError:
                            if line.strip() == "End Elements":
                                break
                            else:
                                raise
                        builder.build_element(eleid, eletype, nodes)
                else:
                    print(line)

        # Finished build
        return

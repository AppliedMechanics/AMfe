#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
GiD json mesh reader for I/O module.
"""

from json import load

from amfe.io.mesh.base import MeshReader

__all__ = [
    'GidJsonMeshReader'
    ]


class GidJsonMeshReader(MeshReader):
    """
    Reader for GiD json files.
    """

    # Eletypes dict:
    # {('shape', 0/1 (=non-quadratic/quadratic)): 'AMfe-name'}
    eletypes = {
        ('Line', 0): 'straight_line',
        ('Line', 1): 'quadratic_line',
        ('Triangle', 0): 'Tri3',
        ('Triangle', 1): 'Tri6',
        ('Triangle', 2): 'Tri10',
        ('Quadrilateral', 0): 'Quad4',
        ('Quadrilateral', 1): 'Quad8',
        ('Tetrahedra', 0): 'Tet4',
        ('Tetrahedra', 1): 'Tet10',
        ('Hexahedra', 0): 'Hexa8',
        ('Hexahedra', 1): 'Hexa20',
        ('Prism', 0): 'Prism6',
        ('Prism', 1): None,
        ('Pyramid', 0): None,
        ('Pyramid', 1): None,
        ('Point', 0): 'point',
        ('Point', 1): 'point',
        ('Sphere', 0): None,
        ('Sphere', 1): None,
        ('Circle', 0): None,
        ('Circle', 1): None
    }

    eletypes_3d = {'Tetrahedra', 'Hexahedra', 'Prism', 'Pyramid'}

    def __init__(self, filename=None):
        super().__init__()
        self._filename = filename
        return

    def parse(self, builder, verbose=False):
        """
        Parse GiD json file to object specified by the builder (MeshConverter object).
        
        Parameters
        ----------
        builder : MeshConverter
        verbose : bool

        Returns
        -------
        object
        """

        with open(self._filename, 'r') as infile:
            json_tree = load(infile)

            dimflag = set([ele_type['ele_type'] for ele_type in json_tree['elements']]).intersection(self.eletypes_3d)
            if not dimflag:
                builder.build_mesh_dimension(2)
            else:
                builder.build_mesh_dimension(3)

            no_of_nodes = json_tree['no_of_nodes']
            no_of_elements = json_tree['no_of_elements']

            builder.build_no_of_nodes(no_of_nodes)
            builder.build_no_of_elements(no_of_elements)

            print("Import nodes...")
            for counter, node in enumerate(json_tree['nodes']):
                builder.build_node(node['id'], node['coords'][0], node['coords'][1], node['coords'][2])
                print("\rImport node no. {} / {}".format(counter, no_of_nodes), end='')

            print("\n...finished")
            print("Import elements")
            for ele_type in json_tree['elements']:
                current_amfe_eletype = self.eletypes[(ele_type['ele_type'], json_tree['quadratic'])]
                print("    Import eletype {} ...".format(current_amfe_eletype))
                for counter, element in enumerate(ele_type['elements']):
                    eleid = element['id']
                    nodes = element['connectivity'][:-1]
                    builder.build_element(eleid, current_amfe_eletype, nodes)
                    print("\rImport element No. {} / {}".format(counter, no_of_elements), end='')
                print("\n    ...finished")
            print("\n...finished")

            print("Import groups...")
            for group in json_tree['groups']:
                builder.build_group(group, nodeids=json_tree['groups'][group]['nodes'],
                                         elementids=json_tree['groups'][group]['elements'])
            print("...finished")

        # Finished build
        return

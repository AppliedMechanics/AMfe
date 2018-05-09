# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
io module of AMfe

It handles input output operations for AMfe
"""

import abc
import re
import numpy as np
import pandas as pd

from amfe import Mesh

__all__ = [
    'GidAsciiMeshReader',
    'AmfeMeshConverter',
]


class MeshReader(abc.ABC):
    '''
    Abstract super class for all MeshReaders.

    The tasks of the MeshReaders are:
    ---------------------------------
    
    - Read line by line a stream (or file)
    - Call MeshConverter function for each line
    
    PLEASE FOLLOW THE BUILDER PATTERN!
    '''

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self._builder = None

    @abc.abstractmethod
    def parse(self):
        pass

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, builder):
        if isinstance(builder, MeshConverter):
            self._builder = builder
        else:
            raise ValueError('No valid builder given.')


class GidAsciiMeshReader(MeshReader):
    '''
    Reads GID-Ascii-Files
    '''

    eletypes = {
                ('Linear',2): 'straight_line',
                ('Linear', 3): 'quadratic_line',
                ('Triangle',3): 'Tri3',
                ('Triangle', 6): 'Tri6',
                ('Triangle', 10): 'Tri10',
                ('Quadrilateral',4): 'Quad4',
                ('Quadrilateral', 8): 'Quad8',
                ('Tetrahedra',4): 'Tet4',
                ('Tetrahedra', 10): 'Tet10',
                ('Hexahedra',8): 'Hexa8',
                ('Hexahedra', 20): 'Hexa20',
                ('Prism',6): 'Prism6',
                ('Pyramid',6): None,
                ('Point',1): 'point',
                ('Sphere',-1): None,
                ('Circle',-1): None,
                }

    def __init__(self, filename=None, builder=None):
        self._filename = filename
        self.builder = builder

    def parse(self, verbose=False):
        with open(self._filename, 'r') as infile:
            line = next(infile)
            pattern = "dimension (\d) ElemType\s([A-Za-z0-9]*)\sNnode\s(\d)"
            match = re.search(pattern, line)
            dimension = int(match.group(1)) # dimension (nodes have two or three coordinates)
            eleshape = match.group(2) # elementtype
            nnodes = int(match.group(3)) # number of nodes per element

            self.builder.build_mesh_dimension(dimension)
            try:
                eletype = self.eletypes[(eleshape, nnodes)]
            except Exception:
                print('Eletype ({},{})  cannot be found in eletypes dictionary, it is not implemented in AMfe'.format(eletype,nnodes))
            if eletype is None:
                raise ValueError('Element ({},{}) is not implemented in AMfe'.format(eletype,nnodes))
            if verbose:
                print('Eletype {} identified'.format(eletype))
            # Coordinates
            for line in infile:
                if line.strip() == 'Coordinates':
                    if verbose:
                        print('Section Coordinates found')
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
                            self.builder.build_node(nodeid,x,y,z)

                elif line.strip() == 'Elements':
                    if verbose:
                        print('Section Elements found')
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
                            self.builder.build_element(eleid, eletype, nodes)
                else:
                    print(line)
        # Finished build, return mesh
        return self.builder.return_mesh()


class MeshConverter():
    '''
    Super class for all MeshConverters.
    '''

    def __init__(self, *args, **kwargs):
        pass

    def build_node(self,id,x,y,z):
        pass

    def build_element(self,id,type,nodes):
        pass

    def build_physical_group(self,type,id,entities):
        pass

    def build_node_group(self, name, nodeids):
        '''
        
        Parameters
        ----------
        name: string
            name identifying the node group
        nodeids: list
            list with node ids

        Returns
        -------

        '''
        pass

    def build_element_type(self,type):
        pass

    def build_material(self,material):
        pass

    def build_partition(self,partition):
        pass

    def build_mesh_dimension(self,dim):
        pass

    def return_mesh(self):
        pass


class AmfeMeshConverter(MeshConverter):
    '''
    Converter for AMfe Meshes
    '''

    # mapping from reader-nodeid to amfe-nodeid
    nodeid2rowidx = dict()

    def __init__(self):
        self._mesh = Mesh()
        self._mesh.el_df.rename(copy=False, inplace=True,
                  columns={0: 'idx',
                           1: 'el_type',
                           3: 'phys_group',
                           4: 'active'
                           })

    def build_mesh_dimension(self,dim):
        self._mesh.no_of_dofs_per_node = dim
        self._mesh.nodes = self._mesh.nodes[:,:dim]

    def build_node(self,id,x,y,z):
        print('ID: {}, X: {}, Y: {}, Z: {}'.format(id,x,y,z))
        amfeid = self._mesh.nodes.shape[0]
        self._mesh.nodes = np.append(self._mesh.nodes, np.array([x,y,z], dtype=float, ndmin=2),axis=0)
        self.nodeid2rowidx.update({id: amfeid})

    def build_element(self,id,type,nodes):
        print('ID: {}, Type: {}, Nodes: {}'.format(id,type,nodes))
        #num_of_nodes = len(nodes)
        #temp = {str(i): nodes[i] for i in range(num_of_nodes)}
        ele = {'idx': id, 'el_type': type, 'phys_group': 0, 'active': False, 'nodes': [np.array(nodes,dtype=np.int64)]}
        #ele.update(temp)
        df = pd.DataFrame(ele, index=[id])
        self._mesh.el_df = self._mesh.el_df.append(df)

    def return_mesh(self):
        self._mesh.nodeid2rowidx = self.nodeid2rowidx
        return self._mesh
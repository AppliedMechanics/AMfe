# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Mesh module of amfe. It handles the mesh from import, defining the dofs for the boundary conditions and the export.
"""


__all__ = [
    'Mesh',
    'create_xdmf_from_hdf5'
]

import os
import copy
# XML stuff
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom



import pandas as pd
import h5py
import numpy as np

from .element import Tet4, \
    Tet10, \
    Tri3, \
    Tri6, \
    Quad4, \
    Quad8, \
    Bar2Dlumped, \
    LineLinearBoundary, \
    LineQuadraticBoundary, \
    Tri3Boundary, \
    Tri6Boundary, \
    Hexa8, \
    Hexa20, \
    Quad4Boundary, \
    Quad8Boundary, \
    Prism6



from .mesh_tying import master_slave_constraint

# Element mapping is described here. If a new element is implemented, the
# features for import and export should work when the followig list will be updated.
element_mapping_list = [
    # internal Name, XMF Key,   gmsh-Key, vtk/ParaView-Key, no_of_nodes, description
    ['Tet4',          'Tetrahedron',   4, 10,  4,
     'Linear Tetraeder / nodes on every corner'],
    ['Tet10',         'Tetrahedron_10',  11, 24, 10,
     'Quadratic Tetraeder / 4 nodes at the corners, 6 nodes at the faces'],
    ['Hexa8',         'Hexahedron', 5, 12, 8,
     'Linear brick element'],
    ['Hexa20',         'Hex_20', 17, 25, 20,
     'Quadratic brick element'],
    ['Tri6',          'Triangle_6',   9, 22,  6,
     'Quadratic triangle / 6 node second order triangle'],
    ['Tri3',          'Triangle',   2,  5,  3,
     'Straight triangle / 3 node first order triangle'],
    ['Tri10',         '',  21, 35, 10,
     'Cubic triangle / 10 node third order triangle'],
    ['Quad4',         'Quadrilateral',   3,  9,  4,
     'Bilinear rectangle / 4 node first order rectangle'],
    ['Quad8',         'Quadrilateral_8',  16, 23,  8,
     'Biquadratic rectangle / 8 node second order rectangle'],
    ['Prism6',         'Wedge', 6, 23,  6,
     'Trilinear 6 node prism'],
    ['straight_line', 'Edge',   1,  3,  2,
     'Straight line composed of 2 nodes'],
    ['quadratic_line', 'Edge_3',  8, 21,  3,
     'Quadratic edge/line composed of 3 nodes'],
    ['point',       '', 15, np.NAN,  1, 'Single Point'],
    # Bars are missing, which are used for simple benfield truss
]

#
# Building the conversion dicts from the element_mapping_list
#
gmsh2amfe        = dict([])
amfe2gmsh        = dict([])
amfe2vtk         = dict([])
amfe2xmf         = dict([])
amfe2no_of_nodes = dict([])

for element in element_mapping_list:
    gmsh2amfe.update({element[2] : element[0]})
    amfe2gmsh.update({element[0] : element[2]})
    amfe2vtk.update( {element[0] : element[3]})
    amfe2xmf.update({element[0] : element[1]})
    amfe2no_of_nodes.update({element[0] : element[4]})

# Some conversion stuff fron NASTRAN to AMFE
nas2amfe = {'CTETRA' : 'Tet10',
            'CHEXA' : 'Hexa8'}

# Same for Abaqus
abaq2amfe = {'C3D10M' : 'Tet10',
             'C3D8' : 'Hexa8',
             'C3D20' : 'Hexa20',
             'C3D4' : 'Tet4',
             'C3D6' : 'Prism6', # 6 node prism
             'C3D8I' : 'Hexa8', # acutally the better version
             'B31' : None,
             'CONN3D2' : None,
            }

# Abaqus faces for identifying surfaces
abaq_faces = {
    'Hexa8': {'S1' : ('Quad4', np.array([0, 1, 2, 3])),
              'S2' : ('Quad4', np.array([4, 7, 6, 5])),
              'S3' : ('Quad4', np.array([0, 4, 5, 1])),
              'S4' : ('Quad4', np.array([1, 5, 6, 2])),
              'S5' : ('Quad4', np.array([2, 6, 7, 3])),
              'S6' : ('Quad4', np.array([3, 7, 4, 0])),
             },

    'Hexa20' : {'S1': ('Quad8', np.array([ 0,  1,  2,  3,  8,  9, 10, 11])),
                'S2': ('Quad8', np.array([ 4,  7,  6,  5, 15, 14, 13, 12])),
                'S3': ('Quad8', np.array([ 0,  4,  5,  1, 16, 12, 17,  8])),
                'S4': ('Quad8', np.array([ 1,  5,  6,  2, 17, 13, 18,  9])),
                'S5': ('Quad8', np.array([ 2,  6,  7,  3, 18, 14, 19, 10])),
                'S6': ('Quad8', np.array([ 3,  7,  4,  0, 19, 15, 16, 11]))},

    'Tet4': {'S1' : ('Tri3', np.array([0, 1, 2])),
             'S2' : ('Tri3', np.array([0, 3, 1])),
             'S3' : ('Tri3', np.array([1, 3, 2])),
             'S4' : ('Tri3', np.array([2, 3, 0])),
            },

    'Tet10': {'S1' : ('Tri6', np.array([0, 1, 2, 4, 5, 6])),
              'S2' : ('Tri6', np.array([0, 3, 1, 7, 8, 4])),
              'S3' : ('Tri6', np.array([1, 3, 2, 8, 9, 5])),
              'S4' : ('Tri6', np.array([2, 3, 0, 9, 7, 6])),
             },

    'Prism6' : {'S1': ('Tri3', np.array([0, 1, 2])),
                'S2': ('Tri3', np.array([3, 5, 4])),
                'S3': ('Quad4', np.array([0, 3, 4, 1])),
                'S4': ('Quad4', np.array([1, 4, 5, 2])),
                'S5': ('Quad4', np.array([2, 5, 3, 0])),
                },
}


def check_dir(*filenames):
    '''
    Check if paths exists; if not, the given paths will be created.

    Parameters
    ----------
    *filenames : string or list of strings
        string containing a path.

    Returns
    -------
    None
    '''
    for filename in filenames:  # loop on files
        dir_name = os.path.dirname(filename)
        # check if directory does not exist; then create directory
        if not os.path.exists(dir_name) or dir_name == '':
            os.makedirs(os.path.dirname(filename))          # then create directory
            print("Created directory: " + os.path.dirname(filename))


def prettify_xml(elem):
    '''
    Return a pretty string from an XML Element-Tree

    Parameters
    ----------
    elem : Instance of xml.etree.ElementTree.Element
        XML element tree

    Returns
    -------
    str : string
        well formatted xml file string
    '''
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml()


def shape2str(tupel):
    '''
    Convert a tupel to a string containing the numbers of the tupel for xml
    export.

    Parameters
    ----------
    tupel : tupel
        tupel containing numbers (usually the shape of an array)

    Returns
    -------
    str : string
        string containing the numbers of the tupel
    '''
    return ' '.join([str(i) for i in tupel])


def h5_set_attributes(h5_object, attribute_dict):
    '''
    Add the attributes from attribute_dict to the h5_object.

    Parameters
    ----------
    h5_object : instance of h5py File, DataSet or Group
        hdf5 object openend with h5py
    attribute_dict : dict
        dictionary with keys and attributes to be added to the h5_object

    Returns
    -------
    None
    '''
    for key in attribute_dict:
        h5_object.attrs[key] = attribute_dict[key]
    return


def create_xdmf_from_hdf5(filename):
    '''
    Create an accompanying xdmf file for a given hdmf file.

    Parameters
    ----------
    filename : str
        filename of the hdf5-file. Produces an XDMF-file of same name with
        .xdmf ending.

    Returns
    -------
    None

    '''
    filename_no_dir = os.path.split(filename)[-1]
    # filename_no_ext = os.path.splitext(filename)[0]

    with h5py.File(filename, 'r') as f:
        h5_topology = f['mesh/topology']
        h5_nodes = f['mesh/nodes']
        h5_time_vals = f['time_vals']

        xml_root = Element('Xdmf', {'Version':'2.2'})
        domain = SubElement(xml_root, 'Domain')
        time_grid = SubElement(domain, 'Grid', {'GridType':'Collection',
                                                'CollectionType':'Temporal'})
        # time loop
        for i, T in enumerate(f['time']):
            spatial_grid= SubElement(time_grid, 'Grid',
                                     {'Type':'Spatial',
                                      'GridType':'Collection'})

            time = SubElement(spatial_grid, 'Time', {'TimeType':'Single',
                                                     'Value':str(T)})
            # loop over all mesh topologies
            for key in h5_topology.keys():
                grid = SubElement(spatial_grid, 'Grid', {'Type':'Uniform'})
                topology = SubElement(grid, 'Topology',
                                      {'TopologyType':h5_topology[key].attrs['TopologyType'],
                                       'NumberOfElements':str(h5_topology[key].shape[0])})
                topology_data = SubElement(topology, 'DataItem',
                                       {'NumberType':'Int',
                                        'Format':'HDF',
                                        'Dimensions':shape2str(h5_topology[key].shape)})
                topology_data.text = filename_no_dir + ':/mesh/topology/' + key

                # Check, if mesh is 2D or 3D
                xdmf_node_type = 'XYZ'
                if h5_nodes.shape[-1] == 2:
                    xdmf_node_type = 'XY'

                geometry = SubElement(grid, 'Geometry',
                                      {'Type':'Uniform',
                                       'GeometryType':xdmf_node_type})
                geometry_data_item = SubElement(geometry, 'DataItem',
                                                {'NumberType':'Float',
                                                 'Format':'HDF',
                                                 'Dimensions':shape2str(h5_nodes.shape)})
                geometry_data_item.text = filename_no_dir + ':/mesh/nodes'

                # Attribute loop for export of displacements, stresses etc.
                for key_t in h5_time_vals.keys():
                    field = h5_time_vals[key_t]
                    if field.attrs['ParaView'] == np.True_:
                        field_attr = SubElement(grid, 'Attribute',
                                                {'Name':field.attrs['Name'],
                                                 'AttributeType':
                                                    field.attrs['AttributeType'],
                                                 'Center':field.attrs['Center']})
                        no_of_components = field.attrs['NoOfComponents']
                        field_dim = (field.shape[0] // no_of_components,
                                     no_of_components)
                        field_data = SubElement(field_attr, 'DataItem',
                                                {'ItemType':'HyperSlab',
                                                 'Dimensions':shape2str(field_dim)})

                        field_hyperslab = SubElement(field_data, 'DataItem',
                                                     {'Dimensions':'3 2',
                                                      'Format':'XML'})

                        # pick the i-th column via hyperslab; If no temporal values
                        # are pumped out, use the first column
                        if i <= field.shape[-1]: # field has time instance
                            col = str(i)
                        else: # field has no time instance, use first col
                            col = '0'
                        field_hyperslab.text = '0 ' + col + ' 1 1 ' + \
                                                str(field.shape[0]) + ' 1'
                        field_hdf = SubElement(field_data, 'DataItem',
                                               {'Format':'HDF',
                                                'NumberType':'Float',
                                                'Dimensions':shape2str(field.shape)})
                        field_hdf.text = filename_no_dir + ':/time_vals/' + key_t

                # Attribute loop for cell values like weights
                if 'time_vals_cell' in f.keys():
                    h5_time_vals_cell = f['time_vals_cell']
                    for key_2 in h5_time_vals_cell.keys():
                        field = h5_time_vals_cell[key_2][key]
                        if field.attrs['ParaView'] == np.True_:
                            field_attr = SubElement(grid, 'Attribute',
                                                    {'Name':field.attrs['Name'],
                                                     'AttributeType':
                                                        field.attrs['AttributeType'],
                                                     'Center':field.attrs['Center']})
                            no_of_components = field.attrs['NoOfComponents']
                            field_dim = (field.shape[0] // no_of_components,
                                         no_of_components)
                            field_data = SubElement(field_attr, 'DataItem',
                                                    {'ItemType':'HyperSlab',
                                                     'Dimensions':shape2str(field_dim)})

                            field_hyperslab = SubElement(field_data, 'DataItem',
                                                         {'Dimensions':'3 2',
                                                          'Format':'XML'})

                            # pick the i-th column via hyperslab; If no temporal values
                            # are pumped out, use the first column
                            if i <= field.shape[-1]: # field has time instance
                                col = str(i)
                            else: # field has no time instance, use first col
                                col = '0'
                            field_hyperslab.text = '0 ' + col + ' 1 1 ' + \
                                                    str(field.shape[0]) + ' 1'
                            field_hdf = SubElement(field_data, 'DataItem',
                                                   {'Format':'HDF',
                                                    'NumberType':'Float',
                                                    'Dimensions':shape2str(field.shape)})
                            field_hdf.text = filename_no_dir + ':/time_vals_cell/' \
                                             + key_2 + '/' + key

    # write xdmf-file
    xdmf_str = prettify_xml(xml_root)
    filename_no_ext, ext = os.path.splitext(filename)
    with open(filename_no_ext + '.xdmf', 'w') as f:
        f.write(xdmf_str)


class Mesh:
    '''
    Class for handling the mesh operations.

    Attributes
    ----------
    nodes : ndarray
        Array of x-y-z coordinates of the nodes when imported. Dimension is
        (no_of_nodes, no_of_dofs_per_node).
        If no_of_dofs_per_node=2: z-direction is dropped!
        Caution! Currently this property describes the node-coordinates of the
        first reference configuration where no mesh morphing techniques have been
        applied.
    connectivity : list
        List of nodes indices belonging to one element.
    constraint_list: ndarray
        Experimental: contains constraints imported from nastran-files via
        import_bdf()
    el_df : pandas.DataFrame
        Pandas Dataframe containing Element-Definitions of the Original file
        (e.g. *.msh or *.bdf-File)
    ele_obj : list
        List of element objects. The list contains actually only the pointers
        pointing to the element object. For each combination of element-type
        and material only one Element object is instanciated.
        ele_obj contains for each element a pointer to one of these Element
        objects.
    neumann_connectivity : list
        list of nodes indices belonging to one element for neumann BCs.
    neumann_obj : list
        List of element objects for the neumann boundary conditions.
    nodes_dirichlet : ndarray
        Array containing the nodes involved in Dirichlet Boundary Conditions.
    dofs_dirichlet : ndarray
        Array containing the dofs which are to be blocked by Dirichlet Boundary
        Conditions.
    no_of_dofs_per_node : int
        Number of dofs per node. Is 3 for 3D-problems, 2 for 2D-problems. If
        rotations are considered, this nubmer can be >3.
    no_of_elements : int
        Number of elements in the whole mesh associated with an element object.
    no_of_nodes : int
        Number of nodes of the whole system.
    no_of_dofs : int
        Number of dofs of the whole system (including constrained dofs).
    element_class_dict : dict
        Dictionary containing objects of elements.
    element_boundary_class_dict : dict
        Dictionary containing objects of skin elements.
    node_idx : int
        index describing, at which position in the Pandas Dataframe `el_df`
        the nodes of the element start.
    '''

    def __init__(self):
        '''
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.nodes = np.array([])
        self.connectivity = []
        self.ele_obj = []
        self.neumann_connectivity = []
        self.neumann_obj = []
        self.nodes_dirichlet = np.array([], dtype=int)
        self.dofs_dirichlet = np.array([], dtype=int)
        self.constraint_list = []  # experimental; Introduced for nastran meshes
        # the displacements; They are stored as a list of numpy-arrays with
        # shape (ndof, no_of_dofs_per_node):
        self.no_of_dofs_per_node = 0
        self.no_of_dofs = 0
        self.no_of_nodes = 0
        self.no_of_elements = 0
        self.el_df = pd.DataFrame()
        self.node_idx = 0

        # Element Class dictionary with all available elements
        # This dictionary is only needed for load_group_to_mesh()-method
        kwargs = { }
        self.element_class_dict = {'Tet4'  : Tet4(**kwargs),
                                   'Tet10' : Tet10(**kwargs),
                                   'Hexa8' : Hexa8(**kwargs),
                                   'Hexa20': Hexa20(**kwargs),
                                   'Prism6' : Prism6(**kwargs),
                                   'Tri3'  : Tri3(**kwargs),
                                   'Tri6'  : Tri6(**kwargs),
                                   'Quad4' : Quad4(**kwargs),
                                   'Quad8' : Quad8(**kwargs),
                                   'Bar2Dlumped' : Bar2Dlumped(**kwargs),
                                  }

        kwargs = {'val' : 1., 'direct' : 'normal'}

        self.element_boundary_class_dict = {
            'straight_line' : LineLinearBoundary(**kwargs),
            'quadratic_line': LineQuadraticBoundary(**kwargs),
            'Tri3'          : Tri3Boundary(**kwargs),
            'Tri6'          : Tri6Boundary(**kwargs),
            'Quad4'         : Quad4Boundary(**kwargs),
            'Quad8'         : Quad8Boundary(**kwargs),}

        # actual set of implemented elements
        self.element_2d_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8', }
        self.element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}

        self.boundary_2d_set = {'straight_line', 'quadratic_line'}
        self.boundary_3d_set = {'straight_line', 'quadratic_line',
                                'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}

    def _update_mesh_props(self):
        '''
        Update the number properties of nodes and elements when the mesh has
        been updated
        
        It updates the following properties of the mesh-class-object:
            - no_of_nodes
            - no_of_dofs
            - no_of_elements
            
        '''
        self.no_of_nodes = len(self.nodes)
        self.no_of_dofs = self.no_of_nodes*self.no_of_dofs_per_node
        self.no_of_elements = len(self.connectivity)

    def import_csv(self, filename_nodes, filename_elements,
                   explicit_node_numbering=False, ele_type=False):
        '''
        Imports the nodes list and elements list from 2 different csv files.

        Parameters
        -----------
        filename_nodes : str
            name of the file containing the nodes in csv-format
        filename_elements : str
            name of the file containing the elements in csv-format
        explicit_node_numbering : bool, optional
            Flag stating, if the nodes are explicitly numbered in the csv-file.
            When set to true, the first column is assumed to have the node numbers
            and is thus ignored.
        ele_type: str
            Spezifiy elements type of the mesh (e.g. for a Tri-Mesh different
            elements types as Tri3, Tri4, Tri6 can be used)
            If not spezified value is set to 'False'

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        print('This function is deprecated! It does not work properly!')
        #######################################################################
        # NODES
        #######################################################################
        try:
            self.nodes = np.genfromtxt(filename_nodes, delimiter = ',', skip_header = 1)
        except:
            ImportError('Error while reading file ' + filename_nodes)
        # when line numbers are erased if they are content of the csv
        if explicit_node_numbering:
            self.nodes = self.nodes[:,1:]

        #######################################################################
        # ELEMENTS
        #######################################################################
        # Dictionary um an Hand der Anzahl der Knoten des Elements auf den Typ
        # des Elements zu schließen
        mesh_type_dict = {3: "Tri3",
                          4: "Quad4",
                          2: "Bar2D"} # Bislang nur 2D-Element aus csv auslesbar

        print('Reading elements from csv...  ', end="")
        self.connectivity = np.genfromtxt(filename_elements,
                                          delimiter = ',',
                                          dtype = int,
                                          skip_header = 1)
        if self.connectivity.ndim == 1: # Wenn nur genau ein Element vorliegt
            self.connectivity = np.array([self.connectivity])
            # Falls erste Spalte die Elementnummer angibt, wird diese hier
        # abgeschnitten, um nur die Knoten des Elements zu erhalten
        if explicit_node_numbering:
            self.connectivity = self.connectivity[:,1:]

        if ele_type:  # If element type is spezified, use this spezified type
            mesh_type = ele_type
        # If element type is not spzezified, try to determine element type
        # depending on the number of nodes per element (see default values for
        # different number of nodes per element in 'mesh_type_dict')
        else:
            try:  # Versuche Elementtyp an Hand von Anzahl der Knoten pro Element auszulesen
                (no_of_ele, no_of_nodes_per_ele) = self.connectivity.shape
                mesh_type = mesh_type_dict[no_of_nodes_per_ele] # Weise Elementtyp zu
            except:
                print('FEHLER beim Einlesen der Elemente. Typ nicht vorhanden.')
                raise

        print('Element type is {0}...  '.format(mesh_type), end="")
        self._update_mesh_props()
        print('Reading elements successful.')
        return

    def import_inp(self, filename, scale_factor=1.):
        '''
        Import Abaqus input file.

        Parameters
        ----------
        filename : string
            filename of the .msh-file
        scale_factor : float, optional
            scale factor for the mesh to adjust the units. The default value is
            1, i.e. no scaling is done.


        Returns
        -------
        None

        Notes
        -----
        This function is heavily experimental. It is just working for a subset
        of Abaqus input files and the goal is to capture the mesh of the model.

        The internal representation of the elements is done via a Pandas
        Dataframe object.

        '''

        print('*************************************************************')
        print('\nLoading Abaqus-mesh from', filename)
        nodes_list = []
        elements_list = []
        surface_list = []
        buf = []  # buffer for handling line continuations

        current_scope = None
        current_type = None
        current_name = None

        #################
        with open(filename, 'r') as infile:
            file_data = infile.read().splitlines()

        # Loop over all lines in the file
        for line in file_data:
            s = [x.strip() for x in line.split(',')]

            # Filter out comments
            if '**' in s[0]:
                continue

            elif s[0] == '*NODE':  # Nodes are coming
                current_scope = 'node'
                print('A node tag has been found:')
                print(s)
                continue

            elif s[0] == '*ELEMENT':
                current_scope = 'element'
                print('An elment tag has been found:')
                print(s)
                current_type = [a[5:] for a in s if a.startswith('TYPE=')][0]
                current_name = [a[6:] for a in s if a.startswith('ELSET=')][0]
                continue

            elif s[0] == '*SURFACE':
                current_scope = 'surface'
                print('A surface tag has been found:')
                print(s)
                current_type = [a[5:] for a in s if a.startswith('TYPE=')][0]
                current_name = [a[5:] for a in s if a.startswith('NAME=')][0]
                continue

            elif s[0].startswith('*'):
                current_scope = None
                continue

            elif current_scope == 'node':
                if s[-1].strip() == '':  # line has comma at its end
                    buf.extend(s[:-1])
                    continue
                else:
                    nodes_list.append(buf + s)
                    buf = []

            elif current_scope == 'element':
                s = line.split(',')
                if s[-1].strip() == '':  # line has comma at its end
                    buf.extend(s[:-1])
                    continue
                else:
                    elements_list.append([current_type, current_name] + buf + s)
                    buf = []

            elif current_scope == 'surface':
                s = line.split(',')
                if s[-1].strip() == '':  # line has comma at its end
                    buf.extend(s[:-1])
                    continue
                else:
                    surface_list.append([current_type, current_name] + buf + s)
                    buf = []


        self.no_of_dofs_per_node = 3 # this is just hard coded right now...
        nodes_arr = np.array(nodes_list, dtype=float)
        self.nodes = nodes_arr[:,1:] * scale_factor

        nodes_dict = pd.Series(index=np.array(nodes_arr[:,0], dtype=int),
                               data=np.arange(nodes_arr.shape[0]))

        for idx, ptr in enumerate(elements_list):
            # pop the first two elements as they are information
            tmp = [abaq2amfe[ptr.pop(0).strip()], ptr.pop(0), ptr.pop(0),]
            tmp.extend([nodes_dict[int(i)] for i in ptr])
            elements_list[idx] = tmp
        self.el_df = df = pd.DataFrame(elements_list, dtype=int)
        df.rename(copy=False, inplace=True,
                  columns={0 : 'el_type',
                           1 : 'phys_group',
                           2 : 'idx_abaqus',
                          })
        self.node_idx = 3

        ele_dict = pd.Series(index=df['idx_abaqus'].values,
                             data=np.arange(df['idx_abaqus'].values.shape[0]))

        # This is a little dirty but works: Add the surfaces to the
        # element dataframe self.el_df
        for row in surface_list:
            if row[0] == 'ELEMENT':
                ele_idx = ele_dict[int(row[2])]
                element = df.iloc[ele_idx, :].values
                face_dict = abaq_faces[element[0]]
                ele_name, node_indices = face_dict[row[3].strip()]
                nodes = element[self.node_idx:][node_indices]
                elements_list.append([ele_name, row[1], None]
                                     + nodes.tolist())
            if row[0] == 'NODE':
                nodes = nodes_dict[int(row[2])]
                elements_list.append(['point', row[1], None, nodes])

        self.el_df = df = pd.DataFrame(elements_list, dtype=int)
        df.rename(copy=False, inplace=True,
                  columns={0 : 'el_type',
                           1 : 'phys_group',
                           2 : 'idx_abaqus',
                          })

        self._update_mesh_props()
        # printing some information regarding the physical groups
        print('Mesh', filename, 'successfully imported.',
              '\nAssign a material to a physical group.')
        print('*************************************************************')
        return

    def import_bdf(self, filename, scale_factor=1.):
        '''
        Import a NASTRAN mesh.

        Parameters
        ----------
        filename : string
            filename of the .bdf-file
        scale_factor : float, optional
            scale factor for the mesh to adjust the units. The default value is
            1, i.e. no scaling is done.

        Returns
        -------
        None

        Notes
        -----
        This function is heavily experimental. It is just working for a subset
        of NASTRAN input files and the goal is to capture the mesh and the
        constraints of the model. The constraints are captured in the
        constraint_list-object of the class.

        The internal representation of the elements is done via a Pandas
        Dataframe object.

        '''
        comment_tag = '$'
        long_format_tag = '*'
        print('*************************************************************')
        print('\nLoading NASTRAN-mesh from', filename)

        nodes_list = []
        elements_list = []
        constraint_list = []
        # Flag indicating, that element was read in previous line
        element_active = False

        with open(filename, 'r') as infile:
            file_data = infile.read().splitlines()

        # Loop over all lines in the file
        for line in file_data:
            # Filter out comments
            if comment_tag in line:
                element_active = False
                continue

            if long_format_tag in line:  # Long format
                s = [line[:8], ]
                s.extend([line[i*16:(i+1)*16] for i in range(len(line)//16)])
                # Note: here some more logics is necessary to handle line
                # continuation
            elif ',' in line:  # Free field format
                s = line.split(',')
            else:  # The regular short format
                s = [line[i*8:(i+1)*8] for i in range(len(line)//8)]

            if len(s) < 1:  # Empty line
                element_active = False
                continue

            # Get the nodes
            if 'GRID' in s[0]:
                nodes_list.append([int(s[1]),
                                   float(s[3]), float(s[4]), float(s[5])])

            elif s[0].strip() in nas2amfe:
                elements_list.append(s)
                element_active = 'Element'
            elif 'RBE' in s[0]:
                constraint_list.append(s)
                element_active = 'Constraint'
            elif s[0] != '        ':  # There is an unknown element
                element_active = False

            # Catch the free lines where elements are continued
            elif s[0] == '        ' and element_active:
                if element_active == 'Element':
                    elements_list[-1].extend(s[1:])
                if element_active == 'Constraint':
                    constraint_list[-1].extend(s[1:])

        self.no_of_dofs_per_node = 3 # this is just hard coded right now...
        self.nodes = np.array(nodes_list, dtype=float)[:,1:]
        self.nodes *= scale_factor # scaling of nodes

        nodes_dict = pd.Series(index=np.array(nodes_list, dtype=int)[:,0],
                               data=np.arange(len(nodes_list)))

        for idx, ptr in enumerate(elements_list):
            tmp = [nas2amfe[ptr.pop(0).strip()],
                   int(ptr.pop(0)), int(ptr.pop(0))]
            tmp.extend([nodes_dict[int(i)] for i in ptr])
            elements_list[idx] = tmp

        for idx, ptr in enumerate(constraint_list):
            tmp = [ptr.pop(0).strip(), int(ptr.pop(0)), int(ptr.pop(1))]
            tmp.append([nodes_dict[int(i)] for i in ptr[4:]])
            constraint_list[idx] = tmp

        self.constraint_list = constraint_list
        self.el_df = df = pd.DataFrame(elements_list, dtype=int)
        df.rename(copy=False, inplace=True,
                  columns={0 : 'el_type',
                           1 : 'idx_nastran',
                           2 : 'phys_group',
                          })
        self.node_idx = 3
        self._update_mesh_props()
        # printing some information regarding the physical groups
        print('Mesh', filename, 'successfully imported.',
              '\nAssign a material to a physical group.')
        print('*************************************************************')
        return


    def import_msh(self, filename, scale_factor=1.):
        '''
        Import a gmsh-mesh.
        
        This method sets the following properties:
            - el_df: Element Definitions as pandas Dataframe
                    (Attention! This property is not the property that defines
                    the elements for later calculation. This is located in
                    the connectivity property)
            - node_idx: First Column in el_df pandas dataframe where the first
                    node-id of each element is stored
            - no_of_dofs_per_node: important to recognize 2D vs. 3D problem
            - nodes: Node Definitions (Locations)

        Parameters
        ----------
        filename : string
            filename of the .msh-file
        scale_factor : float, optional
            scale factor for the mesh to adjust the units. The default value is
            1, i.e. no scaling is done.

        Returns
        -------
        None

        Notes
        -----
        The internal representation of the elements is done via a Pandas Dataframe
        object. This gives the possibility to dynamically choose a part of the mesh
        for boundary conditons etc.

        '''
        tag_format_start   = "$MeshFormat"
        tag_format_end     = "$EndMeshFormat"
        tag_nodes_start    = "$Nodes"
        tag_nodes_end      = "$EndNodes"
        tag_elements_start = "$Elements"
        tag_elements_end   = "$EndElements"

        print('\n*************************************************************')
        print('Loading gmsh-mesh from', filename)

        with open(filename, 'r') as infile:
            data_geometry = infile.read().splitlines()

        for s in data_geometry:
            if s == tag_format_start: # Start Formatliste
                i_format_start   = data_geometry.index(s) + 1
            elif s == tag_format_end: # Ende Formatliste
                i_format_end     = data_geometry.index(s)
            elif s == tag_nodes_start: # Start Knotenliste
                i_nodes_start    = data_geometry.index(s) + 2
                n_nodes          = int(data_geometry[i_nodes_start-1])
            elif s == tag_nodes_end: # Ende Knotenliste
                i_nodes_end      = data_geometry.index(s)
            elif s == tag_elements_start: # Start Elementliste
                i_elements_start = data_geometry.index(s) + 2
                n_elements       = int(data_geometry[i_elements_start-1])
            elif s == tag_elements_end: # Ende Elementliste
                i_elements_end   = data_geometry.index(s)

        # Check inconsistent dimensions
        if (i_nodes_end-i_nodes_start)!=n_nodes \
            or (i_elements_end-i_elements_start)!= n_elements:
            raise ValueError('Error while processing the file!',
                             'Dimensions are not consistent.')

        # extract data from file to lists
        list_imported_mesh_format = data_geometry[i_format_start : i_format_end]
        list_imported_nodes = data_geometry[i_nodes_start : i_nodes_end]
        list_imported_elements = data_geometry[i_elements_start : i_elements_end]

        # conversion of the read strings to integer and floats
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in
                                            list_imported_mesh_format[j].split()]
        for j in range(len(list_imported_nodes)):
            list_imported_nodes[j] = [float(x) for x in
                                      list_imported_nodes[j].split()]
        for j in range(len(list_imported_elements)):
            list_imported_elements[j] = [int(x) for x in
                                         list_imported_elements[j].split()]

        # Construct Pandas Dataframe for the elements (self.el_df and df for shorter code)
        self.el_df = df = pd.DataFrame(list_imported_elements)
        df.rename(copy=False, inplace=True,
                  columns={0 : 'idx_gmsh',
                           1 : 'el_type',
                           2 : 'no_of_tags',
                           3 : 'phys_group',
                           4 : 'geom_entity'})

        # determine the index, where the nodes of the element start in the dataframe
        if len(df[df['no_of_tags'] != 2]) == 0:
            self.node_idx = node_idx = 5
        elif len(df[df['no_of_tags'] != 4]) == 0:
            df.rename(copy=False, inplace=True,
                 columns={5 : 'no_of_mesh_partitions', 6 : 'mesh_partition'})
            self.node_idx = node_idx = 7
        else:
            raise('''The type of mesh is not supported yet.
        Either you have a corrupted .msh-file or you have a too
        complicated mesh partition structure.''')

        # correct the issue wiht gmsh index starting at 1 and amfe index starting with 0
        df.iloc[:, node_idx:] -= 1

        # change the el_type to the amfe convention
        df['el_type'] = df.el_type.map(gmsh2amfe)

        element_types = pd.unique(df['el_type'])
        # Check, if the problem is 2d or 3d and adjust the dimension of the nodes
        # Check, if there is one 3D-Element in the mesh!
        self.no_of_dofs_per_node = 2
        for i in element_types:
            if i in self.element_3d_set:
                self.no_of_dofs_per_node = 3

        # fill the nodes to the array
        self.nodes = np.array(list_imported_nodes)[:,1:1+self.no_of_dofs_per_node]
        self.nodes *= scale_factor # scaling of nodes

        # Change the indices of Tet10-elements, as they are numbered differently
        # from the numbers used in AMFE and ParaView (last two indices permuted)
        if 'Tet10' in element_types:
            row_loc = df['el_type'] == 'Tet10'
            i = self.node_idx
            df.ix[row_loc, i + 9], df.ix[row_loc, i + 8] = \
            df.ix[row_loc, i + 8], df.ix[row_loc, i + 9]
        # Same node nubmering issue with Hexa20
        if 'Hexa20' in element_types:
            row_loc = df['el_type'] == 'Hexa20'
            hexa8_gmsh_swap = np.array([0,1,2,3,4,5,6,7,8,11,13,9,16,18,19,
                                        17,10,12,14,15])
            i = self.node_idx
            df.ix[row_loc, i:] = df.ix[row_loc, i + hexa8_gmsh_swap].values

        self._update_mesh_props()
        # printing some information regarding the physical groups
        print('Mesh', filename, 'successfully imported.',
              '\nAssign a material to a physical group.')
        print('*************************************************************')
        return


    def load_group_to_mesh(self, key, material, mesh_prop='phys_group'):
        '''
        Add a physical group to the main mesh with given material.
        
        It generates the connectivity list (mesh-class-property connectivity)
        which contains the element configuration as array
        and provides a map with pointers to Element-Objects (Tet, Hex etc.)
        which already contain information about material that is passed.
        Each element gets a pointer to such an element object.

        Parameters
        ----------
        key : int
            Key for mesh property which is to be chosen. Matches the group given
            in the gmsh file. For help, the function mesh_information or
            boundary_information gives the groups
        material : Material class
            Material class which will be assigned to the elements
        mesh_prop : {'phys_group', 'geom_entity', 'el_type'}, optional
            label of which the element should be chosen from. Standard is
            physical group.

        Returns
        -------
        None

        '''
        # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        if mesh_prop not in df.columns:
            print('The given mesh property "' + str(mesh_prop) + '" is not valid!',
                  'Please enter a valid mesh prop from the following list:\n')
            for i in df.columns:
                print(i)
            return
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information(mesh_prop)
            print('\nNo valid', mesh_prop, 'is given.\n(Given', mesh_prop,
                  'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop + ' to be used as mesh: '))

        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]

        # add the nodes of the chosen group
        connectivity = [np.nan for i in range(len(elements_df))]
        for i, ele in enumerate(elements_df.values):
            no_of_nodes = amfe2no_of_nodes[elements_df.el_type.iloc[i]]
            connectivity[i] = np.array(ele[self.node_idx :
                                           self.node_idx + no_of_nodes],
                                       dtype=int)

        self.connectivity.extend(connectivity)

        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(self.element_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].material = material
        object_series = elements_df.el_type.map(ele_class_dict)
        self.ele_obj.extend(object_series.values.tolist())
        self._update_mesh_props()

        # print some output stuff
        print('\n', mesh_prop, key, 'with', len(connectivity), \
              'elements successfully added.')
        print('Total number of elements in mesh:', len(self.ele_obj))
        print('*************************************************************')
        print('! Attention: If you have called load group to mesh() directly,\n'
              'do not forget to add a material observer of your mechanical system to the new added materials,'
              'if needed')

    def tie_mesh(self, master_key, slave_key, master_prop='phys_group',
                 slave_prop='phys_group', tying_type='fixed', robustness=4,
                 verbose=False, conform_slave_mesh=True, fix_mesh_dist=1E-3):
        '''
        Tie nonconforming meshes for a given master and slave side.


        Parameters
        ----------
        master_key : int or string
            mesh key of the master face mesh. The master face mesh has to be at
            least the size of the slave mesh. It is better, when the master
            mesh is larger than the slave mesh.
        slave_key : int or string
            mesh key of the slave face mesh or point cloud
        master_prop : string, optional
            mesh property for which master_key is specified.
            Default value: 'phys_group'
        slave_prop : string, optional
            mesh property for which slave_key is specified.
            Default value: 'phys_group'
        tying_type : string {'fixed', 'slide'}
            Mesh tying type. 'fixed' glues the meshes together while 'slide'
            allows for a sliding motion between the meshes.
        robustness : int, optional
            Integer value indicating, how many master elements should be
            considered for one slave node.

        Returns
        -------
        slave_dofs : ndarray, type: int
            slave dofs of the tied mesh
        row : ndarray, type: int
            row indices of the triplet description of the master slave
            conversion
        col : ndarray, type: int
            col indices of the triplet description of the master slave
            conversion
        val : ndarray, type: float
            values of the triplet description of the master slave conversion

        Notes
        -----
        The master mesh has to embrace the full slave mesh. If this is not the
        case, the routine will fail, a slave point outside the master mesh
        cannot be addressed to a specific element.

        '''
        df = self.el_df
        master_elements = df[df[master_prop]  == master_key]
        slave_elements = df[df[slave_prop]  == slave_key]

        master_nodes = master_elements.iloc[:, self.node_idx:].values
        master_obj =  master_elements.el_type.values
        slave_nodes = np.unique(slave_elements.iloc[:,self.node_idx:].values)
        slave_nodes = np.array(slave_nodes[np.isfinite(slave_nodes)], dtype=int)
        slave_dofs, row, col, val = master_slave_constraint(master_nodes,
            master_obj, slave_nodes, nodes=self.nodes, tying_type=tying_type,
            robustness=robustness, verbose=verbose,
            conform_slave_mesh=conform_slave_mesh, fix_mesh_dist=fix_mesh_dist)

        print('*'*80)
        print(('Tied mesh part {0} as master mesh to part {1} as slave mesh. \n'
              + 'In total {2} slave dofs were tied using the tying type {3}.'
              + '').format(master_key, slave_key, len(slave_dofs), tying_type)
             )
        print('*'*80)
        return (slave_dofs, row, col, val)


    def mesh_information(self, mesh_prop='phys_group'):
        '''
        Print some information about the mesh that is being imported
        Attention: This information is not about the mesh that is already
        loaded for further calculation. Instead it is about the mesh that is
        found in an import-file!

        Parameters
        ----------
        mesh_prop : str, optional
            mesh property of the loaded mesh. This mesh property is the basis
            for selection of parts of the mesh for materials and boundary
            conditions. The default value is 'phys_group' which is the physical
            group, if the mesh comes from gmsh.

        Returns
        -------
        None

        '''
        df = self.el_df
        if mesh_prop not in df.columns:
            print('The given mesh property "' + str(mesh_prop) + '" is not valid!',
                  'Please enter a valid mesh prop from the following list:\n')
            for i in df.columns:
                print(i)
            return

        phys_groups = pd.unique(df[mesh_prop])
        print('The loaded mesh contains', len(phys_groups),
              'physical groups:')
        for i in phys_groups:
            print('\nPhysical group', i, ':')
            # print('Number of Nodes:', len(self.phys_group_dict [i]))
            print('Number of Elements:', len(df[df[mesh_prop] == i]))
            print('Element types appearing in this group:',
                  pd.unique(df[df[mesh_prop] == i].el_type))

        return


    def set_neumann_bc(self, key, val, direct, time_func=None,
                       shadow_area=False,
                       mesh_prop='phys_group'):
        '''
        Add group of mesh to neumann boundary conditions.

        Parameters
        ----------
        key : int
            Key of the physical domain to be chosen for the neumann bc
        val : float
            value for the pressure/traction onto the element
        direct : str 'normal' or ndarray
            array giving the direction, in which the traction force should act.
            alternatively, the keyword 'normal' may be given. Default value:
            'normal'.
        time_func : function object
            Function object returning a value between -1 and 1 given the
            input t:

            >>> val = time_func(t)

        shadow_area : bool, optional
            Flat setting, if force should be proportional to the shadow area,
            i.e. the area of the surface projected on the direction. Default
            value: 'False'.
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            label of which the element should be chosen from. Default is
            phys_group.

        Returns
        -------
        None
        '''
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information(mesh_prop)
            print('\nNo valid', mesh_prop, 'is given.\n(Given',
                  mesh_prop, 'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop +
                            ' to be used for the Neumann Boundary conditions: '))

        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]
        ele_type = elements_df['el_type'].values
        # add the nodes of the chosen group
        nm_connectivity = [np.nan for i in range(len(elements_df))]

        for i, ele in enumerate(elements_df.values):
            nm_connectivity[i] = np.array(ele[self.node_idx : self.node_idx
                                          + amfe2no_of_nodes[ele_type[i]]],
                                          dtype=int)

        self.neumann_connectivity.extend(nm_connectivity)

        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(self.element_boundary_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].__init__(val=val, direct=direct,
                                       time_func=time_func,
                                       shadow_area=shadow_area)

        object_series = elements_df['el_type'].map(ele_class_dict)
        self.neumann_obj.extend(object_series.values.tolist())
        # self._update_mesh_props() <- old implementation: not necessary!

        # print some output stuff
        print('\n', mesh_prop, key, 'with', len(nm_connectivity),
              'elements successfully added to Neumann Boundary.')
        print('Total number of neumann elements in mesh:', len(self.neumann_obj))
        print('Total number of elements in mesh:', len(self.ele_obj))
        print('*************************************************************')


    def set_dirichlet_bc(self, key, coord, mesh_prop='phys_group',
                         output='internal'):
        '''
        Add a group of the mesh to the dirichlet nodes to be fixed. It sets the
        mesh-properties 'nodes_dirichlet' and 'dofs_dirichlet'

        Parameters
        ----------
        key : int
            Key for mesh property which is to be chosen. Matches the group given
            in the gmsh file. For help, the function mesh_information or
            boundary_information gives the groups
        coord : str {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'}
            coordinates which should be fixed
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            label of which the element should be chosen from. Default is
            phys_group.
        output : str {'internal', 'external'}
            key stating, boundary information is stored internally or externally


        Returns
        -------
        nodes : ndarray, if output == 'external'
            Array of nodes belonging to the selected group
        dofs : ndarray, if output == 'external'
            Array of dofs respecting the coordinates belonging to the selected
            groups

        '''
        # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information(mesh_prop)
            print('\nNo valid', mesh_prop, 'is given.\n(Given', mesh_prop,
                  'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop +
                            ' to be chosen for Dirichlet BCs: '))

        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]
        # pick the nodes, make them unique and remove NaNs
        all_nodes = elements_df.iloc[:, self.node_idx:]
        unique_nodes = np.unique(all_nodes.values.reshape(-1))
        unique_nodes = unique_nodes[np.isfinite(unique_nodes)]

        # build the dofs_dirichlet, a list containing the dirichlet dofs:
        dofs_dirichlet = []
        if 'x' in coord:
            dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node)
        if 'y' in coord:
            dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node + 1)
        if 'z' in coord and self.no_of_dofs_per_node > 2:
            # TODO: Separate second if and throw error or warning
            dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node + 2)

        dofs_dirichlet = np.array(dofs_dirichlet, dtype=int)

        # TODO: Folgende Zeilen sind etwas umstaendlich, erst conversion to list, dann extend und dann zurueckconversion
        nodes_dirichlet = unique_nodes
        # nodes_dirichlet = self.nodes_dirichlet.tolist()
        # nodes_dirichlet.extend(unique_nodes)
        # nodes_dirichlet = np.array(nodes_dirichlet, dtype=int)

        if output is 'internal':
            dofs_dirichlet = np.append(dofs_dirichlet, self.dofs_dirichlet)
            self.dofs_dirichlet = np.unique(dofs_dirichlet)

            nodes_dirichlet = np.append(nodes_dirichlet, self.nodes_dirichlet)
            self.nodes_dirichlet = np.unique(nodes_dirichlet)

        # print some output stuff
        print('\n', mesh_prop, key, 'with', len(unique_nodes),
              'nodes successfully added to Dirichlet Boundaries.')
        print('Total number of nodes with Dirichlet BCs:', len(self.nodes_dirichlet))
        print('Total number of constrained dofs:', len(self.dofs_dirichlet))
        print('*************************************************************')
        if output is 'external':
            return nodes_dirichlet, dofs_dirichlet

    def deflate_mesh(self):
        '''
        Deflate the mesh, i.e. delete nodes which are not connected to an
        element.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        nodes_vec = np.concatenate(self.connectivity)
        elements_on_node = np.bincount(np.array(nodes_vec, dtype=int))
        mask = np.zeros(self.nodes.shape[0], dtype=bool)
        # all nodes which show up at least once
        mask[:len(elements_on_node)] = elements_on_node != 0
        idx_transform = np.zeros(len(self.nodes), dtype=int)
        idx_transform[mask] = np.arange(len(idx_transform[mask]))
        self.nodes = self.nodes[mask]
        # deflate the connectivities
        for i, nodes in enumerate(self.connectivity):
            self.connectivity[i] = idx_transform[nodes]
        for i, nodes in enumerate(self.neumann_connectivity):
            self.neumann_connectivity[i] = idx_transform[nodes]

        # deflate the element_dataframe
        df = self.el_df
        for col in df.iloc[:,self.node_idx:]:
            nan_mask = np.isfinite(df[col].values)
            indices = np.array(df[col].values[nan_mask], dtype=int)
            df[col].values[nan_mask] = idx_transform[indices]

        self._update_mesh_props()
        print('**************************************************************')
        print('Mesh successfully deflated. ',
              '\nNumber of nodes in old mesh:', len(mask),
              '\nNumber of nodes in deflated mesh:', np.count_nonzero(mask),
              '\nNumer of deflated nodes:', len(mask) - np.count_nonzero(mask))
        print('**************************************************************')


    def save_mesh_xdmf(self, filename, field_list=None, bmat=None, u=None, timesteps=None):
        '''
        Save the mesh in hdf5 and xdmf file format.

        Parameters
        ----------
        filename : str
            String consisting the path and the filename
        field_list : list
            list containing the fields to be exported. The list is a list of
            tupels containing the array with the values in the columns and a
            dictionary with the attribute information:

                >>> # example field list with reduced displacement not to export
                >>> # ParaView and strain epsilon to be exported to ParaView
                >>> field_list = [(q_red, {'ParaView':False, 'Name':'q_red'}),
                                  (eps, {'ParaView':True,
                                         'Name':'epsilon',
                                         'AttributeType':'Tensor6',
                                         'Center':'Node',
                                         'NoOfComponents':6})]
        bmat : csrMatrix
            CSR-Matrix describing the way, how the Dirichlet-BCs are applied:
            u_unconstr = bmat @ u_constr

        u : nparray
            matrix with displacement vectors as columns for different timesteps
            
        timesteps : nparray
            vector with timesteps a displacement vector is stored in u


        Returns
        -------
        None

        Notes
        -----
        Only one homogeneous mesh is exported. Thus only the mesh made of the
        elements which occur most often is exported. The other meshes are
        discarded.

        '''
        # generate a zero displacement if no displacements are passed.
        if not u or not timesteps:
            u = [np.zeros((self.no_of_nodes * self.no_of_dofs_per_node,)),]
            timesteps = np.array([0])


        # determine the part of the mesh which has most elements
        # only this part will be exported!
        ele_types = np.array([obj.name for obj in self.ele_obj], dtype=object)
        el_type_export = np.unique(ele_types)
        connectivties_dict = dict()
        for el_type in el_type_export:
            # Boolean matrix giving the indices for the elements to export
            el_type_ix = (ele_types == el_type)

            # select the nodes to export an make an array of them
            # As the list might be ragged, it has to be put to list and then to
            # array again.
            connectivity_export = np.array(self.connectivity)[el_type_ix]
            connectivity_export = np.array(connectivity_export.tolist())
            connectivties_dict[el_type] = connectivity_export

        # make displacement 3D vector, as paraview only accepts 3D vectors
        q_array = np.array(u, dtype=float).T
        if self.no_of_dofs_per_node == 2:
            tmp_2d = q_array.reshape((self.no_of_nodes,2,-1))
            x, y, z = tmp_2d.shape
            tmp_3d = np.zeros((x,3,z))
            tmp_3d[:,:2,:] = tmp_2d
            q_array = tmp_3d.reshape((-1,z))

        h5_q_dict = {'ParaView':True,
                     'AttributeType':'Vector',
                     'Center':'Node',
                     'Name':'Displacement',
                     'NoOfComponents':3}

        h5_time_dict = {'ParaView':True,
                        'Name':'Time'}

        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()

        new_field_list.append((q_array, h5_q_dict))

        check_dir(filename)

        # write the hdf5 file with the necessary attributes
        with h5py.File(filename + '.hdf5', 'w') as f:
            # export mesh with nodes and topology
            h5_nodes = f.create_dataset('mesh/nodes', data=self.nodes)
            h5_nodes.attrs['ParaView'] = True
            for el_type in connectivties_dict:
                h5_topology = f.create_dataset('mesh/topology/' + el_type,
                                               data=connectivties_dict[el_type],
                                               dtype=np.int)
                h5_topology.attrs['ParaView'] = True
                h5_topology.attrs['TopologyType'] = amfe2xmf[el_type]

            # export timesteps
            h5_time = f.create_dataset('time', data=np.array(timesteps))
            h5_set_attributes(h5_time, h5_time_dict)

            # export bmat if given
            if bmat is not None:
                h5_bmat = f.create_group('mesh/bmat')
                h5_bmat.attrs['ParaView'] = False
                for par in ('data', 'indices', 'indptr', 'shape'):
                    array = np.array(getattr(bmat, par))
                    h5_bmat.create_dataset(par, data=array, dtype=array.dtype)

            # export fields in new_field_list
            for data_array, data_dict in new_field_list:

                # consider heterogeneous meshes:
                # export Cell variables differently than other time variables
                export_cell = False
                if 'Center' in data_dict:
                    if data_dict['Center'] == 'Cell':
                        export_cell = True

                if export_cell:
                    for el_type in connectivties_dict:
                        # Boolean matrix giving the indices for the elements to export
                        el_type_ix = (ele_types == el_type)
                        location = 'time_vals_cell/{}/{}'.format(
                                data_dict['Name'], el_type)
                        h5_dataset = f.create_dataset(location,
                                                      data=data_array[el_type_ix])
                        h5_set_attributes(h5_dataset, data_dict)
                else:
                    h5_dataset = f.create_dataset('time_vals/' + data_dict['Name'],
                                                  data=data_array)
                    h5_set_attributes(h5_dataset, data_dict)

        # Create the xdmf from the hdf5 file
        create_xdmf_from_hdf5(filename + '.hdf5')
        return

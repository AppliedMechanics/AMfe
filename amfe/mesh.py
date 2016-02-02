# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:25:24 2015

@author: johannesr
"""
import os
import copy
# XML stuff
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom



import pandas as pd
import h5py
import numpy as np

from amfe.element import Tet4, Tet10, Tri3, Tri6, Quad4, Quad8, Bar2Dlumped
from amfe.element import LineLinearBoundary, LineQuadraticBoundary, \
    Tri3Boundary, Tri6Boundary

# Element mapping is described here. If a new element is implemented, the
# features for import and export should work when the followig list will be updated.
element_mapping_list = [
    # internal Name, XMF Key,   gmsh-Key, vtk/ParaView-Key, no_of_nodes, description
    ['Tet4',          'Tetrahedron',   4, 10,  4, 
     'Linear Tetraeder / nodes on every corner'],
    ['Tet10',         'Tetrahedron_10',  11, 24, 10, 
     'Quadratic Tetraeder / 4 nodes at the corners, 6 nodes at the faces'],
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
    ['straight_line', 'Edge',   1,  3,  2, 
     'Straight line composed of 2 nodes'],
    ['quadratic_line', 'Edge_3',  8, 21,  3, 
     'Quadratic edge/line composed of 3 nodes'],
    ['point',       '', 15, np.NAN,  1, 'Single Point'],    
    # Bars are missing, which are used for simple benfield truss
]

# Element Class dictionary with all available elements
kwargs = { }
element_class_dict = {'Tet4'  : Tet4(**kwargs),
                      'Tet10' : Tet10(**kwargs),
                      'Tri3'  : Tri3(**kwargs),
                      'Tri6'  : Tri6(**kwargs),
                      'Quad4' : Quad4(**kwargs),
                      'Quad8' : Quad8(**kwargs),
                      'Bar2Dlumped' : Bar2Dlumped(**kwargs),
                     }
                              
kwargs = {'val' : 1., 'direct' : 'normal'}

element_boundary_class_dict = {'straight_line' : LineLinearBoundary(**kwargs), 
                               'quadratic_line': LineQuadraticBoundary(**kwargs),
                               'Tri3'          : Tri3Boundary(**kwargs),
                               'Tri6'          : Tri6Boundary(**kwargs),
                              }
                              
# actual set of implemented elements
element_2d_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8', }
element_3d_set = {'Tet4', 'Tet10'}

boundary_2d_set = {'straight_line', 'quadratic_line'}
boundary_3d_set = {'straight_line', 'quadratic_line',
                   'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}

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



def check_dir(*filenames):
    '''Checkt ob Verzeichnis vorliegt; falls nicht, wird Verzeichnis angelegt'''
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
            grid = SubElement(time_grid, 'Grid', {'Type':'Uniform'})
            
            time = SubElement(grid, 'Time', {'TimeType':'Single', 
                                             'Value':str(T)})
            topology = SubElement(grid, 'Topology', 
                                  {'TopologyType':h5_topology.attrs['TopologyType'],
                                   'NumberOfElements':str(h5_topology.shape[0])})
            topology_data = SubElement(topology, 'DataItem', 
                                       {'NumberType':'Int', 
                                        'Format':'HDF', 
                                        'Dimensions':shape2str(h5_topology.shape)})
            topology_data.text = filename_no_dir + ':/mesh/topology'
            
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
            for key in h5_time_vals.keys():
                field = h5_time_vals[key]
                if field.attrs['ParaView']:
                    field_attr = SubElement(grid, 'Attribute',
                                            {'Name':field.attrs['Name'],
                                             'AttributeType':field.attrs['AttributeType'],
                                             'Center':field.attrs['Center']})
                    no_of_components = field.attrs['NoOfComponents']
                    field_dim = (field.shape[0] // no_of_components, no_of_components)
                    field_data = SubElement(field_attr, 'DataItem', 
                                            {'ItemType':'HyperSlab', 
                                             'Dimensions':shape2str(field_dim)})
                                             
                    field_hyperslab = SubElement(field_data, 'DataItem', 
                                                 {'Dimensions':'3 2', 
                                                  'Format':'XML'})
            
                    # pick the i-th column via hyperslab 
                    field_hyperslab.text = '0 ' + str(i) + ' 1 1 ' + \
                                            str(field.shape[0]) + ' 1'
                    field_hdf = SubElement(field_data, 'DataItem', 
                                           {'Format':'HDF', 
                                            'NumberType':'Float', 
                                            'Dimensions':shape2str(field.shape)})
                    field_hdf.text = filename_no_dir + ':/time_vals/' + key

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
        Array of x-y-z coordinates of the nodes. Dimension is 
        (no_of_nodes, no_of_dofs_per_node). 
    ele_nodes : list
        List of nodes indices belonging to one element. 
    ele_obj : list
        List of element objects. The list contains actually only the pointers
        pointing to the element object
    ele_types : list
        List of strings containing the element types. Basically used for export
        to postprocessing tools. 
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
        Number of dofs of the whole system (ignoring constrained dofs). 
        
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
        self.nodes         = np.array([])
        self.ele_nodes     = []
        self.ele_obj       = []
        self.ele_types     = [] # Element-types for Export
        self.nodes_dirichlet     = np.array([], dtype=int)
        self.dofs_dirichlet      = np.array([], dtype=int)
        # the displacements; They are stored as a list of numpy-arrays with 
        # shape (ndof, no_of_dofs_per_node):
        self.u                   = []
        self.timesteps           = []
        self.no_of_dofs_per_node = 0
        self.no_of_dofs = 0
        self.no_of_nodes = 0
        self.no_of_elements = 0

    def _update_mesh_props(self):
        '''
        Update the number properties of nodes and elements when the mesh has 
        '''
        self.no_of_nodes = len(self.nodes)
        self.no_of_dofs = self.no_of_nodes*self.no_of_dofs_per_node
        self.no_of_elements = len(self.ele_nodes)


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
        #######################################################################
        # NODES
        #######################################################################
        try:
            self.nodes = np.genfromtxt(filename_nodes, delimiter = ',', skip_header = 1)
        except:
            print('FEHLER beim lesen der Datei', filename_nodes, 
                  '\nVermutlich stimmt die erwartete Dimension der Knotenfreiheitsgrade', 
                  self.no_of_dofs_per_node, 'nicht mit der Dimension in der Datei zusammen.')
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
        self.ele_nodes = np.genfromtxt(filename_elements, delimiter = ',', dtype = int, skip_header = 1)
        if self.ele_nodes.ndim == 1: # Wenn nur genau ein Element vorliegt
            self.ele_nodes = np.array([self.ele_nodes])
        # Falls erste Spalte die Elementnummer angibt, wird diese hier 
        # abgeschnitten, um nur die Knoten des Elements zu erhalten
        if explicit_node_numbering: 
            self.ele_nodes = self.ele_nodes[:,1:]

    
        if ele_type: # If element type is spezified, use this spezified type
            mesh_type = ele_type
        # If element type is not spzezified, try to determine element type 
        # depending on the number of nodes per element (see default values for 
        # different number of nodes per element in 'mesh_type_dict')
        else: 
            try: # Versuche Elementtyp an Hand von Anzahl der Knoten pro Element auszulesen
                (no_of_ele, no_of_nodes_per_ele) = self.ele_nodes.shape
                mesh_type = mesh_type_dict[no_of_nodes_per_ele] # Weise Elementtyp zu
            except:
                print('FEHLER beim Einlesen der Elemente. Typ nicht vorhanden.')
                raise

        print('Element type is {0}...  '.format(mesh_type), end="")
        # Hier wird davon ausgegangen, dass genau ein Elementtyp verwendet 
        # wurde, welcher jedem Eintrag des 'element_type'-Vektors zugewiesen wird
        self.ele_types = [mesh_type for i in self.ele_nodes] 
        self._update_mesh_props()
        print('Reading elements successful.')

    def import_msh(self, filename):
        '''
        Import a gmsh-mesh. 
        
        Parameters
        ----------
        filename : string
            filename of the .msh-file
            
        Returns
        -------
        None
        
        Notes
        -----
        The internal representation of the elements is done via a Pandas Dataframe
        object. This gives the possibility to dynamically choose an part of the mesh
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
        list_imported_mesh_format = data_geometry[i_format_start   : i_format_end]
        list_imported_nodes       = data_geometry[i_nodes_start    : i_nodes_end]
        list_imported_elements    = data_geometry[i_elements_start : i_elements_end]
        
        # conversion of the read strings to integer and floats
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in list_imported_mesh_format[j].split()]
        for j in range(len(list_imported_nodes)):
            list_imported_nodes[j] = [float(x) for x in list_imported_nodes[j].split()]
        for j in range(len(list_imported_elements)):
            list_imported_elements[j] = [int(x) for x in list_imported_elements[j].split()]
        
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
            if i in element_3d_set:
                self.no_of_dofs_per_node = 3
                
        # fill the nodes of the selected physical group to the array
        self.nodes = np.array(list_imported_nodes)[:,1:1+self.no_of_dofs_per_node]

        
        # Handling the physical groups 
        all_physical_groups = pd.unique(df.phys_group)
        
        # make a dictionary with the nodes of every physical group
        self.phys_group_dict = nodes_phys_group = {}
        for idx in all_physical_groups:
            gr_nodes = np.array([], dtype=int)
            # pick the elements corresponding to the current physical group from table
            df_phys_group = df[df.phys_group == idx] 
            # assemble all nodes to one huge array
            for series in df_phys_group.iloc[:, node_idx:]:
                gr_nodes = np.append(gr_nodes, df_phys_group[series].unique())
            # make them unique, remove nan (resulting from non-existing entries in pandas)
            # cast and sort the array and put into dict
            gr_nodes = np.unique(gr_nodes)
            gr_nodes = gr_nodes[np.isfinite(gr_nodes)] # remove nan from non-existing entries
            gr_nodes = np.array(gr_nodes, dtype=int) # recast to int as somewhere a float is casted
            gr_nodes.sort()
            nodes_phys_group[idx] = gr_nodes

        self._update_mesh_props()
        # printing some information regarding the physical groups
        print('Mesh', filename, 'successfully imported. \nAssign a material to a physical group.')
        print('*************************************************************')
    
        

    def load_group_to_mesh(self, key, material, mesh_prop='phys_group',
                           element_class_dict=element_class_dict):
        '''
        Add a physical group to the main mesh with given material. 
        
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
        element_class_dict : dict, optional
            Dictionary of elements, where the element keys are mapped to the 
            element objects. 
            
        Returns
        -------
        None
        
        '''
        # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information()
            print('\nNo valid', mesh_prop, 'is given.\n(Given', mesh_prop, 'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop + ' to be used as mesh: '))
        
        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]
        
        # add the nodes of the chosen group
        ele_nodes = [np.nan for i in range(len(elements_df))]
        for i, element in enumerate(elements_df.values):
            ele_nodes[i] = np.array(element[self.node_idx : 
                                    self.node_idx + amfe2no_of_nodes[element[1]]],
                                    dtype=int)
        self.ele_nodes.extend(ele_nodes)
        
        # ele_types for paraview export
        self.ele_types.extend(elements_df['el_type'].values.tolist())
        
        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(element_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].material = material
        object_series = elements_df.el_type.map(ele_class_dict)
        self.ele_obj.extend(object_series.values.tolist())
        self._update_mesh_props()
        
        # print some output stuff
        print('\n', mesh_prop, key, 'with', len(ele_nodes), \
              'elements successfully added.')
        print('Total number of elements in mesh:', len(self.ele_obj))
        print('*************************************************************')
        
    def mesh_information(self):
        df = self.el_df
        print('The loaded mesh contains', len(self.phys_group_dict), 
              'physical groups:')
        for i in self.phys_group_dict :
            print('\nPhysical group', i, ':')
            print('Number of Nodes:', len(self.phys_group_dict [i]))
            print('Number of Elements:', len(df[df.phys_group == i]))
            print('Element types appearing in this group:', 
                  pd.unique(df[df.phys_group == i].el_type))

    def boundary_information(self):
        '''
        Print the information of the boundary stuff
        '''
        print('Voundary nodes sorted by the boundary number:')
        for i in self.phys_group_dict:
            print('Boundary (physical group)', i,
                  'contains the following', len(self.phys_group_dict[i]), 
                  ' nodes:\n', self.phys_group_dict[i])

    def select_neumann_bc(self, key, val, direct, time_func=None, 
                          mesh_prop='phys_group',
                          element_boundary_class_dict=element_boundary_class_dict):
        '''
        Add group of mesh to neumann boundary conditions. 
        
        Parameters
        ----------
        key : int
            Key of the physical domain to be chosen for the neumann bc
        val : float
            value for the pressure/traction onto the element
        direct : str {'normal', 'x_n', 'y_n', 'z_n', 'x', 'y', 'z'}
            direction, in which the traction should point at: 
            
            'normal'
                Pressure acting onto the normal face of the deformed configuration
            'x_n'
                Traction acting in x-direction proportional to the area 
            projected onto the y-z surface
            'y_n'
                Traction acting in y-direction proportional to the area 
                projected onto the x-z surface            
            'z_n'
                Traction acting in z-direction proportional to the area 
                projected onto the x-y surface
            'x'
                Traction acting in x-direction proportional to the area
            'y'
                Traction acting in y-direction proportional to the area
            'z'
                Traction acting in z-direction proportional to the area
            
        time_func : function object
            Function object returning a value between -1 and 1 given the 
            input t: 

            >>> val = time_func(t)
            
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            label of which the element should be chosen from. Default is 
            phys_group. 
        element_boundary_class_dict : dict
            Dictionary containing the skin elements. 
        Returns
        -------
        None
        '''
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information()
            print('\nNo valid', mesh_prop, 'is given.\n(Given', 
                  mesh_prop, 'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop + 
                            ' to be used for the Neumann Boundary conditions: '))
        
        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]
        
        # add the nodes of the chosen group
        ele_nodes = [np.nan for i in range(len(elements_df))]
        for i, element in enumerate(elements_df.values):
            ele_nodes[i] = np.array(element[self.node_idx : 
                                            self.node_idx + amfe2no_of_nodes[element[1]]],
                                    dtype=int)
        self.ele_nodes.extend(ele_nodes)
        
        self.ele_types.extend(elements_df['el_type'].values.tolist())
                
        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(element_boundary_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].__init__(val=val, direct=direct, time_func=time_func)
        object_series = elements_df['el_type'].map(ele_class_dict)
        self.ele_obj.extend(object_series.values.tolist())
        self._update_mesh_props()
        
        # print some output stuff
        print('\n', mesh_prop, key, 'with', len(ele_nodes), 
              'elements successfully added.')
        print('Total number of elements in mesh:', len(self.ele_obj))
        print('*************************************************************')

        
    def select_dirichlet_bc(self, key, coord, mesh_prop='phys_group', 
                            output='internal'):
        '''
        Add a group of the mesh to the dirichlet nodes to be fixed. 
        
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
            Array of dofs respecting the coordinates belonging to the selected groups
        
        '''
        # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information()
            print('\nNo valid', mesh_prop, 'is given.\n(Given', mesh_prop, 'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop + ' to be chosen for Dirichlet BCs: '))
        
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
            dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node + 2)
            
        dofs_dirichlet = np.array(dofs_dirichlet, dtype=int)
        
        nodes_dirichlet = self.nodes_dirichlet.tolist()
        nodes_dirichlet.extend(unique_nodes)
        nodes_dirichlet = np.array(nodes_dirichlet, dtype=int)

        if output is 'internal':
            dofs_dirichlet = np.append(dofs_dirichlet, self.dofs_dirichlet)
            self.dofs_dirichlet = np.unique(dofs_dirichlet)
            self.dofs_dirichlet.sort()
            self.nodes_dirichlet = np.unique(nodes_dirichlet)
            self.nodes_dirichlet.sort()
        
        # print some output stuff
        print('\n', mesh_prop, key, 'with', len(unique_nodes), 
              'nodes successfully added to Dirichlet Boundaries.')
        print('Total number of nodes with Dirichlet BCs:', len(self.nodes_dirichlet))
        print('Total number of constrained dofs:', len(self.dofs_dirichlet))
        print('*************************************************************')
        if output is 'external':
            return nodes_dirichlet, dofs_dirichlet
        

    def set_displacement(self, u):
        '''
        Sets a displacement to the given mesh.

        Parameters
        -----------
        u : ndarray
            displacement of the unconstrained system in voigt notation

        Returns
        --------
        None

        Examples
        ---------
        TODO
        '''
        self.timesteps.append(1)
        self.u.append(np.array(u))
        return


    def set_displacement_with_time(self, u, timesteps):
        '''
        Set the displacement of the mesh with the corresponding timesteps.

        expects for the timesteps a list containing the displacement vector in any shape.

        Parameters
        -----------
        u : list of ndarrays
            list containing the displacements as ndarrays in arbitrary shape
        timesteps : ndarray
            vector containing the time corresponding to the displacements in u

        Returns
        --------
        None

        Examples
        ---------
        TODO
        '''
        self.timesteps = timesteps.copy()
        self.u = []
        for i, timestep in enumerate(self.timesteps):
            self.u.append(np.array(u[i]))
        return


    def save_mesh_xdmf(self, filename):
        '''
        Save the mesh in hdf5 and xdmf file format. 
        
        Parameters
        ----------
        filename : str
            String constisting the path and the filename
        
        Returns
        -------
        None
                
        Note
        ----
        Only the mesh of these elements which occur most ofen are exported. 
        
        '''
        # generate a zero displacement if no displacements are saved. 
        if len(self.timesteps) == 0:
            self.u = [np.zeros((self.no_of_nodes * self.no_of_dofs_per_node,)), ]
            self.timesteps.append(0)

        # determine the part of the mesh which has most elements
        # only this part will be exported! 
        ele_types = np.array(self.ele_types, dtype=object)
        el_type_export = np.unique(ele_types)[0]
        # Boolean matrix giving the indices for the elements to export
        el_type_ix = (ele_types == el_type_export)

        # select the nodes to export an make an array of them
        ele_nodes_export = np.array(self.ele_nodes)[el_type_ix]
        ele_nodes_export = np.array(ele_nodes_export.tolist())
        
        # make displacement 3D vector, as paraview only accepts 3D vectors
        q_array = np.array(self.u, dtype=float).T
        if self.no_of_dofs_per_node == 2:
            tmp_2d = q_array.reshape((self.no_of_nodes,2,-1))
            x, y, z = tmp_2d.shape
            tmp_3d = np.zeros((x,3,z))
            tmp_3d[:,:2,:] = tmp_2d
            q_array = tmp_3d.reshape((-1,z))
        
        h5_q_dict = {'ParaView':True, 
                     'AttributeType':'Vector', 
                     'Center':'Node', 
                     'Name':'Displacment', 
                     'NoOfComponents':3}
        
        h5_time_dict = {'ParaView':True, 
                        'Name':'Time'}
                        
        check_dir(filename)
        
        # write the hdf5 file with the necessary attributes        
        with h5py.File(filename + '.hdf5', 'w') as f:
            h5_nodes = f.create_dataset('mesh/nodes', data=self.nodes)
            h5_nodes.attrs['ParaView'] = True
            h5_topology = f.create_dataset('mesh/topology', 
                                           data=ele_nodes_export, 
                                           dtype=np.int)
            h5_topology.attrs['ParaView'] = True
            h5_topology.attrs['TopologyType'] = amfe2xmf[el_type_export]
            
            h5_displacement = f.create_dataset('time_vals/displacement', 
                                               data=q_array)
            h5_set_attributes(h5_displacement, h5_q_dict)
            
            h5_time = f.create_dataset('time', data=np.array(self.timesteps))
            h5_set_attributes(h5_time, h5_time_dict)
            
        # Create the xdmf from the hdf5 file
        create_xdmf_from_hdf5(filename + '.hdf5')
        
        return 
        
    def save_mesh_for_paraview(self, filename):
        '''
        Saves the mesh and the corresponding displacements to a .pvd file and 
        corresponding .vtu-files readable for ParaView.

        In the .pvd file the information of the timesteps are given. For every 
        timestep a .vtu file is created where the displacement and eventually 
        the stress is saved.

        Parameters
        -----------
        filename : str
            name of the file without file endings.

        Returns
        --------
        None

        Examples
        ---------
        >>> mymesh = Mesh()
        >>> mymesh.import_msh('../meshes/my_gmsh.msh')
        >>> mymesh.save_mesh_for_paraview('../results/gmsh/my_simulation')


        References
        ----------
        www.vtk.org/VTK/img/file-formats.pdf

        '''
        if len(self.timesteps) == 0:
            self.u = [np.zeros((self.no_of_nodes, self.no_of_dofs_per_node))]
            self.timesteps.append(0)

        # Make the pvd-File with the links to vtu-files
        pvd_header = '''<?xml version="1.0"?> \n <VTKFile type="Collection" 
version="0.1" byte_order="LittleEndian">  \n <Collection> \n '''
        pvd_footer = ''' </Collection> \n </VTKFile>'''
        pvd_line_start = '''<DataSet timestep="'''
        pvd_line_middle = '''" group="" part="0" file="'''
        pvd_line_end = '''"/>\n'''
        filename_pvd = filename + '.pvd'

        filename_head, filename_tail = os.path.split(filename)

        check_dir(filename_pvd)
        with open(filename_pvd, 'w') as savefile_pvd:
            savefile_pvd.write(pvd_header)
            for i, t in enumerate(self.timesteps):
                savefile_pvd.write(pvd_line_start + str(t) + pvd_line_middle + 
                                   filename_tail + '_' + str(i).zfill(3) + 
                                   '.vtu' + pvd_line_end)
            savefile_pvd.write(pvd_footer)

        vtu_header = '''<?xml version="1.0"?> \n
        <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
        <UnstructuredGrid>\n'''
        vtu_footer = '''
        </PointData>
        <CellData>
        </CellData>
        </Piece>
        </UnstructuredGrid>
        </VTKFile>'''
        for i, t in enumerate(self.timesteps):
            filename_vtu = filename + '_' + str(i).zfill(3) + '.vtu'
            check_dir(filename_vtu)
            with open(filename_vtu, 'w') as savefile_vtu:
                savefile_vtu.write(vtu_header)
                # Es muss die Anzahl der gesamten Punkte und Elemente angegeben werden
                savefile_vtu.write('<Piece NumberOfPoints="' + str(len(self.nodes)) 
                                   + '" NumberOfCells="' + str(len(self.ele_nodes)) 
                                   + '">\n')
                savefile_vtu.write('<Points>\n')
                savefile_vtu.write('<DataArray type="Float64" Name="Array" \
                    NumberOfComponents="3" format="ascii">\n')
                # bei Systemen mit 2 Knotenfreiheitsgraden wird die dritte 
                # 0-Komponenten noch extra durch die endflag hinzugefuegt...
                if self.no_of_dofs_per_node == 2:
                    endflag = ' 0 \n'
                elif self.no_of_dofs_per_node == 3:
                    endflag = '\n'
                for j in self.nodes:
                    savefile_vtu.write(' '.join(str(x) for x in list(j)) + endflag)
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write('</Points>\n<Cells>\n')
                savefile_vtu.write('<DataArray type="Int32" Name="connectivity" ',
                                   'format="ascii">\n')
                for j in self.ele_nodes:
                    savefile_vtu.write(' '.join(str(x) for x in list(j)) + '\n')
                savefile_vtu.write('\n</DataArray>\n')
                # Writing the offset for the elements; they are ascending by 
                # the number of dofs and have to start with the real integer
                savefile_vtu.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
                i_offset = 0
                for j, el_nodes in enumerate(self.ele_nodes):
                    i_offset += len(el_nodes)
                    savefile_vtu.write(str(i_offset) + ' ')
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write('<DataArray type="Int32" Name="types" format="ascii">\n')
                # Elementtyp ueber Zahl gesetzt
                savefile_vtu.write(' '.join(str(amfe2vtk[el_ty]) 
                                            for el_ty in self.ele_types)) 
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write('</Cells> \n')
                savefile_vtu.write('<PointData Vectors="displacement">\n')
                savefile_vtu.write('<DataArray type="Float64" Name="displacement" ',
                                   'NumberOfComponents="3" format="ascii">\n')
                # pick the i-th timestep
                u_exp = self.u[i].reshape((-1, self.no_of_dofs_per_node))
                for j in u_exp:
                    savefile_vtu.write(' '.join(str(x) for x in list(j)) + endflag)
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write(vtu_footer)
        return


class MeshGenerator:
    '''
    Klasse zum Erzeugen von zweidimensionalen Netzen, die Dreiecks- oder 
    Vierecksstruktur haben. Ausgabe in Netz-Files, die von der Netz-Klasse 
    wieder eingelesen werden können

    '''

    def __init__(self, x_len = 1, y_len = 1, x_no_elements = 2, 
                 y_no_elements = 2, height = 0,
                 x_curve = False, y_curve = False, flat_mesh = True,
                 mesh_style = 'Tri', pos_x0 = 0, pos_y0 = 0):
        self.x_len = x_len
        self.y_len = y_len
        self.x_no_elements = x_no_elements
        self.y_no_elements = y_no_elements
        self.x_curve = x_curve
        self.y_curve = y_curve
        self.mesh_style = mesh_style
        self.flat_mesh = flat_mesh
        self.height = height
        self.pos_x0 = pos_x0
        self.pos_y0 = pos_y0
        self.nodes = []
        self.ele_nodes = []
        # Make mesh 3D, if it is curved in one direction
        if x_curve | y_curve:
            self.flat_mesh = False
        return

    def _curved_mesh_get_phi_r(self, h, l):
        '''
        wenn ein gekruemmtes Netz vorliegt:
        Bestimmung des Winkels phi und des Radiusses r aus der Hoehe und der Laenge

        '''
        # Abfangen, wenn Halbschale vorliegt
        if l - 2*h < 1E-7:
            phi = np.pi
        else:
            phi = 2*np.arctan(2*h*l/(l**2 - 4*h**2))
        # Checkt, wenn die Schale ueber pi hinaus geht:
        if phi<0:
            phi += 2*np.pi
        r = l/(2*np.sin(phi/2))
        return phi, r

    def build_mesh(self):
        '''
        Building the mesh by first producing the points, and secondly the elements
        '''

        def build_tri():
            '''
            Builds a triangular mesh
            '''

            # Length of one element
            l_x = self.x_len / self.x_no_elements
            l_y = self.y_len / self.y_no_elements
            # Generating the nodes
            node_number = 0 # node_number counter; node numbers start with 0
            if self.flat_mesh == True:
                for y_counter in range(self.y_no_elements + 1):
                    for x_counter in range(self.x_no_elements + 1):
                        self.nodes.append([l_x*x_counter, l_y*y_counter])
                        node_number += 1
            else:
                # a 3d-mesh will be generated; 
                # the meshing has to be done with a little calculation in andvance
                r_OO_x = np.array([0, 0, 0])
                r_OO_y = np.array([0, 0, 0])
                if self.x_curve:
                    phi_x, r_x = self._curved_mesh_get_phi_r(self.height, self.x_len)
                    delta_phi_x = phi_x/self.x_no_elements
                    r_OO_x = np.array([0, 0, -r_x])
                if self.y_curve:
                    phi_y, r_y = self._curved_mesh_get_phi_r(self.height, self.y_len)
                    delta_phi_y = phi_y/self.y_no_elements
                    r_OO_y = np.array([0, 0, -r_y])
                # Einfuehren von Ortsvektoren, die Vektorkette zum Element geben:
                r_OP_x = np.array([0, 0, 0])
                r_OP_y = np.array([0, 0, 0])
                r_OO   = np.array([self.x_len/2, self.y_len/2, self.height])
                for y_counter in range(self.y_no_elements + 1):
                    for x_counter in range(self.x_no_elements + 1):
                        if self.x_curve:
                            phi = - phi_x/2 + delta_phi_x*x_counter
                            r_OP_x = np.array([r_x*np.sin(phi), 0, r_x*np.cos(phi)])
                        else:
                            r_OP_x = np.array([- self.x_len/2 + l_x*x_counter, 0, 0])
                        if self.y_curve:
                            phi = - phi_y/2 + delta_phi_y*y_counter
                            r_OP_y = np.array([0, r_y*np.sin(phi), r_y*np.cos(phi)])
                        else:
                            r_OP_y = np.array([0, - self.y_len/2 + l_y*y_counter, 0])
                        r_OP = r_OP_x + r_OP_y + r_OO_x + r_OO_y + r_OO
                        self.nodes.append([x for x in r_OP])
            # ELEMENTS
            # Building the elements which have to be tetrahedron
            element_number = 0 # element_number counter; element numbers start with 0
            for y_counter in range(self.y_no_elements):
                for x_counter in range(self.x_no_elements):
                    # first the lower triangulars
                    first_node  = y_counter*(self.x_no_elements + 1) + x_counter + 0
                    second_node = y_counter*(self.x_no_elements + 1) + x_counter + 1
                    third_node  = (y_counter + 1)*(self.x_no_elements + 1) + \
                                  x_counter + 0
                    self.ele_nodes.append([first_node, second_node, third_node])
                    element_number += 1
                    # second the upper triangulars
                    first_node  = (y_counter + 1)*(self.x_no_elements + 1) + \
                                  x_counter + 1
                    second_node = (y_counter + 1)*(self.x_no_elements + 1) + \
                                  x_counter + 0
                    third_node  = y_counter*(self.x_no_elements + 1) + x_counter + 1
                    self.ele_nodes.append([first_node, second_node, third_node])
                    element_number += 1


        def build_quad4():
            '''
            Builds a rectangular mesh
            '''
            delta_x = self.x_len / self.x_no_elements
            delta_y = self.y_len / self.y_no_elements

            # nodes coordinates
            for counter_y in range(self.y_no_elements+1):
                for counter_x in range(self.x_no_elements+1):
                    self.nodes.append([delta_x*counter_x + self.pos_x0, 
                                       delta_y*counter_y + self.pos_y0])

            # node assignment to quadrilateral elements
            for counter_y in range(self.y_no_elements):
                for counter_x in range(self.x_no_elements):
                    node1 = counter_x     + (counter_y - 0)*(self.x_no_elements + 1)
                    node2 = counter_x + 1 + (counter_y - 0)*(self.x_no_elements + 1)
                    node3 = counter_x + 1 + (counter_y + 1)*(self.x_no_elements + 1)
                    node4 = counter_x     + (counter_y + 1)*(self.x_no_elements + 1)
                    self.ele_nodes.append([node1, node2, node3, node4])


        mesh_type_dict = {"Tri": build_tri,
                          "Quad4": build_quad4}

        mesh_type_dict[self.mesh_style]()
        print('Mesh was generated: mesh_style =', self.mesh_style)




    def save_mesh(self, filename_nodes, filename_elements):
        '''
        Speichert das Netz ab; Funktioniert fuer alle Elementtypen,
        es muss also stets nur eine Liste vorhanden sein
        '''

        delimiter = ','
        newline = '\n'

        check_dir(filename_nodes, filename_elements)
        with open(filename_nodes, 'w') as savefile_nodes: # Save nodes
            # Header for file:
            if self.flat_mesh:
                header = 'x_coord' + delimiter + 'y_coord' + newline
            else:
                header = 'x_coord' + delimiter + 'y_coord' + delimiter + \
                         'z_coord' + newline
            savefile_nodes.write(header)
            for nodes in self.nodes:
                savefile_nodes.write(delimiter.join(str(x) for x in nodes) + newline)

        with open(filename_elements, 'w') as savefile_elements: # Save elements
            # Header for the file:
            number_of_nodes = len(self.ele_nodes[0])
            if number_of_nodes == 3:
                savefile_elements.write('node_1' + delimiter + 'node_2' + 
                                        delimiter + 'node_3' + newline)
            elif number_of_nodes == 4:
                savefile_elements.write('node_1' + delimiter + 'node_2' + 
                                        delimiter + 'node_3' + delimiter + 
                                        'node_4' + newline)
            elif number_of_nodes == 2:
                savefile_elements.write('node_1' + delimiter + 'node_2' + newline)                
            else:
                print("Hier lief etwas falsch. Anzahl der Knoten pro Element", 
                      "konnte nicht bestimmt werden.")

            for elements in self.ele_nodes:
                savefile_elements.write(delimiter.join(str(x) for x in elements) 
                                        + newline)


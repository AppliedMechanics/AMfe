# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

"""
Mesh module of AMfe.

This module provides a mesh class that handles the mesh information: nodes, mesh topology, element shapes, groups, ids.
"""


import numpy as np
import pandas as pd
from collections.abc import Iterable

__all__ = [
    'Mesh'
]

# Describe Element shapes, that can be used in AMfe
# 2D volume elements:
element_2d_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8', }
# 3D volume elements:
element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}
# 2D boundary elements
boundary_2d_set = {'straight_line', 'quadratic_line'}
# 3D boundary elements
boundary_3d_set = {'straight_line', 'quadratic_line', 'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}


class Mesh:
    """
    Class for handling the mesh operations.

    Attributes
    ----------
    nodes_df : pandas.DataFrame
        DataFrame containing the x-y-z coordinates of the nodes in reference configuration. Dimension is
        (no_of_nodes, 2) for 2D problems and (no_of_nodes, 3) for 3D problems.
        z-direction is dropped for 2D problems!
        The Dataframe also provides accessing nodes by arbitrary indices
    _el_df : pandas.DataFrame
        DataFrame with element information
    groups : list
        List of groups containing ids (not row indices!)

    Notes
    -----
    GETTER CLASSES NAMING CONVENTION
    We use the following naming convention for function names:
      get_<node|element><ids|idxs>_by_<groups|ids|idxs>
               |            |     |        |
               |            |     |        - Describe which entity is passed groups, ids or row indices
               |            |     - 'by' keyword
               |            - describes weather ids or row indices are returned
                - describes weather nodes or elements are returned

    """
    def __init__(self, dimension=3):
        """
        Parameters
        ----------
        dimension : int
            describes the dimension of the mesh (2 or 3)

        Returns
        -------
        mesh : Mesh
            a new mesh object
        """
        # -- GENERAL INFORMATION --
        self._dimension = dimension

        # -- NODE INFORMATION --
        if dimension == 3:
            self.nodes_df = pd.DataFrame(columns=('x', 'y', 'z'))
        elif dimension == 2:
            self.nodes_df = pd.DataFrame(columns=('x', 'y'))
        else:
            raise ValueError('Mesh dimension must be 2 or 3')

        # Pandas dataframe for elements:
        self._el_df = pd.DataFrame(columns=('shape', 'is_boundary', 'connectivity'))

        # group dict with names mapping to element ids or node ids, respectively
        self.groups = dict()

        # Flag for lazy evaluation of iconnectivity
        self._changed_iconnectivity = True
        # Cache for lazy evaluation of iconnectivity
        self._iconnectivity_df_cached = pd.DataFrame(columns=('iconnectivity',))

    @property
    def el_df(self):
        return self._el_df

    @el_df.setter
    def el_df(self, df):
        self._el_df = df
        self._changed_iconnectivity = True

    @property
    def no_of_nodes(self):
        """
        Returns the number of nodes

        Returns
        -------
        no_of_nodes: int
            Number of nodes of the whole mesh.
        """
        return self.nodes_df.shape[0]

    @property
    def connectivity(self):
        return self._el_df['connectivity'].values

    @property
    def _iconnectivity_df(self):
        """
        Handles the lazy evaluation of the iconnectivity
        Always access the iconnectivity df by this property

        Returns
        -------
        iconnectivity_df : pandas.DataFrame
            DataFrame containint the iconnectivity inormation of the elements,
            i.e. the connectivity w.r.t. row indices of a nodes ndarray
        """
        if self._changed_iconnectivity:
            self._update_iconnectivity()
            self._changed_iconnectivity = False
        return self._iconnectivity_df_cached

    @property
    def iconnectivity(self):
        """
        Get all iconnectivites, i.e. the row indices in the nodes array of the nodes that belong to the elements

        Returns
        -------
        iconnectivity : ndarray
            iconnectivity
        """
        return self._iconnectivity_df['iconnectivity'].values

    @property
    def nodes(self):
        return self.nodes_df.values

    @property
    def no_of_elements(self):
        """
        Returns the number of volume elements

        Returns
        -------
        no_of_elements : int
            Number of volume elements in the mesh
        """
        return len(self._el_df[self._el_df['is_boundary'] != True].index)

    @property
    def no_of_boundary_elements(self):
        """
        Returns the number of boundary elements

        Returns
        -------
        no_of_elements : int
            Number of boundary elements in the mesh
        """
        return len(self._el_df[self._el_df['is_boundary'] == True].index)

    @property
    def dimension(self):
        """
        Returns the dimension of the mesh

        Returns
        -------
        dimension : int
            Dimension of the mesh
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """
        Sets the dimension of the mesh

        Attention: The dimension should not be modified except you know what you are doing.

        Parameters
        ----------
        dim : int
            Dimension of the mesh

        Returns
        -------
        None

        """
        self._dimension = dim

    @property
    def nodes_voigt(self):
        """
        Returns the nodes in voigt notation

        Returns
        -------
        nodes_voigt : ndarray
            Returns the nodes in voigt-notation
        """
        return self.nodes_df.values.reshape(-1)

    def get_connectivity_by_elementids(self, elementids):
        """

        Parameters
        ----------
        elementids : iterable(int)
            elementids for which the connectivity shall be returned
        Returns
        -------
        connectivity : list of ndarrays
            list containing the connectivity of the desired elements
        """
        return self._el_df.loc[elementids, 'connectivity'].values

    def get_iconnectivity_by_elementids(self, elementids):
        """
        Lazy return of iconnectivity of given elementids

        Parameters
        ----------
        elementids : iterable(int)
            elementids for which the connectivity shall be returned
        Returns
        -------
        iconnectivity : list of ndarrays
            list containing the index based connectivity of the desired elements
            i.e. the row indices of the nodes ndarray
        """
        return self._iconnectivity_df.loc[elementids, 'iconnectivity'].values

    def get_elementidxs_by_groups(self, groups):
        """
        Returns elementindices of the connectivity property belonging to groups

        Parameters
        ----------
        groups : list
            groupnames as strings in a list

        Returns
        -------
            indices of the elements in the connectivity array
        """
        elementids = list()
        for group in groups:
            elementids.extend(self.groups[group]['elements'])
        elementids = np.array(elementids)
        elementids = np.unique(elementids)
        return np.array([self._el_df.index.get_loc(elementid) for elementid in elementids], dtype=int)

    def get_elementids_by_groups(self, groups):
        """
        Returns elementids belonging to a group

        Parameters
        ----------
        groups : list
            groupnames as strings in a list

        Returns
        -------
            indices of the elements in the connectivity array
        """
        elementids = list()
        for group in groups:
            elementids.extend(self.groups[group]['elements'])
        elementids = np.array(elementids)
        elementids = np.unique(elementids)
        return elementids

    def get_elementidxs_by_elementids(self, elementids):
        """
        Returns elementindices of the connectivity property belonging to elementids

        Parameters
        ----------
        elementids : iterable
            elementids as integers

        Returns
        -------
            indices of the elements in the connectivity array
        """
        return np.array([self._el_df.index.get_loc(elementid) for elementid in elementids], dtype=int)

    def get_elementids_by_elementidxs(self, elementidxs):
        """
        Returns elementids belonging to elements with elementidxs in connectivity array

        Parameters
        ----------
        elementidxs : iterable
            elementidxs as integers

        Returns
        -------
            ids of the elements
        """
        return self._el_df.iloc[elementidxs].index.values

    def get_nodeid_by_coordinates(self, x, y, z=None, epsilon=1e-12):
        """

        Parameters
        ----------
        x : float
            x-coordinate
        y : float
            y-coordinate
        z : float
            z-coordinate
        epsilon : float (optional, default 1e-12)
            Allowed tolerance (distance), default

        Returns
        -------
        nodeid : int
            nodeid with the given coordinates
        """
        if self.dimension == 2:
            if z is not None:
                print('Warning: z coordinate is ignored in get_nodeid_by_coordinates')
            nodeid = (self.nodes_df[['x', 'y']] - (x, y)).apply(np.linalg.norm, axis=1).idxmin()
            if np.linalg.norm(self.nodes_df.loc[nodeid, ['x', 'y']] - (x, y)) > epsilon:
                nodeid = None
        else:
            nodeid = (self.nodes_df[['x', 'y', 'z']] - (x, y, z)).apply(np.linalg.norm, axis=1).idxmin()
            if np.linalg.norm(self.nodes_df.loc[nodeid, ['x', 'y', 'z']] - (x, y, z)) > epsilon:
                nodeid = None
        return nodeid

    def get_nodeids_by_x_coordinates(self, x, epsilon):
        """

        Parameters
        ----------
        x : float
            x-coordinate where the searched node is located at
        epsilon : float
            radius that acceptable as tolerance for the location
        Returns
        -------
        nodeids : ndarray
            ndarray the nodeids that fulfill the condition
        """
        nodeids = self.nodes_df.index[((self.nodes_df['x'] - x).abs() - epsilon) <= 0].tolist()
        return np.array(nodeids, dtype=int)

    def get_nodeids_by_lesser_equal_x_coordinates(self, x, epsilon):
        """

        Parameters
        ----------
        x : float
            maximum x coordinate of the desired nodes
        epsilon : float
            radius that acceptable as tolerance for the x location
        Returns
        -------
        nodeids : ndarray
            ndarray the nodeids that fulfill the condition
        """
        nodeids = self.nodes_df.index[(self.nodes_df['x'] - (x + epsilon)) <= 0].tolist()
        return np.array(nodeids, dtype=int)

    def get_nodeids_by_greater_equal_x_coordinates(self, x, epsilon):
        """

        Parameters
        ----------
        x : float
            minimum x coordinate of the desired nodes
        epsilon : float
            radius that acceptable as tolerance for the x location
        Returns
        -------
        nodeids : ndarray
            ndarray the nodeids that fulfill the condition
        """
        nodeids = self.nodes_df.index[(self.nodes_df['x'] - (x - epsilon)) >= 0].tolist()
        return np.array(nodeids, dtype=int)

    def get_nodeids_by_groups(self, groups):
        """
        Returns nodeids of the nodes property belonging to a group

        Parameters
        ----------
        groups : list
            contains the groupnames as strings

        Returns
        -------
        nodeids : ndarray
        
        """
        # Get nodeids from 'nodes' key
        nodeids_from_nodes = []
        for group in groups:
            nodeids_from_nodes.extend(self.groups[group]['nodes'])
        nodeids_from_nodes = np.array(nodeids_from_nodes, dtype=int)

        # Get Nodids from nodes which belong the elements of the group
        elementids = self.get_elementids_by_groups(groups)
        nodeids_from_elements = self.get_nodeids_by_elementids(elementids)

        # remove duplicates
        nodes = np.unique(np.hstack((nodeids_from_nodes, np.array(nodeids_from_elements))))
        return nodes
    
    def get_nodeids_by_elementids(self, elementids):
        """
        Returns nodeids of the nodes property belonging to elements

        Parameters
        ----------
        elementids : ndarray
            contains the elementids as int

        Returns
        -------
        nodeids : ndarray
        """
        if len(elementids) == 0:
            return np.array([], dtype=int)
        nodeids = np.hstack(self.get_connectivity_by_elementids(elementids))
        nodeids = np.unique(nodeids)
        return nodeids
    
    def get_nodeids_by_tags(self, tag_name, tag_value, opt_larger=None):
        """
        Returns nodeids of the nodes property belonging to elements, that are tagged by the assigne tag-value

        Parameters
        ----------
        tag_name : list of str
            tag name for adding column in el_df 
        tag_value : list of str, int, Boolean, float
            current tag value to select the element ids
        opt_larger : list of boolean
            optional parameter for selection by a larger-than-boolean operation for the specified tag 


        Returns
        -------
        nodeids : ndarray
        """
        elementids = self.get_elementids_by_tags(tag_name, tag_value, opt_larger)
        nodeids = self.get_nodeids_by_elementids(elementids)
        return nodeids

    def get_ele_shapes_by_ids(self, elementids):
        """
        Returns list of element_shapes for elementids

        Parameters
        ----------
        elementids : list or ndarray
            contains the ids of elements the ele_shapes are asked for

        Returns
        -------
        ele_shapes : list
            list of element_shapes as string
        """
        return [self._el_df.loc[idx, 'shape'] for idx in elementids]

    def get_ele_shapes_by_elementidxs(self, elementidxes):
        """
        Returns list of element_shapes for elementidxes

        Parameters
        ----------
        elementidxes : list
            contains indices of the desired elements in connectivity array

        Returns
        -------
        ele_shapes : list
            list of element_shapes as string
        """
        return self._el_df.iloc[elementidxes]['shape'].values

    def get_ele_shapes_by_elementids(self, elementids):
        """
        Returns list of element_shapes for elementidxes

        Parameters
        ----------
        elementids : list
            contains indices of the desired elements in connectivity array

        Returns
        -------
        ele_shapes : list
            list of element_shapes as string
        """
        return self._el_df.loc[elementids]['shape'].values

    def get_nodeidxs_by_all(self):
        """
        Returns all nodeidxs

        Returns
        -------
        nodeidxs : ndarray
            returns all nodeidxs
        """
        return np.arange(self.no_of_nodes, dtype=np.int)
    
    def get_nodeidxs_by_nodeids(self, nodeids):
        """
        Parameters
        ----------
        nodeids : ndarray
            nodeids
            
        Returns
        -------
        nodeidxs: ndarray
            rowindices of nodes in nodes dataframe
        """
        nodeidxs = np.array([self.nodes_df.index.get_loc(nodeid) for nodeid in nodeids], dtype=int)
        return nodeidxs

    def get_nodeids_by_nodeidxs(self, nodeidxs):
        """

        Parameters
        ----------
        nodeidxs : list
            rowindices of node array

        Returns
        -------
        id : list
            IDs of the corresponding nodes
        """
        return self.nodes_df.iloc[nodeidxs, :].index.values

    def insert_tag(self, tag_name, tag_value_dict=None):
        """
        This function adds an extra column in the el_df
        with name equal the "tag_name" parameter . By default
        a column will be inserted with None value for every elem_id.
        If a dictionary is provided with tag_value_dict[value] = [list of elements id]
        then, the el_df will populated with this information.

        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 
        tag_value_dict : dict, default None
            a dictionary with tag_value_dict[value] = [list of elements id]
            where value are the associated property to a list a elements.

        Returns
        -------
            None
        """

        self.el_df[tag_name] = None

        if tag_value_dict is not None:
            self.change_tag_values_by_dict(tag_name, tag_value_dict)

        return None

    def remove_tag(self, tag_name):
        """
        This function deletes a columns which has name equal to 
        'tag_name' parameter
        
        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 

        Returns
        -------
            None

        """
        self._el_df = self.el_df.drop(columns=tag_name)
        return None

    def change_tag_values_by_dict(self, tag_name, tag_value_dict):
        """
        This function changes the values of the el_df column
        with name equal to the "tag_name" paramenter . By default 
        The tag_value_dict parameters has the format:
        tag_value_dict[value] = [list of elements id]
        

        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 
        tag_value_dict : dict
            a dictionary with tag_value_dict[value] = [list of elements id]
            where value are the associated property to a list a elements.

        Returns
        -------
            None
        """
        for tag_value, elem_list in tag_value_dict.items():
            try:
                self.el_df.loc[elem_list, (tag_name)] = tag_value
            except:
                temp_list = self.el_df[tag_name].tolist()
                for elem in elem_list:
                    temp_list[elem] = tag_value 
                self.el_df[tag_name] = temp_list
        
        return None

    def replace_tag_values(self, tag_name, current_tag_value, new_tag_value):
        """
        This function replaces tag values of the el_df column named
        given by the "tag_name" parameter. The user must provide the current 
        tag value which will replace by the new tag.
        

        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 
        current_tag_value : str, int, Boolean, float
            current tag value in the tag_name column
        new_tag_value : str, int, Boolean, float
            new tag value to replace the current tag value

        Returns
        -------
            None
        """

        self._el_df = self._el_df.replace({tag_name : current_tag_value}, new_tag_value)
        return None

    def get_elementids_by_tags(self, tag_names, tag_values, opt_larger=None):
        """
        This function returns a list with the element ids given a "tag_name" 
        and the tag value associated with it. 
        

        Parameters
        ----------
        tag_names : list of str
            tag name for adding column in el_df 
        tag_values : list of str, int, Boolean, float
            current tag value to select the element ids
        opt_larger : list of boolean
            optional parameter for selection by a larger-than-boolean operation for the specified tag 
    
        Returns
        -------
            elementids : list
                indices of the elements in the self.el_df

        Example
        -------
            testmesh = amfe.mesh.Mesh()
            elementids_list = testmesh.get_elementids_by_tags('is_boundary','False')   
        """
        
        if not isinstance(tag_names, Iterable) or isinstance(tag_names, str):
            tag_names = [tag_names]
            tag_values = [tag_values]
            
        eleids = []
        for itag, tagname in enumerate(tag_names):
            if opt_larger is not None and opt_larger[itag]:
                eleids += self._el_df[self._el_df[tagname] > tag_values[itag]].index.tolist()
            else:
                eleids += self._el_df[self._el_df[tagname] == tag_values[itag]].index.tolist()
        
        eleids = list(set(eleids))
        eleids.sort()
        return eleids


    def get_elementidxs_by_tags(self, tag_names, tag_values, opt_larger=None):
        """
        This function returns a list with the elementidxs in connectivity array
        given a "tag_name" and the tag value associated with it. 
        

        Parameters
        ----------
        tag_names : list of str
            tag name for adding column in el_df 
        tag_values : list of str, int, Boolean, float
            current tag value to select the element idxs
        opt_larger : list of boolean
            optional parameter for selection by a larger-than-boolean operation for the specified tag 

    
        Returns
        -------
            elementidxs : list
                indices of the elements in the connectivity array

        Example
        -------
            testmesh = amfe.mesh.Mesh()
            elementidxs_list = testmesh.get_elementidxs_by_tag('is_boundary','False')                
        """
        
        rows = self.get_elementids_by_tags(tag_names, tag_values, opt_larger)
        return np.array([self._el_df.index.get_loc(row) for row in rows], dtype=int)
    
    def merge_into_groups(self, groups):        
        """
        Merge a dictionary of groups with node- and element-ids into the mesh's 'groups'-dictionary. The additional dictionary has to be of format
        
        groups = {*groupname* : {'nodes' : [*node-ids*], 'elements' : [*element-ids*]}}
        
        Parameters
        ----------
        groups : dict
            additional dictionary, which is to be merged into the mesh's 'groups'
            
        Returns
        -------
        None
        """
        for key in groups:
            if key in self.groups:
                for secondary_key in ['elements', 'nodes']:
                    if secondary_key in groups[key]:
                        self.groups[key][secondary_key] = list(set(self.groups[key][secondary_key]).union(set(groups[key][secondary_key])))
            else:
                self.groups.update({key: {'elements': groups[key].get('elements', []),
                                          'nodes': groups[key].get('nodes', [])}})
                
    def _get_groups_by_secondary_key(self, values, secondary_key):
        """
        Private method returning list of groups where the given entities are associcated with.

        Parameters
        ----------
        values : list
            list containing the ids of the subset for which the associated groups shall be returned
        secondary_key : str ('elements' or 'nodes')
            mesh entity which is described by the ids of the values parameter

        Returns
        -------
        groups : list
            list containing the groups which are associated with the given entities
        """
        if not isinstance(values, Iterable):
            values = [values]

        groups_selection = []
        for key in self.groups:
            for value in values:
                if value in self.groups[key][secondary_key]:
                    if key not in groups_selection:
                        groups_selection.append(key)

        return groups_selection
        
    def get_groups_by_elementids(self, eleids):
        """
        Provides a selection of groups, where the given elements belong to.
        
        Parameters
        ----------
        eleids : list of int
            list of elements, which group-belongings shall be returned
        
        Returns
        -------
        groups : list of str
            group-names of the specified elements
        """
        return self._get_groups_by_secondary_key(eleids, 'elements')
    
    def get_groups_by_nodeids(self, nodeids):
        """
        Provides a selection of groups, where the given nodes belong to.
        
        Parameters
        ----------
        nodeids : list of int
            list of nodes, which group-belongings shall be returned
        
        Returns
        -------
        groups : list of str
            group-names of the specified nodes
        """

        return self._get_groups_by_secondary_key(nodeids, 'nodes')
    
    def _get_groups_dict_by_secondary_key(self, values, secondary_key):
        """
        Private method returning groups dict for a subset of values and desired mesh entity (elements or nodes)

        Parameters
        ----------
        values : list
            list containing the ids of the subset the groups dict shall be generated for
        secondary_key : str ('elements' or 'nodes')
            mesh entity which is described by the ids of the values parameter

        Returns
        -------
        groups : dict
            A dictionary containing the groups of the given subset.
        """
        if not isinstance(values, Iterable):
            values = [values]

        groups_selection = dict()
        for key in self.groups:
            for eleid in values:
                if eleid in self.groups[key][secondary_key]:
                    if key in groups_selection:
                        elements = groups_selection[key]
                        elements[secondary_key].append(eleid)
                        groups_selection[key] = elements
                    else:
                        groups_selection.update({key: {secondary_key: [eleid]}})

        return groups_selection
    
    def get_groups_dict_by_elementids(self, eleids):
        """
        Provides a selection of groups as a sub-dictionary, where the given elements belong to.
        
        Parameters
        ----------
        eleids : list of int
            list of elements, which group-belongings shall be returned

        Returns
        -------
        groups : dict
            subdictionary of the mesh's groups with the given nodes only
        """
        return self._get_groups_dict_by_secondary_key(eleids, 'elements')
    
    def get_groups_dict_by_nodeids(self, nodeids):
        """
        Provides a selection of groups as a sub-dictionary, where the given nodes belong to.
        
        Parameters
        ----------
        nodeids : list of int
            list of nodes, which group-belongings shall be returned
        
        Returns
        -------
        groups : dict
            subdictionary of the mesh's groups with the given nodes only
        """
        return self._get_groups_dict_by_secondary_key(nodeids, 'nodes')
    
    def add_element_to_groups(self, new_ele, groups_ele, secondary_key = 'elements'):
        for key in groups_ele:
            if new_ele not in self.groups[key][secondary_key]:
                self.groups[key][secondary_key].append(new_ele)

    def add_node_to_groups(self, new_node, groups_node):
        self.add_element_to_groups(new_node, groups_node, 'nodes')
    
    def get_nodes_and_elements_by_partition_id(self, partition_id):
        """
        Provides dataframes with all nodes and elements, which belong to the requested partition.
        
        Parameters
        ----------
        partition_id : int
            id of the requested partition
            
        Returns
        -------
        nodes : pandas.DataFrame
            all and only nodes, that belong to selected partition
            
        elements : pandas.DataFrame
            all and only elements, that belong to selected partition
        """

        ele_ids = self.get_elementids_by_tags('partition_id', partition_id)
        elements = self._el_df.loc[ele_ids]
        node_ids = self.get_nodeids_by_elementids(ele_ids)
        nodes = self.nodes_df.loc[node_ids]
        
        return nodes, elements
    
    def copy_node_by_id(self, node_id):
        return self.add_node(self.nodes_df.loc[node_id])
    
    def add_node(self, node_coordinates, node_id=None):
        """
        Copy node with its coordinates and append it at the node-list's end.
        
        Parameters
        ----------
        node_id : int
            id of that node, which is to be copied
            
        Returns
        -------
        new_node_id : int
            id of the new, copied node
        """
        if node_id is None or node_id in self.nodes_df.index.tolist():
            node_id = self.nodes_df.last_valid_index() + 1
            
        if isinstance(node_coordinates, pd.Series):
            new_node = node_coordinates.rename(node_id)
        else:
            new_node = pd.DataFrame(node_coordinates, index=[node_id])
            
        self.nodes_df = self.nodes_df.append(new_node, ignore_index=False)
        return node_id


    def _update_iconnectivity(self):
        """
        Triggers update mechanism for the iconnectivity, i.e. the connectivity of the elements
        but w.r.t to the row indices in a node ndarray instead of the real nodes_df indices

        Returns
        -------
        None
        """
        self._iconnectivity_df_cached = pd.DataFrame(self._el_df['connectivity'].apply(self.get_nodeidxs_by_nodeids),
                                                     index=self._el_df.index)
        self._iconnectivity_df_cached.columns = ['iconnectivity']

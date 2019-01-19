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
        nodeids = np.hstack(self.get_connectivity_by_elementids(elementids))
        nodeids = np.unique(nodeids)
        return nodeids
    
    def get_nodeids_by_tag(self, tag_name, tag_value):
        """
        Returns nodeids of the nodes property belonging to elements, that are tagged by the assigne tag-value

        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 
        tag_value : str, int, Boolean, float
            current tag value to select the element ids

        Returns
        -------
        nodeids : ndarray
        """
        elementids = self.get_elementids_by_tag(tag_name, tag_value)
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

    def get_elementids_by_tag(self, tag_name, tag_value):
        """
        This function returns a list with the element ids given a "tag_name" 
        and the tag value associated with it. 
        

        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 
        tag_value : str, int, Boolean, float
            current tag value to select the element ids
    
        Returns
        -------
            elementids : list
                indices of the elements in the self.el_df

        Example
        -------
            testmesh = amfe.mesh.Mesh()
            elementids_list = testmesh.get_elementids_by_tag('is_boundary','False')   
        """
        
        return self.el_df[self.el_df[tag_name] == tag_value].index.values

    def get_elementidxs_by_tag(self, tag_name, tag_value):
        """
        This function returns a list with the elementidxs in connectivity array
        given a "tag_name" and the tag value associated with it. 
        

        Parameters
        ----------
        tag_name : str
            tag name for adding column in el_df 
        tag_value : str, int, Boolean, float
            current tag value to select the element idxs
    
        Returns
        -------
            elementidxs : list
                indices of the elements in the connectivity array

        Example
        -------
            testmesh = amfe.mesh.Mesh()
            elementidxs_list = testmesh.get_elementidxs_by_tag('is_boundary','False')                
        """
        
        rows = self.get_elementids_by_tag(tag_name, tag_value)
        return np.array([self._el_df.index.get_loc(row) for row in rows], dtype=int)

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

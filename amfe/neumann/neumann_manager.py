#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import pandas as pd
from copy import deepcopy

from amfe.component.constants import ELEPROTOTYPEHELPERLIST
from .structural_neumann import *


class NeumannManager:
    """
    Manager-class to create, assign and manipulate several different Neumann conditions
    """
    BOUNDARYELEMENTSHAPES = [element[0] for element in ELEPROTOTYPEHELPERLIST if element[2] is not None]
    
    def __init__(self):
        # Dataframe for reconstructing applied conditions
        self._neumann_df = pd.DataFrame(columns=['name', 'tag', 'property_names', 'neumann_obj'])
        
        # Dataframe containing element_objects and their position in connectivity array and their
        # foreign key to the _neumann_df they belong to
        self._neumann_obj_df = pd.DataFrame(columns=['neumann_obj', 'fk_elementid', 'fk_neumann_df'])
        self._neumann_obj_df['fk_neumann_df'] = self._neumann_obj_df['fk_neumann_df'].astype(int)
        self._neumann_obj_df['fk_elementid'] = self._neumann_obj_df['fk_elementid'].astype(int)
        
    def assign_neumann_by_eleids(self, neumannobj, eleidxes, ele_shapes, property_names, tag, name):
        dfindex = self._neumann_df.index.max() + 1
        if pd.isnull(dfindex):
            dfindex = 0

        neumann_prototypes = {element_shape: deepcopy(neumannobj) for element_shape in self.BOUNDARYELEMENTSHAPES}
        for element_shape, neumann_obj in neumann_prototypes.items():
            neumann_obj.set_element(element_shape)

        # Create pointer for each element
        neumann_objects = np.array([neumann_prototypes[ele_shape] for ele_shape in ele_shapes])
        # Create new rows for neumann_obj_df
        df = pd.DataFrame(
            {'neumann_obj': neumann_objects, 'fk_elementid': eleidxes,
             'fk_neumann_df': np.ones(len(neumann_objects), dtype=int) * dfindex}
        )
        self._neumann_obj_df = self._neumann_obj_df.append(df)

        # Create entry for neumann_df describing the whole b.c.
        df_data = {'name': name, 'tag': tag, 'property_names': [property_names], 'neumann_obj': neumannobj}
        self._neumann_df = self._neumann_df.append(pd.DataFrame(df_data, index=[dfindex]), sort=True)

    @staticmethod
    def create_fixed_direction_neumann(direction, time_func=lambda t: 1):
        return FixedDirectionNeumann(direction, time_func)

    @staticmethod
    def create_normal_following_neumann(time_func=lambda t: 1):
        return NormalFollowingNeumann(time_func)

    @staticmethod
    def create_projected_area_neumann(direction, time_func=lambda t: 1):
        return ProjectedAreaNeumann(direction, time_func)

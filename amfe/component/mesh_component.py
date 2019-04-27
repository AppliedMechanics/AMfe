#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from amfe.mesh import Mesh
from amfe.mapping import StandardMapping
from .component_base import ComponentBase
from amfe.assembly.assembly import Assembly
from amfe.component.constants import ELEPROTOTYPEHELPERLIST, SHELLELEPROTOTYPEHELPERLIST
from amfe.neumann.neumann_manager import *
from amfe.constraint.constraint_manager import *
from amfe.tools import make_input_iterable
from amfe.material import ShellMaterial

__all__ = ['MeshComponent']


class MeshComponent(ComponentBase):
    # The following class attributes must be overwritten by subclasses
    ELEMENTPROTOTYPES = dict(((element[0], None) for element in ELEPROTOTYPEHELPERLIST))
    SHELLELEMENTPROTOTYPES = dict(((element[0], None) for element in SHELLELEPROTOTYPEHELPERLIST))

    def __init__(self, mesh=Mesh()):
        super().__init__()
        self._mesh = mesh
        self._mapping = StandardMapping()
        self._ele_obj_df = pd.DataFrame([], columns=['physics', 'fk_mesh', 'ele_obj', 'fk_mapping'])
        self._ele_obj_df['fk_mapping'] = self._ele_obj_df['fk_mapping'].astype(int)
        self._ele_obj_df['fk_mesh'] = self._ele_obj_df['fk_mesh'].astype(int)
        self._neumann = NeumannManager()
        self._assembly = Assembly()
        self._constraints = ConstraintManager()

    # -- PROPERTIES --------------------------------------------------------------------------------------
    @property
    def ele_obj(self):
        return self._ele_obj_df['ele_obj'].values
    
    @property
    def X(self):
        """
        Returns the reference-configuration of each dof
        """
        X = np.zeros(self._mapping.no_of_dofs)
        nodeidxs = self._mesh.get_nodeidxs_by_all()
        nodeids = self._mesh.get_nodeids_by_nodeidxs(nodeidxs)
        X[self._mapping.get_dofs_by_nodeids(nodeids)] = self._mesh.nodes[nodeidxs]
        
        return X

    @property
    def no_of_elements(self):
        return len(self._ele_obj_df.index)

    @property
    def assembly(self):
        return self._assembly

    @assembly.setter
    def assembly(self, assembly):
        self._assembly = assembly

    @property
    def mesh(self):
        return self._mesh

    @property
    def constraints(self):
        return self._constraints

    @property
    def mapping(self):
        return self._mapping

    @property
    def neumann(self):
        return self._neumann

    @property
    def fields(self):
        fields_volume = set([field for ele_obj in self._ele_obj_df['ele_obj'].unique() for field in ele_obj.fields()])
        fields_list = list(fields_volume.union(set(self._neumann.fields)))
        fields_list.sort()
        return fields_list

    # -- ASSIGN MATERIAL METHODS -------------------------------------------------------------------------
    def assign_material(self, materialobj, propertynames, physics, tag='_groups'):
        if tag == '_groups':
            eleids = self._mesh.get_elementids_by_groups(propertynames)
        elif tag == '_eleids':
            eleids = propertynames
        else:
            eleids = self._mesh.get_elementids_by_tags(tag, propertynames)
        self._assign_material_by_eleids(materialobj, eleids, physics)

    def _assign_material_by_eleids(self, materialobj, eleids, physics):
        if isinstance(materialobj, ShellMaterial):
            prototypes = deepcopy(self.SHELLELEMENTPROTOTYPES)
        else:
            prototypes = deepcopy(self.ELEMENTPROTOTYPES)
        for prototype in prototypes.values():
            prototype.material = materialobj
        ele_shapes = self._mesh.get_ele_shapes_by_ids(eleids)
        new_df = pd.DataFrame({'physics': [physics]*len(ele_shapes), 'fk_mesh': eleids,
                               'ele_obj': [prototypes[ele_shape] for ele_shape in ele_shapes],
                               'fk_mapping': np.ones(len(ele_shapes), dtype=int)*-1})
        self._ele_obj_df = self._ele_obj_df.append(new_df, ignore_index=True)
        self._ele_obj_df = self._ele_obj_df.sort_index()
        self._ele_obj_df['fk_mapping'] = self._ele_obj_df['fk_mapping'].astype(int)
        self._ele_obj_df['fk_mesh'] = self._ele_obj_df['fk_mesh'].astype(int)
        self._update_mapping()
        self._C_csr = self._assembly.preallocate(self._mapping.no_of_dofs, self._mapping.elements2global)
        self._M_csr = self._C_csr.copy()
        self._f_glob_int = np.zeros(self._C_csr.shape[1])

    # -- ASSIGN NEUMANN CONDITION METHODS -----------------------------------------------------------------
    def assign_neumann(self, name, condition, tag_values, tag='_groups'):
        print('Assigning Neumann Condition')
        if tag == '_groups':
            eleids = self._mesh.get_elementids_by_groups(tag_values)
        elif tag == '_eleids':
            eleids = tag_values
        else:
            eleids = self._mesh.get_elementids_by_tags(tag, tag_values)
            
        # get ele_shapes of the elements belonging to the passed eleidxes
        ele_shapes = self._mesh.get_ele_shapes_by_elementids(eleids)
            
        self._neumann.assign_neumann_by_eleids(condition, eleids, ele_shapes, tag_values, tag, name)
        self._update_mapping()

    # -- ASSIGN CONSTRAINTS METHODS ------------------------------------------------------------------------
    def assign_constraint(self, name, constraint, dofidxs, nodeidxs):
        self._constraints.add_constraint(name, constraint, dofidxs, nodeidxs)
        
    # -- MAPPING METHODS -----------------------------------------------------------------------------------
    def _update_mapping(self):
        # collect parameters for call of update_mapping
        print('Updating Mapping')
        fields = self.fields
        nodeids = self._mesh.nodes_df.index.values

        volume_connectivites = self._ele_obj_df.join(self._mesh.el_df, on='fk_mesh')['connectivity'].values
        dofs_by_elements = [element.dofs() for element in self._ele_obj_df['ele_obj'].values]
        volume_callbacks = [self.write_mapping_key]*len(dofs_by_elements)
        volume_callbackargs = self._ele_obj_df.index.get_values()

        boundary_connectivities = self._neumann.el_df.join(self._mesh.el_df, on='fk_mesh')['connectivity'].values
        dofs_by_elements_boundary = [element.dofs() for element in self._neumann.el_df['neumann_obj'].values]
        boundary_callbacks = [self._neumann.write_mapping_key]*len(dofs_by_elements_boundary)
        boundary_callbackargs = self._neumann.el_df.index.get_values()

        connectivities = np.concatenate([volume_connectivites, boundary_connectivities])
        dofs_by_elements.extend(dofs_by_elements_boundary)
        callbacks = np.concatenate([volume_callbacks, boundary_callbacks])
        callbackargs = np.concatenate([volume_callbackargs, boundary_callbackargs])

        # call update_mapping
        self._mapping.update_mapping(fields, nodeids, connectivities, dofs_by_elements, callbacks, callbackargs)
        self._constraints.no_of_dofs_unconstrained = self._mapping.no_of_dofs

    def write_mapping_key(self, fk, local_id):
        self._ele_obj_df.at[local_id, 'fk_mapping'] = fk
        
    def get_physics(self):
        return self._ele_obj_df['physics'].unique()
    
    def get_materials(self):
        elements = self._ele_obj_df['ele_obj'].unique()
        material = []
        for element in elements:
            material.append(element.material)
        return material
    
    @make_input_iterable
    def get_elementids_by_physics(self, physics):
        elements = np.array([])
        for phys in physics: 
            elements = np.append(elements, self._ele_obj_df['fk_mesh'][self._ele_obj_df['physics'] == phys])
        return elements
    
    @make_input_iterable
    def get_elementids_by_materials(self, material_obj):
        ele_ids = np.array([])
        for mat in material_obj:
            for eleid, element in self._ele_obj_df.iterrows():
                if element['ele_obj'].material is mat:
                    ele_ids = np.append(ele_ids, element['fk_mesh'])
        return ele_ids

    # -- GETTER FOR SYSTEM MATRICES ------------------------------------------------------------------------
    #
    # MUST BE IMPLEMENTED IN SUBCLASSES
    #

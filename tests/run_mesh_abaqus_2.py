"""
Leaf spring file 2
"""


import amfe

input_file = amfe.amfe_dir('meshes/abaqus/A9603205802.inp')
output_file = amfe.amfe_dir('results/abaqus/leaf_spring_2')

my_system = amfe.MechanicalSystem()

my_mesh = my_system.mesh_class
my_mesh.import_inp(input_file)

steel = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)

my_mesh.load_group_to_mesh('Leaf_1', steel, 'phys_group')
my_mesh.load_group_to_mesh('Leaf_2', steel, 'phys_group')

my_mesh.load_group_to_mesh('NX_3d_mesh(65)', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Lager_Lower_Platte', my_material, 'phys_group')

my_mesh.load_group_to_mesh('LAger_Upper_Platte', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Platte', my_material, 'phys_group')
my_mesh.load_group_to_mesh('P13', my_material, 'phys_group')

my_mesh.save_mesh_xdmf(output_file + '_solid')

#%%
my_mesh.load_group_to_mesh('', my_material, 'phys_group')
my_mesh.load_group_to_mesh('', my_material, 'phys_group')
my_mesh.load_group_to_mesh('', my_material, 'phys_group')
my_mesh.load_group_to_mesh('', my_material, 'phys_group')
my_mesh.load_group_to_mesh('', my_material, 'phys_group')


#%%
my_system.no_of_dofs_per_node = my_mesh.no_of_dofs_per_node
my_system.assembly_class.preallocate_csr()
my_system.dirichlet_class.no_of_unconstrained_dofs = my_mesh.no_of_dofs
my_system.dirichlet_class.update()

K = my_system.K()

M = my_system.M()

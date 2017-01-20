"""
Leaf spring file
"""


import amfe

input_file = amfe.amfe_dir('meshes/abaqus/A9483201605_max_loaded.inp')
output_file = amfe.amfe_dir('results/abaqus/leaf_spring')

my_mesh = amfe.Mesh()
my_mesh.import_inp(input_file)

my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)

my_mesh.load_group_to_mesh('Federbuegel', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Federspannplate', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Zwischenplatte', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Blatt_01', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Blatt_02', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Blatt_03', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Blatt_04', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Blatt_05', my_material, 'phys_group')

my_mesh.save_mesh_xdmf(output_file + '_all')

#%%

#my_system = amfe.MechanicalSystem(stress_recovery=True)



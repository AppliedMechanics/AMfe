"""
Leaf spring file 2
"""


import amfe

input_file = amfe.amfe_dir('meshes/abaqus/A9603205802.inp')
output_file = amfe.amfe_dir('results/abaqus/leaf_spring_2')

my_mesh = amfe.Mesh()
my_mesh.import_inp(input_file)

my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)

#my_mesh.load_group_to_mesh('NX_3d_mesh(65)', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Lager_Lower_Platte', my_material, 'phys_group')

my_mesh.load_group_to_mesh('LAger_Upper_Platte', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Leaf_1', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Leaf_2', my_material, 'phys_group')
my_mesh.load_group_to_mesh('Platte', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Leaf_1', my_material, 'phys_group')


#my_mesh.load_group_to_mesh('Federspannplate', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Zwischenplatte', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Blatt_01', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Blatt_02', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Blatt_03', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Blatt_04', my_material, 'phys_group')
#my_mesh.load_group_to_mesh('Blatt_05', my_material, 'phys_group')

my_mesh.save_mesh_xdmf(output_file + '_all')

#%%

#my_system = amfe.MechanicalSystem(stress_recovery=True)



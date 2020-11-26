#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#


from os.path import splitext, basename
from xml.etree import ElementTree as ET

from amfe.io import XDMFDICT, MESHENTITYTYPE2XDMF, POSTPROCESSDATATYPE2XDMF,\
    POSTPROCESSDATATYPE2XDMFDIMENSION, MeshEntityType
from amfe.io.tools import insert_line_breaks_in_xml

__all__ = ['write_xdmf_from_hdf5'
           ]


def write_xdmf_from_hdf5(xdmffp, hdf5fp, nodesroot, topologyroot, timesteps, fielddict):
    """
    write an xdmf file from hdf5 information

    Parameters
    ----------
    xdmffp : File
        Python file pointer for writing xdmf file
    hdf5fp : h5py.File
        H5py file pointer for reading hdf5 information
    nodesroot : str
        string describing the HDF5 path to the nodes array, e.g. "/mesh/nodes"
    topologyroot : str
        string describing the HDF5 path to the topology information, e.g. "/mesh/topology"
    timesteps : list or ndarray
        timesteps that shall be written to the xdmf file
    fielddict : dict
        dict with information about the fields that are contained in HDF5file and shall be written to xdmf:
        { 'fieldname1' : {
                'mesh_entitiy_type' : MeshEntityType
                'data_type' : PostProcessDatatype
                'hdf_path' : str (describing the HDF5-path to the fielddata
                }
          'fieldname2' : {
                ...
                }
        }

    Returns
    -------

    """
    relative_hdf5_path = splitext(basename(hdf5fp.filename))[0]
    no_of_nodes = hdf5fp[nodesroot].shape[0]
    root = ET.Element('Xdmf', {'Version': '3.0'})
    domain = ET.SubElement(root, 'Domain')  # , {'Type': 'Uniform'})
    temporal = ET.SubElement(domain, 'Grid', {'GridType': 'Collection', 'CollectionType': 'Temporal'})
    for i_t, t in enumerate(timesteps):
        spatial = ET.SubElement(temporal, 'Grid', {'GridType': 'Collection', 'CollectionType': 'Spatial'})
        time = ET.SubElement(spatial, 'Time', {'TimeType': 'Single', 'Value': '{}'.format(t)})
        etypesgroup = hdf5fp[topologyroot]
        etypes = etypesgroup.keys()
        for etype in etypes:
            grid = ET.SubElement(spatial, 'Grid', {'GridType': 'Uniform', 'Name': 'mesh'})
            no_of_elements_of_current_etype = etypesgroup[etype].shape[0]

            topology = ET.SubElement(grid, 'Topology',
                                     {'NumberOfElements': str(no_of_elements_of_current_etype),
                                      'TopologyType': XDMFDICT[etype]['xdmf_name']})
            elementdata = ET.SubElement(topology, 'DataItem',
                                                {'Dimensions': '{} {}'.format(no_of_elements_of_current_etype,
                                                                            XDMFDICT[etype]['no_of_nodes']),
                                                 'Format': 'HDF', 'NumberType': 'Int'})
            elementdata.text = relative_hdf5_path + '.hdf5:' + topologyroot + '/' + etype
            geometry = ET.SubElement(grid, 'Geometry', {'GeometryType': 'XYZ'})
            nodedata = ET.SubElement(geometry, 'DataItem', {'Dimensions': '{} {}'.format(
                no_of_nodes, 3), 'Format': 'HDF', 'NumberType': 'Float'})
            nodedata.text = relative_hdf5_path + '.hdf5:' + nodesroot

            # write cell fields:
            for name in fielddict.keys():
                if fielddict[name]['mesh_entity_type'] == MeshEntityType.ELEMENT:
                    attribute = ET.SubElement(grid, 'Attribute', {'Name': name, 'Center': MESHENTITYTYPE2XDMF[fielddict[name]['mesh_entity_type']],
                                                                  'AttributeType': POSTPROCESSDATATYPE2XDMF[fielddict[name]['data_type']]})
                    attribute_data = ET.SubElement(attribute, 'DataItem',
                                                   {'Dimensions': '{} {}'.format(no_of_elements_of_current_etype,
                                                                                 POSTPROCESSDATATYPE2XDMFDIMENSION[
                                                                                     fielddict[name][
                                                                                         'data_type']]),
                                                    'ItemType': 'HyperSlab'})
                    dataitem1 = ET.SubElement(attribute_data, 'DataItem', {'Dimensions': "3 2", 'Format': 'XML'})
                    dataitem1.text = "0 {} 1 1 {} 1".format(i_t, no_of_elements_of_current_etype)
                    dataitem2 = ET.SubElement(attribute_data, 'DataItem',
                                              {'Dimensions': '{} {}'.format(no_of_elements_of_current_etype, len(timesteps)),
                                               'Format': 'HDF', 'NumberType': 'Float'})
                    dataitem2.text = relative_hdf5_path + '.hdf5:' + fielddict[name]['hdf5path'] + '/' + etype

                elif fielddict[name]['mesh_entity_type'] == MeshEntityType.NODE:
                    attribute = ET.SubElement(grid, 'Attribute', {'Name': name, 'Center': MESHENTITYTYPE2XDMF[fielddict[name]['mesh_entity_type']],
                                                                  'AttributeType': POSTPROCESSDATATYPE2XDMF[fielddict[name]['data_type']]})
                    attribute_data = ET.SubElement(attribute, 'DataItem',
                                                   {'Dimensions': '{} {}'.format(no_of_nodes, POSTPROCESSDATATYPE2XDMFDIMENSION[fielddict[name]['data_type']]),
                                                    'ItemType': 'HyperSlab'})
                    dataitem1 = ET.SubElement(attribute_data, 'DataItem', {'Dimensions': "3 2", 'Format': 'XML'})
                    dataitem1.text = "0 {} 1 1 {} 1".format(i_t, no_of_nodes*3)
                    dataitem2 = ET.SubElement(attribute_data, 'DataItem', {'Dimensions':'{} {}'.format(no_of_nodes*3, len(timesteps)),
                                                                           'Format': 'HDF', 'NumberType': 'Float'})
                    dataitem2.text = relative_hdf5_path + '.hdf5:' + fielddict[name]['hdf5path']
                else:
                    raise ValueError('MeshEntityType must be valid MeshEntityType (node or element)')

    insert_line_breaks_in_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xdmffp, 'UTF-8', True)
    return

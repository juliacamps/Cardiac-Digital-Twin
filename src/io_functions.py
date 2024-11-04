import os

import math
import numpy as np
import json
import scipy.io
import pandas as pd
import pymp, multiprocessing
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.util import numpy_support as VN
from warnings import warn

from ecg_functions import resample_ecg
from utils import initialise_pandas_dataframe, fold_ecg_matrix, unfold_ecg_matrix, get_monoalg_vm_file_name_tag, \
    get_monoalg_vm_file_time, get_monoalg_geo_file_name


def read_alya_ecg_mat(file_path):
    warn('Function read_alya_ecg_mat() is hardcoded for only 8 leads with specific names.')
    nb_leads = 8
    ecgs = scipy.io.loadmat(file_path)['ecgs']
    time_steps = np.squeeze(ecgs['ts'][0][0]) * 1000 # Convert to ms
    # print('min ', np.amin(time_steps))
    # print('max ', np.amax(time_steps))
    # print('time_steps ', time_steps)
    ecg_array = np.zeros((nb_leads, time_steps.shape[0]))
    alya_lead_name_list = ['Is', 'IIs', 'V1s', 'V2s', 'V3s', 'V4s', 'V5s', 'V6s']
    for lead_i in range(len(alya_lead_name_list)):
        lead_name = alya_lead_name_list[lead_i]
        ecg_array[lead_i, :] = np.squeeze(ecgs[lead_name][0][0])
    ecg_array = resample_ecg(desired_freq=1000, original_ecg=ecg_array,
                             original_freq=None, original_x=time_steps)
    # print('ecg_array ', ecg_array.shape)
    return ecg_array


def get_electrode_filename(anatomy_subject_name, geometric_data_dir):
    return geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_electrode_xyz.csv'


# Common path characteristics for geometric .csv files
def get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution):
    return geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + resolution + '/'


def get_geometric_file_prefix(anatomy_subject_name, resolution):
    return anatomy_subject_name + '_' + resolution


# Node XYZ
def get_node_xyz_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_xyz.csv'


# Tetra
def get_tetra_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_tetra.csv'


# Endocardial surfaces
def get_node_lvendo_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_boundarynodefield_lvendo.csv'


def old_get_node_lvendo_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_boundarynodefield_ep-lvnodes.csv'


def get_node_rvendo_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_boundarynodefield_rvendo.csv'


def old_get_node_rvendo_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_boundarynodefield_ep-rvnodes.csv'


# Fibres
def get_fibre_fibre_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_nodefield_fibre.csv'


def get_fibre_sheet_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_nodefield_sheet.csv'


def get_fibre_normal_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_nodefield_normal.csv'


# Materials
def get_material_filename(anatomy_subject_name, geometric_data_dir, resolution):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_material_tetra.csv'


# VC fields
def get_vc_filename(anatomy_subject_name, geometric_data_dir, resolution, vc_name):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_nodefield_' + vc_name + '.csv'


def old_get_vc_filename(anatomy_subject_name, geometric_data_dir, resolution, vc_name):
    geometric_data_path = get_geometric_data_path(anatomy_subject_name, geometric_data_dir, resolution)
    geometric_file_prefix = get_geometric_file_prefix(anatomy_subject_name, resolution)
    return geometric_data_path + geometric_file_prefix + '_nodefield_cobiveco-' + vc_name + '.csv'


'''functionality'''
def save_dictionary(dictionary, filename):
    with open(filename, "w") as fp:
        json.dump(dictionary, fp)


def read_dictionary(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def read_pandas(filename):
    df = pd.read_csv(filename, delimiter=',')
    df = initialise_pandas_dataframe(df)
    return df


def save_pandas(df, filename):
    df.to_csv(filename, sep=',', index=False, header=True)
    # df.to_csv(filename, sep=',', index=False, header=True)


def obtainVTKField(txt, tag_start, tag_end):  # Reads one field as an array in a vtk file-format
    start = txt.find(tag_start)
    end = txt.find(tag_end, start)
    tagged_txt = txt[txt.find('\n', start):end]
    tagged_txt = tagged_txt.replace('\n', ' ').split(
        None)  # PYTHON documentation: If sep is not specified or is None, a different splitting algorithm is applied: runs of consecutive whitespace are regarded
    # as a single separator, and the result will contain no empty strings at the start or end if the string has leading or trailing whitespace.
    return np.array([float(x) for x in tagged_txt])


def read_csv_file(filename, skiprows=0, usecols=None):
    return np.loadtxt(filename, delimiter=',', dtype=float, skiprows=skiprows, usecols=usecols)


def read_ecg_from_csv(filename, nb_leads):
    folded_data = read_csv_file(filename=filename)
    return unfold_ecg_matrix(data=folded_data, nb_leads=nb_leads)


def save_csv_file(data, filename, column_name_list=None):
    if column_name_list is None:
        np.savetxt(filename, data, delimiter=',')
    else:
        np.savetxt(filename, data, delimiter=',', header=','.join(column_name_list), comments='')


def save_ecg_to_csv(data, filename):
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]
    folded_data = fold_ecg_matrix(data=data)
    save_csv_file(data=folded_data, filename=filename)


def save_vtk_to_csv(anatomy_subject_name, geometric_data_dir, target_resolution, vtk_filename):
    if os.path.exists(vtk_filename):
        reader = vtkUnstructuredGridReader()
        reader.SetFileName(vtk_filename)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        unprocessed_node_xyz = VN.vtk_to_numpy(data.GetPoints().GetData())
        # Save node xyz coordinates
        np.savetxt(geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name
                   + '_' + target_resolution + '/' + anatomy_subject_name + '_'
                   + target_resolution + '_xyz.csv', unprocessed_node_xyz, delimiter=',')
        n_tetra = data.GetNumberOfCells()
        unprocessed_tetra = np.reshape(VN.vtk_to_numpy(data.GetCells().GetConnectivityArray()), [n_tetra, 4])
        # Save tetra
        np.savetxt(geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + target_resolution
                   + '/' + anatomy_subject_name + '_' + target_resolution + '_tetra.csv', delimiter=',')

def parse_UKB_ecg(txt):  # TODO
    raise NotImplementedError

# def preprocess_UKB_ecg(subject_name, data_dir): #TODO
#     filename = None
#     raw_txt = read_csv_file(filename)
#     raw_ecg = parse_UKB_ecg(raw_txt)
#     simulated_ecg_normalised = preprocess_raw_ecg(raw_ecg)
#     return simulated_ecg_normalised


# TODO: finish coding this function to save the Purkinje network as a VTK
def write_purkinje_vtk(edge_list, filename, node_xyz, verbose, visualisation_dir):
    if verbose:
        print('Write out Purkinje as VTK ', filename, ' , at ', visualisation_dir)
    # for node_field_i in range(len(node_field_name_list)):
    #     attribute_name = node_field_name_list[node_field_i]
    #     attribute_value = node_field_list[node_field_i]
    #     if attribute_value is not None:
    #         export_ensight_scalar_per_node(dir=visualisation_dir, casename=subject_name,
    #                                        data=attribute_value, dataname=attribute_name)
    #         export_ensight_add_case_node(dir=visualisation_dir, casename=subject_name, dataname=attribute_name)


    # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
    # lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
    # lv_edges = edges[lvedges_indexes, :]
    # lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
    # lv_root_to_PKpath_mat = lv_PK_path_mat[lvActnode_ids, :]
    # for i in range(0, lv_edges.shape[0], 1):
    #     for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
    #         if np.all(np.isin(lv_edges[i, :],
    #                           lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != get_nan_value()]])):
    #             lvPK_edges_indexes[i] = 1
    #             break
    # LV_PK_edges = lv_edges[lvPK_edges_indexes, :]
    # Save the available LV Purkinje network
    with open(visualisation_dir + filename + '.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS ' + str(
            node_xyz.shape[0]) + ' double\n')
        for i in range(0, node_xyz.shape[0], 1):
            f.write('\t' + str(node_xyz[i, 0]) + '\t' + str(node_xyz[i, 1]) + '\t' + str(node_xyz[i, 2]) + '\n')
        f.write('LINES ' + str(edge_list.shape[0]) + ' ' + str(edge_list.shape[0] * 3) + '\n')
        for i in range(0, edge_list.shape[0], 1):
            f.write('2 ' + str(edge_list[i, 0]) + ' ' + str(edge_list[i, 1]) + '\n')
        f.write('POINT_DATA ' + str(node_xyz.shape[0]) + '\n\nCELL_DATA ' + str(edge_list.shape[0]) + '\n')


def save_purkinje_out_of_use(dir, geometry, root_node_meta_bool_index, subject_name):   # TODO This function knows too much about the geometry class
    # conduction system
    attribute_name = 'conduction_system'
    if hasattr(geometry, attribute_name):
        attribute_value = geometry.__dict__[attribute_name]
        nb_candidate_root_node = geometry.get_nb_candidate_root_node()
        if attribute_value is not None and nb_candidate_root_node is not None:
            print('Save candidate root nodes')
            candidate_root_node_meta_bool_index = np.ones((nb_candidate_root_node), dtype=bool)
            lv_pk_candidate_edge, rv_pk_candidate_edge = geometry.get_purkinje_edge(
                root_node_meta_bool_index=candidate_root_node_meta_bool_index)
            if lv_pk_candidate_edge is not None:
                # export_ensight_initialise_case(dir=visualisation_dir, casename=casename)
                # export_ensight_geometry(dir=visualisation_dir, casename=casename, node_xyz=node_xyz, elem=pk_edge)
                print('Export conduction system')
                write_purkinje_vtk(casename=subject_name + '_candidate_lv_pk', dir=dir, edge=lv_pk_candidate_edge,
                                   node_xyz=geometry.get_node_xyz())
            root_node_xyz = geometry.get_node_xyz()[geometry.get_candidate_root_node_index(), :]
            write_node_xyz_csv(casename=subject_name + '_candidate_root_node', dir=dir, node_xyz=root_node_xyz)
            if root_node_meta_bool_index is not None:
                print('Save inferred root nodes')
                lv_pk_edge, rv_pk_edge = geometry.get_purkinje_edge(
                    root_node_meta_bool_index=root_node_meta_bool_index)
                write_purkinje_vtk(casename=subject_name + '_lv_pk', dir=dir, edge=lv_pk_edge,
                                   node_xyz=geometry.get_node_xyz())
                root_node_xyz = geometry.get_node_xyz()[
                                geometry.get_selected_root_node_index(root_node_meta_index=root_node_meta_bool_index), :]
                write_node_xyz_csv(casename=subject_name + '_root_node', dir=dir, node_xyz=root_node_xyz)


def write_purkinje_vtk_bakcup(casename, dir, edge, node_xyz):
    with open(dir + casename + '.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS ' + str(
            node_xyz.shape[0]) + ' double\n')
        for i in range(0, node_xyz.shape[0], 1):
            f.write(
                '\t' + str(node_xyz[i, 0]) + '\t' + str(node_xyz[i, 1]) + '\t' + str(node_xyz[i, 2]) + '\n')
        f.write('LINES ' + str(edge.shape[0]) + ' ' + str(edge.shape[0] * 3) + '\n')
        for i in range(0, edge.shape[0], 1):
            f.write('2 ' + str(edge[i, 0]) + ' ' + str(edge[i, 1]) + '\n')
        f.write(
            'POINT_DATA ' + str(node_xyz.shape[0]) + '\n\nCELL_DATA ' + str(edge.shape[0]) + '\n')


def write_root_node_csv(filename, node_vc_list, node_xyz, root_node_index_list, vc_name_list, verbose,
                        visualisation_dir, xyz_name_list):
    if verbose:
        print('Write out root nodes as CSV ', filename, ' , at ', visualisation_dir)
    # Save the available LV and RV root nodes
    column_name_list = xyz_name_list + vc_name_list
    data = np.zeros((len(root_node_index_list), node_xyz.shape[1]+len(node_vc_list)), dtype=float)
    for root_node_i in range(0, len(root_node_index_list)):
        root_node_index = root_node_index_list[root_node_i]
        root_node_xyz = node_xyz[root_node_index, :]
        root_node_vc = []
        for vc_i in range(len(node_vc_list)):
            node_vc = node_vc_list[vc_i]
            root_node_vc.append(node_vc[root_node_index])
        if len(root_node_vc) > 0:
            root_node_vc = np.asarray(root_node_vc)
            root_node_data = np.concatenate((root_node_xyz, root_node_vc), axis=0)
        else:
            root_node_data = root_node_xyz
        data[root_node_i, :] = root_node_data
    save_csv_file(data=data, filename=visualisation_dir+filename, column_name_list=column_name_list)


# def write_root_node_csv(filename, node_xyz, node_vc_list, root_node_index_list, xyz_name_list, vc_name_list, verbose, visualisation_dir):
#     if verbose:
#         print('Write out root nodes as CSV ', filename, ' , at ', visualisation_dir)
#     # Save the available LV and RV root nodes
#     column_name_list = []
#     save_csv_file(data, filename, column_name_list=None)
#     with open(visualisation_dir + filename + '.csv', 'w') as f:
#         f.write('"x","y","z"\n')
#         for root_node_i in range(0, len(root_node_index_list)):
#             f.write(
#                 str(node_xyz[root_node_index_list[root_node_i], 0]) + ',' + str(node_xyz[root_node_index_list[root_node_i], 1]) + ',' + str(
#                     node_xyz[root_node_index_list[root_node_i], 2]) + '\n')


def save_purkinje_network_backup(dir, casename, edges, lvnodes, ):
    # Save the pseudo-Purkinje networks considered during the inference for visulalisation and plotting purposes - LV
    lvedges_indexes = np.all(np.isin(edges, lvnodes), axis=1)
    lv_edges = edges[lvedges_indexes, :]
    lvPK_edges_indexes = np.zeros(lv_edges.shape[0], dtype=np.bool_)
    lv_root_to_PKpath_mat = lv_PK_path_mat[lvActnode_ids, :]
    for i in range(0, lv_edges.shape[0], 1):
        for j in range(0, lv_root_to_PKpath_mat.shape[0], 1):
            if np.all(np.isin(lv_edges[i, :],
                              lvnodes[lv_root_to_PKpath_mat[j, :][lv_root_to_PKpath_mat[j, :] != get_nan_value()]])):
                lvPK_edges_indexes[i] = 1
                break
    LV_PK_edges = lv_edges[lvPK_edges_indexes, :]
    # Save the available LV Purkinje network
    with open(figPath + meshName + '_available_LV_PKnetwork.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS ' + str(
            nodesXYZ.shape[0]) + ' double\n')
        for i in range(0, nodesXYZ.shape[0], 1):
            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) + '\n')
        f.write('LINES ' + str(LV_PK_edges.shape[0]) + ' ' + str(LV_PK_edges.shape[0] * 3) + '\n')
        for i in range(0, LV_PK_edges.shape[0], 1):
            f.write('2 ' + str(LV_PK_edges[i, 0]) + ' ' + str(LV_PK_edges[i, 1]) + '\n')
        f.write('POINT_DATA ' + str(nodesXYZ.shape[0]) + '\n\nCELL_DATA ' + str(LV_PK_edges.shape[0]) + '\n')
    # Save the available LV and RV root nodes
    with open(figPath + meshName + '_available_root_nodes.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(lvActivationIndexes)):
            f.write(
                str(nodesXYZ[lvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[lvActivationIndexes[i], 1]) + ',' + str(
                    nodesXYZ[lvActivationIndexes[i], 2]) + '\n')
        for i in range(0, len(rvActivationIndexes)):
            f.write(
                str(nodesXYZ[rvActivationIndexes[i], 0]) + ',' + str(nodesXYZ[rvActivationIndexes[i], 1]) + ',' + str(
                    nodesXYZ[rvActivationIndexes[i], 2]) + '\n')
    # RV
    rvedges_indexes = np.all(np.isin(edges, rvnodes), axis=1)
    rv_edges = edges[rvedges_indexes, :]
    rvPK_edges_indexes = np.zeros(rv_edges.shape[0], dtype=np.bool_)
    rv_root_to_PKpath_mat = rv_PK_path_mat[rvActnode_ids, :]
    # rv_edges_crossing_to_roots = rv_edges_crossing[rvActnode_ids, :]
    # rv_edges_crossing_to_roots = rv_edges_crossing_to_roots[np.logical_not(np.any(rv_edges_crossing_to_roots == nan_value, axis=1)), :]
    # rv_edges_crossing_to_roots = rvnodes[rv_edges_crossing_to_roots]
    for i in range(0, rv_edges.shape[0], 1):
        for j in range(0, rv_root_to_PKpath_mat.shape[0], 1):
            if np.all(np.isin(rv_edges[i, :],
                              rvnodes[rv_root_to_PKpath_mat[j, :][rv_root_to_PKpath_mat[j, :] != nan_value]])):
                rvPK_edges_indexes[i] = 1
                break
    RV_PK_edges = rv_edges[rvPK_edges_indexes, :]
    # RV_PK_edges = np.concatenate((RV_PK_edges, rv_edges_his), axis=0)
    # RV_PK_edges = np.concatenate((RV_PK_edges, rv_edges_crossing_to_roots), axis=0)
    # # Save the available RV Purkinje network
    with open(figPath + meshName + '_available_RV_PKnetwork.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS ' + str(
            nodesXYZ.shape[0]) + ' double\n')
        for i in range(0, nodesXYZ.shape[0], 1):
            f.write('\t' + str(nodesXYZ[i, 0]) + '\t' + str(nodesXYZ[i, 1]) + '\t' + str(nodesXYZ[i, 2]) + '\n')
        f.write('LINES ' + str(RV_PK_edges.shape[0]) + ' ' + str(RV_PK_edges.shape[0] * 3) + '\n')
        for i in range(0, RV_PK_edges.shape[0], 1):
            f.write('2 ' + str(RV_PK_edges[i, 0]) + ' ' + str(RV_PK_edges[i, 1]) + '\n')
        f.write('POINT_DATA ' + str(nodesXYZ.shape[0]) + '\n\nCELL_DATA ' + str(RV_PK_edges.shape[0]) + '\n')


def export_ensight_scalar_per_node(dir, casename, data, dataname):
    with open(dir+casename+'.ensi.'+dataname, 'w') as f:
        f.write(casename+' Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
        for i in range(data.shape[0]):
            f.write(str(data[i]) + '\n')


def export_ensight_scalar_per_cell(dir, casename, data, dataname):
    with open(dir+casename+'.ensi.'+dataname, 'w') as f:
        f.write(casename+' Ensight Gold --- Scalar per-cell variables file\npart\n\t1\ntetra4\n')
        for i in range(data.shape[0]):
            f.write(str(data[i]) + '\n')


def export_ensight_geometry(dir, casename, node_xyz, elem):
    if np.amin(elem) == 0:
        elem = elem + 1 # Ensight takes node indices starting from 1.
    with open(dir+casename+'.ensi.geo', 'w') as f:
        f.write(
            'Problem name:  ' + casename +'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n' + str(
                node_xyz.shape[0]) + '\n')
        for node_i in range(0, node_xyz.shape[0], 1):
            f.write(str(node_i + 1) + '\n')
        for xyz_i in range(0, node_xyz.shape[1], 1):
            for node_i in range(0, node_xyz.shape[0]):
                f.write(str(node_xyz[node_i, xyz_i]) + '\n')
        if elem.shape[1] == 4:
            f.write('tetra4\n  ')
        elif elem.shape[1] == 2:
            f.write('bar2\n  ')
        f.write(str(elem.shape[0]) + '\n')
        for elem_i in range(0, elem.shape[0], 1):
            f.write('  ' + str(elem_i + 1) + '\n')
        for elem_i in range(0, elem.shape[0], 1):
            line_str = ''
            for xyz_i in range(0, elem.shape[1], 1):
                line_str = line_str + str(elem[elem_i, xyz_i]) + '\t'
            f.write(line_str + '\n')


def export_ensight_initialise_case(dir, casename):
    with open(dir+casename+'.ensi.case', 'w') as f:
        f.write(
            '#\n# TwavePersonalisation generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\t'+casename+'\n#\n')
        f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t' + casename + '.ensi.geo\nVARIABLE\n')


def export_ensight_add_case_node(dir, casename, dataname):
    with open(dir+casename+'.ensi.case', 'a+') as f:
        f.write('scalar per node: 1	'+dataname+'\t' + casename + '.ensi.'+dataname+'\n')


def export_ensight_add_case_element(dir, casename, dataname):
    with open(dir+casename+'.ensi.case', 'a+') as f:
        f.write('scalar per element: 1	'+dataname+'\t' + casename + '.ensi.'+dataname+'\n')


def export_ensight_timeseries_case(dir, casename, dataname_list, vm_list, dt, nodesxyz, tetrahedrons):
    export_ensight_geometry(dir=dir, casename=casename, node_xyz=nodesxyz, elem=tetrahedrons)
    for dataname_i in range(len(dataname_list)):
        dataname = dataname_list[dataname_i]
        num_steps = vm_list[dataname_i].shape[1]
        with open(dir+casename+dataname+'.ensi.case', 'w') as f:
            f.write(
                '#\n# TwavePersonalisation generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\t' + casename + '\n#\n')
            f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t' + casename + '.ensi.geo\nVARIABLE\n\n')
            f.write('scalar per node: 1	' + dataname + '\t' + casename + '.ensi.' + dataname + '-******\n')
            f.write('\nTIME\ntime set:\t1\nnumber of steps:\t'+str(num_steps-1)+'\nfilename start number:\t1\nfilename increment:\t1\nfilename numbers:\n')
            files_per_row = 12 # This is an Ensight specific number of filename numbers per row.
            for file_row_i in range(int(math.ceil(num_steps / files_per_row))):
                for file_per_row_i in range(files_per_row):
                    file_i = file_row_i * files_per_row + file_per_row_i
                    f.write('\t' +"%d" % (file_i))
                    export_ensight_scalar_per_node(dir=dir, casename=casename, data=vm_list[dataname_i][:, file_i],
                                                   dataname=dataname+'-' + "%06d" % (file_i))
                    if (file_i + 1) == num_steps:  # exit before it tries to write a file that is not there
                        break
                f.write('\n')

            f.write('time values:\n')
            files_per_row = 14 # This is an Ensight specific number of time values per row.
            for file_row_i in range(int(math.ceil(num_steps / files_per_row))):
                for file_per_row_i in range(files_per_row):
                    file_i = file_row_i * files_per_row + file_per_row_i
                    f.write('\t' + "%.3f" % (file_i*dt))
                    if (file_i + 1) == num_steps:  # exit before it tries to write a file that is not there
                        break
                f.write('\n')


def write_geometry_to_ensight_with_fields(geometry, node_field_list, node_field_name_list, subject_name, verbose, visualisation_dir):
    node_xyz = geometry.get_node_xyz()
    tetra = geometry.get_tetra()
    write_geometry_data_to_ensight_with_fields(node_xyz, node_field_list, node_field_name_list, subject_name, tetra,
                                               verbose, visualisation_dir)



def write_geometry_data_to_ensight_with_fields(node_xyz, node_field_list, node_field_name_list, subject_name, tetra,
                                               verbose, visualisation_dir):
    if verbose:
        print('Write out geometric fields as Ensight at ', visualisation_dir)
    export_ensight_initialise_case(dir=visualisation_dir, casename=subject_name)
    export_ensight_geometry(dir=visualisation_dir, casename=subject_name, node_xyz=node_xyz, elem=tetra)
    for node_field_i in range(len(node_field_name_list)):
        attribute_name = node_field_name_list[node_field_i]
        attribute_value = node_field_list[node_field_i]
        if attribute_value is not None:
            export_ensight_scalar_per_node(dir=visualisation_dir, casename=subject_name,
                                           data=attribute_value, dataname=attribute_name)
            export_ensight_add_case_node(dir=visualisation_dir, casename=subject_name, dataname=attribute_name)


def add_attribute_to_ensight(data, data_name, casename, visualisation_dir):
    export_ensight_scalar_per_node(dir=visualisation_dir, casename=casename,
                                   data=data, dataname=data_name)
    export_ensight_add_case_node(dir=visualisation_dir, casename=casename, dataname=data_name)


def write_geometry_to_ensight(geometry, subject_name, resolution, verbose, visualisation_dir):
    # attribute_name_list = list(geometry.__dict__.keys())
    if verbose:
        print('Write out geometric fields as Ensight at ', visualisation_dir)
    casename = subject_name + '_' + resolution
    export_ensight_initialise_case(dir=visualisation_dir, casename=casename)
    node_xyz = geometry.get_node_xyz()
    tetra = geometry.get_tetra()
    export_ensight_geometry(dir=visualisation_dir, casename=casename, node_xyz=node_xyz, elem=tetra)
    # # lat
    # attribute_name = 'lat'
    # if hasattr(geometry, attribute_name):
    #     attribute_value = geometry.__dict__[attribute_name]
    #     if attribute_value is not None:
    #         export_ensight_scalar_per_node(dir=visualisation_dir, casename=casename,
    #                                            data=attribute_value, dataname=attribute_name)
    #         export_ensight_add_case_node(dir=visualisation_dir, casename=casename, dataname=attribute_name)
    '''ventricular coordinates'''
    try:
        node_vc = geometry.get_node_vc()
        key_list = list(node_vc.keys())
        for key in key_list:
            add_attribute_to_ensight(data=node_vc[key], data_name=key, casename=casename, visualisation_dir=visualisation_dir)
    except NotImplementedError:
        print('No node_vc in geometry')

    '''lv endocardial surface'''
    try:
        node_lvendo = geometry.get_node_lvendo()
        lv_endo_field = np.zeros((node_xyz.shape[0]))
        lv_endo_field[node_lvendo] = 1.0
        add_attribute_to_ensight(data=lv_endo_field, data_name='lvendo', casename=casename, visualisation_dir=visualisation_dir)
    except NotImplementedError:
        print('No node_lvendo in geometry')

    '''rv endocardial surface'''
    try:
        node_rvendo = geometry.get_node_rvendo()
        rv_endo_field = np.zeros((node_xyz.shape[0]))
        rv_endo_field[node_rvendo] = 2.0
        add_attribute_to_ensight(data=rv_endo_field, data_name='rvendo', casename=casename,
                                 visualisation_dir=visualisation_dir)
    except NotImplementedError:
        print('No node_rvendo in geometry')

    '''fibre'''
    try:
        axis_name_list = ['x', 'y', 'z']
        node_fibre = geometry.get_node_fibre()
        for axis_i in range(node_fibre.shape[1]):
            axis_name = axis_name_list[axis_i]
            add_attribute_to_ensight(data=node_fibre[:, axis_i], data_name='fibre_'+axis_name, casename=casename,
                                     visualisation_dir=visualisation_dir)
    except NotImplementedError:
        print('No node_fibre in geometry')
    '''sheet'''
    try:
        axis_name_list = ['x', 'y', 'z']
        node_sheet = geometry.get_node_sheet()
        for axis_i in range(node_sheet.shape[1]):
            axis_name = axis_name_list[axis_i]
            add_attribute_to_ensight(data=node_sheet[:, axis_i], data_name='sheet_' + axis_name, casename=casename,
                                     visualisation_dir=visualisation_dir)
    except NotImplementedError:
        print('No node_sheet in geometry')
    '''normal'''
    try:
        axis_name_list = ['x', 'y', 'z']
        node_normal = geometry.get_node_normal()
        for axis_i in range(node_normal.shape[1]):
            axis_name = axis_name_list[axis_i]
            add_attribute_to_ensight(data=node_normal[:, axis_i], data_name='normal_' + axis_name, casename=casename,
                                     visualisation_dir=visualisation_dir)
    except NotImplementedError:
        print('No node_normal in geometry')

    # vc
    # attribute_name = 'node_vc'
    # if hasattr(geometry, attribute_name):
    #     attribute_value = geometry.__dict__[attribute_name]
    #     if attribute_value is not None:
    #         key_list = list(attribute_value.keys())
    #         for key in key_list:
    #             export_ensight_scalar_per_node(dir=visualisation_dir, casename=casename,
    #                                            data=attribute_value[key], dataname=key)
    #             export_ensight_add_case_node(dir=visualisation_dir, casename=casename, dataname=key)
    # endocardial surfaces
    # attribute_name = 'node_lvendo'
    # if hasattr(geometry, attribute_name):
    #     attribute_value = geometry.__dict__[attribute_name]
    #     if attribute_value is not None:
    #         lv_endo_field = np.zeros((node_xyz.shape[0]))
    #         lv_endo_field[attribute_value] = 1.0
    #         export_ensight_scalar_per_node(dir=visualisation_dir, casename=casename,
    #                                            data=lv_endo_field, dataname=attribute_name)
    #         export_ensight_add_case_node(dir=visualisation_dir, casename=casename, dataname=attribute_name)
    # attribute_name = 'node_rvendo'
    # if hasattr(geometry, attribute_name):
    #     attribute_value = geometry.__dict__[attribute_name]
    #     if attribute_value is not None:
    #         rv_endo_field = np.zeros((geometry.node_xyz.shape[0]))
    #         rv_endo_field[attribute_value] = 2.0
    #         export_ensight_scalar_per_node(dir=visualisation_dir, casename=casename,
    #                                        data=rv_endo_field, dataname=attribute_name)
    #         export_ensight_add_case_node(dir=visualisation_dir, casename=casename, dataname=attribute_name)
    # fibres
    # The fibres need to be in the shape of [node, xyz, fibre-sheet-normal] for the Eikonal to operate on them
    # attribute_name = 'node_fibre_sheet_normal'
    # if hasattr(geometry, attribute_name):
    #     attribute_value = geometry.__dict__[attribute_name]
    #     if attribute_value is not None:
    #         coordinate_list = ['x', 'y', 'z']
    #         key_list = ['fibre', 'sheet', 'normal']
    #         for coordinate_i in range(len(coordinate_list)):
    #             coordinate = coordinate_list[coordinate_i]
    #             for key_i in range(len(key_list)):
    #                 key = key_list[key_i]
    #                 dataname = key +'_' + coordinate
    #                 export_ensight_add_case_node(dir=visualisation_dir, casename=casename, dataname=dataname)
    #                 export_ensight_scalar_per_node(dir=visualisation_dir, casename=casename,
    #                                                data=geometry.attribute_value[:, coordinate_i, key_i],
    #                                                dataname=dataname)


# def write_optimised_fields_alya(sim, coarse_indices, best_gradient, dir, alya_name, nodal_mask):
#     print('Writing optimised fields to Alya format')
#     # Writes out the optimised ionic current scaling factor fields for 1) visualisation in paraview and 2) Alya simulations.
#     nodes_biomarkers_dataframe = sim.generate_nodes_biomarkers(apd_gradients=best_gradient)#, nodal_mask=nodal_mask)
#     for column in nodes_biomarkers_dataframe.columns:
#         filename = dir+alya_name+'.'+column
#         data = nodes_biomarkers_dataframe[column].values
#         # print(coarse_indices.shape[0])
#         # print(data.shape[0])
#         # print(max(coarse_indices))
#         # raise
#         # return cobiveco_ab_rt_tm_tv_aprt_rvlv[cobiveco_indices, :]
#         data_fine = data[coarse_indices]
#         np.savetxt(filename, data_fine, delimiter=',')


# def generate_nodes_biomarkers(self, apd_gradients):
#     nodes_apd = np.asarray(self.get_apd_map(apd_gradients), dtype=int)
#     nodes_endo_celltype, nodes_mid_celltype, nodes_epi_celltype = self.assign_nodes_celltype_epi_mid_endo(nodes_apd)
#     return self.get_nodes_biomarkers(nodes_apd=nodes_apd, nodes_endo_celltype=nodes_endo_celltype,
#                                      nodes_mid_celltype=nodes_mid_celltype,
#                                      nodes_epi_celltype=nodes_epi_celltype)


def write_list_to_file(data_list, filename):
    # open file in write mode
    with open(filename, 'w') as fp:
        for item in data_list:
            # write each item on a new line
            fp.write("%s\n" % item)


def write_node_xyz_csv(casename, dir, node_xyz):
    # Save the available LV and RV root nodes
    with open(dir + casename + '.csv', 'w') as f:
        f.write('"x","y","z"\n')
        for i in range(0, node_xyz.shape[0]):
            f.write(str(node_xyz[i, 0]) + ',' + str(node_xyz[i, 1]) + ',' + str(node_xyz[i, 2]) + '\n')


def read_monoalg_vm_file(filename, nb_node):
    # import time
    row_offset = 4
    # nb_row = nb_node+row_offset
    # time_s = time.time()
    vm = np.squeeze(pd.read_csv(filename, delimiter=',', skiprows=row_offset, names=['colA'], nrows=nb_node,
                                dtype=float).values)
    # time_e = time.time()
    # print('1 time ', time_e-time_s)
    # print('vm_2 ', vm_2.shape)
    # # print('vm_2 ', vm_2[:10])
    # print('nb_node ', nb_node)
    vm = vm[:int(nb_node)]

    # time_s = time.time()
    # vm = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=row_offset, skip_footer=0, converters=None,
    #                      missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None,
    #                      replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None,
    #                      usemask=False, loose=True, invalid_raise=True)
    # time_e = time.time()
    # print('2 time ', time_e - time_s)
    # print('vm ', vm.shape)
    # print('vm ', vm[:10])
    #
    # # print('nb_node ', nb_node)
    # # print('1 vm ', vm.shape)
    # vm = vm[:int(nb_node)]
    # print('Check code ', np.all(vm==vm_2))
    # print(np.sum(np.abs(vm-vm_2)))
    # raise()
    # # print('vms ')
    # # print('2 vm ', vm.shape)
    # # print()
    # # print(vm)
    #
    # # start = txt.find(tag_start)
    # # end = txt.find(tag_end, start)
    # # tagged_txt = txt[txt.find('\n', start):end]
    # # tagged_txt = tagged_txt.replace('\n', ' ').split(
    # #     None)  # PYTHON documentation: If sep is not specified or is None, a different splitting algorithm is applied: runs of consecutive whitespace are regarded
    # # # as a single separator, and the result will contain no empty strings at the start or end if the string has leading or trailing whitespace.
    # # return np.array([float(x) for x in tagged_txt])
    # # pd.read_csv(filename, delimiter=',', header=None).values.transpose()[0].astype(float)
    #
    return vm


def read_monoalg_geo_ensight(ensight_dir):
    # READ XYZ
    geo_file_name = ensight_dir + get_monoalg_geo_file_name()
    print('geo_file_name ', geo_file_name)
    assert os.path.isfile(geo_file_name)
    with open(geo_file_name, 'r') as f:
        data = f.readlines()
    nnodes = int(data[8])
    nodes_xyz = np.zeros((nnodes, 3))
    nodes_xyz[:, 0] = data[9:nnodes + 9]
    nodes_xyz[:, 1] = data[nnodes + 9:nnodes * 2 + 9]
    nodes_xyz[:, 2] = data[nnodes * 2 + 9:nnodes * 3 + 9]
    idx = int(np.where(np.array(data) == 'hexa8\n')[0])
    nelems = int(data[idx + 1])

    elems = np.zeros((nelems, 8), dtype=int)
    cell_centres = np.zeros((nelems, 3))
    for i in range(0, nelems):
        elems[i, :] = [int(x) - 1 for x in data[i + idx + 2].split()]
        cell_centres[i, :] = np.mean(nodes_xyz[elems[i, :], :], axis=0)
    # print('cell_centres ', cell_centres.shape)
    return cell_centres


def read_monoalg_vm_ensight(ensight_dir, nb_node):
    # # READ XYZ
    # geo_file_name = ensight_dir + get_monoalg_geo_file_name()
    # print('geo_file_name ', geo_file_name)
    # assert os.path.isfile(geo_file_name)
    # with open(geo_file_name, 'r') as f:
    #     data = f.readlines()
    # nnodes = int(data[8])
    # nodes_xyz = np.zeros((nnodes, 3))
    # nodes_xyz[:, 0] = data[9:nnodes + 9]
    # nodes_xyz[:, 1] = data[nnodes + 9:nnodes * 2 + 9]
    # nodes_xyz[:, 2] = data[nnodes * 2 + 9:nnodes * 3 + 9]
    # idx = int(np.where(np.array(data) == 'hexa8\n')[0])
    # nelems = int(data[idx + 1])
    #
    # elems = np.zeros((nelems, 8), dtype=int)
    # cell_centres = np.zeros((nelems, 3))
    # for i in range(0, nelems):
    #     elems[i, :] = [int(x) - 1 for x in data[i + idx + 2].split()]
    #     cell_centres[i, :] = np.mean(nodes_xyz[elems[i, :], :], axis=0)
    # print('cell_centres ', cell_centres.shape)
    # nb_node = cell_centres.shape[0]

    # READ VMS
    # This function assumes that the VMs are sampled at 1000Hz (time resolution of 1 ms)
    file_name_tag = get_monoalg_vm_file_name_tag()
    filename_list = [filename for filename in os.listdir(ensight_dir) if
                     os.path.isfile(os.path.join(ensight_dir, filename)) and (file_name_tag in filename)]
    time_list = [get_monoalg_vm_file_time(filename) for filename in filename_list]
    # Sort by time tag value
    time_sort = np.argsort(time_list)
    filename_list = [filename_list[time_sort[i]] for i in range(len(time_sort))]
    time_list = [time_list[time_sort[i]] for i in range(len(time_sort))]
    # print('filename_list ', filename_list)
    # print('time_list ', time_list)
    # # print('time_sort ', time_sort)
    # print('filename_list[0] ', filename_list[0])
    # # vm_0 = read_csv_file(filename=ensight_dir + filename_list[0], skiprows=10, usecols=None)
    vm_0 = read_monoalg_vm_file(filename=ensight_dir + filename_list[0], nb_node=nb_node)
    # print('vm_0 ', vm_0.shape)
    assert nb_node == vm_0.shape[0]
    # print('vm_0 ', vm_0.shape)
    # print('1vm_0 ', vm_0[0:10])
    # print('2vm_0 ', vm_0[-10:-1])
    vm = pymp.shared.array((vm_0.shape[0], len(filename_list)), dtype=float)

    vm[:, 0] = vm_0
    threadsNum = multiprocessing.cpu_count()
    with pymp.Parallel(min(threadsNum, len(filename_list)-1)) as p1:
        for time_i in p1.range(1, len(filename_list)):  # Starting from 1 because 0 is done outside the loop
            vm[:, time_i] = read_monoalg_vm_file(filename=ensight_dir + filename_list[time_i], nb_node=nb_node) #read_csv_file(filename=filename_list[0], skiprows=4, usecols=None)
    return vm


def read_time_csv_fields(anatomy_subject_name, csv_dir, file_name_tag, node_xyz_filename):
    print('Reading Vm')
    time = pd.read_csv(csv_dir + 'timeset_1.csv', delimiter=',', header=None).values.transpose()[0].astype(float) #np.loadtxt(csv_dir + 'timeset_1.csv', delimiter=',', dtype=float)
    assert 0.00099 < time[1] - time[0] < 0.0011, "Time resolution is NOT 1 ms"
    time_index = pd.read_csv(csv_dir + 'timeindices_1.csv', delimiter=',', header=None).values.transpose()[0].astype(int) #np.loadtxt(csv_dir + 'timeindices_1.csv', delimiter=',', dtype=float).astype(int)
    assert time.shape == time_index.shape, "Landmine in function read_csv_fields"
    node_xyz = pd.read_csv(node_xyz_filename, delimiter=',', header=None).values #np.loadtxt(node_xyz_filename, delimiter=',', dtype=float)
    filenames = np.array([f for f in os.listdir(csv_dir) if
                          os.path.isfile(os.path.join(csv_dir, f)) and (file_name_tag in f)])
    # TODO this is no longer generic, for example cannot output CALCIUM and INTRA any more!
    # field_names_types = []
    # for filename in filenames:
    #     field_names_types.append([filename.split('.')[2].split('-')[0], filename.split('.')[1]])
    # field_names_types = np.unique(field_names_types, axis=0)
    # TODO change interface to ask for field name and field type
    field_name = file_name_tag.split('.')[1]
    field_type = file_name_tag.split('.')[0]
    # print('field name types: ', field_names_types)
    # for field_i in range(field_names_types.shape[0]):
    temp = np.zeros((node_xyz.shape[0], time.shape[0]))
    # field_name = field_names_types[field_i, 0]
    # field_type = field_names_types[field_i, 1]
    temp_shared = pymp.shared.array((temp.shape), dtype=float)
    threadsNum = multiprocessing.cpu_count()
    with pymp.Parallel(min(threadsNum, time_index.shape[0])) as p1:
        for time_i in p1.range(time_index.shape[0]):
    # if True:
    #     for time_i in range(time_index.shape[0]):
            index = '{:06d}'.format(time_index[time_i])
            filename = csv_dir + anatomy_subject_name + '.' + field_type + '.' + field_name + '-' + index + '.csv'
            temp_shared[:, time_i] = pd.read_csv(filename, delimiter=',', header=None).values.transpose()[0] #np.loadtxt(filename, delimiter=',').astype(float)
    temp = temp_shared
    return temp, node_xyz


def read_csv_field(data_dir, file_name_tag):
    filename_list = np.array([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and (file_name_tag in f)])
    print('filename_list ', filename_list)
    filename_list = sorted(filename_list)
    print('filename_list ', filename_list)
    data_list = []
    for filename in filename_list:
        data_list.append(np.loadtxt(data_dir + filename, delimiter=',', dtype=float))
    return data_list

# EOF

# The conduction system is meant to be part of a geometry, thus, it indexes the geometry with its index values,
# and it indexes itself with meta_index values.
from warnings import warn

import numpy as np

from distance_functions import djikstra
from utils import get_nan_value, find_first, index_list, get_edge_from_path_list, \
    remove_nan_and_make_list_of_array, from_meta_index_list_to_index, get_edge_list_from_path, \
    get_root_node_max_ab_cut_threshold, get_vc_ab_cut_name, get_vc_rt_name, get_base_ab_cut_value, get_apex_ab_cut_value, \
    get_rv_apical_ab_cut_threshold, get_lv_apical_ab_cut_threshold, get_freewall_center_rt_value, \
    get_freewall_posterior_rt_value, get_freewall_anterior_rt_value


class EmptyConductionSystem:
    def __init__(self, verbose):
        self.verbose = verbose

    def get_candidate_root_node_index(self):
        return None

    def get_candidate_root_node_distance(self):
        return None

    def get_candidate_root_node_time(self, purkinje_speed):
        return None

    def get_nb_candidate_root_node(self):
        return None

    def get_lv_rv_candidate_root_node_index(self):
        return None, None

    def get_lv_rv_candidate_purkinje_edge(self):
        return None, None


class ConductionSystem(EmptyConductionSystem):
    """Root node activation is defined as distances to enable parameterising the Purkinje speed on-the-fly
    Root nodes are always assembled as lv-rv"""
    def __init__(self, candidate_root_node_index, candidate_root_node_distance, verbose):
        super().__init__(verbose=verbose)
        if verbose:
            print('Initialising ConductionSystem')
            print('Number of candidate root nodes: ', candidate_root_node_index.shape)
        self.candidate_root_node_index = candidate_root_node_index
        self.candidate_root_node_distance = candidate_root_node_distance

    def get_nb_candidate_root_node(self):
        return self.candidate_root_node_index.shape[0]

    def get_candidate_root_node_meta_bool_index(self):
        return np.ones((self.get_nb_candidate_root_node()), dtype=bool)

    """Root node global indexes"""
    def get_selected_root_node_index(self, root_node_meta_index):
        return self.candidate_root_node_index[root_node_meta_index]

    def get_candidate_root_node_index(self):
        return self.get_selected_root_node_index(root_node_meta_index=self.get_candidate_root_node_meta_bool_index())

    """Root node distances to His"""
    def get_selected_root_node_distance(self, root_node_meta_index):
        return self.candidate_root_node_distance[root_node_meta_index]

    def get_candidate_root_node_distance(self):
        return self.get_selected_root_node_distance(root_node_meta_index=self.get_candidate_root_node_meta_bool_index())

    """Root node activation times"""
    def get_selected_root_node_time(self, purkinje_speed, root_node_meta_index):
        selected_root_node_distance = self.get_selected_root_node_distance(root_node_meta_index=root_node_meta_index)
        return selected_root_node_distance/purkinje_speed

    def get_candidate_root_node_time(self, purkinje_speed):
        return self.get_selected_root_node_time(purkinje_speed=purkinje_speed, root_node_meta_index=self.get_candidate_root_node_meta_bool_index())

    """LV and RV Root node splitting"""
    def get_lv_rv_selected_root_node_meta_index(self, root_node_meta_index):
        raise NotImplementedError

    def get_lv_rv_candidate_root_node_meta_index(self):
        lv_candidate_root_node_meta_index, rv_candidate_root_node_meta_index = self.get_lv_rv_selected_root_node_meta_index(
            root_node_meta_index=self.get_candidate_root_node_meta_bool_index())
        return lv_candidate_root_node_meta_index, rv_candidate_root_node_meta_index

    def get_lv_rv_selected_root_node_index(self, root_node_meta_index):
        raise NotImplementedError

    def get_lv_rv_candidate_root_node_index(self):
        lv_candidate_root_node_index, rv_candidate_root_node_index = self.get_lv_rv_selected_root_node_index(
            root_node_meta_index=self.get_candidate_root_node_meta_bool_index())
        return lv_candidate_root_node_index, rv_candidate_root_node_index

    def get_lv_rv_selected_purkinje_edge(self, root_node_meta_index):
        raise NotImplementedError

    def get_lv_rv_candidate_purkinje_edge(self):
        lv_pk_edge, rv_pk_edge = self.get_lv_rv_selected_purkinje_edge(
            root_node_meta_index=self.get_candidate_root_node_meta_bool_index())
        return lv_pk_edge, rv_pk_edge


class DjikstraConductionSystemVC(ConductionSystem):
    """This class is a conduction system that gets defined from Ventricular Coordinates using Djikstra.
    Root nodes are always assembled as lv-rv"""
    def __init__(self, approx_djikstra_purkinje_max_path_len, geometry, lv_candidate_root_node_meta_index,
                 rv_candidate_root_node_meta_index,
                 purkinje_max_ab_cut_threshold,  # The Purkinje fibres cannot grow all the way to the base
                 verbose):
        if verbose:
            print('Generating Purkinje tree')
        # purkinje_max_cut_ab_threshold = 0.8  # The Purkinje fibres cannot grow all the way to the base
        node_xyz = geometry.get_node_xyz()
        node_lvendo_fast = geometry.get_node_lvendo_fast()
        # node_vc = geometry.get_node_vc()
        # node_vc_ab_cut = node_vc[vc_ab_cut_name]
        # node_vc = None  # Clear Arguments to prevent Argument recycling
        # lv_candidate_root_node_meta_index = generate_candidate_root_nodes_in_cavity(
        #     basal_cavity_nodes_xyz=node_xyz[node_lvendo_fast, :],
        #     basal_cavity_vc_ab_cut=node_vc_ab_cut[node_lvendo_fast],
        #     inter_root_node_distance=lv_inter_root_node_distance,
        #     purkinje_max_cut_ab_threshold=purkinje_max_cut_ab_threshold)
        self.nb_lv_root_node = lv_candidate_root_node_meta_index.shape[0]
        print('lv_candidate_root_node_meta_index ', lv_candidate_root_node_meta_index.shape)
        print()
        print('rv_candidate_root_node_meta_index ', rv_candidate_root_node_meta_index)
        lv_candidate_root_node_index = node_lvendo_fast[lv_candidate_root_node_meta_index]
        node_rvendo_fast = geometry.get_node_rvendo_fast()
        # rv_candidate_root_node_meta_index = generate_candidate_root_nodes_in_cavity(
        #     basal_cavity_nodes_xyz=node_xyz[node_rvendo_fast, :],
        #     basal_cavity_vc_ab_cut=node_vc_ab_cut[node_rvendo_fast],
        #     inter_root_node_distance=rv_inter_root_node_distance,
        #     purkinje_max_cut_ab_threshold=purkinje_max_cut_ab_threshold)
        self.nb_rv_root_node = rv_candidate_root_node_meta_index.shape[0]
        rv_candidate_root_node_index = node_rvendo_fast[rv_candidate_root_node_meta_index]
        candidate_root_node_index = np.concatenate((lv_candidate_root_node_index, rv_candidate_root_node_index), axis=0)
        # Clear Arguments to prevent Argument recycling
        lv_candidate_root_node_index = None
        rv_candidate_root_node_index = None
        node_vc_ab_cut = None
        # Create Purkinje tree
        edge = geometry.get_edge()
        # TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
        # original_approx_djikstra_purkinje_max_path_len = approx_djikstra_purkinje_max_path_len
        # repeat_process = True
        # while repeat_process:
        # TODO give candidate root node indexes to the process so that it has to store and compute less paths
        vc_ab_cut_name = get_vc_ab_cut_name()
        vc_rt_name = get_vc_rt_name()
        node_vc_ab_cut = geometry.get_node_vc_field(vc_name=vc_ab_cut_name)
        node_vc_rt = geometry.get_node_vc_field(vc_name=vc_rt_name)
        lv_pk_distance_to_all, lv_pk_meta_path_to_all_mat, rv_pk_distance_to_all, rv_pk_meta_path_to_all_mat\
            = generate_djikstra_purkinje_tree_from_vc(
            approx_djikstra_max_path_len=approx_djikstra_purkinje_max_path_len, edge=edge, node_lvendo=node_lvendo_fast,
            node_rvendo=node_rvendo_fast, node_xyz=node_xyz, node_ab=node_vc_ab_cut, node_rt=node_vc_rt,
            purkinje_max_ab_cut_threshold=purkinje_max_ab_cut_threshold)
            # # TODO Delete this call and the backup function after all the refactoring is completed
            # lv_pk_distance_to_all_backup, lv_pk_meta_path_to_all_mat_backup, rv_pk_distance_to_all_backup, rv_pk_meta_path_to_all_mat_backup, \
            #     repeat_process_backup = generate_djikstra_purkinje_tree_from_vc_backup(
            #     approx_djikstra_max_path_len=approx_djikstra_purkinje_max_path_len,
            #     edge=edge,
            #     node_lvendo=node_lvendo_fast,
            #     node_rvendo=node_rvendo_fast,
            #     node_xyz=node_xyz,
            #     node_vc=geometry.get_node_vc(),
            #     vc_ab_cut_name=vc_ab_cut_name,
            #     vc_rt_name=vc_rt_name
            #     ,
            #     vc_rvlv_binary_name=vc_rvlv_binary_name
            # )
            # print('Critical consistency check of the Purkinje generation code:')
            # print('lv_pk_distance_to_all ', np.all(lv_pk_distance_to_all == lv_pk_distance_to_all_backup))
            # print('lv_pk_meta_path_to_all_mat ', np.all(lv_pk_meta_path_to_all_mat == lv_pk_meta_path_to_all_mat_backup))
            # print('rv_pk_distance_to_all ', np.all(rv_pk_distance_to_all == rv_pk_distance_to_all_backup))
            # print('rv_pk_meta_path_to_all_mat ', np.all(rv_pk_meta_path_to_all_mat == rv_pk_meta_path_to_all_mat_backup))
            # print('repeat_process ', repeat_process == repeat_process_backup)
            # if not (np.all(lv_pk_distance_to_all == lv_pk_distance_to_all_backup)
            #         and np.all(lv_pk_meta_path_to_all_mat == lv_pk_meta_path_to_all_mat_backup)
            #         and np.all(rv_pk_distance_to_all == rv_pk_distance_to_all_backup)
            #         and np.all(rv_pk_meta_path_to_all_mat == rv_pk_meta_path_to_all_mat_backup)
            #         and repeat_process == repeat_process_backup):
            #     raise()
            # # TODO delete above code
            # if repeat_process:
            #     approx_djikstra_purkinje_max_path_len = approx_djikstra_purkinje_max_path_len*2

        # TODO this test passed successfully on 2023/09/18
        # repeat_process = True
        # approx_djikstra_purkinje_max_path_len = original_approx_djikstra_purkinje_max_path_len
        # while repeat_process:
        #     lv_pk_distance_to_all_old, lv_pk_meta_path_to_all_mat_old, rv_pk_distance_to_all_old, rv_pk_meta_path_to_all_mat_old, \
        #         repeat_process = self.__generate_djikstra_purkinje_tree_from_vc_old(
        #         approx_djikstra_max_path_len=approx_djikstra_purkinje_max_path_len,
        #         edge=edge,
        #         node_lvendo=node_lvendo_fast,
        #         node_rvendo=node_rvendo_fast,
        #         node_xyz=node_xyz,
        #         node_vc=geometry.get_node_vc()
        #     )
        #     if repeat_process:
        #         approx_djikstra_purkinje_max_path_len = approx_djikstra_purkinje_max_path_len * 2
        #
        # print('The most important check in your life: ')
        # print(np.sum(np.abs(lv_pk_distance_to_all-lv_pk_distance_to_all_old)))
        # print(np.sum(np.abs(lv_pk_meta_path_to_all_mat-lv_pk_meta_path_to_all_mat_old)))
        # print(np.sum(np.abs(rv_pk_distance_to_all-rv_pk_distance_to_all_old)))
        # print(np.sum(np.abs(rv_pk_meta_path_to_all_mat-rv_pk_meta_path_to_all_mat_old)))
        # print('hellio')
        # quit()
        # TODO Continue refactoring from here

        # Clear Arguments to prevent Argument recycling
        approx_djikstra_purkinje_max_path_len = None
        geometry = None
        # node_lvendo = None
        # node_rvendo = None
        node_xyz = None
        # Combine biventricular root nodes
        candidate_root_node_distance = np.around(
            np.concatenate((lv_pk_distance_to_all[lv_candidate_root_node_meta_index],
                            rv_pk_distance_to_all[rv_candidate_root_node_meta_index]), axis=0), decimals=4)
        lv_pk_distance_to_all = None    # Clear Arguments to prevent Argument recycling
        rv_pk_distance_to_all = None    # Clear Arguments to prevent Argument recycling
        super().__init__(candidate_root_node_index=candidate_root_node_index,
                         candidate_root_node_distance=candidate_root_node_distance, verbose=verbose)
        # Convert and save purkinje structure as edges
        # TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
        # print('lv_pk_meta_path_to_all_mat ', lv_pk_meta_path_to_all_mat.shape)  # TODO remove
        # aux_print_aux = lv_pk_meta_path_to_all_mat == get_nan_value()  # TODO remove
        # print('aux_print_aux ', aux_print_aux.shape)  # TODO remove
        # aux_print_aux = np.all(aux_print_aux, axis=1)  # TODO remove
        # print('aux_print_aux2 ', aux_print_aux.shape)  # TODO remove
        # print('aux_print_aux3 ', np.sum(aux_print_aux))  # TODO remove
        # print('aux_print_aux ', aux_print_aux)  # TODO remove
        # print('Hey remove this')  # TODO remove
        # raise ()
        lv_pk_meta_path_to_all_mat = remove_nan_and_make_list_of_array(mat=lv_pk_meta_path_to_all_mat)
        rv_pk_meta_path_to_all_mat = remove_nan_and_make_list_of_array(mat=rv_pk_meta_path_to_all_mat)
        # Only candidate root nodes
        # print('lv_pk_meta_path_to_all_mat ', len(lv_pk_meta_path_to_all_mat))
        # print('lv_pk_meta_path_to_all_mat ', lv_pk_meta_path_to_all_mat)
        # print('empty ', np.sum([array_i.size==0 for array_i in lv_pk_meta_path_to_all_mat]))
        # print('Hey remove this')  # TODO remove
        # raise ()
        lv_pk_meta_path_to_candidate_lv_root_node = index_list(list=lv_pk_meta_path_to_all_mat,
                                                               index=lv_candidate_root_node_meta_index)
        rv_pk_meta_path_to_candidate_rv_root_node = index_list(list=rv_pk_meta_path_to_all_mat,
                                                               index=rv_candidate_root_node_meta_index)
        # Clear Arguments to prevent Argument recycling
        lv_pk_meta_path_to_all_mat = None
        lv_candidate_root_node_meta_index = None
        rv_pk_meta_path_to_all_mat = None
        rv_candidate_root_node_meta_index = None
        # From meta endocaridal index to geometry index
        # print('lv_pk_meta_path_to_candidate_lv_root_node ', len(lv_pk_meta_path_to_candidate_lv_root_node))
        # print('lv_pk_meta_path_to_candidate_lv_root_node ', lv_pk_meta_path_to_candidate_lv_root_node)
        # print('Hey remove this') # TODO remove
        # raise()
        self.lv_pk_path_to_candidate_lv_root_node = from_meta_index_list_to_index(
            meta_index_list=lv_pk_meta_path_to_candidate_lv_root_node, index=node_lvendo_fast)
        self.rv_pk_path_to_candidate_rv_root_node = from_meta_index_list_to_index(
            meta_index_list=rv_pk_meta_path_to_candidate_rv_root_node, index=node_rvendo_fast)
        # Clear Arguments to prevent Argument recycling
        node_lvendo_fast = None
        node_rvendo_fast = None
        lv_pk_meta_path_to_candidate_lv_root_node = None
        rv_pk_meta_path_to_candidate_rv_root_node = None
        # From path to edge

        # self.lv_pk_edge = get_edge_from_path_list(path_list=lv_pk_path_to_candidate_lv_root_node)
        # self.rv_pk_edge = get_edge_from_path_list(path_list=rv_pk_path_to_candidate_rv_root_node)
        # Clear Arguments to prevent Argument recycling
        # lv_pk_path_to_candidate_lv_root_node = None  # Clear Arguments to prevent Argument recycling
        # rv_pk_path_to_candidate_rv_root_node = None     # Clear Arguments to prevent Argument recycling

    def get_lv_rv_selected_root_node_meta_index(self, root_node_meta_index):
        # TODO make this part of the code common accross functions
        nb_candidate_root_nodes = self.get_nb_candidate_root_node()
        print('root_node_meta_index.shape[0] ', root_node_meta_index.shape[0])
        print('nb_candidate_root_nodes ', nb_candidate_root_nodes)
        assert nb_candidate_root_nodes == root_node_meta_index.shape[0] # If true, nothing happens
        lv_root_node_meta_bool_index = np.zeros((nb_candidate_root_nodes), dtype=bool)
        rv_root_node_meta_bool_index = np.zeros((nb_candidate_root_nodes), dtype=bool)
        lv_root_node_meta_bool_index[:self.nb_lv_root_node] = root_node_meta_index[:self.nb_lv_root_node]
        rv_root_node_meta_bool_index[self.nb_lv_root_node:] = root_node_meta_index[self.nb_lv_root_node:]
        return lv_root_node_meta_bool_index, rv_root_node_meta_bool_index

    def get_lv_rv_selected_root_node_index(self, root_node_meta_index):
        lv_root_node_meta_bool_index, rv_root_node_meta_bool_index = self.get_lv_rv_selected_root_node_meta_index(root_node_meta_index)
        lv_selected_root_node_index = self.get_selected_root_node_index(
            root_node_meta_index=lv_root_node_meta_bool_index)
        rv_selected_root_node_index = self.get_selected_root_node_index(
            root_node_meta_index=rv_root_node_meta_bool_index)
        # print('nb_candidate_root_nodes ', nb_candidate_root_nodes)
        # print()
        # print('lv_candidate_root_node_index ', lv_selected_root_node_index, ' len ', len(lv_selected_root_node_index))
        # print()
        # print('rv_candidate_root_node_index ', rv_selected_root_node_index, ' len ', len(rv_selected_root_node_index))
        # print()
        # print('self.lv_pk_path_to_candidate_lv_root_node ', len(self.lv_pk_path_to_candidate_lv_root_node))
        # print('self.lv_pk_path_to_candidate_lv_root_node ', self.lv_pk_path_to_candidate_lv_root_node)
        # print()
        # print('self.rv_pk_path_to_candidate_rv_root_node ', len(self.rv_pk_path_to_candidate_rv_root_node))
        # print('self.rv_pk_path_to_candidate_rv_root_node ', self.rv_pk_path_to_candidate_rv_root_node)
        # print()
        return lv_selected_root_node_index, rv_selected_root_node_index

    # def get_lv_rv_candidate_root_node_index(self):
    #     nb_candidate_root_nodes = self.get_nb_candidate_root_node()
    #     lv_candidate_root_node_index, rv_candidate_root_node_index = self.get_lv_rv_selected_root_node_index(
    #         root_node_meta_bool_index=np.ones((nb_candidate_root_nodes), dtype=bool))
    #     return lv_candidate_root_node_index, rv_candidate_root_node_index

    # def get_lv_rv_candidate_purkinje_edge(self):
    #     nb_candidate_root_nodes = self.get_nb_candidate_root_node()
    #     root_node_meta_bool_index = np.ones((nb_candidate_root_nodes), dtype=bool)
    #     lv_pk_edge, rv_pk_edge = self.get_lv_rv_selected_purkinje_edge(
    #         root_node_meta_bool_index=root_node_meta_bool_index)
    #     return lv_pk_edge, rv_pk_edge

    def get_lv_rv_selected_purkinje_edge(self, root_node_meta_index):
        """This function assumes that the root nodes are always assembled as lv-rv"""
        # Get purkinje sub-tree
        lv_index = np.nonzero(root_node_meta_index[:self.nb_lv_root_node])[0]
        lv_pk_path_to_lv_root_node = index_list(list=self.lv_pk_path_to_candidate_lv_root_node,
                                                index=lv_index)
        # print('root_node_meta_index[self.nb_lv_root_node:] ', root_node_meta_bool_index[self.nb_lv_root_node:])
        rv_index = np.nonzero(root_node_meta_index[self.nb_lv_root_node:])[0]
        rv_pk_path_to_rv_root_node = index_list(list=self.rv_pk_path_to_candidate_rv_root_node,
                                                index=rv_index)
        # From path to edge
        # print('lv_pk_path_to_lv_root_node ', lv_pk_path_to_lv_root_node)
        lv_pk_edge = get_edge_from_path_list(path_list=lv_pk_path_to_lv_root_node)
        # print('rv_pk_path_to_rv_root_node ', rv_pk_path_to_rv_root_node)
        rv_pk_edge = get_edge_from_path_list(path_list=rv_pk_path_to_rv_root_node)
        # pk_edge = np.concatenate((lv_pk_edge, rv_pk_edge), axis=0)
        # return np.unique(pk_edge, axis=0)
        return lv_pk_edge, rv_pk_edge

    # def get_edge_from_path_list:

    # TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
    # def get_root_node_xyz(self, root_node_meta_index):
    #     return self.root_node_xyz[root_node_meta_index, :]

    # TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
    # def __generate_djikstra_purkinje_tree_from_vc_old(self, approx_djikstra_max_path_len, edge, node_lvendo, node_rvendo,
    #                                               node_xyz, node_vc):
    #     repeat_process = False
    #     # TODO This function CRASHES when approx_djikstra_max_path_len is too small!!! There should be an adaptive something for this!
    #     # TODO for example, if the approx_djikstra_max_path_len is too small, it should call the function again with twice the value!!
    #     # Prepare for Djikstra - Set LV endocardial edges aside
    #     # No need to use transmural coordinate, because the function is using the surfaces directry read from the geometry creation
    #     # TODO there is a huge amount of repeated code in this function! Split into smaller functions!
    #     nodes_vc_concatenated = np.transpose(np.array([node_vc['ab'], node_vc['rt'], node_vc['tv']], dtype=float))
    #     lvnodes_xyz = node_xyz[node_lvendo, :]
    #     lvedges = edge[np.all(np.isin(edge, node_lvendo), axis=1), :]
    #     lvedges[:, 0] = np.asarray(
    #         [np.flatnonzero(node_lvendo == node_id)[0] for node_id in lvedges[:, 0]]).astype(
    #         int)
    #     lvedges[:, 1] = np.asarray(
    #         [np.flatnonzero(node_lvendo == node_id)[0] for node_id in lvedges[:, 1]]).astype(
    #         int)
    #     lvedgeVEC = lvnodes_xyz[lvedges[:, 0], :] - lvnodes_xyz[lvedges[:, 1], :]  # edge vectors
    #     lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    #     aux = [[] for i in range(0, lvnodes_xyz.shape[0], 1)]
    #     for i in range(0, len(lvunfoldedEdges), 1):
    #         aux[lvunfoldedEdges[i, 0]].append(i)
    #     lvneighbours = [np.array(n, dtype=int) for n in aux]
    #     aux = None  # Clear Memory
    #     # Set RV endocardial edges aside
    #     rvnodes_xyz = node_xyz[node_rvendo, :]
    #     rvedges = edge[np.all(np.isin(edge, node_rvendo), axis=1), :]
    #     rvedges[:, 0] = np.asarray(
    #         [np.flatnonzero(node_rvendo == node_id)[0] for node_id in rvedges[:, 0]]).astype(
    #         int)
    #     rvedges[:, 1] = np.asarray(
    #         [np.flatnonzero(node_rvendo == node_id)[0] for node_id in rvedges[:, 1]]).astype(
    #         int)
    #     rvedgeVEC = rvnodes_xyz[rvedges[:, 0], :] - rvnodes_xyz[rvedges[:, 1], :]  # edge vectors
    #     rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    #     aux = [[] for i in range(0, rvnodes_xyz.shape[0], 1)]
    #     for i in range(0, len(rvunfoldedEdges), 1):
    #         aux[rvunfoldedEdges[i, 0]].append(i)
    #     rvneighbours = [np.array(n, dtype=int) for n in aux]
    #     aux = None  # Clear Memory
    #     # Define Purkinje tree using Cobiveco-based rules - Initialise data structures
    #     lv_PK_distance_mat = np.full(node_lvendo.shape, get_nan_value(), np.float64)
    #     lv_PK_path_mat = np.full((node_lvendo.shape[0], approx_djikstra_max_path_len), get_nan_value(),
    #                              dtype=np.int32)
    #     rv_PK_distance_mat = np.full(node_rvendo.shape, get_nan_value(), np.float64)
    #     rv_PK_path_mat = np.full((node_rvendo.shape[0], approx_djikstra_max_path_len), get_nan_value(),
    #                              dtype=np.int32)
    #     lv_visited = np.zeros(node_lvendo.shape, dtype=bool)
    #     rv_visited = np.zeros(node_rvendo.shape, dtype=bool)
    #     # Rule 1) his-av node at coordinates [1., 0.85, 1., :] == [basal, septal, endo, :]
    #     lv_hisBase_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([1., 0.85, 0.]), ord=2,
    #                        axis=1)))  # [basal, septal, lv]
    #     rv_hisBase_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_rvendo, :] - np.array([1., 0.85, 1.]), ord=2,
    #                        axis=1)))  # [basal, septal, rv]
    #     lv_hisBase_distance_mat, lv_hisBase_path_mat = djikstra(source_id_list=np.asarray([lv_hisBase_index], dtype=int),
    #                                                             djikstra_nodes_xyz=lvnodes_xyz,
    #                                                             djikstra_unfoldedEdges=lvunfoldedEdges,
    #                                                             djikstra_edgeVEC=lvedgeVEC,
    #                                                             djikstra_neighbours=lvneighbours,
    #                                                             approx_max_path_len=approx_djikstra_max_path_len)
    #     rv_hisBase_distance_mat, rv_hisBase_path_mat = djikstra(np.asarray([rv_hisBase_index], dtype=int),
    #                                                             rvnodes_xyz, rvunfoldedEdges, rvedgeVEC,
    #                                                             rvneighbours,
    #                                                             approx_max_path_len=approx_djikstra_max_path_len)
    #     # Rule 2) hisbundle goes down to most apical endocardial point while trying to keep a straight rotation trajectory [0., 0.85, 1., :] == [basal, septal, endo, :]
    #     lv_hisApex_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0., 0.85, 0.]), ord=2,
    #                        axis=1)))  # int(np.argmin(nodesCobiveco[lvnodes, 0])) # [basal, septal, lv]
    #     rv_hisApex_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_rvendo, :] - np.array([0., 0.85, 1.]), ord=2,
    #                        axis=1)))  # int(np.argmin(nodesCobiveco[rvnodes, 0])) # [basal, septal, rv]
    #     lv_hisBundle_indexes = lv_hisBase_path_mat[lv_hisApex_index, 0, :]  # The nodes in this path are the LV his bundle
    #     lv_hisBundle_indexes = lv_hisBundle_indexes[lv_hisBundle_indexes != get_nan_value()]
    #     sorted_indexes = np.argsort(
    #         lv_hisBase_distance_mat[lv_hisBundle_indexes, 0])  # Sort nodes by distance to the reference
    #     lv_hisBundle_indexes = lv_hisBundle_indexes[sorted_indexes]  # Sort nodes by distance to the reference
    #     rv_hisBundle_indexes = rv_hisBase_path_mat[rv_hisApex_index, 0,
    #                            :]  # The nodes in this path are the LV his bundle
    #     rv_hisBundle_indexes = rv_hisBundle_indexes[rv_hisBundle_indexes != get_nan_value()]
    #     sorted_indexes = np.argsort(
    #         rv_hisBase_distance_mat[rv_hisBundle_indexes, 0])  # Sort nodes by distance to the reference
    #     rv_hisBundle_indexes = rv_hisBundle_indexes[sorted_indexes]  # Sort nodes by distance to the reference
    #     # lv_hisBundle_offsets = lv_hisBase_distance_mat[lv_hisBundle_indexes, 0]
    #     # rv_hisBundle_offsets = rv_hisBase_distance_mat[rv_hisBundle_indexes, 0]
    #     # Rule 3) The apical and Lateral/Freewall in the RV can connect directly to their closest point in the hisbundle that has ab < 0.8
    #     rv_hisBundle_ab_values = node_vc['ab'][node_rvendo[rv_hisBundle_indexes]]
    #     rv_hisBundle_meta_indexes = np.nonzero(rv_hisBundle_ab_values < 0.8)[0]
    #     rv_ab_values = node_vc['ab'][node_rvendo]
    #     rv_ab_dist = np.abs(rv_ab_values[:, np.newaxis] - rv_hisBundle_ab_values[rv_hisBundle_meta_indexes])
    #     rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(
    #         np.asarray(rv_hisBundle_indexes[rv_hisBundle_meta_indexes], dtype=int), rvnodes_xyz,
    #         rvunfoldedEdges, rvedgeVEC, rvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #     rv_hisbundle_connections = np.argmin(np.abs(rv_ab_dist),
    #                                          axis=1)  # match root nodes to the hisbundles as a rib-cage (same ab values) #np.argmin(rv_hisbundle_distance_mat, axis=1)
    #     rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                         get_nan_value(),
    #                                         dtype=np.int32)
    #     rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), get_nan_value(),
    #                                             dtype=np.float64)
    #     for i in range(rv_hisbundle_connections.shape[0]):
    #         offset = rv_hisBase_path_mat[
    #                  rv_hisBundle_indexes[rv_hisBundle_meta_indexes[rv_hisbundle_connections[i]]], 0, :]
    #         offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #         path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
    #         path = path[path != get_nan_value()]  # For visualisation only - path offset
    #         path = np.concatenate((offset, path), axis=0)
    #         if rv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
    #             repeat_process = True
    #             break
    #         rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
    #         rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + \
    #                                            rv_hisBase_distance_mat[
    #                                                rv_hisBundle_indexes[
    #                                                    rv_hisBundle_meta_indexes[
    #                                                        rv_hisbundle_connections[i]]], 0]
    #     if not repeat_process:
    #         rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    #         rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    #         # Clear Arguments to avoid recycling
    #         rv_hisbundle_path_mat_aux = None
    #         rv_hisbundle_distance_mat_aux = None
    #
    #         rv_apical_lateral_mask = ((node_vc['ab'][node_rvendo] <= 0.2) | (
    #                 (0.2 <= node_vc['rt'][node_rvendo]) & (
    #                 node_vc['rt'][node_rvendo] <= 0.5))) & np.logical_not(
    #             rv_visited)
    #         rv_PK_distance_mat[rv_apical_lateral_mask] = rv_hisbundle_distance_mat[rv_apical_lateral_mask]
    #         rv_PK_path_mat[rv_apical_lateral_mask, :] = rv_hisbundle_path_mat[rv_apical_lateral_mask, :]
    #         rv_visited[rv_apical_lateral_mask] = True
    #
    #         # Rule 4) The apical hisbundle can directly connects to Septal and Apical (and Paraseptal for the RV) root nodes after it crosses the Apex-to-Base 0.4/0.2 threshold LV/RV
    #         lv_hisMiddle_index = lv_hisBundle_indexes[int(np.argmin(
    #             np.abs(node_vc['ab'][
    #                        node_lvendo[lv_hisBundle_indexes]] - 0.4)))]  # [basal, septal, lv]
    #         rv_hisMiddle_index = rv_hisBundle_indexes[int(np.argmin(
    #             np.abs(node_vc['ab'][
    #                        node_rvendo[rv_hisBundle_indexes]] - 0.2)))]  # [basal, septal, rv]
    #
    #         lv_hisConnected_indexes = lv_hisBundle_indexes[find_first(lv_hisMiddle_index, lv_hisBundle_indexes):]
    #         rv_hisConnected_indexes = rv_hisBundle_indexes[find_first(rv_hisMiddle_index, rv_hisBundle_indexes):]
    #
    #         # Rule 5) Root nodes in the Apical regions of the heart connect to their closest Apical hisbundle node
    #         lv_hisbundle_distance_mat, lv_hisbundle_path_mat = djikstra(
    #             np.asarray(lv_hisConnected_indexes, dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
    #             lvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #         rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(
    #             np.asarray(rv_hisConnected_indexes, dtype=int), rvnodes_xyz, rvunfoldedEdges, rvedgeVEC,
    #             rvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #         lv_hisbundle_connections = np.argmin(lv_hisbundle_distance_mat, axis=1)
    #         rv_hisbundle_connections = np.argmin(rv_hisbundle_distance_mat, axis=1)
    #         lv_hisbundle_path_mat_aux = np.full((lv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                             get_nan_value(),
    #                                             dtype=np.int32)
    #         lv_hisbundle_distance_mat_aux = np.full((lv_hisbundle_distance_mat.shape[0]), get_nan_value(),
    #                                                 dtype=np.float64)
    #         for i in range(lv_hisbundle_connections.shape[0]):
    #             offset = lv_hisBase_path_mat[lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0, :]
    #             offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #             path = lv_hisbundle_path_mat[i, lv_hisbundle_connections[i], :]
    #             path = path[path != get_nan_value()]  # For visualisation only - path offset
    #             path = np.concatenate((offset, path), axis=0)
    #             if lv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
    #                 repeat_process = True
    #                 break
    #             lv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
    #             lv_hisbundle_distance_mat_aux[i] = lv_hisbundle_distance_mat[i, lv_hisbundle_connections[i]] + \
    #                                                lv_hisBase_distance_mat[
    #                                                    lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0]
    #         if not repeat_process:
    #             lv_hisbundle_path_mat = lv_hisbundle_path_mat_aux
    #             lv_hisbundle_distance_mat = lv_hisbundle_distance_mat_aux
    #             # Clear Arguments to avoid recycling
    #             lv_hisbundle_path_mat_aux = None
    #             lv_hisbundle_distance_mat_aux = None
    #
    #             rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                                 get_nan_value(),
    #                                                 dtype=np.int32)
    #             rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), get_nan_value(),
    #                                                     dtype=np.float64)
    #             for i in range(rv_hisbundle_connections.shape[0]):
    #                 offset = rv_hisBase_path_mat[rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0, :]
    #                 offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #                 path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
    #                 path = path[path != get_nan_value()]  # For visualisation only - path offset
    #                 path = np.concatenate((offset, path), axis=0)
    #                 if rv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
    #                     repeat_process = True
    #                     break
    #                 rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
    #                 rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + \
    #                                                    rv_hisBase_distance_mat[
    #                                                        rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0]
    #             if not repeat_process:
    #                 rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    #                 rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    #                 # Clear Arguments to avoid recycling
    #                 rv_hisbundle_path_mat_aux = None
    #                 rv_hisbundle_distance_mat_aux = None
    #
    #                 # Rule 5) Apical|Septal|Paraseptal regions of the heart are defined as AB < 0.4/0.2 in the LV/RV | [0.7 < RT < 1.] | [0. < RT < 0.2] & [0.5 < RT < 0.7], respectively
    #                 lv_apical_septal_mask = ((node_vc['ab'][node_lvendo] <= 0.4) | (
    #                         (0.7 <= node_vc['rt'][node_lvendo]) & (
    #                         node_vc['rt'][node_lvendo] <= 1.))) & np.logical_not(
    #                     lv_visited)
    #                 rv_apical_septal_paraseptal_mask = (((node_vc['ab'][node_rvendo] <= 0.2) | (
    #                         (0.7 <= node_vc['rt'][node_rvendo]) & (
    #                         node_vc['rt'][node_rvendo] <= 1.))) |
    #                                                     (((0.0 <= node_vc['rt'][node_rvendo]) & (
    #                                                             node_vc['rt'][node_rvendo] <= 0.2)) | (
    #                                                              (0.5 <= node_vc['rt'][node_rvendo]) & (
    #                                                              node_vc['rt'][
    #                                                                  node_rvendo] <= 0.7)))) & np.logical_not(
    #                     rv_visited)
    #                 lv_PK_distance_mat[lv_apical_septal_mask] = lv_hisbundle_distance_mat[lv_apical_septal_mask]
    #                 lv_PK_path_mat[lv_apical_septal_mask, :] = lv_hisbundle_path_mat[lv_apical_septal_mask, :]
    #                 lv_visited[lv_apical_septal_mask] = True
    #                 rv_PK_distance_mat[rv_apical_septal_paraseptal_mask] = rv_hisbundle_distance_mat[
    #                     rv_apical_septal_paraseptal_mask]
    #                 rv_PK_path_mat[rv_apical_septal_paraseptal_mask, :] = rv_hisbundle_path_mat[
    #                                                                       rv_apical_septal_paraseptal_mask, :]
    #                 rv_visited[rv_apical_septal_paraseptal_mask] = True
    #
    #                 # Rule 6) Paraseptal regions of the heart are connected from apex to base through either [0.4/0.2, 0.1, 1., :] or  [0.4/0.2, 0.6, 1., :] LV/RV
    #                 lv_ant_paraseptalApex_index = int(np.argmin(
    #                     np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0.4, 0.6, 0.]), ord=2,
    #                                    axis=1)))  # [mid, paraseptal, lv]
    #                 lv_post_paraseptalApex_index = int(np.argmin(
    #                     np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0.4, 0.1, 0.]), ord=2,
    #                                    axis=1)))  # [mid, paraseptal, lv]
    #                 if not lv_visited[lv_ant_paraseptalApex_index]:
    #                     lv_PK_distance_mat[lv_ant_paraseptalApex_index] = lv_hisbundle_distance_mat[
    #                         lv_ant_paraseptalApex_index]
    #                     lv_PK_path_mat[lv_ant_paraseptalApex_index, :] = lv_hisbundle_path_mat[lv_ant_paraseptalApex_index,
    #                                                                      :]
    #                     lv_visited[lv_ant_paraseptalApex_index] = True
    #                 if not lv_visited[lv_post_paraseptalApex_index]:
    #                     lv_PK_distance_mat[lv_post_paraseptalApex_index] = lv_hisbundle_distance_mat[
    #                         lv_post_paraseptalApex_index]
    #                     lv_PK_path_mat[lv_post_paraseptalApex_index, :] = lv_hisbundle_path_mat[
    #                                                                       lv_post_paraseptalApex_index, :]
    #                     lv_visited[lv_post_paraseptalApex_index] = True
    #                 lv_paraseptalApex_offsets = np.array(
    #                     [lv_PK_distance_mat[lv_ant_paraseptalApex_index], lv_PK_distance_mat[lv_post_paraseptalApex_index]],
    #                     dtype=float)
    #                 lv_paraseptal_distance_mat, lv_paraseptal_path_mat = djikstra(
    #                     np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int), lvnodes_xyz,
    #                     lvunfoldedEdges, lvedgeVEC, lvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #                 lv_paraseptal_connections = np.argmin(lv_paraseptal_distance_mat, axis=1)
    #                 lv_paraseptal_path_mat_aux = np.full((lv_paraseptal_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                                      get_nan_value(),
    #                                                      dtype=np.int32)
    #                 lv_paraseptal_distance_mat_aux = np.full((lv_paraseptal_distance_mat.shape[0]), get_nan_value(),
    #                                                          dtype=np.float64)
    #                 for i in range(lv_paraseptal_connections.shape[0]):
    #                     offset = lv_PK_path_mat[
    #                              np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int)[
    #                                  lv_paraseptal_connections[i]], :]
    #                     offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #                     path = lv_paraseptal_path_mat[i, lv_paraseptal_connections[i], :]
    #                     path = path[path != get_nan_value()]  # For visualisation only - path offset
    #                     path = np.concatenate((offset, path), axis=0)
    #                     if lv_paraseptal_path_mat_aux.shape[1] < path.shape[0]:
    #                         repeat_process = True
    #                         break
    #                     lv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
    #                     lv_paraseptal_distance_mat_aux[i] = lv_paraseptal_distance_mat[i, lv_paraseptal_connections[i]] + \
    #                                                         lv_paraseptalApex_offsets[lv_paraseptal_connections[i]]
    #                 if not repeat_process:
    #                     lv_paraseptal_path_mat = lv_paraseptal_path_mat_aux
    #                     lv_paraseptal_distance_mat = lv_paraseptal_distance_mat_aux
    #                     # Clear Arguments to avoid recycling
    #
    #                     # Rule 7) Paraseptal regions of the heart are defined as [0. < rotation-angle (RT) < 0.2] & [0.5 < RT < 0.7], these are connected to their closest paraseptal routing point (anterior or posterior)
    #                     lv_paraseptal_mask = (((0.0 <= node_vc['rt'][node_lvendo]) & (
    #                             node_vc['rt'][node_lvendo] <= 0.2)) | (
    #                                                   (0.5 <= node_vc['rt'][node_lvendo]) & (
    #                                                   node_vc['rt'][node_lvendo] <= 0.7))) & np.logical_not(
    #                         lv_visited)
    #                     lv_PK_distance_mat[lv_paraseptal_mask] = lv_paraseptal_distance_mat[lv_paraseptal_mask]
    #                     lv_PK_path_mat[lv_paraseptal_mask, :] = lv_paraseptal_path_mat[lv_paraseptal_mask, :]
    #                     lv_visited[lv_paraseptal_mask] = True
    #                     # Rule 8) Freewall regions of the heart are connected from apex to base through [0.4, 0.35, 1., :] in the LV
    #                     lv_freewallApex_index = int(np.argmin(
    #                         np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0.4, 0.35, 0.]), ord=2,
    #                                        axis=1)))  # [mid, freewall, endo, lv]
    #                     if not lv_visited[lv_freewallApex_index]:
    #                         lv_PK_distance_mat[lv_freewallApex_index] = lv_hisbundle_distance_mat[lv_freewallApex_index]
    #                         lv_PK_path_mat[lv_freewallApex_index, :] = lv_hisbundle_path_mat[lv_freewallApex_index, :]
    #                         lv_visited[lv_freewallApex_index] = True
    #                     lv_freewallApex_offset = lv_PK_distance_mat[lv_freewallApex_index]
    #                     lv_freewallApex_path_offset = lv_PK_path_mat[lv_freewallApex_index, :]
    #                     lv_freewallApex_path_offset = lv_freewallApex_path_offset[lv_freewallApex_path_offset != get_nan_value()]
    #                     lv_freewall_distance_mat, lv_freewall_path_mat = djikstra(
    #                         np.asarray([lv_freewallApex_index], dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
    #                         lvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #                     lv_freewall_path_mat_aux = np.full((lv_freewall_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                                        get_nan_value(),
    #                                                        dtype=np.int32)
    #                     lv_freewall_distance_mat_aux = np.full((lv_freewall_distance_mat.shape[0]), get_nan_value(),
    #                                                            dtype=np.float64)
    #                     for i in range(lv_freewall_distance_mat.shape[0]):
    #                         path = lv_freewall_path_mat[i, 0, :]
    #                         path = path[path != get_nan_value()]  # For visualisation only - path offset
    #                         path = np.concatenate((lv_freewallApex_path_offset, path), axis=0)
    #                         if lv_freewall_path_mat_aux.shape[1] < path.shape[0]:
    #                             repeat_process = True
    #                             break
    #                         lv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #                         lv_freewall_distance_mat_aux[i] = lv_freewall_distance_mat[i, 0] + lv_freewallApex_offset
    #                     if not repeat_process:
    #                         lv_freewall_path_mat = lv_freewall_path_mat_aux
    #                         lv_freewall_distance_mat = lv_freewall_distance_mat_aux
    #
    #                         # Rule 10) Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
    #                         lv_freewall_mask = ((0.2 <= node_vc['rt'][node_lvendo]) & (
    #                                 node_vc['rt'][node_lvendo] <= 0.5)) & np.logical_not(lv_visited)
    #                         lv_PK_distance_mat[lv_freewall_mask] = lv_freewall_distance_mat[lv_freewall_mask]
    #                         lv_PK_path_mat[lv_freewall_mask, :] = lv_freewall_path_mat[lv_freewall_mask, :]
    #                         lv_visited[lv_freewall_mask] = True
    #
    #     return lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, repeat_process


class PurkinjeSystemVC(DjikstraConductionSystemVC):
    """This class is a conduction system that gets defined from Ventricular Coordinates using Djikstra.
    Root nodes are always assembled as lv-rv"""
    def __init__(self, approx_djikstra_purkinje_max_path_len, geometry, lv_inter_root_node_distance,
                 rv_inter_root_node_distance, verbose):
        if verbose:
            print('Generating Purkinje tree')
        purkinje_max_ab_cut_threshold = get_root_node_max_ab_cut_threshold()  # The Purkinje fibres cannot grow all the way to the base
        print('purkinje_max_ab_cut_threshold ', purkinje_max_ab_cut_threshold)
        node_xyz = geometry.get_node_xyz()
        node_lvendo_fast = geometry.get_node_lvendo_fast()
        # node_vc = geometry.get_node_vc()
        vc_ab_cut_name = get_vc_ab_cut_name()
        vc_rt_name = get_vc_rt_name()
        node_vc_ab_cut = geometry.get_node_vc_field(vc_name=vc_ab_cut_name)
        node_vc_rt = geometry.get_node_vc_field(vc_name=vc_rt_name)
        # node_vc = None  # Clear Arguments to prevent Argument recycling
        lv_candidate_root_node_meta_index = generate_candidate_root_nodes_in_cavity(
            basal_cavity_nodes_xyz=node_xyz[node_lvendo_fast, :],
            basal_cavity_vc_ab_cut=node_vc_ab_cut[node_lvendo_fast],
            inter_root_node_distance=lv_inter_root_node_distance,
            purkinje_max_cut_ab_threshold=purkinje_max_ab_cut_threshold)
        node_rvendo_fast = geometry.get_node_rvendo_fast()
        rv_candidate_root_node_meta_index = generate_candidate_root_nodes_in_cavity(
            basal_cavity_nodes_xyz=node_xyz[node_rvendo_fast, :],
            basal_cavity_vc_ab_cut=node_vc_ab_cut[node_rvendo_fast],
            inter_root_node_distance=rv_inter_root_node_distance,
            purkinje_max_cut_ab_threshold=purkinje_max_ab_cut_threshold)
        super().__init__(approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, geometry=geometry,
                         lv_candidate_root_node_meta_index=lv_candidate_root_node_meta_index,
                         rv_candidate_root_node_meta_index=rv_candidate_root_node_meta_index,
                         purkinje_max_ab_cut_threshold=purkinje_max_ab_cut_threshold, verbose=verbose)


    def get_lv_rv_selected_root_node_meta_index(self, root_node_meta_index):
        # TODO make this part of the code common accross functions
        nb_candidate_root_nodes = self.get_nb_candidate_root_node()
        print('root_node_meta_index.shape[0] ', root_node_meta_index.shape[0])
        print('nb_candidate_root_nodes ', nb_candidate_root_nodes)
        assert nb_candidate_root_nodes == root_node_meta_index.shape[0] # If true, nothing happens
        lv_root_node_meta_bool_index = np.zeros((nb_candidate_root_nodes), dtype=bool)
        rv_root_node_meta_bool_index = np.zeros((nb_candidate_root_nodes), dtype=bool)
        lv_root_node_meta_bool_index[:self.nb_lv_root_node] = root_node_meta_index[:self.nb_lv_root_node]
        rv_root_node_meta_bool_index[self.nb_lv_root_node:] = root_node_meta_index[self.nb_lv_root_node:]
        return lv_root_node_meta_bool_index, rv_root_node_meta_bool_index

    def get_lv_rv_selected_root_node_index(self, root_node_meta_index):
        lv_root_node_meta_bool_index, rv_root_node_meta_bool_index = self.get_lv_rv_selected_root_node_meta_index(root_node_meta_index)
        lv_selected_root_node_index = self.get_selected_root_node_index(
            root_node_meta_index=lv_root_node_meta_bool_index)
        rv_selected_root_node_index = self.get_selected_root_node_index(
            root_node_meta_index=rv_root_node_meta_bool_index)
        # print('nb_candidate_root_nodes ', nb_candidate_root_nodes)
        # print()
        # print('lv_candidate_root_node_index ', lv_selected_root_node_index, ' len ', len(lv_selected_root_node_index))
        # print()
        # print('rv_candidate_root_node_index ', rv_selected_root_node_index, ' len ', len(rv_selected_root_node_index))
        # print()
        # print('self.lv_pk_path_to_candidate_lv_root_node ', len(self.lv_pk_path_to_candidate_lv_root_node))
        # print('self.lv_pk_path_to_candidate_lv_root_node ', self.lv_pk_path_to_candidate_lv_root_node)
        # print()
        # print('self.rv_pk_path_to_candidate_rv_root_node ', len(self.rv_pk_path_to_candidate_rv_root_node))
        # print('self.rv_pk_path_to_candidate_rv_root_node ', self.rv_pk_path_to_candidate_rv_root_node)
        # print()
        return lv_selected_root_node_index, rv_selected_root_node_index

    # def get_lv_rv_candidate_root_node_index(self):
    #     nb_candidate_root_nodes = self.get_nb_candidate_root_node()
    #     lv_candidate_root_node_index, rv_candidate_root_node_index = self.get_lv_rv_selected_root_node_index(
    #         root_node_meta_bool_index=np.ones((nb_candidate_root_nodes), dtype=bool))
    #     return lv_candidate_root_node_index, rv_candidate_root_node_index

    # def get_lv_rv_candidate_purkinje_edge(self):
    #     nb_candidate_root_nodes = self.get_nb_candidate_root_node()
    #     root_node_meta_bool_index = np.ones((nb_candidate_root_nodes), dtype=bool)
    #     lv_pk_edge, rv_pk_edge = self.get_lv_rv_selected_purkinje_edge(
    #         root_node_meta_bool_index=root_node_meta_bool_index)
    #     return lv_pk_edge, rv_pk_edge

    def get_lv_rv_selected_purkinje_edge(self, root_node_meta_index):
        """This function assumes that the root nodes are always assembled as lv-rv"""
        # Get purkinje sub-tree
        lv_index = np.nonzero(root_node_meta_index[:self.nb_lv_root_node])[0]
        lv_pk_path_to_lv_root_node = index_list(list=self.lv_pk_path_to_candidate_lv_root_node,
                                                index=lv_index)
        # print('root_node_meta_index[self.nb_lv_root_node:] ', root_node_meta_bool_index[self.nb_lv_root_node:])
        rv_index = np.nonzero(root_node_meta_index[self.nb_lv_root_node:])[0]
        rv_pk_path_to_rv_root_node = index_list(list=self.rv_pk_path_to_candidate_rv_root_node,
                                                index=rv_index)
        # From path to edge
        # print('lv_pk_path_to_lv_root_node ', lv_pk_path_to_lv_root_node)
        lv_pk_edge = get_edge_from_path_list(path_list=lv_pk_path_to_lv_root_node)
        # print('rv_pk_path_to_rv_root_node ', rv_pk_path_to_rv_root_node)
        rv_pk_edge = get_edge_from_path_list(path_list=rv_pk_path_to_rv_root_node)
        # pk_edge = np.concatenate((lv_pk_edge, rv_pk_edge), axis=0)
        # return np.unique(pk_edge, axis=0)
        return lv_pk_edge, rv_pk_edge

    # def get_edge_from_path_list:

    # TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
    # def get_root_node_xyz(self, root_node_meta_index):
    #     return self.root_node_xyz[root_node_meta_index, :]

    # TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
    # def __generate_djikstra_purkinje_tree_from_vc_old(self, approx_djikstra_max_path_len, edge, node_lvendo, node_rvendo,
    #                                               node_xyz, node_vc):
    #     repeat_process = False
    #     # TODO This function CRASHES when approx_djikstra_max_path_len is too small!!! There should be an adaptive something for this!
    #     # TODO for example, if the approx_djikstra_max_path_len is too small, it should call the function again with twice the value!!
    #     # Prepare for Djikstra - Set LV endocardial edges aside
    #     # No need to use transmural coordinate, because the function is using the surfaces directry read from the geometry creation
    #     # TODO there is a huge amount of repeated code in this function! Split into smaller functions!
    #     nodes_vc_concatenated = np.transpose(np.array([node_vc['ab'], node_vc['rt'], node_vc['tv']], dtype=float))
    #     lvnodes_xyz = node_xyz[node_lvendo, :]
    #     lvedges = edge[np.all(np.isin(edge, node_lvendo), axis=1), :]
    #     lvedges[:, 0] = np.asarray(
    #         [np.flatnonzero(node_lvendo == node_id)[0] for node_id in lvedges[:, 0]]).astype(
    #         int)
    #     lvedges[:, 1] = np.asarray(
    #         [np.flatnonzero(node_lvendo == node_id)[0] for node_id in lvedges[:, 1]]).astype(
    #         int)
    #     lvedgeVEC = lvnodes_xyz[lvedges[:, 0], :] - lvnodes_xyz[lvedges[:, 1], :]  # edge vectors
    #     lvunfoldedEdges = np.concatenate((lvedges, np.flip(lvedges, axis=1))).astype(int)
    #     aux = [[] for i in range(0, lvnodes_xyz.shape[0], 1)]
    #     for i in range(0, len(lvunfoldedEdges), 1):
    #         aux[lvunfoldedEdges[i, 0]].append(i)
    #     lvneighbours = [np.array(n, dtype=int) for n in aux]
    #     aux = None  # Clear Memory
    #     # Set RV endocardial edges aside
    #     rvnodes_xyz = node_xyz[node_rvendo, :]
    #     rvedges = edge[np.all(np.isin(edge, node_rvendo), axis=1), :]
    #     rvedges[:, 0] = np.asarray(
    #         [np.flatnonzero(node_rvendo == node_id)[0] for node_id in rvedges[:, 0]]).astype(
    #         int)
    #     rvedges[:, 1] = np.asarray(
    #         [np.flatnonzero(node_rvendo == node_id)[0] for node_id in rvedges[:, 1]]).astype(
    #         int)
    #     rvedgeVEC = rvnodes_xyz[rvedges[:, 0], :] - rvnodes_xyz[rvedges[:, 1], :]  # edge vectors
    #     rvunfoldedEdges = np.concatenate((rvedges, np.flip(rvedges, axis=1))).astype(int)
    #     aux = [[] for i in range(0, rvnodes_xyz.shape[0], 1)]
    #     for i in range(0, len(rvunfoldedEdges), 1):
    #         aux[rvunfoldedEdges[i, 0]].append(i)
    #     rvneighbours = [np.array(n, dtype=int) for n in aux]
    #     aux = None  # Clear Memory
    #     # Define Purkinje tree using Cobiveco-based rules - Initialise data structures
    #     lv_PK_distance_mat = np.full(node_lvendo.shape, get_nan_value(), np.float64)
    #     lv_PK_path_mat = np.full((node_lvendo.shape[0], approx_djikstra_max_path_len), get_nan_value(),
    #                              dtype=np.int32)
    #     rv_PK_distance_mat = np.full(node_rvendo.shape, get_nan_value(), np.float64)
    #     rv_PK_path_mat = np.full((node_rvendo.shape[0], approx_djikstra_max_path_len), get_nan_value(),
    #                              dtype=np.int32)
    #     lv_visited = np.zeros(node_lvendo.shape, dtype=bool)
    #     rv_visited = np.zeros(node_rvendo.shape, dtype=bool)
    #     # Rule 1) his-av node at coordinates [1., 0.85, 1., :] == [basal, septal, endo, :]
    #     lv_hisBase_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([1., 0.85, 0.]), ord=2,
    #                        axis=1)))  # [basal, septal, lv]
    #     rv_hisBase_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_rvendo, :] - np.array([1., 0.85, 1.]), ord=2,
    #                        axis=1)))  # [basal, septal, rv]
    #     lv_hisBase_distance_mat, lv_hisBase_path_mat = djikstra(source_id_list=np.asarray([lv_hisBase_index], dtype=int),
    #                                                             djikstra_nodes_xyz=lvnodes_xyz,
    #                                                             djikstra_unfoldedEdges=lvunfoldedEdges,
    #                                                             djikstra_edgeVEC=lvedgeVEC,
    #                                                             djikstra_neighbours=lvneighbours,
    #                                                             approx_max_path_len=approx_djikstra_max_path_len)
    #     rv_hisBase_distance_mat, rv_hisBase_path_mat = djikstra(np.asarray([rv_hisBase_index], dtype=int),
    #                                                             rvnodes_xyz, rvunfoldedEdges, rvedgeVEC,
    #                                                             rvneighbours,
    #                                                             approx_max_path_len=approx_djikstra_max_path_len)
    #     # Rule 2) hisbundle goes down to most apical endocardial point while trying to keep a straight rotation trajectory [0., 0.85, 1., :] == [basal, septal, endo, :]
    #     lv_hisApex_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0., 0.85, 0.]), ord=2,
    #                        axis=1)))  # int(np.argmin(nodesCobiveco[lvnodes, 0])) # [basal, septal, lv]
    #     rv_hisApex_index = int(np.argmin(
    #         np.linalg.norm(nodes_vc_concatenated[node_rvendo, :] - np.array([0., 0.85, 1.]), ord=2,
    #                        axis=1)))  # int(np.argmin(nodesCobiveco[rvnodes, 0])) # [basal, septal, rv]
    #     lv_hisBundle_indexes = lv_hisBase_path_mat[lv_hisApex_index, 0, :]  # The nodes in this path are the LV his bundle
    #     lv_hisBundle_indexes = lv_hisBundle_indexes[lv_hisBundle_indexes != get_nan_value()]
    #     sorted_indexes = np.argsort(
    #         lv_hisBase_distance_mat[lv_hisBundle_indexes, 0])  # Sort nodes by distance to the reference
    #     lv_hisBundle_indexes = lv_hisBundle_indexes[sorted_indexes]  # Sort nodes by distance to the reference
    #     rv_hisBundle_indexes = rv_hisBase_path_mat[rv_hisApex_index, 0,
    #                            :]  # The nodes in this path are the LV his bundle
    #     rv_hisBundle_indexes = rv_hisBundle_indexes[rv_hisBundle_indexes != get_nan_value()]
    #     sorted_indexes = np.argsort(
    #         rv_hisBase_distance_mat[rv_hisBundle_indexes, 0])  # Sort nodes by distance to the reference
    #     rv_hisBundle_indexes = rv_hisBundle_indexes[sorted_indexes]  # Sort nodes by distance to the reference
    #     # lv_hisBundle_offsets = lv_hisBase_distance_mat[lv_hisBundle_indexes, 0]
    #     # rv_hisBundle_offsets = rv_hisBase_distance_mat[rv_hisBundle_indexes, 0]
    #     # Rule 3) The apical and Lateral/Freewall in the RV can connect directly to their closest point in the hisbundle that has ab < 0.8
    #     rv_hisBundle_ab_values = node_vc['ab'][node_rvendo[rv_hisBundle_indexes]]
    #     rv_hisBundle_meta_indexes = np.nonzero(rv_hisBundle_ab_values < 0.8)[0]
    #     rv_ab_values = node_vc['ab'][node_rvendo]
    #     rv_ab_dist = np.abs(rv_ab_values[:, np.newaxis] - rv_hisBundle_ab_values[rv_hisBundle_meta_indexes])
    #     rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(
    #         np.asarray(rv_hisBundle_indexes[rv_hisBundle_meta_indexes], dtype=int), rvnodes_xyz,
    #         rvunfoldedEdges, rvedgeVEC, rvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #     rv_hisbundle_connections = np.argmin(np.abs(rv_ab_dist),
    #                                          axis=1)  # match root nodes to the hisbundles as a rib-cage (same ab values) #np.argmin(rv_hisbundle_distance_mat, axis=1)
    #     rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                         get_nan_value(),
    #                                         dtype=np.int32)
    #     rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), get_nan_value(),
    #                                             dtype=np.float64)
    #     for i in range(rv_hisbundle_connections.shape[0]):
    #         offset = rv_hisBase_path_mat[
    #                  rv_hisBundle_indexes[rv_hisBundle_meta_indexes[rv_hisbundle_connections[i]]], 0, :]
    #         offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #         path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
    #         path = path[path != get_nan_value()]  # For visualisation only - path offset
    #         path = np.concatenate((offset, path), axis=0)
    #         if rv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
    #             repeat_process = True
    #             break
    #         rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
    #         rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + \
    #                                            rv_hisBase_distance_mat[
    #                                                rv_hisBundle_indexes[
    #                                                    rv_hisBundle_meta_indexes[
    #                                                        rv_hisbundle_connections[i]]], 0]
    #     if not repeat_process:
    #         rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    #         rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    #         # Clear Arguments to avoid recycling
    #         rv_hisbundle_path_mat_aux = None
    #         rv_hisbundle_distance_mat_aux = None
    #
    #         rv_apical_lateral_mask = ((node_vc['ab'][node_rvendo] <= 0.2) | (
    #                 (0.2 <= node_vc['rt'][node_rvendo]) & (
    #                 node_vc['rt'][node_rvendo] <= 0.5))) & np.logical_not(
    #             rv_visited)
    #         rv_PK_distance_mat[rv_apical_lateral_mask] = rv_hisbundle_distance_mat[rv_apical_lateral_mask]
    #         rv_PK_path_mat[rv_apical_lateral_mask, :] = rv_hisbundle_path_mat[rv_apical_lateral_mask, :]
    #         rv_visited[rv_apical_lateral_mask] = True
    #
    #         # Rule 4) The apical hisbundle can directly connects to Septal and Apical (and Paraseptal for the RV) root nodes after it crosses the Apex-to-Base 0.4/0.2 threshold LV/RV
    #         lv_hisMiddle_index = lv_hisBundle_indexes[int(np.argmin(
    #             np.abs(node_vc['ab'][
    #                        node_lvendo[lv_hisBundle_indexes]] - 0.4)))]  # [basal, septal, lv]
    #         rv_hisMiddle_index = rv_hisBundle_indexes[int(np.argmin(
    #             np.abs(node_vc['ab'][
    #                        node_rvendo[rv_hisBundle_indexes]] - 0.2)))]  # [basal, septal, rv]
    #
    #         lv_hisConnected_indexes = lv_hisBundle_indexes[find_first(lv_hisMiddle_index, lv_hisBundle_indexes):]
    #         rv_hisConnected_indexes = rv_hisBundle_indexes[find_first(rv_hisMiddle_index, rv_hisBundle_indexes):]
    #
    #         # Rule 5) Root nodes in the Apical regions of the heart connect to their closest Apical hisbundle node
    #         lv_hisbundle_distance_mat, lv_hisbundle_path_mat = djikstra(
    #             np.asarray(lv_hisConnected_indexes, dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
    #             lvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #         rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(
    #             np.asarray(rv_hisConnected_indexes, dtype=int), rvnodes_xyz, rvunfoldedEdges, rvedgeVEC,
    #             rvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #         lv_hisbundle_connections = np.argmin(lv_hisbundle_distance_mat, axis=1)
    #         rv_hisbundle_connections = np.argmin(rv_hisbundle_distance_mat, axis=1)
    #         lv_hisbundle_path_mat_aux = np.full((lv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                             get_nan_value(),
    #                                             dtype=np.int32)
    #         lv_hisbundle_distance_mat_aux = np.full((lv_hisbundle_distance_mat.shape[0]), get_nan_value(),
    #                                                 dtype=np.float64)
    #         for i in range(lv_hisbundle_connections.shape[0]):
    #             offset = lv_hisBase_path_mat[lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0, :]
    #             offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #             path = lv_hisbundle_path_mat[i, lv_hisbundle_connections[i], :]
    #             path = path[path != get_nan_value()]  # For visualisation only - path offset
    #             path = np.concatenate((offset, path), axis=0)
    #             if lv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
    #                 repeat_process = True
    #                 break
    #             lv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
    #             lv_hisbundle_distance_mat_aux[i] = lv_hisbundle_distance_mat[i, lv_hisbundle_connections[i]] + \
    #                                                lv_hisBase_distance_mat[
    #                                                    lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0]
    #         if not repeat_process:
    #             lv_hisbundle_path_mat = lv_hisbundle_path_mat_aux
    #             lv_hisbundle_distance_mat = lv_hisbundle_distance_mat_aux
    #             # Clear Arguments to avoid recycling
    #             lv_hisbundle_path_mat_aux = None
    #             lv_hisbundle_distance_mat_aux = None
    #
    #             rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                                 get_nan_value(),
    #                                                 dtype=np.int32)
    #             rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), get_nan_value(),
    #                                                     dtype=np.float64)
    #             for i in range(rv_hisbundle_connections.shape[0]):
    #                 offset = rv_hisBase_path_mat[rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0, :]
    #                 offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #                 path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
    #                 path = path[path != get_nan_value()]  # For visualisation only - path offset
    #                 path = np.concatenate((offset, path), axis=0)
    #                 if rv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
    #                     repeat_process = True
    #                     break
    #                 rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
    #                 rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + \
    #                                                    rv_hisBase_distance_mat[
    #                                                        rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0]
    #             if not repeat_process:
    #                 rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
    #                 rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
    #                 # Clear Arguments to avoid recycling
    #                 rv_hisbundle_path_mat_aux = None
    #                 rv_hisbundle_distance_mat_aux = None
    #
    #                 # Rule 5) Apical|Septal|Paraseptal regions of the heart are defined as AB < 0.4/0.2 in the LV/RV | [0.7 < RT < 1.] | [0. < RT < 0.2] & [0.5 < RT < 0.7], respectively
    #                 lv_apical_septal_mask = ((node_vc['ab'][node_lvendo] <= 0.4) | (
    #                         (0.7 <= node_vc['rt'][node_lvendo]) & (
    #                         node_vc['rt'][node_lvendo] <= 1.))) & np.logical_not(
    #                     lv_visited)
    #                 rv_apical_septal_paraseptal_mask = (((node_vc['ab'][node_rvendo] <= 0.2) | (
    #                         (0.7 <= node_vc['rt'][node_rvendo]) & (
    #                         node_vc['rt'][node_rvendo] <= 1.))) |
    #                                                     (((0.0 <= node_vc['rt'][node_rvendo]) & (
    #                                                             node_vc['rt'][node_rvendo] <= 0.2)) | (
    #                                                              (0.5 <= node_vc['rt'][node_rvendo]) & (
    #                                                              node_vc['rt'][
    #                                                                  node_rvendo] <= 0.7)))) & np.logical_not(
    #                     rv_visited)
    #                 lv_PK_distance_mat[lv_apical_septal_mask] = lv_hisbundle_distance_mat[lv_apical_septal_mask]
    #                 lv_PK_path_mat[lv_apical_septal_mask, :] = lv_hisbundle_path_mat[lv_apical_septal_mask, :]
    #                 lv_visited[lv_apical_septal_mask] = True
    #                 rv_PK_distance_mat[rv_apical_septal_paraseptal_mask] = rv_hisbundle_distance_mat[
    #                     rv_apical_septal_paraseptal_mask]
    #                 rv_PK_path_mat[rv_apical_septal_paraseptal_mask, :] = rv_hisbundle_path_mat[
    #                                                                       rv_apical_septal_paraseptal_mask, :]
    #                 rv_visited[rv_apical_septal_paraseptal_mask] = True
    #
    #                 # Rule 6) Paraseptal regions of the heart are connected from apex to base through either [0.4/0.2, 0.1, 1., :] or  [0.4/0.2, 0.6, 1., :] LV/RV
    #                 lv_ant_paraseptalApex_index = int(np.argmin(
    #                     np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0.4, 0.6, 0.]), ord=2,
    #                                    axis=1)))  # [mid, paraseptal, lv]
    #                 lv_post_paraseptalApex_index = int(np.argmin(
    #                     np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0.4, 0.1, 0.]), ord=2,
    #                                    axis=1)))  # [mid, paraseptal, lv]
    #                 if not lv_visited[lv_ant_paraseptalApex_index]:
    #                     lv_PK_distance_mat[lv_ant_paraseptalApex_index] = lv_hisbundle_distance_mat[
    #                         lv_ant_paraseptalApex_index]
    #                     lv_PK_path_mat[lv_ant_paraseptalApex_index, :] = lv_hisbundle_path_mat[lv_ant_paraseptalApex_index,
    #                                                                      :]
    #                     lv_visited[lv_ant_paraseptalApex_index] = True
    #                 if not lv_visited[lv_post_paraseptalApex_index]:
    #                     lv_PK_distance_mat[lv_post_paraseptalApex_index] = lv_hisbundle_distance_mat[
    #                         lv_post_paraseptalApex_index]
    #                     lv_PK_path_mat[lv_post_paraseptalApex_index, :] = lv_hisbundle_path_mat[
    #                                                                       lv_post_paraseptalApex_index, :]
    #                     lv_visited[lv_post_paraseptalApex_index] = True
    #                 lv_paraseptalApex_offsets = np.array(
    #                     [lv_PK_distance_mat[lv_ant_paraseptalApex_index], lv_PK_distance_mat[lv_post_paraseptalApex_index]],
    #                     dtype=float)
    #                 lv_paraseptal_distance_mat, lv_paraseptal_path_mat = djikstra(
    #                     np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int), lvnodes_xyz,
    #                     lvunfoldedEdges, lvedgeVEC, lvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #                 lv_paraseptal_connections = np.argmin(lv_paraseptal_distance_mat, axis=1)
    #                 lv_paraseptal_path_mat_aux = np.full((lv_paraseptal_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                                      get_nan_value(),
    #                                                      dtype=np.int32)
    #                 lv_paraseptal_distance_mat_aux = np.full((lv_paraseptal_distance_mat.shape[0]), get_nan_value(),
    #                                                          dtype=np.float64)
    #                 for i in range(lv_paraseptal_connections.shape[0]):
    #                     offset = lv_PK_path_mat[
    #                              np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int)[
    #                                  lv_paraseptal_connections[i]], :]
    #                     offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
    #                     path = lv_paraseptal_path_mat[i, lv_paraseptal_connections[i], :]
    #                     path = path[path != get_nan_value()]  # For visualisation only - path offset
    #                     path = np.concatenate((offset, path), axis=0)
    #                     if lv_paraseptal_path_mat_aux.shape[1] < path.shape[0]:
    #                         repeat_process = True
    #                         break
    #                     lv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
    #                     lv_paraseptal_distance_mat_aux[i] = lv_paraseptal_distance_mat[i, lv_paraseptal_connections[i]] + \
    #                                                         lv_paraseptalApex_offsets[lv_paraseptal_connections[i]]
    #                 if not repeat_process:
    #                     lv_paraseptal_path_mat = lv_paraseptal_path_mat_aux
    #                     lv_paraseptal_distance_mat = lv_paraseptal_distance_mat_aux
    #                     # Clear Arguments to avoid recycling
    #
    #                     # Rule 7) Paraseptal regions of the heart are defined as [0. < rotation-angle (RT) < 0.2] & [0.5 < RT < 0.7], these are connected to their closest paraseptal routing point (anterior or posterior)
    #                     lv_paraseptal_mask = (((0.0 <= node_vc['rt'][node_lvendo]) & (
    #                             node_vc['rt'][node_lvendo] <= 0.2)) | (
    #                                                   (0.5 <= node_vc['rt'][node_lvendo]) & (
    #                                                   node_vc['rt'][node_lvendo] <= 0.7))) & np.logical_not(
    #                         lv_visited)
    #                     lv_PK_distance_mat[lv_paraseptal_mask] = lv_paraseptal_distance_mat[lv_paraseptal_mask]
    #                     lv_PK_path_mat[lv_paraseptal_mask, :] = lv_paraseptal_path_mat[lv_paraseptal_mask, :]
    #                     lv_visited[lv_paraseptal_mask] = True
    #                     # Rule 8) Freewall regions of the heart are connected from apex to base through [0.4, 0.35, 1., :] in the LV
    #                     lv_freewallApex_index = int(np.argmin(
    #                         np.linalg.norm(nodes_vc_concatenated[node_lvendo, :] - np.array([0.4, 0.35, 0.]), ord=2,
    #                                        axis=1)))  # [mid, freewall, endo, lv]
    #                     if not lv_visited[lv_freewallApex_index]:
    #                         lv_PK_distance_mat[lv_freewallApex_index] = lv_hisbundle_distance_mat[lv_freewallApex_index]
    #                         lv_PK_path_mat[lv_freewallApex_index, :] = lv_hisbundle_path_mat[lv_freewallApex_index, :]
    #                         lv_visited[lv_freewallApex_index] = True
    #                     lv_freewallApex_offset = lv_PK_distance_mat[lv_freewallApex_index]
    #                     lv_freewallApex_path_offset = lv_PK_path_mat[lv_freewallApex_index, :]
    #                     lv_freewallApex_path_offset = lv_freewallApex_path_offset[lv_freewallApex_path_offset != get_nan_value()]
    #                     lv_freewall_distance_mat, lv_freewall_path_mat = djikstra(
    #                         np.asarray([lv_freewallApex_index], dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
    #                         lvneighbours, approx_max_path_len=approx_djikstra_max_path_len)
    #                     lv_freewall_path_mat_aux = np.full((lv_freewall_path_mat.shape[0], approx_djikstra_max_path_len),
    #                                                        get_nan_value(),
    #                                                        dtype=np.int32)
    #                     lv_freewall_distance_mat_aux = np.full((lv_freewall_distance_mat.shape[0]), get_nan_value(),
    #                                                            dtype=np.float64)
    #                     for i in range(lv_freewall_distance_mat.shape[0]):
    #                         path = lv_freewall_path_mat[i, 0, :]
    #                         path = path[path != get_nan_value()]  # For visualisation only - path offset
    #                         path = np.concatenate((lv_freewallApex_path_offset, path), axis=0)
    #                         if lv_freewall_path_mat_aux.shape[1] < path.shape[0]:
    #                             repeat_process = True
    #                             break
    #                         lv_freewall_path_mat_aux[i, :path.shape[0]] = path
    #                         lv_freewall_distance_mat_aux[i] = lv_freewall_distance_mat[i, 0] + lv_freewallApex_offset
    #                     if not repeat_process:
    #                         lv_freewall_path_mat = lv_freewall_path_mat_aux
    #                         lv_freewall_distance_mat = lv_freewall_distance_mat_aux
    #
    #                         # Rule 10) Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
    #                         lv_freewall_mask = ((0.2 <= node_vc['rt'][node_lvendo]) & (
    #                                 node_vc['rt'][node_lvendo] <= 0.5)) & np.logical_not(lv_visited)
    #                         lv_PK_distance_mat[lv_freewall_mask] = lv_freewall_distance_mat[lv_freewall_mask]
    #                         lv_PK_path_mat[lv_freewall_mask, :] = lv_freewall_path_mat[lv_freewall_mask, :]
    #                         lv_visited[lv_freewall_mask] = True
    #
    #     return lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, repeat_process



def prepare_for_djikstra(edge, node_xyz, sub_node_index):
    sub_node_xyz = node_xyz[sub_node_index, :]
    node_xyz = None
    sub_edge = edge[np.all(np.isin(edge, sub_node_index), axis=1), :]
    edge = None
    aux_edge = sub_edge
    sub_edge[:, 0] = np.asarray([np.flatnonzero(sub_node_index == node_i)[0] for node_i in sub_edge[:, 0]]).astype(int)
    sub_edge[:, 1] = np.asarray([np.flatnonzero(sub_node_index == node_i)[0] for node_i in sub_edge[:, 1]]).astype(int)
    sub_edge_vec = sub_node_xyz[sub_edge[:, 0], :] - sub_node_xyz[sub_edge[:, 1], :]  # edge vectors
    sub_unfolded_edge = np.concatenate((sub_edge, np.flip(sub_edge, axis=1))).astype(int)
    aux = [[] for i in range(0, sub_node_xyz.shape[0], 1)]
    for i in range(0, len(sub_unfolded_edge), 1):
        aux[sub_unfolded_edge[i, 0]].append(i)
    sub_neighbour = [np.array(n, dtype=int) for n in aux]
    return sub_neighbour, sub_node_xyz, sub_unfolded_edge, sub_edge_vec


def find_node_index(node, val):
    if len(node.shape) == 1:
        node = node[:, np.newaxis]
    return int(np.argmin(np.linalg.norm(node - val, ord=2, axis=1)))


def sort_djikstra_by_distance(source_to_all_distance_mat, source_to_all_path_mat):
    for destination_index in range(source_to_all_path_mat.shape[0]):
        for source_i in range(source_to_all_path_mat.shape[1]):
            path_indexes = source_to_all_path_mat[destination_index, source_i, :]
            path_indexes = path_indexes[path_indexes != get_nan_value()]
            source_to_path_distance = source_to_all_distance_mat[path_indexes, source_i]
            sorted_by_distance_indexes = np.argsort(source_to_path_distance)  # Sort nodes by distance to the reference
            source_to_all_path_mat[destination_index, source_i, :path_indexes.shape[0]] = path_indexes[sorted_by_distance_indexes]
    return source_to_all_path_mat


def sorted_djikstra(source_indexes, djikstra_nodes_xyz, djikstra_unfoldedEdges, djikstra_edgeVEC, djikstra_neighbours,
                    approx_djikstra_max_path_len):
    # Djikstra
    source_to_all_distance_mat, source_to_all_path_mat = djikstra(
        source_id_list=np.asarray(source_indexes, dtype=int),
        djikstra_nodes_xyz=djikstra_nodes_xyz,
        djikstra_unfoldedEdges=djikstra_unfoldedEdges,
        djikstra_edgeVEC=djikstra_edgeVEC,
        djikstra_neighbours=djikstra_neighbours,
        approx_max_path_len=approx_djikstra_max_path_len)
    # Sort path djikstra results by distance
    source_to_all_path_mat = sort_djikstra_by_distance(source_to_all_distance_mat=source_to_all_distance_mat,
                                                       source_to_all_path_mat=source_to_all_path_mat)
    return source_to_all_distance_mat, source_to_all_path_mat


def generate_Purkinje_fibre_between_closest_source_to_destination(node_coordinates, destination_coordinate, source_index_list,
                                               djikstra_nodes_xyz, djikstra_unfoldedEdges, djikstra_edgeVEC,
                                               djikstra_neighbours, approx_djikstra_max_path_len):
    # Source
    source_to_all_distance_mat, source_to_all_path_mat = sorted_djikstra(
        source_indexes=np.asarray(source_index_list, dtype=int),
        djikstra_nodes_xyz=djikstra_nodes_xyz,
        djikstra_unfoldedEdges=djikstra_unfoldedEdges,
        djikstra_edgeVEC=djikstra_edgeVEC,
        djikstra_neighbours=djikstra_neighbours,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    closest_source_to_all_index = np.argmin(source_to_all_distance_mat, axis=1)
    # Destination
    destination_index = find_node_index(node=node_coordinates, val=destination_coordinate)
    print('destination_index ', destination_index)
    source_meta_index = closest_source_to_all_index[destination_index]
    path_indexes = source_to_all_path_mat[destination_index, source_meta_index, :]  # The nodes in this path are the LV his bundle
    path_indexes = path_indexes[path_indexes != get_nan_value()]
    # Get distance to indexes in path
    source_to_path_distance = source_to_all_distance_mat[path_indexes, source_meta_index]
    return source_to_path_distance, path_indexes, source_meta_index


def recover_unfolded_edges_from_path_indexes(path_indexes):
    unfolded_edges = []
    for i in range(len(path_indexes)-1):
        a = np.array([path_indexes[i], path_indexes[i+1]])
        unfolded_edges.append(a)
        b = np.array([path_indexes[i + 1], path_indexes[i]])
        unfolded_edges.append(b)
    return unfolded_edges


def generate_Purkinje_fibre_between_two_points(node_coordinates, destination_coordinate, source_coordinate,
                                               djikstra_nodes_xyz, djikstra_unfoldedEdges, djikstra_edgeVEC,
                                               djikstra_neighbours, approx_djikstra_max_path_len):
    # Source
    source_index = find_node_index(node=node_coordinates, val=source_coordinate)
    source_to_all_distance_mat, source_to_all_path_mat = sorted_djikstra(
        source_indexes=np.asarray([source_index], dtype=int),
        djikstra_nodes_xyz=djikstra_nodes_xyz,
        djikstra_unfoldedEdges=djikstra_unfoldedEdges,
        djikstra_edgeVEC=djikstra_edgeVEC,
        djikstra_neighbours=djikstra_neighbours,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Destination
    destination_index = find_node_index(node=node_coordinates, val=destination_coordinate)
    path_indexes = source_to_all_path_mat[destination_index, 0, :]  # The nodes in this path are the LV his bundle
    path_indexes = path_indexes[path_indexes != get_nan_value()]
    # Get distance to indexes in path
    source_to_path_distance = source_to_all_distance_mat[path_indexes, 0]
    return source_to_path_distance, path_indexes


def generate_Purkinje_fibres(
        from_previous_available_distance_mat, from_previous_available_path_mat,
        node_connected_previous_available_meta_index, node_mask, previous_available_indexes,
        previous_complete_fibre_distances, previous_complete_fibre_indexes, pk_distance_mat, pk_path_mat,
        # to_previous_available_distances,
        visited):
    # Raise a warning if the code is being asked to assign multpile paths to the same node!
    aux_count = np.sum(node_mask)
    node_mask = node_mask & np.logical_not(visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    if aux_count != np.sum(node_mask):
        warn('In generate_Purkinje_fibres: trying to set multiple connecting rules for the same nodes!')
    aux_count = None   # Clear memory to avoid recycling
    # print_warn_1 = True
    print_warn_2 = True
    for node_i in range(node_connected_previous_available_meta_index.shape[0]):
        # Check Rule!
        if node_mask[node_i]:
            # Process case and mark as visited
            correspondent_previous_available_meta_index = node_connected_previous_available_meta_index[node_i]
            correspondent_previous_source_index = previous_available_indexes[correspondent_previous_available_meta_index]
            previous_complete_fibre_meta_index = find_first(correspondent_previous_source_index, previous_complete_fibre_indexes)
            to_source_path = previous_complete_fibre_indexes[:previous_complete_fibre_meta_index]   # Path from origin to source
            to_source_distance = previous_complete_fibre_distances[previous_complete_fibre_meta_index]  # Distance from origin to source
            # if print_warn_1 and not(to_source_distance == to_previous_available_distances[correspondent_previous_available_meta_index]):
            #     print('Check to_source_distance == to_previous_available_distances[correspondent_previous_available_meta_index] ', to_source_distance, to_previous_available_distances[correspondent_previous_available_meta_index])
            #     print_warn_1 = False
            from_previous_path = from_previous_available_path_mat[node_i, correspondent_previous_available_meta_index, :]
            from_previous_path = from_previous_path[from_previous_path != get_nan_value()]  # For visualisation only - path offset
            path_to_node = np.concatenate((to_source_path, from_previous_path), axis=0)
            aux = np.unique(path_to_node)
            if print_warn_2 and not np.all(aux.shape==path_to_node.shape):
                print('warn_2: np.unique(path_to_node) ', path_to_node.shape, aux.shape)
                print_warn_2 = False
            aux = None
            # Check if the data structure is large enough
            if pk_path_mat.shape[1] < path_to_node.shape[0]:
                # If not - then increase its size
                max_path_len_tmp = path_to_node.shape[0] * 2
                PK_path_mat_tmp = np.full((pk_path_mat.shape[0], max_path_len_tmp),
                                          get_nan_value(), dtype=np.int32)
                PK_path_mat_tmp[:, :pk_path_mat.shape[1]] = pk_path_mat
                # Update structures
                pk_path_mat = PK_path_mat_tmp
                # Clear memory to avoid recycling
                max_path_len_tmp = None
                PK_path_mat_tmp = None
            # Add path
            pk_path_mat[node_i, :path_to_node.shape[0]] = path_to_node
            # Clear memory to avoid recycling
            path_to_node = None
            # Add distance cost
            pk_distance_mat[node_i] = from_previous_available_distance_mat[
                                          node_i, node_connected_previous_available_meta_index[node_i]] \
                                      + to_source_distance
            # Add node to the list of visited ones
            visited[node_i] = True
    return pk_distance_mat, pk_path_mat, visited


# TODO: refactor lv_PK_path_mat and rv_PK_path_mat into lists of arrays without nan values
def generate_djikstra_purkinje_tree_from_vc(approx_djikstra_max_path_len, edge, node_lvendo, node_rvendo, node_xyz,
                                            node_ab, node_rt, purkinje_max_ab_cut_threshold):#, vc_rvlv_binary_name):
    '''This funciton creates a Purkinje tree bound to the endocardial surface following the rules described in Camps & Berg et al. (2024)'''
    # No need to use transmural coordinate, because the function is using the surfaces directry read from the geometry creation
    # Future versions that wish to consider intramural Purkinje, would require incorporating the transmural coordinates
    # print('In conduction system')

    ## Initialise data structures for results
    # LV
    lv_his_to_all_distance_mat = np.full((node_lvendo.shape[0]), get_nan_value(), np.float64)
    lv_his_to_all_path_mat = np.full((node_lvendo.shape[0], approx_djikstra_max_path_len*2), get_nan_value(), dtype=np.int32)
    lv_visited = np.zeros((node_lvendo.shape[0]), dtype=bool)   # This will serve to check weather all nodes got assigned a connecting fibre and weather some nodes are being attempted to connect in multiple ways
    # RV
    rv_his_to_all_distance_mat = np.full((node_rvendo.shape[0]), get_nan_value(), np.float64)
    rv_his_to_all_path_mat = np.full((node_rvendo.shape[0], approx_djikstra_max_path_len*2), get_nan_value(), dtype=np.int32)
    rv_visited = np.zeros((node_rvendo.shape[0]), dtype=bool)   # This will serve to check weather all nodes got assigned a connecting fibre and weather some nodes are being attempted to connect in multiple ways

    ### Fixed coordinate values:
    ## Apex to Base (only between apex and base, no valves)
    base_ab_value = get_base_ab_cut_value()
    apex_ab_value = get_apex_ab_cut_value()
    lv_apical_ab_threshold = get_lv_apical_ab_cut_threshold()        # This threshold delineates the nodes that are close to the apex in the LV
    rv_apical_ab_threshold = get_rv_apical_ab_cut_threshold()        # This threshold delineates the nodes that are close to the apex in the RV
    ## Rotational (Cobiveco)
    # Free-wall/Lateral
    freewall_center_rt_value = get_freewall_center_rt_value()
    freewall_posterior_rt_value = get_freewall_posterior_rt_value()
    freewall_anterior_rt_value = get_freewall_anterior_rt_value()
    # Septum # TODO Define these values globally in utils.py
    septal_center_rt_value = 0.85
    septal_anterior_rt_value = 0.7
    septal_posterior_rt_value = 1.
    # Paraseptal # TODO Define these values globally in utils.py
    paraseptal_anterior_center_rt_value = 0.6
    paraseptal_posterior_center_rt_value = 0.1
    paraseptal_septal_posterior_rt_value = 0.   # Be careful with this discontinuity in Cobiveco coordinates - Not the same value as septal_posterior_rt_value
    paraseptal_freewall_posterior_rt_value = freewall_posterior_rt_value
    paraseptal_septal_anterior_rt_value = septal_anterior_rt_value
    paraseptal_freewall_anterior_rt_value = freewall_anterior_rt_value

    ### Define routing points
    # His-av node at coordinates [1., 0.85] == [base, septal] (symmetric between LV and RV)
    his_av_ab_rt = np.array([base_ab_value, septal_center_rt_value])  # [basal, septal]
    # His-apex node at coordinates [0., 0.85] == [apex, septal] (symmetric between LV and RV)
    his_apex_ab_rt = np.array([apex_ab_value, septal_center_rt_value])  # [apical, septal]
    # LV's Paraseptal routing points
    lv_paraseptal_anterior_center_ab_rt = np.array([lv_apical_ab_threshold, paraseptal_anterior_center_rt_value])
    lv_paraseptal_posterior_center_ab_rt = np.array([lv_apical_ab_threshold, paraseptal_posterior_center_rt_value])
    # LV's Freewall routing point
    lv_freewall_center_ab_rt = np.array([lv_apical_ab_threshold, freewall_center_rt_value])

    ## Prepare coordinates
    # node_ab = node_vc[vc_ab_cut_name]
    # node_rt = node_vc[vc_rt_name]
    # LV
    lv_node_ab = node_ab[node_lvendo]
    lv_node_rt = node_rt[node_lvendo]
    lv_node_ab_rt = np.transpose(np.array([lv_node_ab, lv_node_rt], dtype=float))
    # RV
    rv_node_ab = node_ab[node_rvendo]
    rv_node_rt = node_rt[node_rvendo]
    rv_node_ab_rt = np.transpose(np.array([rv_node_ab, rv_node_rt], dtype=float))
    # Clear memory and prevent recycling
    node_ab_rt = None
    node_ab = None
    node_rt = None

    ## Prepare inputs for Djikstra
    # Set LV endocardial edges aside
    lv_neighbour, lvnodes_xyz, lvunfoldedEdges, lvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
                                                                                 sub_node_index=node_lvendo)
    # Set RV endocardial edges aside
    rv_neighbour, rvnodes_xyz, rvunfoldedEdges, rvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
                                                                                 sub_node_index=node_rvendo)
    # Clear memory and prevent recycling
    # edge = None
    # node_lvendo = None
    # node_rvendo = None
    # node_xyz = None

    ### Define regions in the heart (masks)
    ## Apex to Base Masks
    # Baseline mask within the valid region to grow Purkinje fibres
    lv_baseline_purkinje_mask = (apex_ab_value <= lv_node_ab) & (lv_node_ab <= base_ab_value) \
                                & (lv_node_ab <= purkinje_max_ab_cut_threshold)  # Global Purkinje growth restriction
    rv_baseline_purkinje_mask = (apex_ab_value <= rv_node_ab) & (rv_node_ab <= base_ab_value) \
                                & (rv_node_ab <= purkinje_max_ab_cut_threshold)  # Global Purkinje growth restriction
    # Apical regions of the heart are defined as [apex-to-base (AB) < 0.4/0.2] in the LV/RV
    lv_apical_mask = lv_baseline_purkinje_mask & (lv_node_ab <= lv_apical_ab_threshold)
    # RV
    rv_apical_mask = rv_baseline_purkinje_mask & (rv_node_ab <= rv_apical_ab_threshold)

    ## Rotational Masks
    # Freewall/Lateral regions of the heart are defined as [freewall_posterior_rt_value < rotation-angle (RT) < freewall_anterior_rt_value] [0.2 < RT < 0.5]
    lv_freewall_mask = lv_baseline_purkinje_mask & (     # Global Purkinje growth restriction
            (freewall_posterior_rt_value <= lv_node_rt) & (lv_node_rt <= freewall_anterior_rt_value))
    rv_freewall_mask = rv_baseline_purkinje_mask & (     # Global Purkinje growth restriction
            (freewall_posterior_rt_value <= rv_node_rt) & (rv_node_rt <= freewall_anterior_rt_value))

    # Septal regions of the heart are defined as [septal_anterior_rt_value < rotation-angle (RT) < septal_posterior_rt_value] [0.7 < RT < 1.]
    lv_septal_mask = lv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
            (septal_anterior_rt_value <= lv_node_rt) & (lv_node_rt <= septal_posterior_rt_value)))
    rv_septal_mask = rv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
            (septal_anterior_rt_value <= rv_node_rt) & (rv_node_rt <= septal_posterior_rt_value)))

    # Paraseptal posterior regions of the heart are defined as [paraseptal_septal_posterior_rt_value < rotation-angle (RT) < paraseptal_freewall_posterior_rt_value] [0. < RT < 0.2]
    lv_paraseptal_posterior_mask = lv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
            (paraseptal_septal_posterior_rt_value <= lv_node_rt) & (lv_node_rt <= paraseptal_freewall_posterior_rt_value)))
    rv_paraseptal_posterior_mask = rv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
            (paraseptal_septal_posterior_rt_value <= rv_node_rt) & (rv_node_rt <= paraseptal_freewall_posterior_rt_value)))

    # Paraseptal anterior regions of the heart are defined as [paraseptal_freewall_anterior_rt_value < rotation-angle (RT) < paraseptal_septal_anterior_rt_value] [0.5 < RT < 0.7]
    lv_paraseptal_anterior_mask = lv_baseline_purkinje_mask & ((  # Global Purkinje growth restriction
            (paraseptal_freewall_anterior_rt_value <= lv_node_rt) & (lv_node_rt <= paraseptal_septal_anterior_rt_value)))
    rv_paraseptal_anterior_mask = rv_baseline_purkinje_mask & ((  # Global Purkinje growth restriction
            (paraseptal_freewall_anterior_rt_value <= rv_node_rt) & (rv_node_rt <= paraseptal_septal_anterior_rt_value)))

    #### Generation of the Purkinje tree using physiological rules
    ### The His-bundel
    ## Rule 1) The Purkinje tree begins symmetrically at the his-av node in both ventricles
    ## Rule 2) The hisbundle goes down to the apex through while trying to keep a straight rotational trajectory
    lv_hisBundle_distances, lv_hisBundle_indexes = generate_Purkinje_fibre_between_two_points(
        node_coordinates=lv_node_ab_rt, destination_coordinate=his_apex_ab_rt, source_coordinate=his_av_ab_rt,
        djikstra_nodes_xyz=lvnodes_xyz, djikstra_unfoldedEdges=lvunfoldedEdges, djikstra_edgeVEC=lvedgeVEC,
        djikstra_neighbours=lv_neighbour, approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    rv_hisBundle_distances, rv_hisBundle_indexes = generate_Purkinje_fibre_between_two_points(
        node_coordinates=rv_node_ab_rt, destination_coordinate=his_apex_ab_rt, source_coordinate=his_av_ab_rt,
        djikstra_nodes_xyz=rvnodes_xyz, djikstra_unfoldedEdges=rvunfoldedEdges, djikstra_edgeVEC=rvedgeVEC,
        djikstra_neighbours=rv_neighbour, approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # TODO Improve this section - Right now it only prevents reusing of the Bundles, but this is a problem that could hapen elsewere
    # When translating the inference results to MonoAlg3D and using the Shocker algorithm, there can be problems
    # One of the problems is that Shocker cannot deal with Purkinje trees that go back through the same edges (like a U-turn)
    # The Eikonal doesn't care about this, because it only uses the final distance of the path to compute the LAT at the root node.
    # But Shocker needs branches to be separated in space or when running the monodomain later, it will ignore the extra
    # lenght from doing the U-turn.
    # Eikonal tree: Goes down to the apex and goes back on itself.
    # H
    # |  R
    # | /
    # |/
    # |
    # |
    # |
    # |
    # A
    #
    # Monodomain equivalent: Goes back to the apex and needs to take a different path when coming back
    # H
    # |  R
    # |  |
    # |  |
    # |  |
    # |  |
    # | /
    # |/
    # A
    #
    # To deal with this, we are going to remove the edges in the his-bundle from the list of edges and hope that this
    # will solve most of this type of problems. However, a better solution would be to properly integrate Shocker with this
    # algorithm and have shared constrains and rules, or to run Shocker during the inference process.
    # Remove previous edges from graph LV
    visited_edge_lv = node_lvendo[get_edge_list_from_path(path=lv_hisBundle_indexes)]
    visited_edge_rv = node_rvendo[get_edge_list_from_path(path=rv_hisBundle_indexes)]
    visited_edge = np.concatenate((visited_edge_lv, visited_edge_rv))
    visited_edge = np.concatenate((visited_edge, np.flip(visited_edge)))
    edge_remove_index = np.zeros((edge.shape[0]), dtype=bool)
    for visited_edge_i in range(visited_edge.shape[0]):
        edge_remove_index = edge_remove_index + np.all(edge == visited_edge[visited_edge_i, :], axis=1)
    edge = edge[np.logical_not(edge_remove_index), :]  # Remove the His-bundles from the
    ## Re-Run Prepare inputs for Djikstra
    # Re-Run Set LV endocardial edges aside
    lv_neighbour, lvnodes_xyz, lvunfoldedEdges, lvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
                                                                                 sub_node_index=node_lvendo)
    # Re-Run Set RV endocardial edges aside
    rv_neighbour, rvnodes_xyz, rvunfoldedEdges, rvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
                                                                                 sub_node_index=node_rvendo)


    ## Rule 3) (#1 in the paper) The RV rib-cage-like Purkinje fibres. The apical and Lateral/Freewall in the RV can connect directly to their closest point in the ab coordinate in the hisbundle that has ab < 0.8
    # TODO This rule could be updated to incorporate moderator bands, but it requires later the hability to use Purkinje fibres that are not bound to the endocardium
    # Filter the (root) nodes that are connected according to this rule
    rv_apical_freewall_mask = (rv_apical_mask | rv_freewall_mask) & np.logical_not(rv_visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    # Filter the source nodes that are included in this rule
    rv_hisBundle_ab_values = rv_node_ab[rv_hisBundle_indexes]
    rv_hisBundle_meta_available_indexes = np.nonzero(
        (rv_hisBundle_ab_values >= apex_ab_value) & (rv_hisBundle_ab_values <= base_ab_value)   # Purkinje can only grow between apex and base
        & (rv_hisBundle_ab_values <= purkinje_max_ab_cut_threshold)     # Purkinje can only connect to and from nodes below the Purkinje growth area
    )[0]
    rv_hisBundle_available_indexes = rv_hisBundle_indexes[rv_hisBundle_meta_available_indexes]
    rv_hisBundle_available_ab_values = rv_hisBundle_ab_values[rv_hisBundle_meta_available_indexes]
    # rv_hisBundle_available_distances = rv_hisBundle_distances[rv_hisBundle_meta_available_indexes]
    # Clear memory and prevent recycling
    rv_hisBundle_ab_values = None
    rv_hisBundle_meta_available_indexes = None

    # Calculate Djikstra from all source nodes in this rule to all the endo RV nodes (without filtering by rule)
    rv_hisbundle_distance_mat_rule3, rv_hisbundle_path_mat_rule3 = sorted_djikstra(
        source_indexes=np.asarray(rv_hisBundle_available_indexes, dtype=int),
        djikstra_nodes_xyz=rvnodes_xyz,
        djikstra_unfoldedEdges=rvunfoldedEdges,
        djikstra_edgeVEC=rvedgeVEC,
        djikstra_neighbours=rv_neighbour,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Match endo RV nodes to their closest hisBundle node using apex-to-base difference as the distance between them
    rv_ab_dist = np.abs(rv_node_ab[:, np.newaxis] - rv_hisBundle_available_ab_values)
    rv_hisBundle_available_meta_index_connected_rule3 = np.argmin(np.abs(rv_ab_dist), axis=1)
    # Clear memory and prevent recycling
    rv_hisBundle_available_ab_values = None

    # Add the new fibres to the final tree structure
    rv_his_to_all_distance_mat, rv_his_to_all_path_mat, rv_visited = generate_Purkinje_fibres(
        from_previous_available_distance_mat=rv_hisbundle_distance_mat_rule3,
        from_previous_available_path_mat=rv_hisbundle_path_mat_rule3,
        node_connected_previous_available_meta_index=rv_hisBundle_available_meta_index_connected_rule3,
        node_mask=rv_apical_freewall_mask,  # Here is where Rule 3 is applied
        previous_available_indexes=rv_hisBundle_available_indexes,
        previous_complete_fibre_distances=rv_hisBundle_distances,
        previous_complete_fibre_indexes=rv_hisBundle_indexes,
        pk_distance_mat=rv_his_to_all_distance_mat,
        pk_path_mat=rv_his_to_all_path_mat,
        visited=rv_visited)
    # Clear memory and prevent recycling
    rv_hisBundle_available_meta_index_connected_rule3 = None
    rv_hisBundle_available_distances = None
    rv_hisBundle_available_indexes = None
    rv_hisbundle_distance_mat_rule3 = None
    rv_hisbundle_path_mat_rule3 = None
    rv_apical_freewall_mask = None

    ## Rule 4) (#2 in the paper) The LV's apical hisbundle can directly connects to Septal and Apical nodes using the shortest distance between hisBundle and (root) node
    # Filter the (root) nodes that are connected according to this rule
    lv_apical_or_septal_mask = (lv_apical_mask | lv_septal_mask) & np.logical_not(lv_visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    # Filter the source nodes that are included in this rule
    lv_hisBundle_ab_values = lv_node_ab[lv_hisBundle_indexes]
    lv_hisBundle_apical_meta_indexes = np.nonzero(
        (lv_hisBundle_ab_values >= apex_ab_value) & (lv_hisBundle_ab_values <= base_ab_value)  # Purkinje can only grow between apex and base
        & (lv_hisBundle_ab_values <= purkinje_max_ab_cut_threshold)  # Purkinje can only connect to and from nodes below the Purkinje growth area
        & (lv_hisBundle_ab_values <= lv_apical_ab_threshold)        # Specific SOURCE RULE
    )[0]
    lv_hisBundle_apical_available_indexes = lv_hisBundle_indexes[lv_hisBundle_apical_meta_indexes]
    lv_hisBundle_apical_available_distances = lv_hisBundle_distances[lv_hisBundle_apical_meta_indexes]
    # Clear memory and prevent recycling
    lv_hisBundle_ab_values = None

    # Calculate Djikstra from all source nodes in this rule to all the endo LV nodes (without filtering by rule)
    lv_hisbundle_apical_distance_mat_rule4, lv_hisbundle_apical_path_mat_rule4 = sorted_djikstra(
        source_indexes=np.asarray(lv_hisBundle_apical_available_indexes, dtype=int),
        djikstra_nodes_xyz=lvnodes_xyz,
        djikstra_unfoldedEdges=lvunfoldedEdges,
        djikstra_edgeVEC=lvedgeVEC,
        djikstra_neighbours=lv_neighbour,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Match endo LV nodes to their closest hisBundle node using Djikstra distance between them
    lv_hisBundle_apical_available_meta_index_connected_rule4 = np.argmin(lv_hisbundle_apical_distance_mat_rule4, axis=1)

    # Add the new fibres to the final tree structure
    lv_his_to_all_distance_mat, lv_his_to_all_path_mat, lv_visited = generate_Purkinje_fibres(
        from_previous_available_distance_mat=lv_hisbundle_apical_distance_mat_rule4,
        from_previous_available_path_mat=lv_hisbundle_apical_path_mat_rule4,
        node_connected_previous_available_meta_index=lv_hisBundle_apical_available_meta_index_connected_rule4,
        node_mask=lv_apical_or_septal_mask,
        previous_available_indexes=lv_hisBundle_apical_available_indexes,
        previous_complete_fibre_distances=lv_hisBundle_distances,
        previous_complete_fibre_indexes=lv_hisBundle_indexes,
        # to_previous_available_distances=lv_hisBundle_apical_available_distances,
        pk_distance_mat=lv_his_to_all_distance_mat,
        pk_path_mat=lv_his_to_all_path_mat,
        visited=lv_visited)
    # Clear memory and prevent recycling
    lv_hisBundle_apical_available_meta_index_connected_rule4 = None
    lv_hisBundle_available_distances = None
    lv_hisBundle_available_indexes = None
    lv_hisbundle_apical_distance_mat_rule4 = None
    lv_hisbundle_apical_path_mat_rule4 = None
    lv_apical_or_septal_mask = None


    ## Rule 5) (#3 in the paper) The RV's apical hisbundle can directly connects to Non-Apical-Septal and Non-Apical-Paraseptal nodes using the shortest distance between hisBundle and (root) node
    # Filter the (root) nodes that are connected according to this rule
    rv_non_apical_septal_or_paraseptal_mask = (rv_septal_mask | rv_paraseptal_anterior_mask | rv_paraseptal_posterior_mask) \
                                              & np.logical_not(rv_visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    # Filter the source nodes that are included in this rule
    rv_hisBundle_ab_values = rv_node_ab[rv_hisBundle_indexes]
    rv_hisBundle_apical_meta_indexes = np.nonzero(
        (apex_ab_value <= rv_hisBundle_ab_values) & (rv_hisBundle_ab_values <= base_ab_value)  # Purkinje can only grow between apex and base
        & (rv_hisBundle_ab_values <= purkinje_max_ab_cut_threshold)  # Purkinje can only connect to and from nodes below the Purkinje growth area
        & (rv_hisBundle_ab_values <= rv_apical_ab_threshold)  # Specific SOURCE RULE
    )[0]
    rv_hisBundle_apical_available_indexes = rv_hisBundle_indexes[rv_hisBundle_apical_meta_indexes]
    rv_hisBundle_apical_available_distances = rv_hisBundle_distances[rv_hisBundle_apical_meta_indexes]
    # Clear memory and prevent recycling
    rv_hisBundle_ab_values = None

    # Calculate Djikstra from all source nodes in this rule to all the endo RV nodes (without filtering by rule)
    # print('rv_hisBundle_apical_available_indexes ', rv_hisBundle_apical_available_indexes)
    rv_hisbundle_apical_distance_mat_rule5, rv_hisbundle_apical_path_mat_rule5 = sorted_djikstra(
        source_indexes=np.asarray(rv_hisBundle_apical_available_indexes, dtype=int),
        djikstra_nodes_xyz=rvnodes_xyz,
        djikstra_unfoldedEdges=rvunfoldedEdges,
        djikstra_edgeVEC=rvedgeVEC,
        djikstra_neighbours=rv_neighbour,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Match endo RV nodes to their closest hisBundle node using Djikstra distance between them
    # print('rv_hisbundle_apical_distance_mat_rule5 ', rv_hisbundle_apical_distance_mat_rule5)
    rv_hisBundle_apical_available_meta_index_connected_rule5 = np.argmin(rv_hisbundle_apical_distance_mat_rule5, axis=1)

    # Add the new fibres to the final tree structure
    rv_his_to_all_distance_mat, rv_his_to_all_path_mat, rv_visited = generate_Purkinje_fibres(
        from_previous_available_distance_mat=rv_hisbundle_apical_distance_mat_rule5,
        from_previous_available_path_mat=rv_hisbundle_apical_path_mat_rule5,
        node_connected_previous_available_meta_index=rv_hisBundle_apical_available_meta_index_connected_rule5,
        node_mask=rv_non_apical_septal_or_paraseptal_mask,
        previous_available_indexes=rv_hisBundle_apical_available_indexes,
        previous_complete_fibre_distances=rv_hisBundle_distances,
        previous_complete_fibre_indexes=rv_hisBundle_indexes,
        pk_distance_mat=rv_his_to_all_distance_mat,
        pk_path_mat=rv_his_to_all_path_mat,
        visited=rv_visited)
    # Clear memory and prevent recycling
    rv_hisBundle_apical_available_meta_index_connected_rule5 = None
    rv_hisBundle_available_distances = None
    rv_hisBundle_available_indexes = None
    rv_hisbundle_apical_distance_mat_rule5 = None
    rv_hisbundle_apical_path_mat_rule5 = None
    rv_non_apical_septal_or_paraseptal_mask = None


    ## Rule 6) (#4.1 in the paper) LV Paraseptal anterior and posterior routing points connect to their closest hisBundle Apical point
    # Filter the source nodes that are included in this rule
    lv_hisBundle_ab_values = lv_node_ab[lv_hisBundle_indexes]
    lv_hisBundle_apical_meta_indexes = np.nonzero(
        (apex_ab_value <= lv_hisBundle_ab_values) & (lv_hisBundle_ab_values <= base_ab_value)  # Purkinje can only grow between apex and base
        & (lv_hisBundle_ab_values <= purkinje_max_ab_cut_threshold)  # Purkinje can only connect to and from nodes below the Purkinje growth area
        & (lv_hisBundle_ab_values <= lv_apical_ab_threshold)  # Specific SOURCE RULE
    )[0]
    lv_hisBundle_apical_available_indexes = lv_hisBundle_indexes[lv_hisBundle_apical_meta_indexes]
    # lv_hisBundle_apical_available_distances = rv_hisBundle_distances[rv_hisBundle_apical_meta_indexes]
    # Clear memory and prevent recycling
    lv_hisBundle_ab_values = None

    lv_hisBundle_to_paraseptal_anterior_distances, lv_hisBundle_to_paraseptal_anterior_indexes, \
        lv_hisBundle_apical_available_anterior_source_meta_index = \
        generate_Purkinje_fibre_between_closest_source_to_destination(
            node_coordinates=lv_node_ab_rt, destination_coordinate=lv_paraseptal_anterior_center_ab_rt,
            source_index_list=lv_hisBundle_apical_available_indexes, djikstra_nodes_xyz=lvnodes_xyz,
            djikstra_unfoldedEdges=lvunfoldedEdges, djikstra_edgeVEC=lvedgeVEC, djikstra_neighbours=lv_neighbour,
            approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Recover the previous path from the his-av to the source in the his-bundle
    lv_hisBundle_source_index = lv_hisBundle_apical_available_indexes[lv_hisBundle_apical_available_anterior_source_meta_index]
    lv_hisBundle_source_meta_index = find_first(lv_hisBundle_source_index, lv_hisBundle_indexes)
    to_source_path = lv_hisBundle_indexes[:lv_hisBundle_source_meta_index]  # Path within the his-bundle
    to_source_distance = lv_hisBundle_distances[lv_hisBundle_source_meta_index]     # Distance withing the his-bundle
    # Add distance and path from his-bundle
    lv_hisAv_to_paraseptal_anterior_distances = np.concatenate(
        (lv_hisBundle_distances[:lv_hisBundle_source_meta_index],
         lv_hisBundle_to_paraseptal_anterior_distances + to_source_distance), axis=0)
    lv_hisAv_to_paraseptal_anterior_indexes = np.concatenate((to_source_path, lv_hisBundle_to_paraseptal_anterior_indexes), axis=0)
    # Clear memory and prevent recycling
    lv_hisBundle_apical_available_anterior_source_meta_index = None
    lv_hisBundle_source_index = None
    lv_hisBundle_source_meta_index = None
    lv_hisBundle_to_paraseptal_anterior_distances = None
    lv_hisBundle_to_paraseptal_anterior_indexes = None
    to_source_path = None
    to_source_distance = None

    lv_hisBundle_to_paraseptal_posterior_distances, lv_hisBundle_to_paraseptal_posterior_indexes, \
        lv_hisBundle_apical_available_posterior_source_meta_index = \
        generate_Purkinje_fibre_between_closest_source_to_destination(
            node_coordinates=lv_node_ab_rt, destination_coordinate=lv_paraseptal_posterior_center_ab_rt,
            source_index_list=lv_hisBundle_apical_available_indexes, djikstra_nodes_xyz=lvnodes_xyz,
            djikstra_unfoldedEdges=lvunfoldedEdges, djikstra_edgeVEC=lvedgeVEC, djikstra_neighbours=lv_neighbour,
            approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Recover the previous path from the his-av to the source in the his-bundle
    lv_hisBundle_source_index = lv_hisBundle_apical_available_indexes[lv_hisBundle_apical_available_posterior_source_meta_index]
    lv_hisBundle_source_meta_index = find_first(lv_hisBundle_source_index, lv_hisBundle_indexes)
    to_source_path = lv_hisBundle_indexes[:lv_hisBundle_source_meta_index]  # Path within the his-bundle
    to_source_distance = lv_hisBundle_distances[lv_hisBundle_source_meta_index]  # Distance withing the his-bundle
    # Add distance and path from his-bundle
    lv_hisAv_to_paraseptal_posterior_distances = np.concatenate(
        (lv_hisBundle_distances[:lv_hisBundle_source_meta_index],
         lv_hisBundle_to_paraseptal_posterior_distances + to_source_distance), axis=0)
    lv_hisAv_to_paraseptal_posterior_indexes = np.concatenate(
        (to_source_path, lv_hisBundle_to_paraseptal_posterior_indexes), axis=0)
    # Clear memory and prevent recycling
    lv_hisBundle_apical_available_posterior_source_meta_index = None
    lv_hisBundle_source_index = None
    lv_hisBundle_source_meta_index = None
    lv_hisBundle_to_paraseptal_posterior_distances = None
    lv_hisBundle_to_paraseptal_posterior_indexes = None
    to_source_path = None
    to_source_distance = None


    ## Rule 7) (#4.2 in the paper) LV Paraseptal anterior regions are connected from apex to base through the anterior routing point
    # Filter the (root) nodes that are connected according to this rule
    lv_non_apical_paraseptal_anterior_mask = lv_paraseptal_anterior_mask & np.logical_not(lv_apical_mask) \
                                             & np.logical_not(lv_visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    # Filter the source nodes that are included in this rule
    lv_paraseptal_anterior_routing_index = lv_hisAv_to_paraseptal_anterior_indexes[-1]

    # Calculate Djikstra from all source nodes in this rule to all the endo LV nodes (without filtering by rule)
    lv_paraseptal_anterior_distance_mat, lv_paraseptal_anterior_path_mat = sorted_djikstra(
        source_indexes=np.asarray([lv_paraseptal_anterior_routing_index], dtype=int),
        djikstra_nodes_xyz=lvnodes_xyz,
        djikstra_unfoldedEdges=lvunfoldedEdges,
        djikstra_edgeVEC=lvedgeVEC,
        djikstra_neighbours=lv_neighbour,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)

    # Add the new fibres to the final tree structure
    node_connected_previous_available_meta_index = np.zeros((lv_paraseptal_anterior_distance_mat.shape[0]), dtype=int)
    lv_his_to_all_distance_mat, lv_his_to_all_path_mat, lv_visited = generate_Purkinje_fibres(
        from_previous_available_distance_mat=lv_paraseptal_anterior_distance_mat,
        from_previous_available_path_mat=lv_paraseptal_anterior_path_mat,
        node_connected_previous_available_meta_index=node_connected_previous_available_meta_index,  # There is only one possible source point
        node_mask=lv_non_apical_paraseptal_anterior_mask,
        previous_available_indexes=np.asarray([lv_paraseptal_anterior_routing_index], dtype=int),
        previous_complete_fibre_distances=lv_hisAv_to_paraseptal_anterior_distances,
        previous_complete_fibre_indexes=lv_hisAv_to_paraseptal_anterior_indexes,
        # to_previous_available_distances=lv_hisAv_to_paraseptal_anterior_distances,
        pk_distance_mat=lv_his_to_all_distance_mat,
        pk_path_mat=lv_his_to_all_path_mat,
        visited=lv_visited)
    # Clear memory and prevent recycling
    lv_paraseptal_anterior_routing_index = None
    lv_paraseptal_anterior_distance_mat = None
    lv_paraseptal_anterior_path_mat = None
    lv_hisAv_to_paraseptal_anterior_distances = None
    lv_hisAv_to_paraseptal_anterior_indexes = None
    lv_non_apical_paraseptal_anterior_mask = None


    ## Rule 8) (#4.3 in the paper) LV Paraseptal posterior regions are connected from apex to base through the posterior routing point
    # Filter the (root) nodes that are connected according to this rule
    lv_non_apical_paraseptal_posterior_mask = lv_paraseptal_posterior_mask & np.logical_not(lv_apical_mask) \
                                             & np.logical_not(
        lv_visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    # Filter the source nodes that are included in this rule
    lv_paraseptal_posterior_routing_index = lv_hisAv_to_paraseptal_posterior_indexes[-1]

    # Calculate Djikstra from all source nodes in this rule to all the endo LV nodes (without filtering by rule)
    lv_paraseptal_posterior_distance_mat, lv_paraseptal_posterior_path_mat = sorted_djikstra(
        source_indexes=np.asarray([lv_paraseptal_posterior_routing_index], dtype=int),
        djikstra_nodes_xyz=lvnodes_xyz,
        djikstra_unfoldedEdges=lvunfoldedEdges,
        djikstra_edgeVEC=lvedgeVEC,
        djikstra_neighbours=lv_neighbour,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)

    # Add the new fibres to the final tree structure
    lv_his_to_all_distance_mat, lv_his_to_all_path_mat, lv_visited = generate_Purkinje_fibres(
        from_previous_available_distance_mat=lv_paraseptal_posterior_distance_mat,
        from_previous_available_path_mat=lv_paraseptal_posterior_path_mat,
        node_connected_previous_available_meta_index=np.zeros((lv_paraseptal_posterior_distance_mat.shape[0]), dtype=int),
        # There is only one possible source point
        node_mask=lv_non_apical_paraseptal_posterior_mask,
        previous_available_indexes=np.asarray([lv_paraseptal_posterior_routing_index], dtype=int),
        previous_complete_fibre_distances=lv_hisAv_to_paraseptal_posterior_distances,
        previous_complete_fibre_indexes=lv_hisAv_to_paraseptal_posterior_indexes,
        pk_distance_mat=lv_his_to_all_distance_mat,
        pk_path_mat=lv_his_to_all_path_mat,
        visited=lv_visited)
    # Clear memory and prevent recycling
    lv_paraseptal_posterior_routing_index = None
    lv_paraseptal_posterior_distance_mat = None
    lv_paraseptal_posterior_path_mat = None
    lv_hisAv_to_paraseptal_posterior_distances = None
    lv_hisAv_to_paraseptal_posterior_indexes = None
    lv_non_apical_paraseptal_posterior_mask = None
    lv_paraseptal_posterior_routing_index = None

    ## Rule 9) (#5.1 in the paper) The LV Freewall routing point at lv_freewall_center_ab_rt is connected to the apical his-bundle by the shortest distance
    # Filter the source nodes that are included in this rule
    lv_hisBundle_ab_values = lv_node_ab[lv_hisBundle_indexes]
    lv_hisBundle_apical_meta_indexes = np.nonzero(
        (apex_ab_value <= lv_hisBundle_ab_values) & (lv_hisBundle_ab_values <= base_ab_value)  # Purkinje can only grow between apex and base
        & (lv_hisBundle_ab_values <= purkinje_max_ab_cut_threshold)  # Purkinje can only connect to and from nodes below the Purkinje growth area
        & (lv_hisBundle_ab_values <= lv_apical_ab_threshold)  # Specific SOURCE RULE
    )[0]
    lv_hisBundle_apical_available_indexes = lv_hisBundle_indexes[lv_hisBundle_apical_meta_indexes]
    # Clear memory and prevent recycling
    lv_hisBundle_ab_values = None
    lv_hisBundle_apical_meta_indexes = None

    lv_hisBundle_to_freewall_distances, lv_hisBundle_to_freewall_indexes, \
        lv_hisBundle_apical_available_source_meta_index = \
        generate_Purkinje_fibre_between_closest_source_to_destination(
            node_coordinates=lv_node_ab_rt, destination_coordinate=lv_freewall_center_ab_rt,
            source_index_list=lv_hisBundle_apical_available_indexes, djikstra_nodes_xyz=lvnodes_xyz,
            djikstra_unfoldedEdges=lvunfoldedEdges, djikstra_edgeVEC=lvedgeVEC, djikstra_neighbours=lv_neighbour,
            approx_djikstra_max_path_len=approx_djikstra_max_path_len)
    # Recover the previous path from the his-av to the source in the his-bundle
    lv_hisBundle_source_index = lv_hisBundle_apical_available_indexes[
        lv_hisBundle_apical_available_source_meta_index]
    lv_hisBundle_source_meta_index = find_first(lv_hisBundle_source_index, lv_hisBundle_indexes)
    to_source_path = lv_hisBundle_indexes[:lv_hisBundle_source_meta_index]  # Path within the his-bundle
    to_source_distance = lv_hisBundle_distances[lv_hisBundle_source_meta_index]  # Distance withing the his-bundle
    # Add distance and path from his-bundle
    lv_hisAv_to_freewall_distances = np.concatenate(
        (lv_hisBundle_distances[:lv_hisBundle_source_meta_index],
         lv_hisBundle_to_freewall_distances + to_source_distance), axis=0)
    lv_hisAv_to_freewall_indexes = np.concatenate(
        (to_source_path, lv_hisBundle_to_freewall_indexes), axis=0)
    # Clear memory and prevent recycling
    lv_hisBundle_apical_available_source_meta_index = None
    lv_hisBundle_source_index = None
    lv_hisBundle_source_meta_index = None
    lv_hisBundle_to_freewall_distances = None
    lv_hisBundle_to_freewall_indexes = None
    to_source_path = None
    to_source_distance = None

    ## Rule 10) (#5.2 in the paper) LV Freewall/Lateral regions of the heart (defined as [0.2 < rotation-angle (RT) < 0.5]) are connected from apex to base through the routing point at lv_freewall_center_ab_rt
    # Filter the (root) nodes that are connected according to this rule
    lv_non_apical_freewall_mask = lv_freewall_mask & np.logical_not(lv_apical_mask) \
                                  & np.logical_not(lv_visited)  # Try to make sure that a node does not get assigned two paths using a register of those visited
    # Filter the source nodes that are included in this rule
    lv_freewall_routing_index = lv_hisAv_to_freewall_indexes[-1]

    # Calculate Djikstra from all source nodes in this rule to all the endo LV nodes (without filtering by rule)
    lv_freewall_to_all_distance_mat, lv_freewall_to_all_path_mat = sorted_djikstra(
        source_indexes=np.asarray([lv_freewall_routing_index], dtype=int),
        djikstra_nodes_xyz=lvnodes_xyz,
        djikstra_unfoldedEdges=lvunfoldedEdges,
        djikstra_edgeVEC=lvedgeVEC,
        djikstra_neighbours=lv_neighbour,
        approx_djikstra_max_path_len=approx_djikstra_max_path_len)

    # Add the new fibres to the final tree structure
    lv_his_to_all_distance_mat, lv_his_to_all_path_mat, lv_visited = generate_Purkinje_fibres(
        from_previous_available_distance_mat=lv_freewall_to_all_distance_mat,
        from_previous_available_path_mat=lv_freewall_to_all_path_mat,
        node_connected_previous_available_meta_index=np.zeros((lv_freewall_to_all_distance_mat.shape[0]), dtype=int),     # There is only one possible source point
        node_mask=lv_non_apical_freewall_mask,
        previous_available_indexes=np.asarray([lv_freewall_routing_index], dtype=int),
        previous_complete_fibre_distances=lv_hisAv_to_freewall_distances,
        previous_complete_fibre_indexes=lv_hisAv_to_freewall_indexes,
        pk_distance_mat=lv_his_to_all_distance_mat,
        pk_path_mat=lv_his_to_all_path_mat,
        visited=lv_visited)
    # Clear memory and prevent recycling
    lv_freewall_routing_index = None
    lv_freewall_to_all_distance_mat = None
    lv_freewall_to_all_path_mat = None
    lv_hisBundle_available_indexes = None
    lv_hisAv_to_freewall_distances = None
    lv_hisAv_to_freewall_indexes = None
    lv_non_apical_freewall_mask = None

    # Verify that all candidate root nodes have been assigned a path and distance to the his-av
    print('Check LV completion: ', lv_visited.shape[0] - np.sum(np.logical_not(lv_baseline_purkinje_mask))
          - np.sum(lv_visited[lv_baseline_purkinje_mask]))
    print('Check RV completion: ', rv_visited.shape[0] - np.sum(np.logical_not(rv_baseline_purkinje_mask))
          - np.sum(rv_visited[rv_baseline_purkinje_mask]))

    return lv_his_to_all_distance_mat, lv_his_to_all_path_mat, rv_his_to_all_distance_mat, rv_his_to_all_path_mat


# def generate_djikstra_purkinje_tree_from_vc_backup_2(approx_djikstra_max_path_len,
#                                             edge, node_lvendo,
#                                               node_rvendo, node_xyz, node_vc, vc_ab_cut_name, vc_rt_name):#, vc_rvlv_binary_name):
#     '''This funciton creates a Purkinje tree bound to the endocardial surface following the rules described in Camps et al. (2024)'''
#     # No need to use transmural coordinate, because the function is using the surfaces directry read from the geometry creation
#     # Future versions that wish to consider intramural Purkinje, would require incorporating the transmural coordinates
#
#     ### Initialise variables for the iteration process:
#     lv_approx_djikstra_max_path_len = approx_djikstra_max_path_len
#     rv_approx_djikstra_max_path_len = approx_djikstra_max_path_len
#     nb_lvendo_nodes = node_lvendo.shape[0]
#     nb_rvendo_nodes = node_rvendo.shape[0]
#
#     ## Data structures for results
#     # LV
#     lv_PK_distance_mat = np.full((nb_lvendo_nodes), get_nan_value(), np.float64)
#     lv_PK_path_mat = np.full((nb_lvendo_nodes, lv_approx_djikstra_max_path_len), get_nan_value(), dtype=np.int32)
#     lv_visited = np.zeros((nb_lvendo_nodes), dtype=bool)
#     # RV
#     rv_PK_distance_mat = np.full((nb_rvendo_nodes), get_nan_value(), np.float64)
#     rv_PK_path_mat = np.full((nb_rvendo_nodes, rv_approx_djikstra_max_path_len), get_nan_value(), dtype=np.int32)
#     rv_visited = np.zeros((nb_rvendo_nodes), dtype=bool)
#
#     ### Fixed coordinate values:
#     ## Apex to Base (only between apex and base, no valves)
#     base_ab_value = 1.
#     apex_ab_value = 0.
#     lv_apical_ab_threshold = 0.4    # This threshold delineates the nodes that are close to the apex in the LV
#     # TODO Change this when Ruben gets the new field done
#     rv_apical_ab_threshold = 0.5#0.2    # This threshold delineates the nodes that are close to the apex in the RV
#     purkinje_max_ab_threshold = 0.8 # The Purkinje fibres cannot grow all the way to the base
#     ## Rotational (Cobiveco)
#     # Free-wall/Lateral
#     freewall_posterior_rt_value = 0.2
#     freewall_anterior_rt_value = 0.5
#     # Septum
#     septal_center_rt_value = 0.85
#     septal_anterior_rt_value = 0.7
#     septal_posterior_rt_value = 1.
#     # Paraseptal
#     paraseptal_anterior_center_rt_value = 0.6
#     paraseptal_posterior_center_rt_value = 0.1
#     paraseptal_septal_posterior_rt_value = 0.   # Be careful with this discontinuity in Cobiveco coordinates - Not the same value as septal_posterior_rt_value
#     paraseptal_freewall_posterior_rt_value = freewall_posterior_rt_value
#     paraseptal_septal_anterior_rt_value = septal_anterior_rt_value
#     paraseptal_freewall_anterior_rt_value = freewall_anterior_rt_value
#
#     # # # Transventricular (Cobiveco)
#     # lv_rvlv_binary_value = 0.
#     # rv_rvlv_binary_value = 1.
#
#     ### Define routing points
#     # His-av node at coordinates [1., 0.85] == [base, septal] (symmetric between LV and RV)
#     his_av_ab_rt = np.array([base_ab_value, septal_center_rt_value])  # [basal, septal]
#     # His-apex node at coordinates [0., 0.85] == [apex, septal] (symmetric between LV and RV)
#     his_apex_ab_rt = np.array([apex_ab_value, septal_center_rt_value])  # [apical, septal]
#
#     ## Prepare coordinates
#     node_ab_values = node_vc[vc_ab_cut_name]
#     node_rt_values = node_vc[vc_rt_name]
#     # LV
#     lv_ab_values = node_ab_values[node_lvendo]
#     lv_rt_values = node_rt_values[node_lvendo]
#     lv_node_ab_rt = np.transpose(np.array([lv_ab_values, lv_rt_values], dtype=float))
#     # RV
#     rv_ab_values = node_ab_values[node_rvendo]
#     rv_rt_values = node_rt_values[node_rvendo]
#     rv_node_ab_rt = np.transpose(np.array([rv_ab_values, rv_rt_values], dtype=float))
#     # Clear memory and prevent recycling
#     node_ab_rt = None
#     node_ab_values = None
#     node_rt_values = None
#
#     ## Prepare inputs for Djikstra
#     # Set LV endocardial edges aside
#     lv_neighbour, lvnodes_xyz, lvunfoldedEdges, lvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
#                                                                                  sub_node_index=node_lvendo)
#     # Set RV endocardial edges aside
#     rv_neighbour, rvnodes_xyz, rvunfoldedEdges, rvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
#                                                                                  sub_node_index=node_rvendo)
#     # Clear memory and prevent recycling
#     edge = None
#     node_lvendo = None
#     node_rvendo = None
#     node_xyz = None
#
#     ### Define regions in the heart (masks)
#     ## Apex to Base Masks
#     # Baseline mask within the valid region to grow Purkinje fibres - Global Purkinje growth restriction
#     lv_baseline_purkinje_mask = (lv_ab_values >= apex_ab_value) & (lv_ab_values <= base_ab_value) \
#                              & (lv_ab_values <= purkinje_max_ab_threshold)
#     rv_baseline_purkinje_mask = (rv_ab_values >= apex_ab_value) & (rv_ab_values <= base_ab_value) \
#                                 & (rv_ab_values <= purkinje_max_ab_threshold)
#     # Apical regions of the heart are defined as apex-to-base (AB) < 0.4/0.2 in the LV/RV
#     # print('lv_baseline_purkinje_mask ', np.sum(lv_baseline_purkinje_mask))
#     # print('rv_baseline_purkinje_mask ', np.sum(rv_baseline_purkinje_mask))
#     lv_apical_mask = lv_baseline_purkinje_mask & (lv_ab_values <= lv_apical_ab_threshold)
#     # print('(lv_ab_values <= lv_apical_ab_threshold) ', np.sum((lv_ab_values <= lv_apical_ab_threshold)))
#     # print('(rv_ab_values <= rv_apical_ab_threshold) ', np.sum((rv_ab_values <= rv_apical_ab_threshold)))
#     # RV
#     rv_apical_mask = rv_baseline_purkinje_mask & (rv_ab_values <= rv_apical_ab_threshold)
#     # print('lv_apical_mask ', np.sum(lv_apical_mask))
#     # print('rv_apical_mask ', np.sum(rv_apical_mask))
#
#     ## Rotational Masks
#     # Freewall/Lateral regions of the heart are defined as [freewall_posterior_rt_value < rotation-angle (RT) < freewall_anterior_rt_value] [0.2 < RT < 0.5]
#     lv_freewall_mask = lv_baseline_purkinje_mask & (     # Global Purkinje growth restriction
#                 (freewall_posterior_rt_value <= lv_rt_values) & (lv_rt_values <= freewall_anterior_rt_value))
#     rv_freewall_mask = rv_baseline_purkinje_mask & (     # Global Purkinje growth restriction
#             (freewall_posterior_rt_value <= rv_rt_values) & (rv_rt_values <= freewall_anterior_rt_value))
#     # print('lv_freewall_mask ', np.sum(lv_freewall_mask))
#     # print('rv_freewall_mask ', np.sum(rv_freewall_mask))
#
#     # Septal regions of the heart are defined as [septal_anterior_rt_value < rotation-angle (RT) < septal_posterior_rt_value] [0.7 < RT < 1.]
#     lv_septal_mask = lv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
#                 (septal_anterior_rt_value <= lv_rt_values) & (lv_rt_values <= septal_posterior_rt_value)))
#     rv_septal_mask = rv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
#             (septal_anterior_rt_value <= rv_rt_values) & (rv_rt_values <= septal_posterior_rt_value)))
#
#     # Paraseptal posterior regions of the heart are defined as [paraseptal_septal_posterior_rt_value < rotation-angle (RT) < paraseptal_freewall_posterior_rt_value] [0. < RT < 0.2]
#     lv_paraseptal_posterior_mask = lv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
#             (paraseptal_septal_posterior_rt_value <= lv_rt_values) & (lv_rt_values <= paraseptal_freewall_posterior_rt_value)))
#     rv_paraseptal_posterior_mask = rv_baseline_purkinje_mask & ((     # Global Purkinje growth restriction
#             (paraseptal_septal_posterior_rt_value <= rv_rt_values) & (rv_rt_values <= paraseptal_freewall_posterior_rt_value)))
#
#     # Paraseptal anterior regions of the heart are defined as [paraseptal_freewall_anterior_rt_value < rotation-angle (RT) < paraseptal_septal_anterior_rt_value] [0.5 < RT < 0.7]
#     lv_paraseptal_anterior_mask = lv_baseline_purkinje_mask & ((  # Global Purkinje growth restriction
#             (paraseptal_freewall_anterior_rt_value <= lv_rt_values) & (lv_rt_values <= paraseptal_septal_anterior_rt_value)))
#     rv_paraseptal_anterior_mask = rv_baseline_purkinje_mask & ((  # Global Purkinje growth restriction
#             (paraseptal_freewall_anterior_rt_value <= rv_rt_values) & (rv_rt_values <= paraseptal_septal_anterior_rt_value)))
#
#     #### Generation of the Purkinje tree using physiological rules
#     ### The His-bundel
#     ## Rule 1) The Purkinje tree begins symmetrically at the his-av node in both ventricles
#     ## Rule 2) The hisbundle goes down to the apex through while trying to keep a straight rotational trajectory
#     lv_hisBundle_distances, lv_hisBundle_indexes = generate_Purkinje_fibre_between_two_points(
#         node_coordinates=lv_node_ab_rt, destination_coordinate=his_apex_ab_rt, source_coordinate=his_av_ab_rt,
#         djikstra_nodes_xyz=lvnodes_xyz, djikstra_unfoldedEdges=lvunfoldedEdges, djikstra_edgeVEC=lvedgeVEC,
#         djikstra_neighbours=lv_neighbour, approx_djikstra_max_path_len=approx_djikstra_max_path_len)
#     rv_hisBundle_distances, rv_hisBundle_indexes = generate_Purkinje_fibre_between_two_points(
#         node_coordinates=rv_node_ab_rt, destination_coordinate=his_apex_ab_rt, source_coordinate=his_av_ab_rt,
#         djikstra_nodes_xyz=rvnodes_xyz, djikstra_unfoldedEdges=rvunfoldedEdges, djikstra_edgeVEC=rvedgeVEC,
#         djikstra_neighbours=rv_neighbour, approx_djikstra_max_path_len=approx_djikstra_max_path_len)
#     # rv_hisBase_index = find_node_index(node=rv_node_ab_rt, val=his_av_ab_rt)
#     # rv_hisBase_distance_mat, rv_hisBase_path_mat = djikstra(
#     #     source_id_list=np.asarray([rv_hisBase_index], dtype=int),
#     #     djikstra_nodes_xyz=rvnodes_xyz,
#     #     djikstra_unfoldedEdges=rvunfoldedEdges,
#     #     djikstra_edgeVEC=rvedgeVEC,
#     #     djikstra_neighbours=rv_neighbour,
#     #     approx_max_path_len=approx_djikstra_max_path_len)
#
#     ## Rule 3) The RV rib-cage. The apical and Lateral/Freewall in the RV can connect directly to their closest point in the ab coordinate in the hisbundle that has ab < 0.8
#     # TODO This rule could be updated to incorporate moderator bands, but it requires later the hability to use Purkinje fibres that are not bound to the endocardium
#     rv_hisBundle_ab_values = rv_ab_values[rv_hisBundle_indexes]
#     # rv_hisBundle_meta_indexes = np.nonzero(rv_hisBundle_ab_values < purkinje_max_ab_threshold)[0]  # Purkinje can only connect to and from nodes below the Purkinje growth area
#     rv_hisBundle_meta_indexes = np.nonzero(
#         (rv_hisBundle_ab_values >= apex_ab_value) & (rv_hisBundle_ab_values <= base_ab_value)   # Purkinje can only grow between apex and base
#         & (rv_hisBundle_ab_values <= purkinje_max_ab_threshold)     # Purkinje can only connect to and from nodes below the Purkinje growth area
#     )[0]
#     # print('Check ', np.all(rv_hisBundle_meta_indexes==rv_hisBundle_meta_indexes_2))
#     rv_hisBundle_available_indexes = rv_hisBundle_indexes[rv_hisBundle_meta_indexes]
#     rv_hisBundle_available_ab_values = rv_hisBundle_ab_values[rv_hisBundle_meta_indexes]
#     rv_hisBundle_available_distances = rv_hisBundle_distances[rv_hisBundle_meta_indexes]
#     # Clear memory and prevent recycling
#     rv_hisBundle_ab_values = None
#     rv_hisBundle_meta_indexes = None
#
#     # Calculate Djikstra to all the nodes first
#     rv_hisbundle_distance_mat_rule3, rv_hisbundle_path_mat_rule3 = sorted_djikstra(
#         source_indexes=np.asarray(rv_hisBundle_available_indexes, dtype=int),
#         djikstra_nodes_xyz=rvnodes_xyz,
#         djikstra_unfoldedEdges=rvunfoldedEdges,
#         djikstra_edgeVEC=rvedgeVEC,
#         djikstra_neighbours=rv_neighbour,
#         approx_djikstra_max_path_len=approx_djikstra_max_path_len)
#     # Have a look at the aligned ab coordinate nodes in the his-bundle
#     rv_ab_dist = np.abs(rv_ab_values[:, np.newaxis] - rv_hisBundle_available_ab_values)
#     rv_hisBundle_available_meta_index_connected_rule3 = np.argmin(np.abs(rv_ab_dist), axis=1)  # match endo nodes to the hisbundles as a rib-cage (same ab values)
#     # Clear memory and prevent recycling
#     rv_hisBundle_available_ab_values = None
#
#     ## Rule 3) The RV rib-cage. The apical and Lateral/Freewall in the RV can connect directly to their closest point in the ab coordinate in the hisbundle that has ab < 0.8
#     rule3_mask = (rv_apical_mask | rv_freewall_mask) & np.logical_not(rv_visited)   # Try to make sure that a node does not get assigned two paths using a register of those visited
#     rv_PK_distance_mat_2, rv_PK_path_mat_2, rv_visited_2 = generate_Purkinje_fibres(
#         from_previous_available_distance_mat=rv_hisbundle_distance_mat_rule3,
#         from_previous_available_path_mat=rv_hisbundle_path_mat_rule3,
#         node_connected_previous_available_meta_index=rv_hisBundle_available_meta_index_connected_rule3,
#         node_mask=rule3_mask,
#         previous_available_indexes=rv_hisBundle_available_indexes,
#         previous_complete_fibre_indexes=rv_hisBundle_indexes,
#         to_previous_available_distances=rv_hisBundle_available_distances,
#         pk_distance_mat=rv_PK_distance_mat,
#         pk_path_mat=rv_PK_path_mat,
#         visited=rv_visited)
#
#
#
#     print('rule3_mask ', np.sum(rule3_mask))
#     for rv_endo_node_i in range(rv_hisBundle_available_meta_index_connected_rule3.shape[0]):
#         # Check Rule!
#         if rule3_mask[rv_endo_node_i]:
#             # Process case and mark as visited
#             correspondent_hisBundle_meta_index = rv_hisBundle_available_meta_index_connected_rule3[rv_endo_node_i]
#             correspondent_hisBundle_index = rv_hisBundle_available_indexes[correspondent_hisBundle_meta_index]
#             path_to_hisBundle = rv_hisBundle_indexes[:find_first(correspondent_hisBundle_index,
#                                                                  rv_hisBundle_indexes) + 1]  # Trim the indexes that connect in this way in the his-bundel
#             path_from_hisBundle = rv_hisbundle_path_mat_rule3[rv_endo_node_i, correspondent_hisBundle_meta_index, :]
#             path_from_hisBundle = path_from_hisBundle[
#                 path_from_hisBundle != get_nan_value()]  # For visualisation only - path offset
#             path_to_node = np.concatenate((path_to_hisBundle, path_from_hisBundle), axis=0)
#             # Check if the data structure is large enough
#             if rv_approx_djikstra_max_path_len < path_to_node.shape[0]:
#                 # If not - then increase its size
#                 rv_approx_djikstra_max_path_len_tmp = path_to_node.shape[0] * 2
#                 rv_PK_path_mat_tmp = np.full(
#                     (nb_rvendo_nodes, rv_approx_djikstra_max_path_len_tmp),
#                     get_nan_value(), dtype=np.int32)
#                 rv_PK_path_mat_tmp[:, :rv_approx_djikstra_max_path_len] = rv_PK_path_mat
#                 # Update structures
#                 rv_approx_djikstra_max_path_len = rv_approx_djikstra_max_path_len_tmp
#                 rv_PK_path_mat = rv_PK_path_mat_tmp
#                 # Clear memory to avoid recycling
#                 rv_approx_djikstra_max_path_len_tmp = None
#                 rv_PK_path_mat_tmp = None
#             # Add path
#             rv_PK_path_mat[rv_endo_node_i, :path_to_node.shape[0]] = path_to_node
#             # Clear memory to avoid recycling
#             path_to_node = None
#             # Add distance cost
#             rv_PK_distance_mat[rv_endo_node_i] = rv_hisbundle_distance_mat_rule3[rv_endo_node_i, rv_hisBundle_available_meta_index_connected_rule3[rv_endo_node_i]] \
#                                                  + rv_hisBundle_available_distances[correspondent_hisBundle_meta_index]
#             # print('rv_hisBase_distance_mat[correspondent_hisBundle_index, 0] ', rv_hisBase_distance_mat[correspondent_hisBundle_index, 0])
#             # print('rv_hisBundle_available_distances[correspondent_hisBundle_meta_index] ', rv_hisBundle_available_distances[correspondent_hisBundle_meta_index])
#             # if not (rv_hisBase_distance_mat[correspondent_hisBundle_index, 0] == rv_hisBundle_available_distances[correspondent_hisBundle_meta_index]):
#             #     print('Check ', rv_hisBase_distance_mat[correspondent_hisBundle_index, 0] == rv_hisBundle_available_distances[correspondent_hisBundle_meta_index])
#             # Add node to the list of visited ones
#             rv_visited[rv_endo_node_i] = True
#     # Clear memory and prevent recycling
#     rule3_mask = None
#     rv_hisBundle_available_meta_index_connected_rule3 = None
#     rv_hisBundle_available_indexes = None
#     rv_hisBundle_available_distances = None
#
#     print('Check ', np.all(rv_PK_distance_mat == rv_PK_distance_mat_2), np.all(rv_PK_path_mat == rv_PK_path_mat_2),
#           np.all(rv_visited == rv_visited_2))
#     raise ()
#
#     # print('done')
#     # raise()
#     ## Rule 4) The LV's apical hisbundle can directly connects to Septal and Apical nodes using the shortest distance between hisBundle and node
#     lv_hisBundle_ab_values = lv_ab_values[lv_hisBundle_indexes]
#     # TODO rename using a descriptive name about the thresholds (everywhere)
#     lv_hisBundle_apical_meta_indexes = np.nonzero(
#         (lv_hisBundle_ab_values >= apex_ab_value) & (lv_hisBundle_ab_values <= base_ab_value)   # Purkinje can only grow between apex and base
#         & (lv_hisBundle_ab_values <= purkinje_max_ab_threshold)     # Purkinje can only connect to and from nodes below the Purkinje growth area
#         & (lv_hisBundle_ab_values <= lv_apical_ab_threshold)        # Specific SOURCE RULE
#     )[0]
#     lv_hisBundle_apical_available_indexes = lv_hisBundle_indexes[lv_hisBundle_apical_meta_indexes]
#     # lv_hisBundle_apical_available_ab_values = lv_hisBundle_ab_values[lv_hisBundle_apical_meta_indexes]
#     lv_hisBundle_apical_available_distances = lv_hisBundle_distances[lv_hisBundle_apical_meta_indexes]
#     # Clear memory and prevent recycling
#     lv_hisBundle_ab_values = None
#
#     # Calculate Djikstra to all the nodes first
#     ### Rule 5) Root nodes in the Apical and Septal regions of the heart connect to their closest Apical hisbundle node
#     # LV
#     lv_hisbundle_apical_distance_mat_rule5, lv_hisbundle_apical_path_mat_rule5 = sorted_djikstra(
#         source_indexes=np.asarray(lv_hisBundle_apical_available_indexes, dtype=int),
#         djikstra_nodes_xyz=lvnodes_xyz,
#         djikstra_unfoldedEdges=lvunfoldedEdges,
#         djikstra_edgeVEC=lvedgeVEC,
#         djikstra_neighbours=lv_neighbour,
#         approx_djikstra_max_path_len=approx_djikstra_max_path_len)
#     lv_hisBundle_apical_available_meta_index_connected_rule5 = np.argmin(lv_hisbundle_apical_distance_mat_rule5, axis=1)     # match endo nodes to their closest hisBundle node
#
#
#     # ## Rule 4) The apical hisbundle can directly connects to Septal and Apical (and Paraseptal for the RV) root nodes after it crosses the Apex-to-Base 0.4/0.2 threshold LV/RV
#     # # LV
#     # lv_his_middle_meta_index = find_node_index(node=lv_ab_values[lv_hisBundle_indexes, np.newaxis],
#     #                                            val=lv_apical_ab_threshold)
#     # lv_his_middle_index = lv_hisBundle_indexes[lv_his_middle_meta_index]
#     # lv_hisConnected_indexes = lv_hisBundle_indexes[find_first(lv_his_middle_index,
#     #                                                           lv_hisBundle_indexes):]  # Trim the indexes that connect in this way in the his-bundel
#     # Clear memory and avoid recycling
#     # lv_his_middle_meta_index = None
#     # lv_his_middle_index = None
#     # # RV
#     # rv_his_middle_meta_index = find_node_index(node=rv_ab_values[rv_hisBundle_indexes, np.newaxis],
#     #                                            val=rv_apical_ab_threshold)
#     # rv_his_middle_index = rv_hisBundle_indexes[rv_his_middle_meta_index]
#     # rv_hisConnected_indexes = rv_hisBundle_indexes[find_first(rv_his_middle_index,
#     #                                                           rv_hisBundle_indexes):]  # Trim the indexes that connect in this way in the his-bundel
#     # Clear memory and avoid recycling
#     # rv_his_middle_meta_index = None
#     # rv_his_middle_index = None
#
#     ### Rule 5) Root nodes in the Apical regions of the heart connect to their closest Apical hisbundle node
#     # LV
#     # lv_hisbundle_apical_distance_mat_rule5, lv_hisbundle_apical_path_mat_rule5 = djikstra(
#     #     np.asarray(lv_hisConnected_indexes, dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
#     #     lv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#     # lv_hisbundle_apical_path_mat_rule5 = sort_djikstra_by_distance(
#     #     source_to_all_distance_mat=lv_hisbundle_apical_distance_mat_rule5,
#     #     source_to_all_path_mat=lv_hisbundle_apical_path_mat_rule5)
#     # lv_hisBundle_apical_available_meta_index_connected_rule5 = np.argmin(lv_hisbundle_apical_distance_mat_rule5, axis=1)
#     # RV
#     # rv_hisbundle_distance_mat_rule5, rv_hisbundle_path_mat_rule5 = djikstra(
#     #     np.asarray(rv_hisConnected_indexes, dtype=int), rvnodes_xyz, rvunfoldedEdges, rvedgeVEC,
#     #     rv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#     # rv_hisbundle_path_mat_rule5 = sort_djikstra_by_distance(
#     #     source_to_all_distance_mat=rv_hisbundle_distance_mat_rule5,
#     #     source_to_all_path_mat=rv_hisbundle_path_mat_rule5)
#     # rv_hisbundle_connections_rule5 = np.argmin(rv_hisbundle_distance_mat_rule5, axis=1)
#
#     # Initialise temporary data structures
#
#
#
#
#     # lv_PK_distance_mat_2, lv_PK_path_mat_2, lv_visited_2 = generate_Purkinje_fibres(
#     #     from_previous_available_distance_mat, from_previous_available_path_mat,
#     #     node_connected_previous_available_meta_index, node_mask, previous_available_indexes,
#     #     previous_complete_fibre_indexes, to_previous_available_distances, pk_distance_mat, pk_path_mat, visited)
#
#     apical_or_septal_mask = (lv_apical_mask | lv_septal_mask) & np.logical_not(lv_visited)     # Try to make sure that a node does not get assigned two paths using a register of those visited
#     print('rule5_mask ', np.sum(apical_or_septal_mask))
#     for lv_endo_node_i in range(lv_hisBundle_apical_available_meta_index_connected_rule5.shape[0]):
#         # Check Rule!
#         if apical_or_septal_mask[lv_endo_node_i]:
#             # Process case and mark as visited
#             correspondent_hisBundle_meta_index = lv_hisBundle_apical_available_meta_index_connected_rule5[lv_endo_node_i]
#             correspondent_hisBundle_index = lv_hisBundle_apical_available_indexes[correspondent_hisBundle_meta_index]
#             path_to_hisBundle = lv_hisBundle_indexes[:find_first(correspondent_hisBundle_index,
#                                                                  lv_hisBundle_indexes) + 1]  # Trim the indexes that connect in this way in the his-bundel
#             path_from_hisBundle = lv_hisbundle_apical_path_mat_rule5[lv_endo_node_i, correspondent_hisBundle_meta_index, :]
#             path_from_hisBundle = path_from_hisBundle[
#                 path_from_hisBundle != get_nan_value()]  # For visualisation only - path offset
#             path_to_node = np.concatenate((path_to_hisBundle, path_from_hisBundle), axis=0)
#             # Check if the data structure is large enough
#             if lv_approx_djikstra_max_path_len < path_to_node.shape[0]:
#                 # If not - then increase its size
#                 lv_approx_djikstra_max_path_len_tmp = path_to_node.shape[0] * 2
#                 lv_PK_path_mat_tmp = np.full(
#                     (nb_lvendo_nodes, lv_approx_djikstra_max_path_len_tmp),
#                     get_nan_value(), dtype=np.int32)
#                 lv_PK_path_mat_tmp[:, :rv_approx_djikstra_max_path_len] = lv_PK_path_mat
#                 # Update structures
#                 lv_approx_djikstra_max_path_len = lv_approx_djikstra_max_path_len_tmp
#                 lv_PK_path_mat = lv_PK_path_mat_tmp
#                 # Clear memory to avoid recycling
#                 lv_approx_djikstra_max_path_len_tmp = None
#                 rv_PK_path_mat_tmp = None
#             # Add path
#             lv_PK_path_mat[lv_endo_node_i, :path_to_node.shape[0]] = path_to_node
#             # Clear memory to avoid recycling
#             path_to_node = None
#             # Add distance cost
#             lv_PK_distance_mat[lv_endo_node_i] = lv_hisbundle_apical_distance_mat_rule5[
#                                                      lv_endo_node_i, lv_hisBundle_apical_available_meta_index_connected_rule5[
#                                                          lv_endo_node_i]] \
#                                                  + lv_hisBundle_apical_available_distances[correspondent_hisBundle_meta_index]
#             # print('rv_hisBase_distance_mat[correspondent_hisBundle_index, 0] ', rv_hisBase_distance_mat[correspondent_hisBundle_index, 0])
#             # print('rv_hisBundle_available_distances[correspondent_hisBundle_meta_index] ', rv_hisBundle_available_distances[correspondent_hisBundle_meta_index])
#             # if not (rv_hisBase_distance_mat[correspondent_hisBundle_index, 0] == rv_hisBundle_available_distances[correspondent_hisBundle_meta_index]):
#             #     print('Check ', rv_hisBase_distance_mat[correspondent_hisBundle_index, 0] == rv_hisBundle_available_distances[correspondent_hisBundle_meta_index])
#             # Add node to the list of visited ones
#             lv_visited[lv_endo_node_i] = True
#     # Clear memory and prevent recycling
#     rule3_mask = None
#     rv_hisBundle_available_meta_index_connected_rule3 = None
#     rv_hisBundle_available_indexes = None
#     rv_hisBundle_available_distances = None
#
#     print('Check ', np.all(lv_PK_distance_mat==lv_PK_distance_mat_2), np.all(lv_PK_path_mat==lv_PK_path_mat_2), np.all(lv_visited==lv_visited_2))
#     raise()
#
#     lv_hisbundle_path_mat_rule5_aux = np.full((nb_lvendo_nodes, approx_djikstra_max_path_len),
#                                               get_nan_value(), dtype=np.int32)
#     lv_hisbundle_distance_mat_rule5_aux = np.full((nb_lvendo_nodes), get_nan_value(),
#                                                   dtype=np.float64)
#
#
#
#     # TODO JUST UPDATE THE SIZE OF THE STRUCTURES AND REMOVE THE CHECKS AND IF ELSE STATEMENTS
#     for lv_endo_node_i in range(lv_hisBundle_apical_available_meta_index_connected_rule5.shape[0]):
#         offset = lv_hisBase_path_mat[lv_hisConnected_indexes[lv_hisBundle_apical_available_meta_index_connected_rule5[lv_endo_node_i]], 0, :]
#         offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#         path = lv_hisbundle_apical_path_mat_rule5[lv_endo_node_i, lv_hisBundle_apical_available_meta_index_connected_rule5[lv_endo_node_i], :]
#         path = path[path != get_nan_value()]  # For visualisation only - path offset
#         path = np.concatenate((offset, path), axis=0)
#         if lv_hisbundle_path_mat_rule5_aux.shape[1] < path.shape[0]:
#             approx_djikstra_max_path_len_is_too_short = True
#             break
#         lv_hisbundle_path_mat_rule5_aux[lv_endo_node_i, :path.shape[0]] = path
#         lv_hisbundle_distance_mat_rule5_aux[lv_endo_node_i] = lv_hisbundle_apical_distance_mat_rule5[lv_endo_node_i,
#         lv_hisBundle_apical_available_meta_index_connected_rule5[lv_endo_node_i]] + lv_hisBase_distance_mat[
#                                                lv_hisConnected_indexes[lv_hisBundle_apical_available_meta_index_connected_rule5[lv_endo_node_i]], 0]
#     # TODO more or less checked up to here! Needs a bit more descriptions of what is going on in each part of the code
#     if not approx_djikstra_max_path_len_is_too_short:
#         lv_hisbundle_apical_path_mat_rule5 = lv_hisbundle_path_mat_rule5_aux
#         lv_hisbundle_apical_distance_mat_rule5 = lv_hisbundle_distance_mat_rule5_aux
#         # Clear Arguments to avoid recycling
#         lv_hisbundle_path_mat_rule5_aux = None
#         lv_hisbundle_distance_mat_rule5_aux = None
#
#         rv_hisbundle_path_mat_rule3_aux = np.full((rv_hisbundle_path_mat_rule5.shape[0], approx_djikstra_max_path_len),
#                                                   get_nan_value(),
#                                                   dtype=np.int32)
#         rv_hisbundle_distance_mat_rule3_aux = np.full((rv_hisbundle_distance_mat_rule5.shape[0]), get_nan_value(),
#                                                       dtype=np.float64)
#         for rv_endo_node_i in range(rv_hisbundle_connections_rule5.shape[0]):
#             offset = rv_hisBase_path_mat[rv_hisConnected_indexes[rv_hisbundle_connections_rule5[rv_endo_node_i]], 0, :]
#             offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#             path = rv_hisbundle_path_mat_rule5[rv_endo_node_i, rv_hisbundle_connections_rule5[rv_endo_node_i], :]
#             path = path[path != get_nan_value()]  # For visualisation only - path offset
#             path = np.concatenate((offset, path), axis=0)
#             if rv_hisbundle_path_mat_rule3_aux.shape[1] < path.shape[0]:
#                 approx_djikstra_max_path_len_is_too_short = True
#                 break
#             rv_hisbundle_path_mat_rule5_aux[rv_endo_node_i, :path.shape[0]] = path
#             rv_hisbundle_distance_mat_rule3_aux[rv_endo_node_i] = rv_hisbundle_distance_mat_rule3[rv_endo_node_i, rv_hisbundle_connections_rule5[rv_endo_node_i]] + \
#                                                                   rv_hisBase_distance_mat[
#                                                    rv_hisConnected_indexes[rv_hisbundle_connections_rule5[rv_endo_node_i]], 0]
#         if not approx_djikstra_max_path_len_is_too_short:
#             rv_hisbundle_path_mat_rule3 = rv_hisbundle_path_mat_rule3_aux
#             rv_hisbundle_distance_mat_rule3 = rv_hisbundle_distance_mat_rule3_aux
#             # Clear Arguments to avoid recycling
#             rv_hisbundle_path_mat_rule3_aux = None
#             rv_hisbundle_distance_mat_rule3_aux = None
#
#             # Rule 5) Apical|Septal|Paraseptal regions of the heart are defined as AB < 0.4/0.2 in the LV/RV | [0.7 < RT < 1.] | [0. < RT < 0.2] & [0.5 < RT < 0.7], respectively
#             lv_apical_septal_mask = lv_baseline_purkinje_mask & ((node_vc[vc_ab_cut_name][node_lvendo] <= lv_apical_ab_threshold) | (
#                     (0.7 <= node_vc[vc_rt_name][node_lvendo]) & (
#                     node_vc[vc_rt_name][node_lvendo] <= 1.))) & np.logical_not(
#                 lv_visited)
#             rv_apical_septal_paraseptal_mask = rv_baseline_purkinje_mask & (((node_vc[vc_ab_cut_name][node_rvendo] <= rv_apical_ab_threshold) | (
#                     (0.7 <= node_vc[vc_rt_name][node_rvendo]) & (
#                     node_vc[vc_rt_name][node_rvendo] <= 1.))) |
#                                                 (((0.0 <= node_vc[vc_rt_name][node_rvendo]) & (
#                                                         node_vc[vc_rt_name][node_rvendo] <= 0.2)) | (
#                                                          (0.5 <= node_vc[vc_rt_name][node_rvendo]) & (
#                                                          node_vc[vc_rt_name][
#                                                              node_rvendo] <= 0.7)))) & np.logical_not(
#                 rv_visited)
#             lv_PK_distance_mat[lv_apical_septal_mask] = lv_hisbundle_apical_distance_mat_rule5[lv_apical_septal_mask]
#             lv_PK_path_mat[lv_apical_septal_mask, :] = lv_hisbundle_path_mat[lv_apical_septal_mask, :]
#             lv_visited[lv_apical_septal_mask] = True
#             rv_PK_distance_mat[rv_apical_septal_paraseptal_mask] = rv_hisbundle_distance_mat_rule3[
#                 rv_apical_septal_paraseptal_mask]
#             rv_PK_path_mat[rv_apical_septal_paraseptal_mask, :] = rv_hisbundle_path_mat_rule3[
#                                                                   rv_apical_septal_paraseptal_mask, :]
#             rv_visited[rv_apical_septal_paraseptal_mask] = True
#
#             # Rule 6) Paraseptal regions of the heart are connected from apex to base through either [0.4/0.2, 0.1, 1., :] or  [0.4/0.2, 0.6, 1., :] LV/RV
#             lv_ant_paraseptalApex_index = int(np.argmin(
#                 np.linalg.norm(node_ab_rt[node_lvendo, :] - np.array([lv_apical_ab_threshold, paraseptal_anterior_center_rt_value]), ord=2,
#                                axis=1)))  # [mid, paraseptal, lv]
#             lv_post_paraseptalApex_index = int(np.argmin(
#                 np.linalg.norm(node_ab_rt[node_lvendo, :] - np.array([lv_apical_ab_threshold, paraseptal_posterior_center_rt_value]), ord=2,
#                                axis=1)))  # [mid, paraseptal, lv]
#             if not lv_visited[lv_ant_paraseptalApex_index]:
#                 lv_PK_distance_mat[lv_ant_paraseptalApex_index] = lv_hisbundle_apical_distance_mat_rule5[
#                     lv_ant_paraseptalApex_index]
#                 lv_PK_path_mat[lv_ant_paraseptalApex_index, :] = lv_hisbundle_path_mat[
#                                                                  lv_ant_paraseptalApex_index,
#                                                                  :]
#                 lv_visited[lv_ant_paraseptalApex_index] = True
#             if not lv_visited[lv_post_paraseptalApex_index]:
#                 lv_PK_distance_mat[lv_post_paraseptalApex_index] = lv_hisbundle_apical_distance_mat_rule5[
#                     lv_post_paraseptalApex_index]
#                 lv_PK_path_mat[lv_post_paraseptalApex_index, :] = lv_hisbundle_path_mat[
#                                                                   lv_post_paraseptalApex_index, :]
#                 lv_visited[lv_post_paraseptalApex_index] = True
#             lv_paraseptalApex_offsets = np.array(
#                 [lv_PK_distance_mat[lv_ant_paraseptalApex_index],
#                  lv_PK_distance_mat[lv_post_paraseptalApex_index]],
#                 dtype=float)
#             lv_paraseptal_distance_mat, lv_paraseptal_path_mat = djikstra(
#                 np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int),
#                 lvnodes_xyz,
#                 lvunfoldedEdges, lvedgeVEC, lv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#             lv_paraseptal_connections = np.argmin(lv_paraseptal_distance_mat, axis=1)
#             lv_paraseptal_path_mat_aux = np.full(
#                 (lv_paraseptal_path_mat.shape[0], approx_djikstra_max_path_len),
#                 get_nan_value(),
#                 dtype=np.int32)
#             lv_paraseptal_distance_mat_aux = np.full((lv_paraseptal_distance_mat.shape[0]), get_nan_value(),
#                                                      dtype=np.float64)
#             for rv_endo_node_i in range(lv_paraseptal_connections.shape[0]):
#                 offset = lv_PK_path_mat[
#                          np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int)[
#                              lv_paraseptal_connections[rv_endo_node_i]], :]
#                 offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#                 path = lv_paraseptal_path_mat[rv_endo_node_i, lv_paraseptal_connections[rv_endo_node_i], :]
#                 path = path[path != get_nan_value()]  # For visualisation only - path offset
#                 path = np.concatenate((offset, path), axis=0)
#                 if lv_paraseptal_path_mat_aux.shape[1] < path.shape[0]:
#                     approx_djikstra_max_path_len_is_too_short = True
#                     break
#                 lv_paraseptal_path_mat_aux[rv_endo_node_i, :path.shape[0]] = path
#                 lv_paraseptal_distance_mat_aux[rv_endo_node_i] = lv_paraseptal_distance_mat[
#                                                         rv_endo_node_i, lv_paraseptal_connections[rv_endo_node_i]] + \
#                                                                  lv_paraseptalApex_offsets[lv_paraseptal_connections[rv_endo_node_i]]
#             if not approx_djikstra_max_path_len_is_too_short:
#                 lv_paraseptal_path_mat = lv_paraseptal_path_mat_aux
#                 lv_paraseptal_distance_mat = lv_paraseptal_distance_mat_aux
#                 # Clear Arguments to avoid recycling
#
#                 # Rule 7) Paraseptal regions of the heart are defined as [0. < rotation-angle (RT) < 0.2] & [0.5 < RT < 0.7], these are connected to their closest paraseptal routing point (anterior or posterior)
#                 lv_paraseptal_mask = lv_baseline_purkinje_mask & (((0.0 <= node_vc[vc_rt_name][node_lvendo]) & (
#                         node_vc[vc_rt_name][node_lvendo] <= 0.2)) | (
#                                               (0.5 <= node_vc[vc_rt_name][node_lvendo]) & (
#                                               node_vc[vc_rt_name][node_lvendo] <= 0.7))) & np.logical_not(
#                     lv_visited)
#                 lv_PK_distance_mat[lv_paraseptal_mask] = lv_paraseptal_distance_mat[lv_paraseptal_mask]
#                 lv_PK_path_mat[lv_paraseptal_mask, :] = lv_paraseptal_path_mat[lv_paraseptal_mask, :]
#                 lv_visited[lv_paraseptal_mask] = True
#                 # Rule 8) Freewall regions of the heart are connected from apex to base through [0.4, 0.35, 1., :] in the LV
#                 lv_freewallApex_index = int(np.argmin(
#                     np.linalg.norm(node_ab_rt_rvlv_binary[node_lvendo, :] - np.array([lv_apical_ab_threshold, 0.35, lv_rvlv_binary_value]), ord=2,
#                                    axis=1)))  # [mid, freewall, endo, lv]
#                 if not lv_visited[lv_freewallApex_index]:
#                     lv_PK_distance_mat[lv_freewallApex_index] = lv_hisbundle_apical_distance_mat_rule5[
#                         lv_freewallApex_index]
#                     lv_PK_path_mat[lv_freewallApex_index, :] = lv_hisbundle_path_mat[lv_freewallApex_index,
#                                                                :]
#                     lv_visited[lv_freewallApex_index] = True
#                 lv_freewallApex_offset = lv_PK_distance_mat[lv_freewallApex_index]
#                 lv_freewallApex_path_offset = lv_PK_path_mat[lv_freewallApex_index, :]
#                 lv_freewallApex_path_offset = lv_freewallApex_path_offset[
#                     lv_freewallApex_path_offset != get_nan_value()]
#                 lv_freewall_distance_mat, lv_freewall_path_mat = djikstra(
#                     np.asarray([lv_freewallApex_index], dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
#                     lv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#                 lv_freewall_path_mat_aux = np.full(
#                     (lv_freewall_path_mat.shape[0], approx_djikstra_max_path_len),
#                     get_nan_value(),
#                     dtype=np.int32)
#                 lv_freewall_distance_mat_aux = np.full((lv_freewall_distance_mat.shape[0]), get_nan_value(),
#                                                        dtype=np.float64)
#                 for rv_endo_node_i in range(lv_freewall_distance_mat.shape[0]):
#                     path = lv_freewall_path_mat[rv_endo_node_i, 0, :]
#                     path = path[path != get_nan_value()]  # For visualisation only - path offset
#                     path = np.concatenate((lv_freewallApex_path_offset, path), axis=0)
#                     if lv_freewall_path_mat_aux.shape[1] < path.shape[0]:
#                         approx_djikstra_max_path_len_is_too_short = True
#                         break
#                     lv_freewall_path_mat_aux[rv_endo_node_i, :path.shape[0]] = path
#                     lv_freewall_distance_mat_aux[rv_endo_node_i] = lv_freewall_distance_mat[
#                                                           rv_endo_node_i, 0] + lv_freewallApex_offset
#                 if not approx_djikstra_max_path_len_is_too_short:
#                     lv_freewall_path_mat = lv_freewall_path_mat_aux
#                     lv_freewall_distance_mat = lv_freewall_distance_mat_aux
#
#                     # Rule 10) Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
#                     lv_freewall_mask = lv_baseline_purkinje_mask & ((0.2 <= node_vc[vc_rt_name][node_lvendo]) & (
#                             node_vc[vc_rt_name][node_lvendo] <= 0.5)) & np.logical_not(lv_visited)
#                     lv_PK_distance_mat[lv_freewall_mask] = lv_freewall_distance_mat[lv_freewall_mask]
#                     lv_PK_path_mat[lv_freewall_mask, :] = lv_freewall_path_mat[lv_freewall_mask, :]
#                     lv_visited[lv_freewall_mask] = True
#
#     return lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, approx_djikstra_max_path_len_is_too_short



# def generate_djikstra_purkinje_tree_from_vc_backup(approx_djikstra_max_path_len, edge, node_lvendo,
#                                               node_rvendo, node_xyz, node_vc, vc_ab_cut_name, vc_rt_name, vc_rvlv_binary_name):
#     repeat_process = False
#     # Define regions in the heart
#     # Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
#     lv_freewall_mask = ((0.2 <= node_vc[vc_rt_name][node_lvendo]) & (node_vc[vc_rt_name][node_lvendo] <= 0.5))
#     # Rule 1) his-av node at coordinates [1., 0.85, 1., :] == [basal, septal, endo, :]
#     lv_his_coord = np.array([1., 0.85, 0.]) # [basal, septal, lv]
#     rv_his_coord = np.array([1., 0.85, 1.]) # [basal, septal, rv]
#     # Rule 2) hisbundle goes down to most apical endocardial point while trying to keep a straight rotation trajectory [0., 0.85, 1., :] == [apical, septal, endo, :]
#     lv_hisApex_coord = np.array([0., 0.85, 0.])  # [apical, septal, lv]
#     rv_hisApex_coord = np.array([0., 0.85, 1.])  # [apical, septal, rv]
#     # TODO This function CRASHES when approx_djikstra_max_path_len is too small!!! There should be an adaptive something for this!
#     # TODO for example, if the approx_djikstra_max_path_len is too small, it should call the function again with twice the value!!
#
#     # No need to use transmural coordinate, because the function is using the surfaces directry read from the geometry creation
#     # TODO there is a huge amount of repeated code in this function! Split into smaller functions!
#     node_ab_rt_rvlv = np.transpose(np.array([node_vc[vc_ab_cut_name], node_vc[vc_rt_name], node_vc[vc_rvlv_binary_name]], dtype=float))
#     # Prepare for Djikstra - Set LV endocardial edges aside
#     lv_neighbour, lvnodes_xyz, lvunfoldedEdges, lvedgeVEC  = prepare_for_djikstra(edge=edge, node_xyz=node_xyz, sub_node_index=node_lvendo)
#     # Set RV endocardial edges aside
#     rv_neighbour, rvnodes_xyz, rvunfoldedEdges, rvedgeVEC = prepare_for_djikstra(edge=edge, node_xyz=node_xyz,
#                                                                                  sub_node_index=node_rvendo)
#     # Define Purkinje tree using Cobiveco-based rules - Initialise data structures
#     lv_PK_distance_mat = np.full(node_lvendo.shape, get_nan_value(), np.float64)
#     lv_PK_path_mat = np.full((node_lvendo.shape[0], approx_djikstra_max_path_len), get_nan_value(), dtype=np.int32)
#     rv_PK_distance_mat = np.full(node_rvendo.shape, get_nan_value(), np.float64)
#     rv_PK_path_mat = np.full((node_rvendo.shape[0], approx_djikstra_max_path_len), get_nan_value(), dtype=np.int32)
#     lv_visited = np.zeros(node_lvendo.shape, dtype=bool)
#     rv_visited = np.zeros(node_rvendo.shape, dtype=bool)
#     # Rule 1) his-av node at coordinates [1., 0.85, 1., :] == [basal, septal, endo, :]
#     lv_hisBase_index = find_node_index(node=node_ab_rt_rvlv[node_lvendo, :], val=lv_his_coord)
#     rv_hisBase_index = find_node_index(node=node_ab_rt_rvlv[node_rvendo, :], val=rv_his_coord)
#     lv_hisBase_distance_mat, lv_hisBase_path_mat = djikstra(
#         source_id_list=np.asarray([lv_hisBase_index], dtype=int),
#         djikstra_nodes_xyz=lvnodes_xyz,
#         djikstra_unfoldedEdges=lvunfoldedEdges,
#         djikstra_edgeVEC=lvedgeVEC,
#         djikstra_neighbours=lv_neighbour,
#         approx_max_path_len=approx_djikstra_max_path_len)
#     rv_hisBase_distance_mat, rv_hisBase_path_mat = djikstra(
#         source_id_list=np.asarray([rv_hisBase_index], dtype=int),
#         djikstra_nodes_xyz=rvnodes_xyz,
#         djikstra_unfoldedEdges=rvunfoldedEdges,
#         djikstra_edgeVEC=rvedgeVEC,
#         djikstra_neighbours=rv_neighbour,
#         approx_max_path_len=approx_djikstra_max_path_len)
#     # Rule 2) hisbundle goes down to most apical endocardial point while trying to keep a straight rotation trajectory [0., 0.85, 1., :] == [apical, septal, endo, :]
#     lv_hisApex_index = find_node_index(node=node_ab_rt_rvlv[node_lvendo, :], val=lv_hisApex_coord)
#     rv_hisApex_index = find_node_index(node=node_ab_rt_rvlv[node_rvendo, :], val=rv_hisApex_coord)
#     lv_hisBundle_indexes = lv_hisBase_path_mat[lv_hisApex_index, 0, :]  # The nodes in this path are the LV his bundle
#     lv_hisBundle_indexes = lv_hisBundle_indexes[lv_hisBundle_indexes != get_nan_value()]
#     sorted_indexes = np.argsort(
#         lv_hisBase_distance_mat[lv_hisBundle_indexes, 0])  # Sort nodes by distance to the reference
#     lv_hisBundle_indexes = lv_hisBundle_indexes[sorted_indexes]  # Sort nodes by distance to the reference
#     rv_hisBundle_indexes = rv_hisBase_path_mat[rv_hisApex_index, 0,
#                            :]  # The nodes in this path are the LV his bundle
#     rv_hisBundle_indexes = rv_hisBundle_indexes[rv_hisBundle_indexes != get_nan_value()]
#     sorted_indexes = np.argsort(
#         rv_hisBase_distance_mat[rv_hisBundle_indexes, 0])  # Sort nodes by distance to the reference
#     rv_hisBundle_indexes = rv_hisBundle_indexes[sorted_indexes]  # Sort nodes by distance to the reference
#     # lv_hisBundle_offsets = lv_hisBase_distance_mat[lv_hisBundle_indexes, 0]
#     # rv_hisBundle_offsets = rv_hisBase_distance_mat[rv_hisBundle_indexes, 0]
#     # Rule 3) The apical and Lateral/Freewall in the RV can connect directly to their closest point in the hisbundle that has ab < 0.8
#     rv_hisBundle_ab_values = node_vc[vc_ab_cut_name][node_rvendo[rv_hisBundle_indexes]]
#     rv_hisBundle_meta_indexes = np.nonzero(rv_hisBundle_ab_values < 0.8)[0]
#     rv_ab_values = node_vc[vc_ab_cut_name][node_rvendo]
#     rv_ab_dist = np.abs(rv_ab_values[:, np.newaxis] - rv_hisBundle_ab_values[rv_hisBundle_meta_indexes])
#     rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(
#         np.asarray(rv_hisBundle_indexes[rv_hisBundle_meta_indexes], dtype=int), rvnodes_xyz,
#         rvunfoldedEdges, rvedgeVEC, rv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#     rv_hisbundle_connections = np.argmin(np.abs(rv_ab_dist),
#                                          axis=1)  # match root nodes to the hisbundles as a rib-cage (same ab values) #np.argmin(rv_hisbundle_distance_mat, axis=1)
#     rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
#                                         get_nan_value(),
#                                         dtype=np.int32)
#     rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), get_nan_value(),
#                                             dtype=np.float64)
#     for i in range(rv_hisbundle_connections.shape[0]):
#         offset = rv_hisBase_path_mat[
#                  rv_hisBundle_indexes[rv_hisBundle_meta_indexes[rv_hisbundle_connections[i]]], 0, :]
#         offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#         path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
#         path = path[path != get_nan_value()]  # For visualisation only - path offset
#         path = np.concatenate((offset, path), axis=0)
#         if rv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
#             repeat_process = True
#             break
#         rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
#         rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + \
#                                            rv_hisBase_distance_mat[
#                                                rv_hisBundle_indexes[
#                                                    rv_hisBundle_meta_indexes[
#                                                        rv_hisbundle_connections[i]]], 0]
#     if not repeat_process:
#         rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
#         rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
#         # Clear Arguments to avoid recycling
#         rv_hisbundle_path_mat_aux = None
#         rv_hisbundle_distance_mat_aux = None
#
#         rv_apical_lateral_mask = ((node_vc[vc_ab_cut_name][node_rvendo] <= 0.2) | (
#                 (0.2 <= node_vc[vc_rt_name][node_rvendo]) & (
#                 node_vc[vc_rt_name][node_rvendo] <= 0.5))) & np.logical_not(
#             rv_visited)
#         rv_PK_distance_mat[rv_apical_lateral_mask] = rv_hisbundle_distance_mat[rv_apical_lateral_mask]
#         rv_PK_path_mat[rv_apical_lateral_mask, :] = rv_hisbundle_path_mat[rv_apical_lateral_mask, :]
#         rv_visited[rv_apical_lateral_mask] = True
#
#         # Rule 4) The apical hisbundle can directly connects to Septal and Apical (and Paraseptal for the RV) root nodes after it crosses the Apex-to-Base 0.4/0.2 threshold LV/RV
#         lv_hisMiddle_index = lv_hisBundle_indexes[int(np.argmin(
#             np.abs(node_vc[vc_ab_cut_name][
#                        node_lvendo[lv_hisBundle_indexes]] - 0.4)))]  # [apical, septal, lv]
#         rv_hisMiddle_index = rv_hisBundle_indexes[int(np.argmin(
#             np.abs(node_vc[vc_ab_cut_name][
#                        node_rvendo[rv_hisBundle_indexes]] - 0.2)))]  # [apical, septal, rv]
#
#         lv_hisConnected_indexes = lv_hisBundle_indexes[find_first(lv_hisMiddle_index, lv_hisBundle_indexes):]
#         rv_hisConnected_indexes = rv_hisBundle_indexes[find_first(rv_hisMiddle_index, rv_hisBundle_indexes):]
#
#         # Rule 5) Root nodes in the Apical regions of the heart connect to their closest Apical hisbundle node
#         lv_hisbundle_distance_mat, lv_hisbundle_path_mat = djikstra(
#             np.asarray(lv_hisConnected_indexes, dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
#             lv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#         rv_hisbundle_distance_mat, rv_hisbundle_path_mat = djikstra(
#             np.asarray(rv_hisConnected_indexes, dtype=int), rvnodes_xyz, rvunfoldedEdges, rvedgeVEC,
#             rv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#         lv_hisbundle_connections = np.argmin(lv_hisbundle_distance_mat, axis=1)
#         rv_hisbundle_connections = np.argmin(rv_hisbundle_distance_mat, axis=1)
#         lv_hisbundle_path_mat_aux = np.full((lv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
#                                             get_nan_value(),
#                                             dtype=np.int32)
#         lv_hisbundle_distance_mat_aux = np.full((lv_hisbundle_distance_mat.shape[0]), get_nan_value(),
#                                                 dtype=np.float64)
#         for i in range(lv_hisbundle_connections.shape[0]):
#             offset = lv_hisBase_path_mat[lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0, :]
#             offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#             path = lv_hisbundle_path_mat[i, lv_hisbundle_connections[i], :]
#             path = path[path != get_nan_value()]  # For visualisation only - path offset
#             path = np.concatenate((offset, path), axis=0)
#             if lv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
#                 repeat_process = True
#                 break
#             lv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
#             lv_hisbundle_distance_mat_aux[i] = lv_hisbundle_distance_mat[i, lv_hisbundle_connections[i]] + \
#                                                lv_hisBase_distance_mat[
#                                                    lv_hisConnected_indexes[lv_hisbundle_connections[i]], 0]
#         if not repeat_process:
#             lv_hisbundle_path_mat = lv_hisbundle_path_mat_aux
#             lv_hisbundle_distance_mat = lv_hisbundle_distance_mat_aux
#             # Clear Arguments to avoid recycling
#             lv_hisbundle_path_mat_aux = None
#             lv_hisbundle_distance_mat_aux = None
#
#             rv_hisbundle_path_mat_aux = np.full((rv_hisbundle_path_mat.shape[0], approx_djikstra_max_path_len),
#                                                 get_nan_value(),
#                                                 dtype=np.int32)
#             rv_hisbundle_distance_mat_aux = np.full((rv_hisbundle_distance_mat.shape[0]), get_nan_value(),
#                                                     dtype=np.float64)
#             for i in range(rv_hisbundle_connections.shape[0]):
#                 offset = rv_hisBase_path_mat[rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0, :]
#                 offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#                 path = rv_hisbundle_path_mat[i, rv_hisbundle_connections[i], :]
#                 path = path[path != get_nan_value()]  # For visualisation only - path offset
#                 path = np.concatenate((offset, path), axis=0)
#                 if rv_hisbundle_path_mat_aux.shape[1] < path.shape[0]:
#                     repeat_process = True
#                     break
#                 rv_hisbundle_path_mat_aux[i, :path.shape[0]] = path
#                 rv_hisbundle_distance_mat_aux[i] = rv_hisbundle_distance_mat[i, rv_hisbundle_connections[i]] + \
#                                                    rv_hisBase_distance_mat[
#                                                        rv_hisConnected_indexes[rv_hisbundle_connections[i]], 0]
#             if not repeat_process:
#                 rv_hisbundle_path_mat = rv_hisbundle_path_mat_aux
#                 rv_hisbundle_distance_mat = rv_hisbundle_distance_mat_aux
#                 # Clear Arguments to avoid recycling
#                 rv_hisbundle_path_mat_aux = None
#                 rv_hisbundle_distance_mat_aux = None
#
#                 # Rule 5) Apical|Septal|Paraseptal regions of the heart are defined as AB < 0.4/0.2 in the LV/RV | [0.7 < RT < 1.] | [0. < RT < 0.2] & [0.5 < RT < 0.7], respectively
#                 lv_apical_septal_mask = ((node_vc[vc_ab_cut_name][node_lvendo] <= 0.4) | (
#                         (0.7 <= node_vc[vc_rt_name][node_lvendo]) & (
#                         node_vc[vc_rt_name][node_lvendo] <= 1.))) & np.logical_not(
#                     lv_visited)
#                 rv_apical_septal_paraseptal_mask = (((node_vc[vc_ab_cut_name][node_rvendo] <= 0.2) | (
#                         (0.7 <= node_vc[vc_rt_name][node_rvendo]) & (
#                         node_vc[vc_rt_name][node_rvendo] <= 1.))) |
#                                                     (((0.0 <= node_vc[vc_rt_name][node_rvendo]) & (
#                                                             node_vc[vc_rt_name][node_rvendo] <= 0.2)) | (
#                                                              (0.5 <= node_vc[vc_rt_name][node_rvendo]) & (
#                                                              node_vc[vc_rt_name][
#                                                                  node_rvendo] <= 0.7)))) & np.logical_not(
#                     rv_visited)
#                 lv_PK_distance_mat[lv_apical_septal_mask] = lv_hisbundle_distance_mat[lv_apical_septal_mask]
#                 lv_PK_path_mat[lv_apical_septal_mask, :] = lv_hisbundle_path_mat[lv_apical_septal_mask, :]
#                 lv_visited[lv_apical_septal_mask] = True
#                 rv_PK_distance_mat[rv_apical_septal_paraseptal_mask] = rv_hisbundle_distance_mat[
#                     rv_apical_septal_paraseptal_mask]
#                 rv_PK_path_mat[rv_apical_septal_paraseptal_mask, :] = rv_hisbundle_path_mat[
#                                                                       rv_apical_septal_paraseptal_mask, :]
#                 rv_visited[rv_apical_septal_paraseptal_mask] = True
#
#                 # Rule 6) Paraseptal regions of the heart are connected from apex to base through either [0.4/0.2, 0.1, 1., :] or  [0.4/0.2, 0.6, 1., :] LV/RV
#                 lv_ant_paraseptalApex_index = int(np.argmin(
#                     np.linalg.norm(node_ab_rt_rvlv[node_lvendo, :] - np.array([0.4, 0.6, 0.]), ord=2,
#                                    axis=1)))  # [mid, paraseptal, lv]
#                 lv_post_paraseptalApex_index = int(np.argmin(
#                     np.linalg.norm(node_ab_rt_rvlv[node_lvendo, :] - np.array([0.4, 0.1, 0.]), ord=2,
#                                    axis=1)))  # [mid, paraseptal, lv]
#                 if not lv_visited[lv_ant_paraseptalApex_index]:
#                     lv_PK_distance_mat[lv_ant_paraseptalApex_index] = lv_hisbundle_distance_mat[
#                         lv_ant_paraseptalApex_index]
#                     lv_PK_path_mat[lv_ant_paraseptalApex_index, :] = lv_hisbundle_path_mat[
#                                                                      lv_ant_paraseptalApex_index,
#                                                                      :]
#                     lv_visited[lv_ant_paraseptalApex_index] = True
#                 if not lv_visited[lv_post_paraseptalApex_index]:
#                     lv_PK_distance_mat[lv_post_paraseptalApex_index] = lv_hisbundle_distance_mat[
#                         lv_post_paraseptalApex_index]
#                     lv_PK_path_mat[lv_post_paraseptalApex_index, :] = lv_hisbundle_path_mat[
#                                                                       lv_post_paraseptalApex_index, :]
#                     lv_visited[lv_post_paraseptalApex_index] = True
#                 lv_paraseptalApex_offsets = np.array(
#                     [lv_PK_distance_mat[lv_ant_paraseptalApex_index],
#                      lv_PK_distance_mat[lv_post_paraseptalApex_index]],
#                     dtype=float)
#                 lv_paraseptal_distance_mat, lv_paraseptal_path_mat = djikstra(
#                     np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int),
#                     lvnodes_xyz,
#                     lvunfoldedEdges, lvedgeVEC, lv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#                 lv_paraseptal_connections = np.argmin(lv_paraseptal_distance_mat, axis=1)
#                 lv_paraseptal_path_mat_aux = np.full(
#                     (lv_paraseptal_path_mat.shape[0], approx_djikstra_max_path_len),
#                     get_nan_value(),
#                     dtype=np.int32)
#                 lv_paraseptal_distance_mat_aux = np.full((lv_paraseptal_distance_mat.shape[0]), get_nan_value(),
#                                                          dtype=np.float64)
#                 for i in range(lv_paraseptal_connections.shape[0]):
#                     offset = lv_PK_path_mat[
#                              np.asarray([lv_ant_paraseptalApex_index, lv_post_paraseptalApex_index], dtype=int)[
#                                  lv_paraseptal_connections[i]], :]
#                     offset = offset[offset != get_nan_value()]  # For visualisation only - path offset
#                     path = lv_paraseptal_path_mat[i, lv_paraseptal_connections[i], :]
#                     path = path[path != get_nan_value()]  # For visualisation only - path offset
#                     path = np.concatenate((offset, path), axis=0)
#                     if lv_paraseptal_path_mat_aux.shape[1] < path.shape[0]:
#                         repeat_process = True
#                         break
#                     lv_paraseptal_path_mat_aux[i, :path.shape[0]] = path
#                     lv_paraseptal_distance_mat_aux[i] = lv_paraseptal_distance_mat[
#                                                             i, lv_paraseptal_connections[i]] + \
#                                                         lv_paraseptalApex_offsets[lv_paraseptal_connections[i]]
#                 if not repeat_process:
#                     lv_paraseptal_path_mat = lv_paraseptal_path_mat_aux
#                     lv_paraseptal_distance_mat = lv_paraseptal_distance_mat_aux
#                     # Clear Arguments to avoid recycling
#
#                     # Rule 7) Paraseptal regions of the heart are defined as [0. < rotation-angle (RT) < 0.2] & [0.5 < RT < 0.7], these are connected to their closest paraseptal routing point (anterior or posterior)
#                     lv_paraseptal_mask = (((0.0 <= node_vc[vc_rt_name][node_lvendo]) & (
#                             node_vc[vc_rt_name][node_lvendo] <= 0.2)) | (
#                                                   (0.5 <= node_vc[vc_rt_name][node_lvendo]) & (
#                                                   node_vc[vc_rt_name][node_lvendo] <= 0.7))) & np.logical_not(
#                         lv_visited)
#                     lv_PK_distance_mat[lv_paraseptal_mask] = lv_paraseptal_distance_mat[lv_paraseptal_mask]
#                     lv_PK_path_mat[lv_paraseptal_mask, :] = lv_paraseptal_path_mat[lv_paraseptal_mask, :]
#                     lv_visited[lv_paraseptal_mask] = True
#                     # Rule 8) Freewall regions of the heart are connected from apex to base through [0.4, 0.35, 1., :] in the LV
#                     lv_freewallApex_index = int(np.argmin(
#                         np.linalg.norm(node_ab_rt_rvlv[node_lvendo, :] - np.array([0.4, 0.35, 0.]), ord=2,
#                                        axis=1)))  # [mid, freewall, endo, lv]
#                     if not lv_visited[lv_freewallApex_index]:
#                         lv_PK_distance_mat[lv_freewallApex_index] = lv_hisbundle_distance_mat[
#                             lv_freewallApex_index]
#                         lv_PK_path_mat[lv_freewallApex_index, :] = lv_hisbundle_path_mat[lv_freewallApex_index,
#                                                                    :]
#                         lv_visited[lv_freewallApex_index] = True
#                     lv_freewallApex_offset = lv_PK_distance_mat[lv_freewallApex_index]
#                     lv_freewallApex_path_offset = lv_PK_path_mat[lv_freewallApex_index, :]
#                     lv_freewallApex_path_offset = lv_freewallApex_path_offset[
#                         lv_freewallApex_path_offset != get_nan_value()]
#                     lv_freewall_distance_mat, lv_freewall_path_mat = djikstra(
#                         np.asarray([lv_freewallApex_index], dtype=int), lvnodes_xyz, lvunfoldedEdges, lvedgeVEC,
#                         lv_neighbour, approx_max_path_len=approx_djikstra_max_path_len)
#                     lv_freewall_path_mat_aux = np.full(
#                         (lv_freewall_path_mat.shape[0], approx_djikstra_max_path_len),
#                         get_nan_value(),
#                         dtype=np.int32)
#                     lv_freewall_distance_mat_aux = np.full((lv_freewall_distance_mat.shape[0]), get_nan_value(),
#                                                            dtype=np.float64)
#                     for i in range(lv_freewall_distance_mat.shape[0]):
#                         path = lv_freewall_path_mat[i, 0, :]
#                         path = path[path != get_nan_value()]  # For visualisation only - path offset
#                         path = np.concatenate((lv_freewallApex_path_offset, path), axis=0)
#                         if lv_freewall_path_mat_aux.shape[1] < path.shape[0]:
#                             repeat_process = True
#                             break
#                         lv_freewall_path_mat_aux[i, :path.shape[0]] = path
#                         lv_freewall_distance_mat_aux[i] = lv_freewall_distance_mat[
#                                                               i, 0] + lv_freewallApex_offset
#                     if not repeat_process:
#                         lv_freewall_path_mat = lv_freewall_path_mat_aux
#                         lv_freewall_distance_mat = lv_freewall_distance_mat_aux
#
#                         # Rule 10) Freewall/Lateral regions of the heart are defined as [0.2 < rotation-angle (RT) < 0.5], these are connected to the lateral routing point
#                         lv_freewall_mask = ((0.2 <= node_vc[vc_rt_name][node_lvendo]) & (
#                                 node_vc[vc_rt_name][node_lvendo] <= 0.5)) & np.logical_not(lv_visited)
#                         lv_PK_distance_mat[lv_freewall_mask] = lv_freewall_distance_mat[lv_freewall_mask]
#                         lv_PK_path_mat[lv_freewall_mask, :] = lv_freewall_path_mat[lv_freewall_mask, :]
#                         lv_visited[lv_freewall_mask] = True
#
#     return lv_PK_distance_mat, lv_PK_path_mat, rv_PK_distance_mat, rv_PK_path_mat, repeat_process



def select_random_root_nodes(nb_root_nodes, candidate_root_node_indexes):
    rand_root_node_meta_indexes = np.zeros(candidate_root_node_indexes.shape, dtype=bool)
    indexes = np.random.randint(0, candidate_root_node_indexes.shape[0], nb_root_nodes)
    rand_root_node_meta_indexes[indexes] = True
    return rand_root_node_meta_indexes


# Create equally spaced potential root-node locations in the mesh before 28/06/2021
def generate_candidate_root_nodes_in_cavity(basal_cavity_nodes_xyz, basal_cavity_vc_ab_cut,
                                            inter_root_node_distance, purkinje_max_cut_ab_threshold):
    print('Inside generate_candidate_root_nodes_in_cavity')
    # Include all nodes that are at least at the specified distance
    # permutated_node_indexes = np.random.permutation(np.arange(start=0, stop=cavity_nodes_xyz.shape[0], step=1))
    # The root node regneration cannot be at random otherwise when reading the inferred properties, there is no way to regenerate the same Purkinje tree.
    # TODO change the way the root nodes are saved so that these trees can be generated at random and still be read back in.
    permutated_node_indexes = np.arange(start=0, stop=basal_cavity_nodes_xyz.shape[0], step=1)
    candidate_root_node_meta_indexes = []
    for node_i in range(0, permutated_node_indexes.shape[0], 1):    # range() takes no keyword arguments
        next_node_index = permutated_node_indexes[node_i]
        next_node_ab_cut = basal_cavity_vc_ab_cut[next_node_index]
        if next_node_ab_cut <= purkinje_max_cut_ab_threshold:    # Check if apex-to-base value is below the threshold
            next_node_xyz = basal_cavity_nodes_xyz[next_node_index]
            if not candidate_root_node_meta_indexes:    # Check if list is empty (boolean value of empty list is false)
                # if []:
                #     print('True')
                # else:
                #     print('False')
                candidate_root_node_meta_indexes.append(next_node_index)
            else:
                selected_root_nodes_xyz = basal_cavity_nodes_xyz[candidate_root_node_meta_indexes, :]
                euclidean_distance_to_next_node = np.amin(np.linalg.norm(selected_root_nodes_xyz - next_node_xyz, axis=1))
                if euclidean_distance_to_next_node > inter_root_node_distance:
                    candidate_root_node_meta_indexes.append(next_node_index)
    return np.asarray(candidate_root_node_meta_indexes, dtype=int)


if __name__ == '__main__':
    import os
    from geometry_functions import EikonalGeometry  # Cannot be imported at the start or there is a dependency cycle between this module and geometry_functions.py
    from path_config import get_path_mapping
    from io_functions import write_ensight, save_purkinje

    print(
        'Caution, all the hyper-parameters are set assuming resolutions of 1000 Hz in all time-series.')  # TODO: make hyper-parameters relative to the time series resolutions.
    # TODO: enable having different resolutions in different time-series in the code.
    # Simulate and plot QRS
    if os.path.isfile('../.custom_config/.your_path_mapping.txt'):
        path_dict = get_path_mapping()
    else:
        raise 'Missing data and results configuration file at: ../.custom_config/.your_path_mapping.txt'
    ####################################################################################################################
    # Step 0: Reproducibility:
    random_seed_value = 7  # Ensures reproducibility and turns off stochasticity
    np.random.seed(seed=random_seed_value)  # Ensures reproducibility and turns off stochasticity
    ####################################################################################################################
    # Step 1: Define paths and other environment variables.
    # General settings:
    anatomy_subject_name = 'rodero_05'
    print('anatomy_subject_name: ', anatomy_subject_name)
    resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    geometric_data_dir = data_dir + 'geometric_data/'
    # Output Paths:
    results_dir = path_dict["results_path"] + 'personalisation_data/' + anatomy_subject_name + '/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    theta_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_inferred_population.csv'
    # Module names:
    propagation_module_name = 'propagation_module'
    # Clear Arguments to prevent Argument recycling
    clinical_data_dir_tag = None
    data_dir = None
    ecg_subject_name = None
    results_dir = None
    ####################################################################################################################
    # Step 2: Generate an Eikonal-friendly geometry.
    # Argument setup: (in Alphabetical order)
    boundary_data_dir_tag = 'boundary_data/'
    vc_name_list = ['ab', 'tm', 'rt', 'tv']#, 'aprt', 'rvlv']
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(conduction_system=EmptyConductionSystem(verbose=verbose), geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    dummy_conduction_system = None
    vc_name_list = None
    ####################################################################################################################
    # # Step 3: Create conduction system for the propagation model to be initialised.
    # # Arguments for Conduction system:
    # approx_djikstra_purkinje_max_path_len = 200
    # lv_inter_root_node_distance = 2.5  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    # rv_inter_root_node_distance = 2.5  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    # # Create conduction system
    # conduction_system = DjikstraConductionSystemVC(
    #     approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, geometry=geometry,
    #     lv_candidate_root_node_meta_index=, rv_candidate_root_node_meta_index=,
    #     purkinje_max_ab_cut_threshold=get_purkinje_max_ab_cut_threshold(),
    #     vc_ab_cut_name=, vc_rt_name=, verbose=verbose)
    # # Clear Arguments to prevent Argument recycling
    # approx_djikstra_purkinje_max_path_len = None
    # lv_inter_root_node_distance = None
    # rv_inter_root_node_distance = None
    # # Assign conduction_system to its geometry
    # geometry.set_conduction_system(conduction_system)
    # conduction_system = None    # Clear Arguments to prevent Argument recycling
    # # Save geometry into visualisation directory
    # visualisation_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + resolution + '/ensight/'
    # if not os.path.exists(visualisation_dir):
    #     os.mkdir(visualisation_dir)
    # write_ensight(subject_name=anatomy_subject_name, visualisation_dir=visualisation_dir, geometry=geometry,
    #               verbose=verbose)
    # # TODO code this in a nicer way: This is a super CHAPUZA!
    # # best_parameter_result_file_name = path_dict["results_path"] + 'personalisation_data/' + anatomy_subject_name + '/' + anatomy_subject_name + '_' + resolution + '_inferred-best-parameter.csv'
    # # best_parameter = np.loadtxt(best_parameter_result_file_name, delimiter=',')
    # # print('best_parameter ', best_parameter.shape)
    # # print('best_parameter ', best_parameter)
    # # root_node_meta_bool_index = np.asarray(best_parameter[6:], dtype=bool)
    # root_node_meta_bool_index = np.ones((geometry.get_nb_candidate_root_node()), dtype=bool)
    # print('root_node_meta_bool_index ',root_node_meta_bool_index.shape)
    # print('root_node_meta_bool_index ',root_node_meta_bool_index)
    # save_purkinje(dir=visualisation_dir, geometry=geometry, root_node_meta_bool_index=root_node_meta_bool_index, subject_name=anatomy_subject_name)
    # # Clear Arguments to prevent Argument recycling
    # geometric_data_dir = None
    # geometry = None
    # resolution = None
    # anatomy_subject_name = None

    print('End of Conduction system test case')

    # TODO add here the functions that write out the PK tree as a vtk or something that can be visualised in Paraview.

# EOF



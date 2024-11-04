import math
from warnings import warn
import numpy as np

from postprocess_functions import scatter_visualise_point_cloud
from utils import get_nan_value, map_indexes, check_vc_field_ranges, normalise_field_to_range, get_vc_ab_cut_name, \
    get_vc_rt_name, get_apex_ab_cut_value, get_endo_fast_and_purkinje_max_ab_cut_threshold, \
    get_valid_vc_rt_range, get_lv_apical_ab_cut_threshold, get_rv_apical_ab_cut_threshold, \
    get_freewall_posterior_rt_value, get_freewall_anterior_rt_value
from io_functions import read_csv_file, get_electrode_filename, get_material_filename, \
    get_node_xyz_filename, get_tetra_filename, get_node_lvendo_filename, get_node_rvendo_filename, \
    get_fibre_fibre_filename, get_fibre_sheet_filename, get_fibre_normal_filename, get_vc_filename


class RawEmptyCardiacGeoPointCloud:
    def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, verbose):
        if verbose:
            print('Reading geometry')
        self.verbose = verbose
        self.conduction_system = conduction_system
        # Read all data into the geometry
        # Define file structure
        self.data_path = geometric_data_dir + subject_name + '/' + subject_name + '_' + resolution + '/'
        self.file_prefix = subject_name + '_' + resolution
        # Read node xyz coordinates
        node_xyz_file_path = get_node_xyz_filename(anatomy_subject_name=subject_name,
                                                   geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_node_xyz = read_csv_file(filename=node_xyz_file_path)
        node_xyz_file_path = None  # Clear Arguments to prevent Argument recycling

    def get_node_xyz(self):
        return self.unprocessed_node_xyz

    def get_electrode_xyz(self):
        raise NotImplementedError

    def get_empty_node_field(self):
        return np.zeros((self.get_node_xyz().shape[0]))

    def set_conduction_system(self, conduction_system):
        self.conduction_system = conduction_system

    def get_node_lvendo(self):
        raise NotImplementedError

    def get_node_rvendo(self):
        raise NotImplementedError

    def get_node_vc(self):
        raise NotImplementedError

    def spatial_smoothing_of_time_field_using_adjacentcies(self, original_field_data, ghost_distance_to_self):
        raise NotImplementedError

    def visualise(self):
        scatter_visualise_point_cloud(xyz=self.get_node_xyz(), title=None)

    """Root node xyz coordinates"""
    def get_selected_root_node_xyz(self, root_node_index):
        node_xyz = self.get_node_xyz()
        return node_xyz[root_node_index, :]

    """FUNCTIONS FROM CONDUCTION SYSTEM"""
    # CANDIDATE ROOT NODES
    def get_nb_candidate_root_node(self):
        return self.conduction_system.get_nb_candidate_root_node()

    def get_candidate_root_node_index(self):
        return self.conduction_system.get_candidate_root_node_index()

    def get_candidate_root_node_distance(self):
        return self.conduction_system.get_candidate_root_node_distance()

    def get_candidate_root_node_time(self, purkinje_speed):
        return self.conduction_system.get_candidate_root_node_time(purkinje_speed=purkinje_speed)

    def get_lv_rv_candidate_root_node_index(self):
        lv_candidate_root_node_index, rv_candidate_root_node_index = self.conduction_system.get_lv_rv_candidate_root_node_index()
        return lv_candidate_root_node_index, rv_candidate_root_node_index

    def get_lv_rv_candidate_purkinje_edge(self):
        lv_pk_edge, rv_pk_edge = self.conduction_system.get_lv_rv_candidate_purkinje_edge()
        return lv_pk_edge, rv_pk_edge

    # SELECTED ROOT NODES
    def get_selected_root_node_index(self, root_node_meta_index):
        return self.conduction_system.get_selected_root_node_index(root_node_meta_index=root_node_meta_index)

    def get_selected_root_node_distance(self, root_node_meta_index):
        return self.conduction_system.get_selected_root_node_distance(root_node_meta_index=root_node_meta_index)

    def get_selected_root_node_time(self, root_node_meta_index, purkinje_speed):
        return self.conduction_system.get_selected_root_node_time(root_node_meta_index=root_node_meta_index,
                                                                  purkinje_speed=purkinje_speed)

    def get_lv_rv_selected_root_node_meta_index(self, root_node_meta_index):
        return self.conduction_system.get_lv_rv_selected_root_node_meta_index(root_node_meta_index=root_node_meta_index)

    def get_lv_rv_selected_root_node_index(self, root_node_meta_index):
        lv_selected_root_node_index, rv_selected_root_node_index = \
            self.conduction_system.get_lv_rv_selected_root_node_index(root_node_meta_index=root_node_meta_index)
        return lv_selected_root_node_index, rv_selected_root_node_index

    def get_lv_rv_selected_purkinje_edge(self, root_node_meta_index):
        lv_pk_edge, rv_pk_edge = self.conduction_system.get_lv_rv_selected_purkinje_edge(
            root_node_meta_index=root_node_meta_index)
        return lv_pk_edge, rv_pk_edge


class RawEmptyCardiacGeoTet(RawEmptyCardiacGeoPointCloud):
    def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, verbose):
        super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir,
                         resolution=resolution, subject_name=subject_name, verbose=verbose)
        # Read tetrahedral indices
        tetra_file_path = get_tetra_filename(anatomy_subject_name=subject_name,
                                                  geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_tetra = read_csv_file(filename=tetra_file_path).astype(int)
        tetra_file_path = None  # Clear Arguments to prevent Argument recycling
        # Determine correction of indexing
        self.index_correction_to_zero = self.__define_indexing_correction()  # value to be added to any unprocessed index field to make it start from zero
        # Correct tetra node indexes to make them start from zero
        self.unprocessed_tetra = self.unprocessed_tetra + self.index_correction_to_zero # tetra is always with the correct indexes

    def get_tetra(self):
        return self.unprocessed_tetra

    def get_tetra_centre(self):
        return calculate_centre(array_xyz=self.get_node_xyz(), array_index=self.get_tetra())

    def __define_indexing_correction(self):
        if np.amin(self.unprocessed_tetra) == 0:
            index_correction = 0
        elif np.amin(self.unprocessed_tetra) == 1:
            index_correction = - 1
        elif np.amin(self.unprocessed_tetra) > 1:
            warn('Minimum tetra indexes is larger than 1')
            raise() # TODO should it just be (-1) * np.amin(self.unprocessed_tetra) ?
        else:
            warn('Minimum tetra indexes is smaller than 0')
            raise ()  # TODO should it just be (-1) * np.amin(self.unprocessed_tetra) ?
        return int(index_correction)


# class AlyaFormatRawCardiacGeoTet(RawEmptyCardiacGeoTet):
#     def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, verbose):
#         super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir,
#                          resolution=resolution, subject_name=subject_name, verbose=verbose)
#         # Fibre
#         self.unprocessed_node_fibre_fibre = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_fibre.csv', delimiter=',')
#         self.unprocessed_node_fibre_sheet = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_sheet.csv', delimiter=',')
#         self.unprocessed_node_fibre_normal = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_normal.csv', delimiter=',')
#         # self.unprocessed_node_fibre_fibre = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_fibre.csv')[:,
#         #                                     1:]
#         # self.unprocessed_node_fibre_sheet = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_sheet.csv')[:,
#         #                                     1:]
#         # self.unprocessed_node_fibre_normal = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_normal.csv')[:,
#         #                                      1:]
#
#     def get_node_fibre(self):
#         return self.unprocessed_node_fibre_fibre
#
#     def get_node_sheet(self):
#         return self.unprocessed_node_fibre_sheet
#
#     def get_node_normal(self):
#         return self.unprocessed_node_fibre_normal


class RawVCCardiacGeoTet(RawEmptyCardiacGeoTet):
    def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list, verbose):
        super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir,
                         resolution=resolution, subject_name=subject_name, verbose=verbose)
        # Ventricular Coordinates
        self.unprocessed_node_vc = {}
        for vc_name in vc_name_list:
            vc_file_path = get_vc_filename(anatomy_subject_name=subject_name, geometric_data_dir=geometric_data_dir,
                                           resolution=resolution, vc_name=vc_name)
            # self.set_node_vc_field(vc_data=read_csv_file(filename=vc_file_path), vc_name=vc_name)
            # self.unprocessed_node_vc[vc_name] = read_csv_file(filename=vc_file_path)
            vc_field = read_csv_file(filename=vc_file_path)

            # TODO Delete the following lines after processing Rodero 13
            warn('This part of the code needs to be deleted! Ruben already fixed this for future usage of the code! 19/01/2024')  # TODO DELETE LINE
            if vc_name == 'tm' and subject_name == 'rodero_13':  # TODO DELETE LINE
                vc_field = normalise_field_to_range(field=vc_field, range=[0., 1.])  # TODO DELETE LINE
            # TODO Delete the above lines after processing Rodero 13

            self.unprocessed_node_vc[vc_name] = vc_field
            check_vc_field_ranges(vc_field=vc_field, vc_name=vc_name)
            # Clear memory and avoid recycling
            vc_field = None

            # self.unprocessed_node_vc[vc_name] = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_'
            #                                                + vc_name + '.csv', delimiter=',')
            vc_file_path = None  # Clear Arguments to prevent Argument recycling
        # Endocardial surfaces and correct node indexes to make them start from zero # TODO: Extract endocardial surfaces from ventricular coordinates
        ## LV
        node_lvendo_file_path = get_node_lvendo_filename(anatomy_subject_name=subject_name,
                                                         geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_node_lvendo = read_csv_file(filename=node_lvendo_file_path).astype(int) \
                                       + self.index_correction_to_zero     # lv endocardium nodes
        # self.unprocessed_node_lvendo = np.loadtxt(
        #     self.data_path + self.file_prefix + '_boundarynodefield_lvendo.csv', delimiter=',').astype(int) \
        #                                + self.index_correction_to_zero     # lv endocardium nodes
        node_lvendo_file_path = None  # Clear Arguments to prevent Argument recycling
        ## RV
        node_rvendo_file_path = get_node_rvendo_filename(anatomy_subject_name=subject_name,
                                                         geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_node_rvendo = read_csv_file(filename=node_rvendo_file_path).astype(int) \
                                       + self.index_correction_to_zero     # rv endocardium nodes
        # self.unprocessed_node_rvendo = np.loadtxt(
        #     self.data_path + self.file_prefix + '_boundarynodefield_rvendo.csv', delimiter=',').astype(int) \
        #                                + self.index_correction_to_zero     # rv endocardium nodes
        node_rvendo_file_path = None  # Clear Arguments to prevent Argument recycling

    def get_node_lvendo(self):
        return self.unprocessed_node_lvendo

    def get_node_rvendo(self):
        return self.unprocessed_node_rvendo

    def get_node_vc(self):
        return self.unprocessed_node_vc

    # def set_node_vc_field(self, vc_data, vc_name):
    #     self.unprocessed_node_vc[vc_name] = vc_data

    def get_node_vc_field(self, vc_name):
        return self.unprocessed_node_vc[vc_name]

    """Root node vc"""
    def get_selected_root_node_vc_field(self, root_node_index, vc_name):
        node_vc = self.get_node_vc_field(vc_name)
        return node_vc[root_node_index]


"""Class for the ML Mechanics project"""
class InvariantRawVCCardiacGeoTet(RawEmptyCardiacGeoTet):
    def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list, verbose):
        super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir,
                         resolution=resolution, subject_name=subject_name, verbose=verbose)
        # Ventricular Coordinates
        self.unprocessed_node_vc = {}
        for vc_name in vc_name_list:
            vc_field = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_' + vc_name + '.csv', delimiter=',')
            self.unprocessed_node_vc[vc_name] = vc_field
            check_vc_field_ranges(vc_field=vc_field, vc_name=vc_name)
            vc_field = None     # Clear memory and avoid recycling


"""Class for the ML Mechanics project"""
class InvariantRawVCFibreCardiacGeoTet(InvariantRawVCCardiacGeoTet):
    '''For Adrien's work with GNNs'''
    def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list, verbose):
        super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir,
                         resolution=resolution, subject_name=subject_name, vc_name_list=vc_name_list, verbose=verbose)
        # Epicardial surfaces and correct node indexes to make them start from zero # TODO: Extract endocardial surfaces from ventricular coordinates
        # self.unprocessed_node_lvendo = np.loadtxt(
        #     self.data_path + self.file_prefix + '_boundarynodefield_epi.csv', delimiter=',').astype(int) \
        #                                + self.index_correction_to_zero  # epicardial nodes
        self.unprocessed_node_epi = np.loadtxt(
            self.data_path + self.file_prefix + '_nodefield_epi.csv', delimiter=',').astype(int)
        self.unprocessed_node_lvendo = np.loadtxt(
            self.data_path + self.file_prefix + '_nodefield_lvendo.csv', delimiter=',').astype(int)
        self.unprocessed_node_rvendo = np.loadtxt(
            self.data_path + self.file_prefix + '_nodefield_rvendo.csv', delimiter=',').astype(int)
        # Fibre
        self.unprocessed_node_fibre_fibre = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_fibre_invariant_projection.csv', delimiter=',')
        self.unprocessed_node_fibre_sheet = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_sheet_invariant_projection.csv', delimiter=',')
        self.unprocessed_node_fibre_normal = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_normal_invariant_projection.csv', delimiter=',')
        # Material
        self.unprocessed_node_material = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_material.csv',  delimiter=',')
        # Clear Arguments to prevent Argument recycling
        self.data_path = None
        self.file_prefix = None

    # TODO Delete this
    def get_node_lvendo(self):
        warn('should this be deleted?')     # TODO revise why there is a delete TODO in this part of the code 19/01/2024
        return self.unprocessed_node_lvendo

    def get_node_rvendo(self):
        return self.unprocessed_node_rvendo

    def get_node_epi(self):
        return self.unprocessed_node_epi

    def get_node_vc(self):
        return self.unprocessed_node_vc

    def get_node_vc_field(self, vc_name):
        return self.unprocessed_node_vc[vc_name]

    # def set_node_vc_field(self, vc_data, vc_name):
    #     self.unprocessed_node_vc[vc_name] = vc_data

    def get_node_fibre(self):
        return self.unprocessed_node_fibre_fibre

    def get_node_sheet(self):
        return self.unprocessed_node_fibre_sheet

    def get_node_normal(self):
        return self.unprocessed_node_fibre_normal

    def get_node_material(self):
        return self.unprocessed_node_material


class RawVCFibreCardiacGeoTet(RawVCCardiacGeoTet):
    def __init__(self, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list, verbose):
        super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir,
                         resolution=resolution, subject_name=subject_name, vc_name_list=vc_name_list, verbose=verbose)
        # Fibre
        fibre_fibre_file_path = get_fibre_fibre_filename(anatomy_subject_name=subject_name,
                                                         geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_node_fibre_fibre = read_csv_file(filename=fibre_fibre_file_path)
        fibre_fibre_file_path = None  # Clear Arguments to prevent Argument recycling
        # Sheet
        fibre_sheet_file_path = get_fibre_sheet_filename(anatomy_subject_name=subject_name,
                                                         geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_node_fibre_sheet = read_csv_file(filename=fibre_sheet_file_path)
        fibre_sheet_file_path = None  # Clear Arguments to prevent Argument recycling
        # Normal
        fibre_normal_file_path = get_fibre_normal_filename(anatomy_subject_name=subject_name,
                                                           geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_node_fibre_normal = read_csv_file(filename=fibre_normal_file_path)
        fibre_normal_file_path = None  # Clear Arguments to prevent Argument recycling
        # self.unprocessed_node_fibre_fibre = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_fibre.csv', delimiter=',')
        # self.unprocessed_node_fibre_sheet = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_sheet.csv', delimiter=',')
        # self.unprocessed_node_fibre_normal = np.loadtxt(self.data_path + self.file_prefix + '_nodefield_normal.csv', delimiter=',')
        # Material
        material_file_path = get_material_filename(anatomy_subject_name=subject_name,
                                                   geometric_data_dir=geometric_data_dir, resolution=resolution)
        self.unprocessed_tetra_material = read_csv_file(filename=material_file_path)
        # Clear Arguments to prevent Argument recycling
        self.data_path = None
        self.file_prefix = None
        self.index_correction_to_zero = None
        material_file_path = None

    # # TODO Delete the following lines
    # def get_node_fibre(self):
    #     return self.unprocessed_node_fibre_fibre
    #
    # def get_node_sheet(self):
    #     return self.unprocessed_node_fibre_sheet
    #
    # def get_node_normal(self):
    #     return self.unprocessed_node_fibre_normal


class SimpleCardiacGeoTet(RawVCFibreCardiacGeoTet):
    """This cardiac geometry removes the valbular plug from itself using the materials field."""
    def __init__(self, cellular_model, celltype_vc_info, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list, verbose):
        super().__init__(conduction_system=conduction_system, geometric_data_dir=geometric_data_dir, resolution=resolution, subject_name=subject_name,
                         vc_name_list=vc_name_list, verbose=verbose)
        # Electrodes
        electrode_file_path = get_electrode_filename(geometric_data_dir=geometric_data_dir,
                                                     anatomy_subject_name=subject_name)
        self.electrode_xyz = read_csv_file(filename=electrode_file_path)
        electrode_file_path = None  # Clear Arguments to prevent Argument recycling
        # Remove duplicate tetra
        self.unprocessed_tetra, unique_tetra_index = np.unique(self.unprocessed_tetra, axis=0, return_index=True)
        self.unprocessed_tetra_material = self.unprocessed_tetra_material[unique_tetra_index]
        unique_tetra_index = None  # Clear Arguments to prevent Argument recycling
        # Remove valvular plugs from the geometry
        valvular_plug_tetra_mask = self.unprocessed_tetra_material != 2  # The valvular plugs are coded as 2 for Alya simulations
        self.unprocessed_tetra = self.unprocessed_tetra[valvular_plug_tetra_mask, :]
        self.unprocessed_tetra_material = self.unprocessed_tetra_material[valvular_plug_tetra_mask]  # Pretend like the valvular plug never existed
        valvular_plug_tetra_mask = None  # Clear Arguments to prevent Argument recycling
        #################
        # Reindex nodes # - Pretend like the valvular plug never existed
        #################
        remaining_old_node_index = np.sort(np.unique(self.unprocessed_tetra))
        new_node_index = np.arange(0, remaining_old_node_index.shape[0], 1)
        # Node xyz
        self.node_xyz = self.unprocessed_node_xyz[remaining_old_node_index, :]
        self.unprocessed_node_xyz = None  # Clear Arguments to prevent Argument recycling
        # Ventricular Coordinates
        self.node_vc = {}
        for vc_name in vc_name_list:
            # self.set_node_vc_field(vc_data=self._get_unprocessed_node_vc_field(vc_name=vc_name)[remaining_old_node_index],
            #                        vc_name=vc_name)
            self.node_vc[vc_name] = self.unprocessed_node_vc[vc_name][remaining_old_node_index]
        self.unprocessed_node_vc = None  # Clear Arguments to prevent Argument recycling
        # Fibre
        self.unprocessed_node_fibre_fibre = self.unprocessed_node_fibre_fibre[remaining_old_node_index, :]
        self.unprocessed_node_fibre_sheet = self.unprocessed_node_fibre_sheet[remaining_old_node_index, :]
        self.unprocessed_node_fibre_normal = self.unprocessed_node_fibre_normal[remaining_old_node_index, :]
        normalised_node_fibre_fibre, normalised_node_fibre_sheet, normalised_node_fibre_normal = correct_and_normlise_ortho_fibre(
            fibre=self.unprocessed_node_fibre_fibre, sheet=self.unprocessed_node_fibre_sheet,
            normal=self.unprocessed_node_fibre_normal)
        self.unprocessed_node_fibre_fibre = None  # Clear Arguments to prevent Argument recycling
        self.unprocessed_node_fibre_sheet = None  # Clear Arguments to prevent Argument recycling
        self.unprocessed_node_fibre_normal = None  # Clear Arguments to prevent Argument recycling
        self.normalised_node_fibre_fibre = normalised_node_fibre_fibre
        self.normalised_node_fibre_sheet = normalised_node_fibre_sheet
        self.normalised_node_fibre_normal = normalised_node_fibre_normal
        normalised_node_fibre_fibre = None  # Clear Arguments to prevent Argument recycling
        normalised_node_fibre_sheet = None  # Clear Arguments to prevent Argument recycling
        normalised_node_fibre_normal = None  # Clear Arguments to prevent Argument recycling
        # The fibres need to be in the shape of [node, xyz, fibre-sheet-normal] for the Eikonal to operate on them
        self.normalised_node_fibre_sheet_normal = np.stack([self.normalised_node_fibre_fibre, self.normalised_node_fibre_sheet, self.normalised_node_fibre_normal], axis=2)  # The fibres need to be in the shape of [node, xyz, fibre-sheet-normal] for the Eikonal to operate on them
        # print('node_fibre_fibre ', self.node_fibre_fibre[0, :])
        # print('node_fibre_fibre ', np.linalg.norm(self.node_fibre_fibre[0, :]))
        #
        # print('node_fibre_sheet ', np.linalg.norm(self.node_fibre_sheet[0, :]))
        # print('node_fibre_normal ', np.linalg.norm(self.node_fibre_normal[0, :]))
        # print('self.node_fibre_sheet_normal ', self.node_fibre_sheet_normal[0, :, :])
        # print('self.node_fibre_sheet_normal ', np.linalg.norm(self.node_fibre_sheet_normal[0, 0, :]))
        # print('self.node_fibre_sheet_normal ',np.linalg.norm( self.node_fibre_sheet_normal[0, 1, :]))
        # print('self.node_fibre_sheet_normal ', np.linalg.norm(self.node_fibre_sheet_normal[0, 2, :]))
        # Tetra
        self.tetra = self.__correct_node_index(new_node_index=new_node_index, old_node_index=self.unprocessed_tetra,
                                               remaining_node_index=remaining_old_node_index)
        self.unprocessed_tetra = None  # Clear Arguments to prevent Argument recycling
        # Endocardial surfaces
        self.node_lvendo = self.__correct_node_index(new_node_index=new_node_index,
                                                     old_node_index=self.unprocessed_node_lvendo,
                                                     remaining_node_index=remaining_old_node_index)
        self.unprocessed_node_lvendo = None  # Clear Arguments to prevent Argument recycling
        self.node_rvendo = self.__correct_node_index(new_node_index=new_node_index,
                                                     old_node_index=self.unprocessed_node_rvendo,
                                                     remaining_node_index=remaining_old_node_index)
        self.unprocessed_node_rvendo = None  # Clear Arguments to prevent Argument recycling
        # Clear Arguments to prevent Argument recycling
        new_node_index = None
        remaining_old_node_index = None
        #####################################
        # Process other aspects of material #
        #####################################
        self.ecg_tetra_mask = self.unprocessed_tetra_material == 1  # Materials contains ones where it should be used for the ECG computation
        self.unprocessed_tetra_material = None  # Clear Arguments to prevent Argument recycling
        # Other Not worth re-indexing, just better to recompute
        self.tetra_centre = calculate_centre(array_xyz=self.node_xyz, array_index=self.tetra)  # Needed for the ecg computation
        #####################################################
        # Assign celltypes using a dictionary with vc rules #
        #####################################################
        self.__assign_celltype(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info)

    def get_node_xyz(self):
        return self.node_xyz

    def get_tetra(self):
        return self.tetra

    def get_electrode_xyz(self):
        return self.electrode_xyz

    def get_node_normalised_fibre_sheet_normal(self):
        print('Function get_node_fibre_sheet_normal() returns a node field where each entry includes the components in the fibre direction, the sheet, then normal, rather than the original '
             'fibre_fibre, fibre_sheet and fibre_normal vectors!')
        return self.normalised_node_fibre_sheet_normal

    def get_node_normalised_fibre_fibre(self):
        return self.normalised_node_fibre_fibre

    def get_node_normalised_fibre_sheet(self):
        return self.normalised_node_fibre_sheet

    def get_node_normalised_fibre_normal(self):
        return self.normalised_node_fibre_normal

    def get_node_lvendo(self):
        return self.node_lvendo

    def get_node_rvendo(self):
        return self.node_rvendo

    def get_node_vc(self):
        return self.node_vc

    # def _get_unprocessed_node_vc_field(self, vc_name):
    #     return self.unprocessed_node_vc[vc_name]

    def get_node_vc_field(self, vc_name):
        return self.node_vc[vc_name]

    # def set_node_vc_field(self, vc_data, vc_name):
    #     self.node_vc[vc_name] = vc_data

    def get_node_celltype(self):
        return self.node_celltype

    def __correct_node_index(self, new_node_index, old_node_index, remaining_node_index):
        # print('old_node_index ', old_node_index.shape)
        if len(old_node_index.shape) == 1:
            old_node_index = old_node_index[np.isin(old_node_index, remaining_node_index)]
        else:
            old_node_index = old_node_index[np.all(np.isin(old_node_index, remaining_node_index), axis=1), :]
        # print('old_node_index after ', old_node_index.shape)
        corrected_node_index = np.zeros(old_node_index.shape, dtype=int) - 100  # Initialise index to a value that won't be matched in the search
        for remaining_node_index_i in range(remaining_node_index.shape[0]):
            corrected_node_index[old_node_index == remaining_node_index[remaining_node_index_i]] = new_node_index[remaining_node_index_i]
        return corrected_node_index

    def __assign_celltype(self, cellular_model, celltype_vc_info):
        nb_node = self.get_node_xyz().shape[0]
        node_celltype = np.zeros((nb_node), dtype=int) + cellular_model.get_celltype_invalid_value()
        node_vc = self.get_node_vc()
        vc_name_list = list(node_vc.keys())
        celltype_name_list = list(celltype_vc_info.keys())
        for node_i in range(self.get_node_xyz().shape[0]):
            for celltype_name in celltype_name_list:
                this_celltype = True
                celltype_vc = celltype_vc_info[celltype_name]
                for vc_name in vc_name_list:
                    if vc_name in celltype_vc:
                        vc_value = node_vc[vc_name][node_i]
                        celltype_vc_range = celltype_vc[vc_name]
                        if celltype_vc_range[0] <= vc_value <= celltype_vc_range[1]:
                            continue
                        else:
                            this_celltype = False
                            break
                if this_celltype:
                    node_celltype[node_i] = cellular_model.get_celltype_id(celltype_name=celltype_name)
                    break
        self.node_celltype = node_celltype


class SimulationGeometry(SimpleCardiacGeoTet):
    """This private class extends the CardiacGeoTet class to include Eikonal specific data fields"""
    def __init__(self, cellular_model, celltype_vc_info, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list,
                 verbose):
        super().__init__(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info, conduction_system=conduction_system,
                         geometric_data_dir=geometric_data_dir, resolution=resolution, subject_name=subject_name,
                         vc_name_list=vc_name_list, verbose=verbose)
        # Generate Edges: This section is inefficient in the sake of clarity
        edge_list = []
        for tetra_i in range(self.tetra.shape[0]):
            for vertex_i in range(self.tetra.shape[1]):
                for vertex_j in range(0, vertex_i, 1):
                    edge_list.append(np.sort(np.array(
                        [self.tetra[tetra_i, vertex_i], self.tetra[tetra_i, vertex_j]])))
        edge = np.stack(edge_list, axis=0)
        edge_list = None    # Clear Arguments to prevent Argument recycling
        self.edge = np.unique(edge, axis=0)
        edge = None     # Clear Arguments to prevent Argument recycling
        # # Exclude disconnected surface indexes
        # if verbose:
        #     print('Check all nodes in the LV surface are connected')
        # self.node_lvendo = find_disconnected_nodes(node_index=self.node_lvendo, edge=self.edge)
        # if verbose:
        #     print('Check all nodes in the RV surface are connected')
        # self.node_rvendo = find_disconnected_nodes(node_index=self.node_rvendo, edge=self.edge)
        # # Interpolate fibres from nodes to edges
        # edge_centre = calculate_centre(array_xyz=self.node_xyz, array_index=self.edge)
        # edge_mapped_indexes = map_indexes(points_to_map_xyz=edge_centre, reference_points_xyz=self.node_xyz)
        # edge_centre = None  # Clear Arguments to prevent Argument recycling
        # self.edge_fibre_sheet_normal = self.get_node_normalised_fibre_sheet_normal()[edge_mapped_indexes, :, :]
        # edge_mapped_indexes = None  # Clear Arguments to prevent Argument recycling
        # # self.node_fibre_sheet_normal = None   # Clear Arguments to prevent Argument recycling
        # if verbose:
        #     print('Check all nodes are connected')
        #     find_disconnected_nodes(node_index=np.arange(start=0, stop=self.node_xyz.shape[0], step=1), edge=self.edge)
        # Prepaper for Eikonal and Smoothing computations
        self.edge_vec = self.node_xyz[self.edge[:, 0], :] - self.node_xyz[self.edge[:, 1], :]  # edge vectors
        self.unfolded_edge = np.concatenate((self.edge, np.flip(self.edge, axis=1))).astype(int)
        if verbose:
            print('Precomputing Eikonal adjacentcies')
        self.__build_adjacentcies()
        # # Exclude disconnected surface indexes
        # if verbose:
        #     print('Check all nodes in the LV surface are connected')
        # self.node_lvendo = find_disconnected_nodes(node_index=self.node_lvendo, edge=self.edge)
        # if verbose:
        #     print('Check all nodes in the RV surface are connected')
        # self.node_rvendo = find_disconnected_nodes(node_index=self.node_rvendo, edge=self.edge)
        # # Interpolate fibres from nodes to edges
        # edge_centre = calculate_centre(array_xyz=self.node_xyz, array_index=self.edge)
        # edge_mapped_indexes = map_indexes(points_to_map_xyz=edge_centre, reference_points_xyz=self.node_xyz)
        # edge_centre = None  # Clear Arguments to prevent Argument recycling
        # self.edge_fibre_sheet_normal = self.get_node_normalised_fibre_sheet_normal()[edge_mapped_indexes, :, :]
        # edge_mapped_indexes = None  # Clear Arguments to prevent Argument recycling
        # # self.node_fibre_sheet_normal = None   # Clear Arguments to prevent Argument recycling
        # if verbose:
        #     print('Check all nodes are connected')
        #     find_disconnected_nodes(node_index=np.arange(start=0, stop=self.node_xyz.shape[0], step=1), edge=self.edge)
        # # Check that all nodes in the endocardial surfaces are connected to each other
        # if verbose:
        #     print('Defining dense and sparse regions in the endocardium using ventricular coordinates')
        # self.__define_fast_endocardial_layer_dense_sparse_regions_vc()
        # Precompute smoothing
        self.node_neighbour_distance_weights = None

    def _get_unprocessed_node_vc_field(self, vc_name):
        raise NotImplementedError   # Prevent this function from being called in the future

    def get_edge(self):
        return self.edge

    def get_unfolded_edge(self):
        return self.unfolded_edge

    # TODO: refactor this so that it doesn't get created inside another function
    #TODO these two properties may not even exist! Should they just be part of the conduction system information?
    def get_node_lvendo_fast(self):
        return self.node_lvendo_fast

    def get_node_rvendo_fast(self):
        return self.node_rvendo_fast

    def get_neighbours(self):
        return self.neighbours


    # TODO Refactor and be consistent with the use of naming for "neighbours", also, take care of plural vs singular
    def __build_adjacentcies(self):
        # TODO Refactor and be consistent with the use of naming for "neighbours", also, take care of plural vs singular
        # Build adjacentcies
        edge_index_list_per_node = [[] for node_i in range(0, self.get_node_xyz().shape[0], 1)]
        unfolded_edge = self.get_unfolded_edge()
        for unfolded_edge_i in range(0, len(unfolded_edge), 1):
            edge_index_list_per_node[unfolded_edge[unfolded_edge_i, 0]].append(unfolded_edge_i)
        # make edge_index_array_per_node Numba friendly
        edge_index_array_per_node = [np.array(edge_index_list) for edge_index_list in edge_index_list_per_node]
        edge_index_list_per_node = None  # Clear Memory
        max_len_edge_index_array = 0
        for edge_index_array in edge_index_array_per_node:
            max_len_edge_index_array = max(max_len_edge_index_array, len(edge_index_array))
        edge_index_matrix = np.full((len(edge_index_array_per_node), max_len_edge_index_array), get_nan_value(),
                                    np.int32)  # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
        for edge_index_matrix_i in range(edge_index_matrix.shape[0]):
            edge_index_array = edge_index_array_per_node[edge_index_matrix_i]
            edge_index_matrix[edge_index_matrix_i, :edge_index_array.shape[0]] = edge_index_array
        self.neighbours = edge_index_matrix

    # def spatial_smoothing_of_time_field_using_adjacentcies(
    #         self, fibre_speed, sheet_speed, normal_speed,
    #                                                        ghost_distance_to_self, original_field_data):
    #     smoothed_field_data = np.zeros(original_field_data.shape)
    #     for node_i in range(0, self.get_node_xyz().shape[0], 1):
    #         node_unfolded_edge_meta_indexes = self.neighbours[node_i,
    #                                           :] != get_nan_value()  # These are the edges with the first element being the current node and the second element being the adjacent one.
    #         node_unfolded_edge_indexes = self.neighbours[node_i, node_unfolded_edge_meta_indexes]
    #         node_edge_indexes = node_unfolded_edge_indexes % self.get_edge().shape[0]  # Fold back the unfolded indexes
    #         node_neighbour_distances = np.linalg.norm(self.edge_vec[node_edge_indexes, :], axis=1)
    #         node_neighbour_distances = np.concatenate((node_neighbour_distances, np.asarray([ghost_distance_to_self])),
    #                                                   axis=0)
    #         node_neighbour_inverse_distances = 1. / node_neighbour_distances
    #         node_neighbour_distance_weights = node_neighbour_inverse_distances / np.sum(
    #             node_neighbour_inverse_distances)  # Normalise to add to 1 and flip values to favour small distances
    #         node_neighbour_indexes = self.unfolded_edge[node_unfolded_edge_indexes, 1]
    #         # This function takes into account that the second index of the data matrix is time
    #         node_neighbour_values = original_field_data[node_neighbour_indexes, :]
    #         current_node_value = original_field_data[node_i:node_i + 1, :]
    #         node_neighbour_values = np.concatenate((node_neighbour_values, current_node_value), axis=0)
    #         node_new_weighted_values = node_neighbour_values * node_neighbour_distance_weights[:, np.newaxis]
    #         node_new_value = np.sum(node_new_weighted_values, axis=0)
    #         smoothed_field_data[node_i, :] = node_new_value
    #     return smoothed_field_data

    def spatial_smoothing_of_time_field_using_adjacentcies_orthotropic_fibres(self,
                                                                              #fibre_speed, sheet_speed,
                                                                              #normal_speed, ghost_distance_to_self,
                                                                              original_field_data):
        smoothed_field_data = np.zeros(original_field_data.shape)
        for node_i in range(0, self.get_node_xyz().shape[0], 1):
            node_unfolded_edge_meta_indexes = self.neighbours[node_i,
                                              :] != get_nan_value()  # These are the edges with the first element being the current node and the second element being the adjacent one.
            node_unfolded_edge_indexes = self.neighbours[node_i, node_unfolded_edge_meta_indexes]
            # Check if the smoothing has been precomputed
            if self.node_neighbour_distance_weights is None:
                raise('This part of the code is dangerous, because it doesnt actually give the hability to adapt the '
                      'smoothing to new conduction speeds, but it seems it does that.')
                # TODO An alternative to this, could be to conduct the QT inference using a discrepancy metric that mostly
                # TODO Ignores the contribution of the smoothing and refine the population at a later stage using smoothing and
                # TODO inferred conduction speeds
                warn('Precomputing smoothing!')
                warn('CHANGES IN CONDUCTION SPEEDS CANNOT AFFECT THE SMOOTHING!!')
                self.precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(fibre_speed=fibre_speed,
                                                                                        sheet_speed=sheet_speed,
                                                                                        normal_speed=normal_speed,
                                                                           ghost_distance_to_self=ghost_distance_to_self)
            neighbour_distance_weights = self.node_neighbour_distance_weights[node_i]
            # Get the neighbour node indexes
            node_neighbour_indexes = self.unfolded_edge[node_unfolded_edge_indexes, 1]
            # This function takes into account that the second index of the data matrix is time
            node_neighbour_values = original_field_data[node_neighbour_indexes, :]
            current_node_value = original_field_data[node_i:node_i + 1, :]
            node_neighbour_values = np.concatenate((node_neighbour_values, current_node_value), axis=0)
            node_new_weighted_values = node_neighbour_values * neighbour_distance_weights[:, np.newaxis]
            node_new_value = np.sum(node_new_weighted_values, axis=0)
            smoothed_field_data[node_i, :] = node_new_value
        return smoothed_field_data

    def precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(self, fibre_speed, sheet_speed, normal_speed,
                                                                           ghost_distance_to_self):
        warn('CHANGES IN CONDUCTION SPEEDS DURING THE INFERENCE CANNOT AFFECT THE SMOOTHING!!')
        # TODO reduce the amount of precomputing so that changes in conduction speeds can affect the smoothing
        self.node_neighbour_distance_weights = []
        for node_i in range(0, self.get_node_xyz().shape[0], 1):
            node_unfolded_edge_meta_indexes = self.neighbours[node_i,
                                              :] != get_nan_value()  # These are the edges with the first element being the current node and the second element being the adjacent one.
            node_unfolded_edge_indexes = self.neighbours[node_i, node_unfolded_edge_meta_indexes]
            node_edge_indexes = node_unfolded_edge_indexes % self.get_edge().shape[0]  # Fold back the unfolded indexes
            # node_neighbour_distances_unweighted_by_fibre = np.linalg.norm(self.edge_vec[node_edge_indexes, :], axis=1)
            normalised_fibre_fibre_vector = self.get_node_normalised_fibre_fibre()[node_i, :]
            normalised_fibre_sheet_vector = self.get_node_normalised_fibre_sheet()[node_i, :]
            normalised_fibre_normal_vector = self.get_node_normalised_fibre_normal()[node_i, :]

            # print('orthogonal 1 ', np.dot(normalised_fibre_fibre_vector, normalised_fibre_sheet_vector))
            # print('orthogonal 2 ', np.dot(normalised_fibre_fibre_vector, normalised_fibre_normal_vector))
            # print('orthogonal 3 ', np.dot(normalised_fibre_normal_vector, normalised_fibre_sheet_vector))
            #
            # print('normalised 1 ', np.linalg.norm(normalised_fibre_fibre_vector))
            # print('normalised 2 ', np.linalg.norm(normalised_fibre_sheet_vector))
            # print('normalised 3 ', np.linalg.norm(normalised_fibre_normal_vector))
            #
            # print('node_neighbour_distances_unweighted_by_fibre ', node_neighbour_distances_unweighted_by_fibre.shape)
            edge_vector_list = self.edge_vec[node_edge_indexes, :]

            # Here we calculate the distances by first projecting to f,n,s coordinate space, and then stretching the distances by the
            # relative speed compared to the other directions
            neighbour_distances_weighted = np.zeros((edge_vector_list.shape[0]))
            max_speed = np.amax([fibre_speed, sheet_speed, normal_speed])
            for edge_i in range(edge_vector_list.shape[0]):
                neighbour_distances_weighted[edge_i] = math.sqrt(
                    (np.dot(edge_vector_list[edge_i, :], normalised_fibre_fibre_vector) * max_speed / fibre_speed) ** 2
                    + (np.dot(edge_vector_list[edge_i, :], normalised_fibre_sheet_vector) * max_speed / sheet_speed) ** 2
                    + (np.dot(edge_vector_list[edge_i, :], normalised_fibre_normal_vector) * max_speed / normal_speed) ** 2)
            # print('node_weighted_distance ', node_neighbour_distances_weighted)
            # print()
            # assert np.all(max_speed / np.amin([fibre_speed, sheet_speed, normal_speed])
            #               * node_neighbour_distances_unweighted_by_fibre > neighbour_distances_weighted), 'Not complying with equation!'
            neighbour_distances = np.concatenate((neighbour_distances_weighted,
                                                  np.asarray([ghost_distance_to_self])), axis=0)
            neighbour_inverse_distances = 1. / neighbour_distances
            neighbour_distance_weights = neighbour_inverse_distances / np.sum(
                neighbour_inverse_distances)  # Normalise to add to 1 and flip values to favour small distances
            self.node_neighbour_distance_weights.append(neighbour_distance_weights)


class EikonalGeometry(SimulationGeometry):
    """This private class extends the CardiacGeoTet class to include Eikonal specific data fields"""
    def __init__(self, cellular_model, celltype_vc_info, conduction_system, geometric_data_dir, resolution, subject_name, vc_name_list,
                 verbose):
        super().__init__(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info, conduction_system=conduction_system,
                         geometric_data_dir=geometric_data_dir, resolution=resolution, subject_name=subject_name,
                         vc_name_list=vc_name_list, verbose=verbose)
        # Exclude disconnected surface indexes
        if verbose:
            print('Check all nodes in the LV surface are connected')
        self.node_lvendo = find_disconnected_nodes(node_index=self.node_lvendo, edge=self.edge)
        if verbose:
            print('Check all nodes in the RV surface are connected')
        self.node_rvendo = find_disconnected_nodes(node_index=self.node_rvendo, edge=self.edge)
        # Interpolate fibres from nodes to edges
        edge_centre = calculate_centre(array_xyz=self.node_xyz, array_index=self.edge)
        edge_mapped_indexes = map_indexes(points_to_map_xyz=edge_centre, reference_points_xyz=self.node_xyz)
        edge_centre = None  # Clear Arguments to prevent Argument recycling
        self.edge_fibre_sheet_normal = self.get_node_normalised_fibre_sheet_normal()[edge_mapped_indexes, :, :]
        edge_mapped_indexes = None  # Clear Arguments to prevent Argument recycling
        # self.node_fibre_sheet_normal = None   # Clear Arguments to prevent Argument recycling
        if verbose:
            print('Check all nodes are connected')
            find_disconnected_nodes(node_index=np.arange(start=0, stop=self.node_xyz.shape[0], step=1), edge=self.edge)
        # Check that all nodes in the endocardial surfaces are connected to each other
        if verbose:
            print('Defining dense and sparse regions in the endocardium using ventricular coordinates')
        self.__define_fast_endocardial_layer_dense_sparse_regions_vc()

    def __define_fast_endocardial_layer_dense_sparse_regions_vc(self):
        # TODO make sure that this configuration is defined in a better place and carried over to where its needed
        # TODO Something is not working here! The root nodes should only be generated in the lower parts of the cavity
        '''This function defines the fast endocardial reagions in the ventricles,
        and also delimits where the Purkinje will be allowed to grow.'''

        # Load vc data
        vc_ab_cut_name = get_vc_ab_cut_name()
        vc_rt_name = get_vc_rt_name()
        node_vc_ab = self.get_node_vc_field(vc_name=vc_ab_cut_name)
        node_vc_rt = self.get_node_vc_field(vc_name=vc_rt_name)

        # Load endocardial data
        ## LV
        node_lvendo = self.get_node_lvendo()
        node_lvendo_vc_ab = node_vc_ab[node_lvendo]
        node_lvendo_vc_rt = node_vc_rt[node_lvendo]
        ## RV
        node_rvendo = self.get_node_rvendo()
        node_rvendo_vc_ab = node_vc_ab[node_rvendo]
        node_rvendo_vc_rt = node_vc_rt[node_rvendo]

        # Fast endocardial region in the ventricles
        ## Apex-to-Base Cut Note that, ab_cut ventricular coordinate will only cover from apex-to-base and will have invalid values for the valves (usually -1)
        apex_ab_cut_value = get_apex_ab_cut_value()  # Apex value
        ### LV
        lv_endo_fast_ab_min = apex_ab_cut_value
        lv_endo_fast_ab_max = get_endo_fast_and_purkinje_max_ab_cut_threshold()
        assert lv_endo_fast_ab_min < lv_endo_fast_ab_max
        ### RV
        rv_endo_fast_ab_min = apex_ab_cut_value
        rv_endo_fast_ab_max = get_endo_fast_and_purkinje_max_ab_cut_threshold()
        assert rv_endo_fast_ab_min < rv_endo_fast_ab_max
        ## Rotational
        valid_vc_rt_range = get_valid_vc_rt_range()  # Just check that the values are valid in the candidate root nodes
        ### LV
        lv_endo_fast_rt_min = valid_vc_rt_range[0]
        lv_endo_fast_rt_max = valid_vc_rt_range[1]
        assert lv_endo_fast_rt_min < lv_endo_fast_rt_max
        ### RV
        rv_endo_fast_rt_min = valid_vc_rt_range[0]
        rv_endo_fast_rt_max = valid_vc_rt_range[1]
        assert rv_endo_fast_rt_min < rv_endo_fast_rt_max

        # Initialise data structure
        lv_endo_fast_meta_index = np.zeros(self.get_node_lvendo().shape, dtype=bool)
        rv_endo_fast_meta_index = np.zeros(self.get_node_rvendo().shape, dtype=bool)

        # LV endo fast available
        lv_endo_fast_mask = ((node_lvendo_vc_ab >= lv_endo_fast_ab_min) &
                             (node_lvendo_vc_ab <= lv_endo_fast_ab_max) &
                             (node_lvendo_vc_rt >= lv_endo_fast_rt_min) &
                             (node_lvendo_vc_rt <= lv_endo_fast_rt_max)
                             )
        lv_endo_fast_meta_index[lv_endo_fast_mask] = True
        lv_endo_fast_index = node_lvendo[lv_endo_fast_meta_index]
        lv_endo_fast_meta_index = None   # Clear memory and prevent recycling
        # RV apical
        rv_endo_fast_mask = ((node_rvendo_vc_ab >= rv_endo_fast_ab_min) &
                             (node_rvendo_vc_ab <= rv_endo_fast_ab_max) &
                             (node_rvendo_vc_rt >= rv_endo_fast_rt_min) &
                             (node_rvendo_vc_rt <= rv_endo_fast_rt_max)
                             )
        rv_endo_fast_meta_index[rv_endo_fast_mask] = True
        rv_endo_fast_index = node_rvendo[rv_endo_fast_meta_index]
        rv_endo_fast_meta_index = None  # Clear memory and prevent recycling

        # self.node_lvendo_fast = lv_endo_fast_index
        # self.node_rvendo_fast = rv_endo_fast_index


        # Dense thresholds # The lower third of the LV and the lower sixth of the RV are more densely connected
        ## Apex-to-Base
        ### LV
        lv_dense_ab_min = lv_endo_fast_ab_min
        lv_dense_ab_max = get_lv_apical_ab_cut_threshold()
        assert lv_dense_ab_min < lv_dense_ab_max
        # lv_dense_ab_range = [lv_dense_ab_min, lv_dense_ab_max]
        ### RV
        rv_dense_ab_min = rv_endo_fast_ab_min
        rv_dense_ab_max = get_rv_apical_ab_cut_threshold()
        assert rv_dense_ab_min < rv_dense_ab_max
        # rv_dense_ab_range = [rv_dense_ab_min, rv_dense_ab_max]
        ## Rotational
        ### LV
        ### RV
        rv_dense_rt_min = get_freewall_posterior_rt_value()
        rv_dense_rt_max = get_freewall_anterior_rt_value()
        assert rv_dense_rt_min < rv_dense_rt_max
        # Sparse thresholds
        ## Apex-to-Base
        ### LV
        lv_sparse_ab_min = lv_dense_ab_max
        lv_sparse_ab_max = lv_endo_fast_ab_max
        assert lv_sparse_ab_min < lv_sparse_ab_max
        # lv_sparse_ab_range = [lv_dense_ab_max, lv_sparse_ab_max]
        ### RV
        rv_sparse_ab_min = rv_dense_ab_max
        rv_sparse_ab_max = rv_endo_fast_ab_max
        assert rv_sparse_ab_min < rv_sparse_ab_max
        # rv_sparse_ab_range = [rv_dense_ab_max, rv_sparse_ab_max]
        ## Rotational
        ### LV
        ### RV

        # Define the actual fast regions in the endocardium while respecting the generic bounds
        # Initialise data structure
        lv_dense_meta_index = np.zeros(lv_endo_fast_index.shape, dtype=bool)
        rv_dense_meta_index = np.zeros(rv_endo_fast_index.shape, dtype=bool)

        # Load VC endocardial fields
        ## LV
        node_lv_fast_endo_vc_ab = node_vc_ab[lv_endo_fast_index]
        # node_lv_fast_endo_vc_rt = node_vc_rt[lv_endo_fast_index]
        ## RV
        node_rv_fast_endo_vc_ab = node_vc_ab[rv_endo_fast_index]
        node_rv_fast_endo_vc_rt = node_vc_rt[rv_endo_fast_index]

        # Clear memory and prevent recycling
        node_vc_ab = None
        node_vc_rt = None

        # TODO check that python is doing what I expect
        print('hey, check this')
        lv_desnse_apical_mask = ((node_lv_fast_endo_vc_ab >= lv_dense_ab_min) &
                                 (node_lv_fast_endo_vc_ab <= lv_dense_ab_max))
        lv_dense_meta_index[lv_desnse_apical_mask] = True
        # RV apical
        rv_dense_apical_mask = ((node_rv_fast_endo_vc_ab >= rv_dense_ab_min) &
                                (node_rv_fast_endo_vc_ab <= rv_dense_ab_max))
        # The free wall of the RV is more densely connected
        rv_freewall_mask = ((rv_dense_rt_min <= node_rv_fast_endo_vc_rt) &
                            (node_rv_fast_endo_vc_rt <= rv_dense_rt_max) &
                            (node_rv_fast_endo_vc_ab <= rv_sparse_ab_max))
        # 27/01/2023 - Changed the RV ab threshold to 0.2 for the RV as from Myerburg's paper
        # (TODO: check this in Paraview!)
        rv_dense_meta_index[rv_dense_apical_mask] = True
        rv_dense_meta_index[rv_freewall_mask] = True
        lv_dense_index = lv_endo_fast_index[lv_dense_meta_index]
        rv_dense_index = rv_endo_fast_index[rv_dense_meta_index]
        self.is_dense_endocardial = np.logical_or(np.all(np.isin(self.edge, lv_dense_index), axis=1),
                                                  np.all(np.isin(self.edge, rv_dense_index), axis=1))
        # Sparse
        lv_sparse_meta_index = np.zeros(lv_endo_fast_index.shape, dtype=bool)
        rv_sparse_meta_index = np.zeros(rv_endo_fast_index.shape, dtype=bool)
        # The lower third of the LV and the lower sixth of the RV are more sparsely connected
        lv_sparse_apical_mask = ((node_lv_fast_endo_vc_ab >= lv_sparse_ab_min) &
                                 (node_lv_fast_endo_vc_ab <= lv_sparse_ab_max) &
                                 (node_lv_fast_endo_vc_ab <= lv_sparse_ab_max))
        lv_sparse_meta_index[lv_sparse_apical_mask] = True
        lv_sparse_meta_index[lv_dense_meta_index] = False # it cannot be sparse and also dense
        rv_sparse_apical_mask = ((node_rv_fast_endo_vc_ab >= rv_sparse_ab_min) &
                                (node_rv_fast_endo_vc_ab <= rv_sparse_ab_max) &
                                (node_rv_fast_endo_vc_ab <= rv_sparse_ab_max))
        rv_sparse_meta_index[rv_sparse_apical_mask] = True
        rv_sparse_meta_index[rv_dense_meta_index] = False  # it cannot be sparse and also dense
        lv_sparse_index = lv_endo_fast_index[lv_sparse_meta_index]
        rv_sparse_index = rv_endo_fast_index[rv_sparse_meta_index]
        self.is_sparse_endocardial = np.logical_or(np.all(np.isin(self.edge, lv_sparse_index), axis=1),
                                                   np.all(np.isin(self.edge, rv_sparse_index), axis=1))
        # Purkinje area
        self.node_lvendo_fast = np.concatenate((lv_sparse_index, lv_dense_index), axis=0)
        self.node_rvendo_fast = np.concatenate((rv_sparse_index, rv_dense_index), axis=0)
        # np.logical_or(np.all(np.isin(self.edge, self.node_lvendo), axis=1),
        #               np.all(np.isin(self.edge, self.node_rvendo), axis=1))



# def temporal_smoothing_of_time_field(original_field_data, past_present_smoothing_window):
#     """past_present_smoothing_window: should be a 1D array of two positions (t-1, t) with the weight values
#     to be used. The weights must be positive values, and will be made to add to 1."""
#     warn('Why is this used?')
#     smoothing_window = past_present_smoothing_window/np.sum(past_present_smoothing_window)
#     smoothed_field_data = np.copy(original_field_data)
#     for time_i in range(1, smoothed_field_data.shape[1], 1):  # 1 index to do as a special case
#         smoothed_field_data[:, time_i] = np.sum(smoothed_field_data[:, time_i-1:time_i+1]*smoothing_window, axis=1)
#     return smoothed_field_data


def calculate_centre(array_xyz, array_index):
    return np.mean(array_xyz[array_index, :], axis=1)


# TODO: From chatgpt: needs correcting but it's a good start (more or less)
def F_cobiveco_scar_3d(mesh, node_rnd, scar_rad, peri_rad, scar_center_tv):
    node_dist_scar = (mesh['cobiveco_points'][:, 0] - mesh['cobiveco_points'][node_rnd, 0])**2/scar_rad[0]**2 + (mesh['cobiveco_points'][:, 1] - mesh['cobiveco_points'][node_rnd, 1])**2/scar_rad[1]**2 + (mesh['cobiveco_points'][:, 2] - mesh['cobiveco_points'][node_rnd, 2])**2/scar_rad[2]**2
    node_dist_peri = (mesh['cobiveco_points'][:, 0] - mesh['cobiveco_points'][node_rnd, 0])**2/peri_rad[0]**2 + (mesh['cobiveco_points'][:, 1] - mesh['cobiveco_points'][node_rnd, 1])**2/peri_rad[1]**2 + (mesh['cobiveco_points'][:, 2] - mesh['cobiveco_points'][node_rnd, 2])**2/peri_rad[2]**2

    mesh_properties = mesh['pointData']

    xyzscar = ((node_dist_scar <= 1) & (mesh_properties['rt'] >= 0.68) & (mesh_properties['rt'] <= 0.98)) | (
                (node_dist_scar <= 1) & ((mesh_properties['rt'] < 0.68) | (mesh_properties['rt'] > 0.98)) & (
                    mesh_properties['tv'] == scar_center_tv))
    xyzperi = ((node_dist_peri <= 1) & (mesh_properties['rt'] >= 0.68) & (mesh_properties['rt'] <= 0.98)) | (
                (node_dist_peri <= 1) & ((mesh_properties['rt'] < 0.68) | (mesh_properties['rt'] > 0.98)) & (
                    mesh_properties['tv'] == scar_center_tv))

    triscar = np.all(xyzscar[mesh['cells']], axis=1)
    triperi = np.all(xyzperi[mesh['cells']], axis=1)

    xyzperi = xyzperi & ~xyzscar
    triperi = triperi & ~triscar

    mesh['xyzIds'] = 1 * xyzscar + 2 * xyzperi
    mesh['triIds'] = 1 * triscar + 2 * triperi


# TODO: From Lei: Check if there is a newer version available
def generate_scar():
    import os
    import numpy as np

    foldname = 'E:\2022_ECG_inference\dataset_MI_inference\data_cobiveco_mesh\\'
    foldname_new = foldname.replace('data_cobiveco_mesh', 'data_cobiveco_scar')

    a = [f.name for f in os.scandir(foldname) if f.name.startswith('1')]
    MI_type_index = 1

    for i in range(len(a)):
        meshName = '1000268'

        new_path = foldname_new + meshName + '\\'
        if not os.path.isdir(new_path):
            os.makedirs(new_path)

        vol = np.genfromtxt(foldname + meshName + '\\' + meshName + '_heart_cobiveco.vtu', delimiter=',')

        for MI_type_index in range(7):
            if MI_type_index == 3:
                vol.cobiveco_points = np.column_stack(
                    (vol.pointData.tm, vol.pointData.tv, vol.pointData.ab, vol.pointData.rt, vol.pointData.rtCos))
            else:
                vol.cobiveco_points = np.column_stack(
                    (vol.pointData.tm, vol.pointData.tv, vol.pointData.ab, vol.pointData.rt))

            if MI_type_index == 1:
                MI_type = 'A1'
                cobiveco_coord = [1, 0, 7 / 14, 10 / 12]
                scar_center = np.argwhere(np.all(np.isclose(vol.cobiveco_points, cobiveco_coord), axis=1))
                scar_rad = [3, 0.2, 0.07]
                peri_rad = [3, 0.25, 0.12]
            elif MI_type_index == 2:
                MI_type = 'A2'
                cobiveco_coord = [1, 0, 0, 0]
                scar_center = np.argwhere(np.all(np.isclose(vol.cobiveco_points, cobiveco_coord), axis=1))
                scar_rad = [3, 0.3, 3]
                peri_rad = [3, 0.4, 3]
            elif MI_type_index == 3:
                MI_type = 'A3'
                cobiveco_coord = [1, 0, 3 / 7, 0.92, 0.87]
                scar_center = np.argwhere(np.all(np.isclose(vol.cobiveco_points, cobiveco_coord), axis=1))
                scar_rad = [3, 0.25, 0.5]
                peri_rad = [3, 0.3, 0.7]
            elif MI_type_index == 4:
                MI_type = 'A4'
                cobiveco_coord = [1, 0, 4 / 7, 2 / 12]
                # scar_center = np.


def get_edges_for_node_indexes(node_index, edge):
    # Select edges that only include node_index.
    new_edges = edge[np.all(np.isin(edge, node_index), axis=1), :]
    # Reindex these arrays.
    new_edges[:, 0] = np.asarray([np.flatnonzero(node_index == node_id)[0] for node_id in new_edges[:, 0]]).astype(int)
    new_edges[:, 1] = np.asarray([np.flatnonzero(node_index == node_id)[0] for node_id in new_edges[:, 1]]).astype(int)
    return np.unique(new_edges, axis=1)


def find_disconnected_nodes(node_index, edge):
    node_index = np.unique(node_index)
    edge = np.unique(edge, axis=0)
    meta_edge = get_edges_for_node_indexes(node_index=node_index, edge=edge)
    aux_edge = edge[np.all(np.isin(edge, node_index), axis=1), :]
    # print('meta_edge ', node_index[meta_edge])
    # This could fail if the first root node of the endocardial surface is not connected to the rest
    next_nodes = np.random.randint(low=0, high=node_index.shape[0], size=1).tolist()
    # print('next_nodes ', np.array(next_nodes))
    # print('global ', node_index[np.array(next_nodes)])
    # print('new test ', aux_edge[np.any(np.isin(aux_edge, node_index[np.array(next_nodes)]), axis=1), :])
    # print('neighbours ', edge[np.any(np.isin(edge, node_index[np.array(next_nodes)]), axis=1), :])
    # print('neighbours lv ', meta_edge[np.any(np.isin(meta_edge, np.array(next_nodes)), axis=1), :])
    # Check mesh integrity
    unfolded_edges = np.concatenate((meta_edge, np.flip(meta_edge, axis=1))).astype(int)
    aux = [[] for i in range(0, node_index.shape[0], 1)]
    for next_node_i in range(0, len(unfolded_edges), 1):
        aux[unfolded_edges[next_node_i, 0]].append(next_node_i)
    neighbours = [np.array(n).astype(int) for n in aux]
    # print('neigh ', neighbours[next_nodes[0]])
    aux = None  # Clear Memory
    connected_node_meta_indexes = np.zeros((node_index.shape[0]), dtype=np.bool_)
    while np.sum(connected_node_meta_indexes) < len(connected_node_meta_indexes):
        if len(next_nodes) == 0:
            break
        for next_node_i in range(len(next_nodes)):
            active_node = next_nodes.pop()
            if not connected_node_meta_indexes[active_node]:
                connected_node_meta_indexes[active_node] = True
                for edge_i in neighbours[active_node]:
                    node = unfolded_edges[edge_i, 1]
                    if not connected_node_meta_indexes[node]:
                        next_nodes.append(node)
        # print('-----')
        # print(len(remove_nodes)-np.sum(remove_nodes))
        # print(len(next_nodes))
        next_nodes = np.sort(np.unique(next_nodes)).tolist()
    # print('Remomve this many nodes: ' + str(np.sum(connected_node_meta_indexes)))
    # print('node_index ', node_index.shape)
    print('Remomve this many nodes: ' + str(len(connected_node_meta_indexes) - np.sum(connected_node_meta_indexes)))
    # return node_index[np.logical_not(connected_node_meta_indexes)]
    return node_index[connected_node_meta_indexes]


def correct_and_normlise_ortho_fibre(fibre, sheet, normal):
    # If the fibres are not well defined, this can cause singular matrix errors, because the determinant of the fibre matrix
    # becomes zero and then the Eikonal cannot compute the inverse of that matrix. To amend this issue, we replace incomplete
    # fibres by values at random that comply with the orthotropic nature of our fibres in xyz.
    fibre_norm = np.linalg.norm(fibre, axis=1)
    sheet_norm = np.linalg.norm(sheet, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    # This part of the code warns the user that some fibre items are being replaced by random values.
    if np.sum(fibre_norm <= 0.):
        warn('Number of wrong fibre ' + str(np.sum(fibre_norm <= 0.)))
    if np.sum(sheet_norm <= 0.):
        warn('Number of wrong sheet ' + str(np.sum(sheet_norm <= 0.)))
    if np.sum(normal_norm <= 0.):
        warn('Number of wrong normal ' + str(np.sum(normal_norm <= 0.)))
    # The condition needs to match in shape both x and y arguments of the function numpy.where(condition, x, y)
    # For the fibre, if it's not defined, then we define it at random
    fibre = np.where(np.repeat(fibre_norm[:, np.newaxis] > 0., repeats=fibre.shape[1], axis=1), fibre, np.random.rand(fibre.shape[0], fibre.shape[1]))
    # For the sheet, if it's not defined, then we define it at random but orthogonal to the fibre
    sheet_random = np.cross(fibre, np.random.rand(sheet.shape[0], sheet.shape[1]), axis=1)
    sheet = np.where(np.repeat(sheet_norm[:, np.newaxis] > 0., repeats=sheet.shape[1], axis=1), sheet, sheet_random)
    # For the normal, if it's not defined, then we define it using the cross product of the other two vectors
    normal_random = np.cross(fibre, sheet, axis=1)
    normal = np.where(np.repeat(normal_norm[:, np.newaxis] > 0., repeats=normal.shape[1], axis=1), normal, normal_random)
    # Recompute normals to make sure that everything has been fixed
    fibre_norm = np.linalg.norm(fibre, axis=1)
    sheet_norm = np.linalg.norm(sheet, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    # print('count ', np.sum(fibre_norm <= 0.))
    if np.sum(fibre_norm <= 0.):
        warn('This was not fixed!\nNumber of wrong fibre ' + str(np.sum(fibre_norm <= 0.)))
    if np.sum(sheet_norm <= 0.):
        warn('This was not fixed!\nNumber of wrong sheet ' + str(np.sum(sheet_norm <= 0.)))
    if np.sum(normal_norm <= 0.):
        warn('This was not fixed!\nNumber of wrong normal ' + str(np.sum(normal_norm <= 0.)))
    # Normalise fibres so that the normals are 1.0
    return fibre / fibre_norm[:, np.newaxis], sheet / sheet_norm[:, np.newaxis], normal / normal_norm[:, np.newaxis]


# TEST CASE FOR CREATING A NEW GEOMETRY
if __name__ == '__main__':
    import os

    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields
    from conduction_system import EmptyConductionSystem
    # from utils import get_vc_ab_name, get_vc_aprt_name, get_vc_rt_name, get_vc_tm_name, get_vc_tv_name, get_vc_rvlv_name
    from cellular_models import CellularModelBiomarkerDictionary

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
    hyperparameter_dict = {}  # Save hyperparameters for reproducibility
    ####################################################################################################################
    # Step 1: Define paths and other environment variables.
    # General settings:
    anatomy_subject_name = 'DTI004'
    print('anatomy_subject_name: ', anatomy_subject_name)
    resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    geometric_data_dir = data_dir + 'geometric_data/'
    # APD dictionary configuration:
    ep_model = 'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    cellular_stim_amp = 11
    cellular_model_convergence = 'not_converged'
    stimulation_protocol = 'diffusion'
    cellular_data_relative_path = 'cellular_data/' + cellular_model_convergence + '_' + stimulation_protocol + '_' + str(
        cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model + '/'
    cellular_data_dir_complete = data_dir + cellular_data_relative_path
    # Output Paths:
    results_dir_root = path_dict["results_path"]
    results_dir = results_dir_root + anatomy_subject_name + '/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # Directory to save the configuration of the inference before it runs to allow manual inspection:
    visualisation_dir = results_dir + 'geometry_checkpoint/'  # GLOBAL DEFINITION
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    # Clear Arguments to prevent Argument recycling
    cellular_data_relative_path = None
    cellular_stim_amp = None
    clinical_data_dir_tag = None
    clinical_data_filename = None
    cellular_model_convergence = None
    data_dir = None
    ecg_subject_name = None
    ep_model = None
    experiment_type = None
    gradient_ion_channel_list = None
    results_dir = None
    results_dir_root = None
    # ####################################################################################################################
    # # Step 2: Generate an Eikonal-friendly geometry.
    # # Argument setup: (in Alphabetical order)
    # anatomy_subject_name = 'rodero_05'  # 'rodero_01' # 'DTI004' #
    # resolution = 'coarse'
    # vc_name_list = ['ab', 'tm', 'rt', 'tv']#, 'aprt', 'rvlv']
    # # Create geometry with a dummy conduction system to allow initialising the geometry.
    # geometry = EikonalGeometry(conduction_system=EmptyConductionSystem(verbose=verbose), geometric_data_dir=geometric_data_dir, resolution=resolution, subject_name=anatomy_subject_name,
    #                            vc_name_list=vc_name_list, verbose=verbose)
    # # Save geometry into visualisation directory
    # visualisation_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + resolution + '/ensight/'
    # if not os.path.exists(visualisation_dir):
    #     os.mkdir(visualisation_dir)
    # write_ensight(subject_name=anatomy_subject_name, visualisation_dir=visualisation_dir, geometry=geometry, verbose=verbose)
    # # Clear Arguments to prevent Argument recycling
    # geometric_data_dir = None
    # geometry = None
    # resolution = None
    # anatomy_subject_name = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
    print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
    # Arguments for cellular model:
    # APD dictionary configuration:
    cellular_model_name = 'torord_calibrated_pom_1000Hz'
    endo_celltype_name = 'endo'
    epi_celltype_name = 'epi'
    list_celltype_name = [endo_celltype_name, epi_celltype_name]
    biomarker_upstroke_name = 'activation_time'  # TODO consider chaning to something different with the name upstroke in it
    biomarker_apd90_name = 'apd90'
    biomarker_celltype_name = 'celltype'
    # Create cellular model instance.
    cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                      biomarker_apd90_name=biomarker_apd90_name,
                                                      biomarker_celltype_name=biomarker_celltype_name,
                                                      cellular_data_dir=cellular_data_dir_complete,
                                                      cellular_model_name=cellular_model_name,
                                                      list_celltype_name=list_celltype_name, verbose=verbose)
    apd_min_min, apd_max_max = cellular_model.get_biomarker_range(biomarker_name=biomarker_apd90_name)
    print('apd_min_min ', apd_min_min)
    print('apd_max_max ', apd_max_max)
    assert apd_max_max > apd_min_min
    # Clear Arguments to prevent Argument recycling
    biomarker_apd90_name = None
    biomarker_celltype_name = None
    biomarker_upstroke_name = None
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_model_name = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 3: Vendtricular Coordinate Geometry fields.
    # Arguments for VC information:
    ## AB
    vc_ab_name = get_vc_ab_name()
    ## APRT
    vc_aprt_name = get_vc_aprt_name()
    ## RT
    vc_rt_name = get_vc_rt_name()
    ## RVLV
    vc_rvlv_name = get_vc_rvlv_name()
    ## TM
    vc_tm_name = get_vc_tm_name()
    ## TV
    vc_tv_name = get_vc_tv_name()
    vc_name_list = [vc_ab_name, vc_aprt_name, vc_rt_name, vc_rvlv_name, vc_tm_name, vc_tv_name]
    # Pre-assign celltype spatial correspondence.
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 1.]}}
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    celltype_vc_info = None
    cellular_model = None
    geometric_data_dir = None
    ####################################################################################################################
    # Step 9: Visualise geometric fields.
    # Arguments for geometric information:
    vc_nodefield_list = []
    for vc_name in vc_name_list:
        vc_nodefield_list.append(geometry.node_vc[vc_name])
    # Save geometry as a check point
    write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=vc_nodefield_list,
                                          node_field_name_list=vc_name_list,
                                          subject_name=anatomy_subject_name + '_' + resolution + '_checkpoint',
                                          verbose=verbose,
                                          visualisation_dir=visualisation_dir)
    print('Saved geometry before inference in ', visualisation_dir)
    # Clear Arguments to prevent Argument recycling
    geometry = None
    visualisation_dir = None
    raw_ukb_geometric_data_path = None
    print('End of Geometry test case')
    ####################################################################################################################

# EOF

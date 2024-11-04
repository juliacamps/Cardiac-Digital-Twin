"""Interpolate GKs to cube centres for MonoAlg3D simulations"""
import os
import sys
from warnings import warn

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import pymp, multiprocessing

if __name__ == '__main__':
    anatomy_subject_name_list = ['DTI024', 'DTI004', 'DTI032']
    for anatomy_subject_name in anatomy_subject_name_list:
        if len(sys.argv) < 2:
            anatomy_subject_name = anatomy_subject_name
            ecg_subject_name = anatomy_subject_name   # Allows using a different ECG for the personalisation than for the anatomy
        else:
            anatomy_subject_name = sys.argv[1]
            ecg_subject_name = sys.argv[1]
        print('anatomy_subject_name: ', anatomy_subject_name)
        print('ecg_subject_name: ', ecg_subject_name)
        # ####################################################################################################################
        # # TODO THIs kills all the processes every time you run the inference because it tries to exceed the allowed memory
        # # Set the memory limit to 100GB (in bytes) - Heartsrv has 126GB
        # memory_limit = 60 * 1024 * 1024 * 1024
        # # Set the memory limit for the current process
        # resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        # ####################################################################################################################
        print(
            'Caution, all the hyper-parameters are set assuming resolutions of 1000 Hz in all time-series.')  # TODO: make hyper-parameters relative to the time series resolutions.
        # TODO: enable having different resolutions in different time-series in the code.
        # Get the directory of the script
        script_directory = os.path.dirname(os.path.realpath(__file__))
        print('Script directory:', script_directory)
        # Change the current working directory to the script dierctory
        os.chdir(script_directory)
        working_directory = os.getcwd()
        print('Working directory:', working_directory)
        # Clear Arguments to prevent Argument recycling
        script_directory = None
        working_directory = None
        ####################################################################################################################
        # LOAD FUNCTIONS AFTER DEFINING THE WORKING DIRECTORY
        from conduction_system import DjikstraConductionSystemVC, EmptyConductionSystem, PurkinjeSystemVC
        from ecg_functions import PseudoEcgTetFromVM, get_cycle_length
        from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet, RawEmptyCardiacGeoPointCloud, \
            SimulationGeometry
        from propagation_models import EikonalDjikstraTet, PrescribedLAT
        from simulator_functions import SimulateECG, SimulateEP
        from adapter_theta_params import AdapterThetaParams, RoundTheta
        from discrepancy_functions import DiscrepancyECG, BiomarkerFromOnlyECG
        from evaluation_functions import DiscrepancyEvaluator, ParameterSimulator
        from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
        from electrophysiology_functions import ElectrophysiologyAPDmap
        from path_config import get_path_mapping
        from io_functions import write_geometry_to_ensight_with_fields, read_dictionary, save_ecg_to_csv, \
        export_ensight_timeseries_case, save_pandas, save_csv_file, read_ecg_from_csv, read_csv_file, write_purkinje_vtk, \
        write_root_node_csv, read_pandas
        from utils import map_indexes, remap_pandas_from_row_index, get_qtc_dur_name, \
        get_t_pe_name, get_t_peak_name, get_tpeak_dispersion_name, get_qtpeak_dur_name, \
        get_t_polarity_name, get_root_node_meta_index_population_from_pandas, translate_from_pandas_to_array, \
        get_purkinje_speed_name, get_lat_biomarker_name, get_repol_biomarker_name, get_best_str, \
        convert_from_monoalg3D_to_cm_and_translate, get_apd90_biomarker_name, get_sf_iks_biomarker_name, get_nan_value
        from postprocess_functions import generate_repolarisation_map, visualise_ecg, generate_activation_map

        print('All imports done!')
        ####################################################################################################################
        # Load the path configuration in the current server
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
        inference_resolution = 'coarse'
        monodomain_resolution = 'hex500'
        verbose = True
        # Input Paths:
        data_dir = path_dict["data_path"]
        cellular_data_dir = data_dir + 'cellular_data/'
        geometric_data_dir = data_dir + 'geometric_data/'
        results_dir_root = path_dict["results_path"]
        # Intermediate Paths: # e.g., results from the QRS inference
        experiment_type = 'personalisation'
        # TODO have a single definition of the heart rate or the cycle length for the whole pipeline
        # TODO the heart rate is clinical data and should be stored in the clinical data folder
        if anatomy_subject_name == 'DTI024':  # Subject 1
            heart_rate = 66
        elif anatomy_subject_name == 'DTI004':  # Subject 2
            heart_rate = 48
        elif anatomy_subject_name == 'DTI032':  # Subject 3
            heart_rate = 74
        cycle_length = get_cycle_length(heart_rate=heart_rate)
        cycle_length_str = str(int(cycle_length))
        ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_' + cycle_length_str
        # ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_909' #'GKs5_GKr0.6_tjca60'  # 'MitchellSchaefferEP' #'no_rescale' #'GKs5_GKr0.6_tjca60'
        gradient_ion_channel_list = ['sf_IKs']
        gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
        # Build results folder structure
        results_dir_part = results_dir_root + experiment_type + '_data/'
        assert os.path.exists(results_dir_part)  # Path should already exist from running the Twave inference
        results_dir_part = results_dir_part + anatomy_subject_name + '/'
        assert os.path.exists(results_dir_part)  # Path should already exist from running the Twave inference
        results_dir_part_twave = results_dir_part + 'twave_' + gradient_ion_channel_str + '_' + ep_model_twave_name + '/'
        assert os.path.exists(results_dir_part_twave)  # Path should already exist from running the Twave inference
        # Use date to name the result folder to preserve some history of results
        current_month_text = 'Jun'#datetime.now().strftime('%h')  # e.g., Feb
        current_year_full = datetime.now().strftime('%Y')  # e.g., 2024
        date_str = current_month_text + '_' + current_year_full
        results_dir_twave = results_dir_part_twave + date_str + '_fixed_filter/'
        assert os.path.exists(results_dir_twave)  # Path should already exist from running the Twave inference
        results_dir_part_twave = None  # Clear Arguments to prevent Argument recycling
        # Read hyperparamter dictionary
        hyperparameter_result_file_name = results_dir_twave + anatomy_subject_name + '_' + inference_resolution + '_hyperparameter.txt'
        hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
        # Load QRS inference result # Intermediate Paths: # e.g., results from the QRS inference
        ep_model_qrs_name = hyperparameter_dict['ep_model_qrs']
        results_dir_part_qrs = results_dir_part + 'qrs_' + ep_model_qrs_name + '/'
        ep_model_qrs_name = None  # Clear Arguments to prevent Argument recycling
        assert os.path.exists(results_dir_part_qrs)  # Path should already exist from running the QRS inference
        results_dir_part = None  # Clear Arguments to prevent Argument recycling
        results_dir_qrs = results_dir_part_qrs + date_str + '/best_discrepancy/'
        assert os.path.exists(results_dir_qrs)  # Path should already exist from running the QRS inference
        results_dir_part_qrs = None  # Clear Arguments to prevent Argument recycling
        qrs_lat_prescribed_filename = hyperparameter_dict['qrs_lat_prescribed_filename']
        qrs_lat_prescribed_filename_path = results_dir_qrs + qrs_lat_prescribed_filename
        results_dir_qrs = None  # Clear Arguments to prevent Argument recycling
        if not os.path.isfile(qrs_lat_prescribed_filename_path):
            print('qrs_lat_prescribed_filename_path: ', qrs_lat_prescribed_filename_path)
            raise Exception(
                "This inference needs to be run after the QRS inference and need the correct path with those results.")
        # Continue defining results paths and configuration
        result_tag = hyperparameter_dict['result_tag']
        parameter_result_file_name = results_dir_twave + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_parameter_population.csv'
        # Output Paths:
        # Uncertainty Translation to Monodomain
        for_monodomain_dir = results_dir_twave + 'for_translation_to_monodomain_recalculate/'
        if not os.path.exists(for_monodomain_dir):
            os.mkdir(for_monodomain_dir)
        for_monodomain_biomarker_result_file_name_start = for_monodomain_dir + anatomy_subject_name + '_' \
                                                          + monodomain_resolution + '_nodefield_' + result_tag + '-biomarker_'
        for_monodomain_biomarker_result_file_name_end = '.csv'
        for_monodomain_parameter_population_file_name = for_monodomain_dir + anatomy_subject_name + '_' \
                                                        + inference_resolution + '_' + result_tag + '_selected_parameter_population.csv'
        for_monodomain_figure_result_file_name = for_monodomain_dir + anatomy_subject_name + '_' \
                                                 + inference_resolution + '_' + result_tag + '_population.png'
        translation_dir_tag = for_monodomain_dir + 'translation_'
        # Precomputed subfolder
        inference_precomputed_dir = for_monodomain_dir + 'precomputed/'
        if not os.path.exists(inference_precomputed_dir):
            os.mkdir(inference_precomputed_dir)
        inference_ecg_uncertainty_population_filename = inference_precomputed_dir + anatomy_subject_name + '_' \
                                                        + inference_resolution + '_' + result_tag + '_selected_pseudo_ecg_population.csv'
        inference_ecg_inferred_population_filename = inference_precomputed_dir + anatomy_subject_name + '_' \
                                                     + inference_resolution + '_' + result_tag + '_inferred_pseudo_ecg_population.csv'
        inference_repol_uncertainty_population_filename = inference_precomputed_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_selected_repol_population.csv'
        preprocessed_clinical_ecg_file_name = inference_precomputed_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_ecg_clinical.csv'
        inference_precomputed_dir = None  # Clear Arguments to prevent Argument recycling
        # Module names:
        propagation_module_name = 'propagation_module'
        electrophysiology_module_name = 'electrophysiology_module'
        # Read hyperparameters
        clinical_data_filename = hyperparameter_dict['clinical_data_filename']
        clinical_data_filename_path = data_dir + clinical_data_filename
        # Clear Arguments to prevent Argument recycling
        clinical_data_filename = None
        data_dir = None
        ecg_subject_name = None
        qrs_lat_prescribed_filename = None
        results_dir_root = None
        ####################################################################################################################
        # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
        print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
        # Arguments for cellular model:
        # Read hyperparameters
        biomarker_apd90_name = hyperparameter_dict['biomarker_apd90_name']
        biomarker_celltype_name = hyperparameter_dict['biomarker_celltype_name']
        biomarker_upstroke_name = hyperparameter_dict['biomarker_upstroke_name']
        cellular_model_name = hyperparameter_dict['cellular_model_name']
        cellular_stim_amp = hyperparameter_dict['cellular_stim_amp']
        cellular_model_convergence = hyperparameter_dict['cellular_model_convergence']
        ep_model_twave_name = hyperparameter_dict['ep_model_twave']
        list_celltype_name = hyperparameter_dict['list_celltype_name']
        stimulation_protocol = hyperparameter_dict['stimulation_protocol']
        cellular_data_dir_complete = cellular_data_dir + cellular_model_convergence + '_' + stimulation_protocol + '_' + str(
            cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model_twave_name + '/'
        apd_max_max = hyperparameter_dict['apd_max_max']
        apd_min_min = hyperparameter_dict['apd_min_min']
        # Create cellular model instance.
        print('ep_model ', ep_model_twave_name)
        # print('apd_resolution ', apd_resolution)
        # print('cycle_length ', cycle_length)
        if ep_model_twave_name == 'MitchellSchaefferEP':
            apd_resolution = hyperparameter_dict['apd_resolution']
            cycle_length = hyperparameter_dict['cycle_length']
            vm_max = hyperparameter_dict['vm_max']
            vm_min = hyperparameter_dict['vm_min']
            cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                            apd_resolution=apd_resolution, cycle_length=cycle_length,
                                                            list_celltype_name=list_celltype_name, verbose=verbose,
                                                            vm_max=vm_max, vm_min=vm_min)
        else:
            cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                              biomarker_apd90_name=biomarker_apd90_name,
                                                              biomarker_celltype_name=biomarker_celltype_name,
                                                              cellular_data_dir=cellular_data_dir_complete,
                                                              cellular_model_name=cellular_model_name,
                                                              list_celltype_name=list_celltype_name, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        apd_max_max = None
        apd_min_min = None
        apd_resolution = None
        # biomarker_apd90_name = None
        biomarker_upstroke_name = None
        cellular_data_dir = None
        cellular_data_dir_complete = None
        cellular_model_name = None
        cellular_stim_amp = None
        cellular_model_convergence = None
        cycle_length = None
        ep_model_twave_name = None
        stimulation_protocol = None
        vm_max = None
        vm_min = None
        ####################################################################################################################
        # Step 3: Generate a cardiac geometry.
        print('Step 3: Generate a cardiac geometry.')
        # Argument setup: (in Alphabetical order)
        # Read hyperparameters
        vc_ab_cut_name = hyperparameter_dict['vc_ab_cut_name']
        vc_aprt_name = hyperparameter_dict['vc_aprt_name']
        vc_rvlv_name = hyperparameter_dict['vc_rvlv_name']
        vc_tm_name = hyperparameter_dict['vc_tm_name']
        celltype_vc_info = hyperparameter_dict['celltype_vc_info']
        vc_name_list = hyperparameter_dict['vc_name_list']
        # Create geometry with a dummy conduction system to allow initialising the geometry.
        geometry = SimulationGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                   conduction_system=EmptyConductionSystem(verbose=verbose),
                                   geometric_data_dir=geometric_data_dir, resolution=inference_resolution,
                                   subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
        raw_geometry_point_cloud = RawEmptyCardiacGeoPointCloud(conduction_system=EmptyConductionSystem(verbose=verbose),
                                                                geometric_data_dir=geometric_data_dir, resolution=monodomain_resolution,
                                                                subject_name=anatomy_subject_name, verbose=verbose)
        # TODO DELETE THE FOLLOWING CODE
        warn(
            'This should not be done in here!\nThis hack will only work for DTI... meshes, and should be done before calling the script in the futrure.')
        print('min max ', np.amin(raw_geometry_point_cloud.unprocessed_node_xyz),
              np.amax(raw_geometry_point_cloud.unprocessed_node_xyz))
        for_monodomain_translation_scale = np.array([1., 1., 1.])
        raw_geometry_point_cloud.unprocessed_node_xyz = convert_from_monoalg3D_to_cm_and_translate(
            monoalg3D_xyz=raw_geometry_point_cloud.get_node_xyz(), inference_xyz=geometry.get_node_xyz(),
            scale=for_monodomain_translation_scale)
        print('min max ', np.amin(raw_geometry_point_cloud.unprocessed_node_xyz),
              np.amax(raw_geometry_point_cloud.unprocessed_node_xyz))
        print('geometry min max ', np.amin(geometry.get_node_xyz()),
              np.amax(geometry.get_node_xyz()))
        # TODO DELETE THE ABOVE CODE

        # Clear Arguments to prevent Argument recycling
        geometric_data_dir = None
        list_celltype_name = None
        # inference_resolution = None
        # vc_name_list = None
        ####################################################################################################################
        # Step 4: Prepare smoothing configuration to resemble diffusion effects
        print('Step 4: Prepare smoothing configuration to resemble diffusion effects.')
        # Define the speeds used during the fibre-based smoothing
        warn(
            'Inference from QT can, but does NOT, update the speeds in the smoothing function!\nAlso, it requires some initial fixed values!')
        fibre_speed_name = hyperparameter_dict['fibre_speed_name']
        sheet_speed_name = hyperparameter_dict['sheet_speed_name']
        normal_speed_name = hyperparameter_dict['normal_speed_name']
        fibre_speed = hyperparameter_dict[fibre_speed_name]
        sheet_speed = hyperparameter_dict[sheet_speed_name]
        print('sheet_speed ', sheet_speed)
        normal_speed = hyperparameter_dict[normal_speed_name]
        # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
        smoothing_ghost_distance_to_self = hyperparameter_dict['smoothing_ghost_distance_to_self']  # cm # This parameter enables to control how much spatial smoothing happens and
        print('smoothing_ghost_distance_to_self ', smoothing_ghost_distance_to_self)
        # smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 5% and the present as 95%
        # full_smoothing_time_index = 400  # (ms) assumming 1000Hz
        # warn('Precompuing the smoothing, change this please!')  # TODO refactor
        geometry.precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(
            fibre_speed=fibre_speed, sheet_speed=sheet_speed, normal_speed=normal_speed,
            ghost_distance_to_self=smoothing_ghost_distance_to_self)
        ####################################################################################################################
        # Step 5: Create propagation model instance, this will be a static dummy propagation model.
        print('Step 5: Create propagation model instance, this will be a static dummy propagation model.')
        # Arguments for propagation model:
        # Read hyperparameters
        # propagation_parameter_name_list_in_order = hyperparameter_dict['propagation_parameter_name_list_in_order']
        lat_prescribed = (np.loadtxt(qrs_lat_prescribed_filename_path, delimiter=',')).astype(int)
        propagation_model = PrescribedLAT(geometry=geometry, lat_prescribed=lat_prescribed,
                                          module_name=propagation_module_name, verbose=verbose)

        # Clear Arguments to prevent Argument recycling
        qrs_lat_prescribed_filename_path = None
        # lat_prescribed = None
        ####################################################################################################################
        # Step 6: Create Whole organ Electrophysiology model.
        print('Step 6: Create Whole organ Electrophysiology model.')
        # Read hyperparameters
        apd_max_name = hyperparameter_dict['apd_max_name']
        apd_min_name = hyperparameter_dict['apd_min_name']
        g_vc_ab_name = hyperparameter_dict['g_vc_ab_name']
        g_vc_aprt_name = hyperparameter_dict['g_vc_aprt_name']
        g_vc_rvlv_name = hyperparameter_dict['g_vc_rvlv_name']
        g_vc_tm_name = hyperparameter_dict['g_vc_tm_name']
        electrophysiology_parameter_name_list_in_order = hyperparameter_dict['electrophysiology_parameter_name_list_in_order']
        # Spatial and temporal smoothing parameters:
        smoothing_dt = hyperparameter_dict['smoothing_dt']
        print('smoothing_dt ', smoothing_dt)
        start_smoothing_time_index = hyperparameter_dict['start_smoothing_time_index']
        end_smoothing_time_index = hyperparameter_dict['end_smoothing_time_index']
        print('end_smoothing_time_index ', end_smoothing_time_index)
        electrophysiology_model = ElectrophysiologyAPDmap(apd_max_name=apd_max_name, apd_min_name=apd_min_name,
                                                          cellular_model=cellular_model,
                                                          fibre_speed_name=fibre_speed_name,
                                                          end_smoothing_time_index=end_smoothing_time_index,
                                                          module_name=electrophysiology_module_name,
                                                          normal_speed_name=normal_speed_name,
                                                          parameter_name_list_in_order=electrophysiology_parameter_name_list_in_order,
                                                          propagation_model=propagation_model,
                                                          sheet_speed_name=sheet_speed_name,
                                                          smoothing_dt=smoothing_dt,
                                                          smoothing_ghost_distance_to_self=smoothing_ghost_distance_to_self,
                                                          start_smoothing_time_index=start_smoothing_time_index,
                                                          verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        cellular_model = None
        end_smoothing_time_index = None
        propagation_model = None
        smoothing_dt = None
        start_smoothing_time_index = None
        ####################################################################################################################
        # Step 7: Create ECG calculation method.
        print('Step 7: Create ECG calculation method.')
        # Arguments for ECG calculation:
        # Read hyperparameters
        lead_names = hyperparameter_dict['lead_names']
        nb_leads = hyperparameter_dict['nb_leads']
        v3_name = hyperparameter_dict['v3_name']
        v5_name = hyperparameter_dict['v5_name']
        lead_v3_i = lead_names.index(v3_name)
        lead_v5_i = lead_names.index(v5_name)
        assert nb_leads == len(lead_names)
        filtering = hyperparameter_dict['filtering']
        max_len_qrs = hyperparameter_dict['max_len_qrs']
        max_len_ecg = hyperparameter_dict['max_len_ecg']
        normalise = hyperparameter_dict['normalise']
        zero_align = hyperparameter_dict['zero_align']
        frequency = hyperparameter_dict['frequency']
        if frequency != 1000:
            warn(
                'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
        low_freq_cut = hyperparameter_dict['low_freq_cut']
        print('low_freq_cut ', low_freq_cut)
        # low_freq_cut = 0.001
        high_freq_cut = hyperparameter_dict['high_freq_cut']
        print('high_freq_cut ', high_freq_cut)
        # high_freq_cut = 100
        # Read clinical data
        clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
        # Create ECG model
        ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                       frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                       low_freq_cut=low_freq_cut,
                                       max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                       nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                       reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                       tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
        clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
        # save_ecg_to_csv(data=clinical_ecg[np.newaxis, :, :], filename=preprocessed_clinical_ecg_file_name)
        # print('Saved preprocessed clinical ECG at ', preprocessed_clinical_ecg_file_name)
        # Clear Arguments to prevent Argument recycling
        clinical_data_filename_path = None
        clinical_ecg_raw = None
        filtering = None
        max_len_ecg = None
        max_len_qrs = None
        normalise = None
        # preprocessed_clinical_ecg_file_name = None
        v3_name = None
        v5_name = None
        zero_align = None
        ####################################################################################################################
        # Step 8: Define instance of the simulation method.
        print('Step 8: Define instance of the simulation method.')
        simulator_ecg = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
        simulator_ep = SimulateEP(electrophysiology_model=electrophysiology_model, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        electrophysiology_model = None
        ecg_model = None
        ####################################################################################################################
        # Step 9: Define Adapter to translate between theta and parameters.
        print('Step 9: Define Adapter to translate between theta and parameters.')
        # Read hyperparameters
        # TODO make the following code into a for loop!!
        # Theta resolutions
        apd_max_resolution = hyperparameter_dict['apd_max_resolution']
        apd_min_resolution = hyperparameter_dict['apd_min_resolution']
        g_vc_ab_resolution = hyperparameter_dict['g_vc_ab_resolution']
        g_vc_aprt_resolution = hyperparameter_dict['g_vc_aprt_resolution']
        g_vc_rvlv_resolution = hyperparameter_dict['g_vc_rvlv_resolution']
        g_vc_tm_resolution = hyperparameter_dict['g_vc_tm_resolution']
        theta_adjust_function_list_in_order = [RoundTheta(resolution=apd_max_resolution),
                                           RoundTheta(resolution=apd_min_resolution),
                                           RoundTheta(resolution=g_vc_ab_resolution),
                                           RoundTheta(resolution=g_vc_aprt_resolution),
                                           RoundTheta(resolution=g_vc_rvlv_resolution),
                                           RoundTheta(resolution=g_vc_tm_resolution)
                                           ]
        nb_discrete_theta = hyperparameter_dict['nb_discrete_theta']
        for root_i in range(nb_discrete_theta):
            theta_adjust_function_list_in_order.append(None)
        theta_name_list_in_order = hyperparameter_dict['theta_name_list_in_order']
        if len(theta_adjust_function_list_in_order) != len(theta_name_list_in_order):
            print('theta_name_list_in_order ', len(theta_name_list_in_order))
            print('theta_adjust_function_list_in_order ', len(theta_adjust_function_list_in_order))
            raise Exception('Different number of adjusting functions and theta for the inference')
        # Create an adapter that can translate between theta and parameters
        # Paramter destinations
        destination_module_name_list_in_order = hyperparameter_dict['destination_module_name_list_in_order']
        parameter_destination_module_dict = hyperparameter_dict['parameter_destination_module_dict']
        parameter_name_list_in_order = hyperparameter_dict['parameter_name_list_in_order']
        # Parameter pre-fixed values
        parameter_fixed_value_dict = hyperparameter_dict['parameter_fixed_value_dict']
        physiological_rules_larger_than_dict = hyperparameter_dict['physiological_rules_larger_than_dict']
        adapter = AdapterThetaParams(destination_module_name_list_in_order=destination_module_name_list_in_order,
                                     parameter_fixed_value_dict=parameter_fixed_value_dict,
                                     parameter_name_list_in_order=parameter_name_list_in_order,
                                     parameter_destination_module_dict=parameter_destination_module_dict,
                                     theta_name_list_in_order=theta_name_list_in_order,
                                     physiological_rules_larger_than_dict=physiological_rules_larger_than_dict,
                                     theta_adjust_function_list_in_order=theta_adjust_function_list_in_order,
                                     verbose=verbose)
        nb_theta = len(theta_name_list_in_order)
        # Clear Arguments to prevent Argument recycling
        speed_parameter_name_list_in_order = None
        candidate_root_node_names = None
        fibre_speed_name = None
        sheet_speed_name = None
        normal_speed_name = None
        endo_dense_speed_name = None
        endo_sparse_speed_name = None
        parameter_fixed_value_dict = None
        print('theta_name_list_in_order ', theta_name_list_in_order)
        theta_name_list_in_order = None
        ####################################################################################################################
        # Step 10: Create evaluators for the ECG, LAT and VM.
        print('Step 10: Create evaluators for the ECG, LAT and VM.')
        evaluator_ecg = ParameterSimulator(adapter=adapter, simulator=simulator_ecg, verbose=verbose)
        evaluator_ep = ParameterSimulator(adapter=adapter, simulator=simulator_ep, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        adapter = None
        simulator_ecg = None
        simulator_ep = None
        ####################################################################################################################
        # Step 11: Read the values inferred for parameters.
        print('Step 11: Read the values inferred for parameters.')
        # TODO save candidate root nodes and their times so that the meta-indexes can be used to point at them.
        pandas_parameter_population = pd.read_csv(parameter_result_file_name, delimiter=',')
        # print('pandas_parameter_population ', pandas_parameter_population)
        # print('parameter_name_list_in_order ', parameter_name_list_in_order)
        # for param_name in parameter_name_list_in_order:
        #     print('\nparam_name ', param_name)
        #     param_value = pandas_parameter_population.get(param_name).tolist()
        #     print('param_value ', param_value)
        #     print('mean ', np.mean(param_value))
        #     print('std ', np.std(param_value))
        # print('Done printing inferred parameter-set values')
        # # raise()

        # root_node_meta_index_population = get_root_node_meta_index_population_from_pandas(pandas_parameter_population=pandas_parameter_population)
        # purkinje_speed_population = translate_from_pandas_to_array(name_list_in_order=[get_purkinje_speed_name()], pandas_data=pandas_parameter_population)
        # print('root_node_meta_index_population ', root_node_meta_index_population.shape)
        parameter_population = evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
        unique_parameter_population, unique_index = np.unique(parameter_population, axis=0, return_index=True)
        # unique_root_node_meta_index_population = root_node_meta_index_population[unique_index, :]
        # unique_purkinje_speed_population = purkinje_speed_population[unique_index]
        # Clear Arguments to prevent Argument recycling.
        pandas_parameter_population = None
        parameter_result_file_name = None
        purkinje_speed_population = None
        root_node_meta_index_population = None
        unique_index = None
        ####################################################################################################################
        # unique_parameter_population = unique_parameter_population[0:30, :] #TODO REVERT
        # print('unique_parameter_population ', unique_parameter_population.shape)
        # # Step 12: Evaluate their ECG.
        print('Step 12: Evaluate inferred ECGs.')
        # Simulate the parameter population from the inference
        unique_population_ecg = evaluator_ecg.simulate_parameter_population(parameter_population=unique_parameter_population)
        print('unique_population_ecg ', unique_population_ecg.shape)
        save_ecg_to_csv(data=unique_population_ecg, filename=inference_ecg_inferred_population_filename)
        print('Saved inferred ECGs at ', inference_ecg_inferred_population_filename)
        # Clear Arguments to prevent Argument recycling.
        inference_ecg_inferred_population_filename = None
        ####################################################################################################################
        # Step 13: Define the discrepancy metric and make sure that the result is the same when calling the evaluator.
        print('Step 13: Define the discrepancy metric.')
        # Arguments for discrepancy metrics:
        # Read hyperparameters
        error_method_name_inference_metric = hyperparameter_dict['error_method_name']

        error_method_name_inference_metric = 'pcc'
        # Create discrepancy metric instance using the inference metric:
        discrepancy_metric_inference = DiscrepancyECG(
            error_method_name=error_method_name_inference_metric)
        # Evaluate discrepancy:
        unique_discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
            predicted_data_population=unique_population_ecg, target_data=clinical_ecg)

        plt.plot(unique_discrepancy_population_inference)
        print(anatomy_subject_name, ' Mean PCC ', np.mean(unique_discrepancy_population_inference))
        print(anatomy_subject_name, ' STD PCC ', np.std(unique_discrepancy_population_inference))

        error_method_name_inference_metric = 'rmse'
        # Create discrepancy metric instance using the inference metric:
        discrepancy_metric_inference = DiscrepancyECG(
            error_method_name=error_method_name_inference_metric)
        # Evaluate discrepancy:
        unique_discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
            predicted_data_population=unique_population_ecg, target_data=clinical_ecg)

        print(anatomy_subject_name, ' Mean RMSE ', np.mean(unique_discrepancy_population_inference))
        print(anatomy_subject_name, ' STD RMSE ', np.std(unique_discrepancy_population_inference))

    # if False:
        # Generate discrepancy
        error_method_name_inference_metric = hyperparameter_dict['error_method_name']
        # Create discrepancy metric instance using the inference metric:
        discrepancy_metric_inference = DiscrepancyECG(
            error_method_name=error_method_name_inference_metric)
        # Evaluate discrepancy:
        unique_discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
            predicted_data_population=unique_population_ecg, target_data=clinical_ecg)

        # The following piece of code is to test that different ways of calculating the discrepancy yield the same results
        # # Create discrepancy evaluator to assess code correctness!!!
        # evaluator_ecg_inference_metric = DiscrepancyEvaluator(
        #     adapter=adapter, discrepancy_metric=discrepancy_metric_inference, simulator=simulator_ecg,
        #     target_data=clinical_ecg, verbose=verbose)
        # discrepancy_population_inference_from_evaluator = evaluator_ecg_inference_metric.evaluate_parameter_population(
        #     parameter_population=unique_parameter_population)
        # if not (np.all(unique_discrepancy_population_inference == discrepancy_population_inference_from_evaluator)):
        #     warn('These should be identical: discrepancy_population_inference '
        #          + str(unique_discrepancy_population_inference.shape)
        #          + ' discrepancy_population_inference_from_evaluator '
        #          + str(discrepancy_population_inference_from_evaluator.shape))
        # Clear Arguments to prevent Argument recycling.
        discrepancy_metric_inference = None
        discrepancy_population_inference = None
        error_method_name_inference_metric = None
        evaluator_ecg_inference_metric = None
        population_discrepancy_inference_from_evaluator = None
        ####################################################################################################################
        # Step 14: Select best discrepancy particle and save best parameter.
        print('Step 14: Select best discrepancy particle.')
        unique_best_index = np.argmin(unique_discrepancy_population_inference)
        print('Best discrepancy ', unique_discrepancy_population_inference[unique_best_index])
        best_parameter = unique_parameter_population[unique_best_index]
        print('Best parameter ', best_parameter)
        print()
        print()
        print(anatomy_subject_name)
        print('unique_best_index ', unique_best_index)
        print()
        print()

        # Clear Arguments to prevent Argument recycling.
        best_parameter = None
        # best_parameter_result_file_name = None
        unique_discrepancy_population_inference = None
        ####################################################################################################################
        # Step 15: Randomly select some % of the particles in the final population and save their biomarkers.
        print('Step 15: Randomly select some % of the particles in the final population and save their biomarkers.')
        # Arguments for uncertainty quantification:
        uncertainty_proportion = 0.1   # 10% of the population size
        population_size = parameter_population.shape[0]  # The population size is computed with respect to the initial population
        # unique_parameter_population = np.unique(parameter_population, axis=0)
        unique_population_size = unique_parameter_population.shape[0]
        nb_uncertainty_particles = math.ceil(uncertainty_proportion * population_size)
        assert nb_uncertainty_particles < unique_population_size, 'We cannot sample more particles than the unique ones available in the final population!'
        uncertainty_index = np.random.permutation(unique_population_size)[:nb_uncertainty_particles]
        print('Adding the best index to the begining of the list to be translated into Monodomain simulations')
        # Add the best index to the begining of the list to be translated into Monodomain simulations
        uncertainty_index = np.append(np.array([unique_best_index]), uncertainty_index, axis=0)
        # Print out indexes
        print('best index ', unique_best_index)
        print('uncertainty_index ', uncertainty_index)
        print('num translations ', len(uncertainty_index))
        # Index parameter population
        uncertainty_parameter_population = unique_parameter_population[uncertainty_index, :]  # Parameter values
        # Print out indexes
        print('uncertainty_parameter_population ', uncertainty_parameter_population.shape)
        # Save parameter values for the translation to monodomain simulations
        save_csv_file(data=uncertainty_parameter_population, filename=for_monodomain_parameter_population_file_name,
                      column_name_list=parameter_name_list_in_order)
        print('uncertainty_parameter_population ', uncertainty_parameter_population.shape)
        # Simulate ECGs from uncertainty selection of parameters
        uncertainty_population_ecg = unique_population_ecg[uncertainty_index, :, :]
        print('uncertainty_population_ecg ', uncertainty_population_ecg.shape)
        # save_ecg_to_csv(data=uncertainty_population_ecg, filename=inference_ecg_uncertainty_population_filename)
        # print('Saved selected inference ECGs at ', inference_ecg_uncertainty_population_filename)
        # inference_ecg_uncertainty_population_filename = None  # Clear Arguments to prevent Argument recycling.
        # Simulate LAT and VMs form uncertainty selection of parameters
        _, uncertainty_population_vm = evaluator_ep.simulate_parameter_population(
            parameter_population=uncertainty_parameter_population)
        # TODO I AM HERE
        # print('uncertainty_population_lat_prescribed ', uncertainty_population_lat_prescribed.shape)
        print('uncertainty_population_vm ', uncertainty_population_vm.shape)
        # # Make sure that the LATs are calculated in the same way that they will be calculated for the monodomain simulations
        # uncertainty_population_lat = np.zeros(uncertainty_population_lat_prescribed.shape)
        # uncertainty_population_earliest_lat = np.zeros((uncertainty_population_lat_prescribed.shape[0]))
        # for uncertainty_i in range(uncertainty_parameter_population.shape[0]):
        #     # TODO make the percentage for claculating the LATs into a global varibale to be consistent
        #
        #     ## Calculate LATs
        #     uncertainty_lat = generate_activation_map(
        #         vm=uncertainty_population_vm[uncertainty_i, :, :], percentage=70)
        #     # We want the frist LAT value to be 1 ms
        #     earliest_activation_time = int(max(np.amin(uncertainty_lat) - 1, 0))
        #     uncertainty_population_earliest_lat[uncertainty_i] = earliest_activation_time
        #     # if earliest_activation_time > 0:
        #     #     warn('The Eikonal ECG may not have been aligned using the same LAT as the one used later for the monodomain!')
        #     # print()
        #     # print('earliest_activation_time ', earliest_activation_time)
        #     # print('uncertainty_population_lat[uncertainty_i, :] ', np.amin(uncertainty_population_lat[uncertainty_i, :])-1)
        #     # print()
        #     ## Correct LATs
        #     uncertainty_lat = uncertainty_lat - earliest_activation_time
        #     ## Save LATs
        #     uncertainty_population_lat[uncertainty_i, :] = uncertainty_lat
        #     # ## Correct the VMs using the new LAT start - This will enable generating aligned ECGs and REPOLs
        #     # uncertainty_vm = uncertainty_population_vm[uncertainty_i, :, earliest_activation_time:]
        #     # uncertainty_population_vm[uncertainty_i, :, :uncertainty_vm.shape[1]] = uncertainty_vm
        #     # uncertainty_population_vm[uncertainty_i, :, uncertainty_vm.shape[1]:] = uncertainty_vm[:, -1:]
        # print('uncertainty_population_lat ', uncertainty_population_lat.shape)
        # print('uncertainty_population_vm ', uncertainty_population_vm.shape)
        # Clear Arguments to prevent Argument recycling.
        unique_root_node_meta_index_population = None
        unique_purkinje_speed_population = None
        uncertainty_index = None
        unique_best_index = None
        # ####################################################################################################################
        # Step 16: Interpolate simulation results to have the same indexing that the input data files.
        print('16: Interpolate simulation results to have the same indexing that the input data files.')
        # Interpolate nodefield
        unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=raw_geometry_point_cloud.get_node_xyz(),
                                                     reference_points_xyz=geometry.get_node_xyz())
        # ####################################################################################################################
        # Step 17: Iterate for all particles chosen to represent the uncertainty of the inference.
        print('17: Iterate for all particles chosen to represent the uncertainty of the inference.')
        # Biomarker names
        activation_time_map_biomarker_name = get_lat_biomarker_name()  # TODO make these names globally defined in utils.py
        repolarisation_time_map_biomarker_name = get_repol_biomarker_name()  # TODO make these names globally defined in utils.py
        apd90_biomarker_name = get_apd90_biomarker_name()
        sf_iks_biomarker_name = get_sf_iks_biomarker_name()
        # Iterate for all particles chosen to represent the uncertainty of the inference
        # inference_repol_population = []
        lat_corrected_uncertainty_population_ecg = pymp.shared.array(uncertainty_population_ecg.shape, dtype=np.float64) \
                                                   + get_nan_value()
        lat_correction_list = pymp.shared.array((uncertainty_parameter_population.shape[0]), dtype=np.float64) \
                                                   + get_nan_value()
        # if True:
        #     for uncertainty_i in range(uncertainty_parameter_population.shape[0]):
        threadsNum = multiprocessing.cpu_count()
        with pymp.Parallel(min(threadsNum, uncertainty_parameter_population.shape[0])) as p1:
            for uncertainty_i in p1.range(uncertainty_parameter_population.shape[0]):
                p1.print('uncertainty_i ', uncertainty_i)
                if uncertainty_i == 0:
                    iteration_str_tag = get_best_str()
                else:
                    iteration_str_tag = str(uncertainty_i)

                # CREATE RESULT DIRECTORY
                current_translation_dir = translation_dir_tag + iteration_str_tag + '/'
                if not os.path.exists(current_translation_dir):
                    os.mkdir(current_translation_dir)

                # BIOMARKERS
                # Calculate the effect of uncertainty in the biomarkers and save them
                uncertainty_biomarker_result_file_name = for_monodomain_biomarker_result_file_name_start + iteration_str_tag + for_monodomain_biomarker_result_file_name_end
                # if not os.path.exists(uncertainty_biomarker_result_file_name):
                uncertainty_parameter_particle = uncertainty_parameter_population[uncertainty_i, :]
                unprocessed_node_biomarker = evaluator_ep.biomarker_parameter_particle(
                    parameter_particle=uncertainty_parameter_particle)
                # LAT AND REPOL MAPS
                # Make sure that the LATs are calculated in the same way that they will be calculated for the monodomain
                ## Calculate LATs
                uncertainty_lat = generate_activation_map(
                    vm=uncertainty_population_vm[uncertainty_i, :, :], percentage=70) # TODO make the percentage for claculating the LATs into a global varibale to be consistent
                # We want the frist LAT value to be 1 ms
                earliest_activation_time = int(max(np.amin(uncertainty_lat) - 1, 0))
                ## Save earliest_activation_time
                lat_correction_list[uncertainty_i] = earliest_activation_time
                ## Correct LATs
                uncertainty_lat = uncertainty_lat - earliest_activation_time
                ## Save LATs
                unprocessed_node_biomarker[activation_time_map_biomarker_name] = uncertainty_lat
                ## Calculate REPOLs
                unprocessed_node_repol = generate_repolarisation_map(vm=uncertainty_population_vm[uncertainty_i, :, :])
                ## Correct REPOL using earliest LAT
                unprocessed_node_repol = unprocessed_node_repol - earliest_activation_time
                ## Save REPOLs
                unprocessed_node_biomarker[repolarisation_time_map_biomarker_name] = unprocessed_node_repol

                ## Correct ECGs
                uncertainty_ecg = uncertainty_population_ecg[uncertainty_i, :, earliest_activation_time:]
                lat_corrected_uncertainty_population_ecg[uncertainty_i, :, :uncertainty_ecg.shape[1]] = uncertainty_ecg
                lat_corrected_uncertainty_population_ecg[uncertainty_i, :, uncertainty_ecg.shape[1]:] = uncertainty_ecg[:, -1:]

                if not os.path.exists(uncertainty_biomarker_result_file_name):
                    p1.print('Interpolate and save the results for ', iteration_str_tag)
                    # Save biomarkers to allow translation to MonoAlg3D and Alya
                    p1.print('Saving biomarkers for uncertainty_i ', uncertainty_i)
                    node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker,
                                                                 row_index=unprocessed_node_mapping_index)
                    save_pandas(df=node_biomarker, filename=uncertainty_biomarker_result_file_name)
                    p1.print('Saved: ', uncertainty_biomarker_result_file_name)

                    # SAVE LAT and REPOL comparison
                    # Save non-time depenedent fields for comparison
                    eikonal_field_tag = 'RE_'
                    write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=[
                        unprocessed_node_biomarker[activation_time_map_biomarker_name],
                        unprocessed_node_biomarker[repolarisation_time_map_biomarker_name],
                        lat_prescribed,
                        unprocessed_node_biomarker[apd90_biomarker_name],
                        unprocessed_node_biomarker[sf_iks_biomarker_name],
                        unprocessed_node_biomarker[repolarisation_time_map_biomarker_name] -
                        unprocessed_node_biomarker[activation_time_map_biomarker_name]
                    ],
                                                          node_field_name_list=[
                                                              eikonal_field_tag + activation_time_map_biomarker_name,
                                                              eikonal_field_tag + repolarisation_time_map_biomarker_name,
                                                              'read_' + activation_time_map_biomarker_name,
                                                              eikonal_field_tag + apd90_biomarker_name,
                                                              eikonal_field_tag + sf_iks_biomarker_name,
                                                              eikonal_field_tag + 'ARI'
                                                          ],
                                                          subject_name=anatomy_subject_name + '_' + inference_resolution + '_TRANSLATION',
                                                          verbose=verbose,
                                                          visualisation_dir=current_translation_dir)

                    # VM translation
                    unprocessed_node_vm = uncertainty_population_vm[uncertainty_i, :, :]
                    ## Correct the VMs using the new LAT start - This will enable generating aligned ECGs and REPOLs
                    unprocessed_node_vm = unprocessed_node_vm[:, earliest_activation_time:]
                    export_ensight_timeseries_case(dir=current_translation_dir,
                                                   casename=anatomy_subject_name + '_' + inference_resolution + '_simulation',
                                                   dataname_list=['vm'],
                                                   vm_list=[unprocessed_node_vm], dt=1. / frequency,
                                                   nodesxyz=geometry.get_node_xyz(),
                                                   tetrahedrons=geometry.get_tetra())
                else:
                    p1.print('Skip ', iteration_str_tag)
                    # Load Biomarkers from inference selected particles for translation to monodomain
                    # node_biomarker = read_pandas(filename=uncertainty_biomarker_result_file_name)
                    # unprocessed_node_vm = uncertainty_population_vm[uncertainty_i, :, :]

        # Clear Arguments to prevent Argument recycling.
        anatomy_subject_name = None
        best_theta = None
        ecg_population_file_name = None
        evaluator_ecg = None
        evaluator_ep = None
        # for_monodomain_figure_result_file_name = None
        frequency = None
        geometry = None
        inferred_theta_population = None
        max_lat_population_file_name = None
        node_repol = None
        population_metric_dir = None
        raw_geometry_point_cloud = None
        for_monodomain_parameter_population_file_name = None
        unique_population_lat = None
        unique_population_vm = None
        uncertainty_population_lat = None
        uncertainty_population_vm = None
        unprocessed_node_mapping_index = None
        ####################################################################################################################
        # Step 18: Save ECGs for the final population.
        print('Step 18: Save Clinical and Simulated ECGs for the final population.')
        earliest_activation_time = int(round(np.mean(lat_correction_list)))
        print('earliest_activation_time ', earliest_activation_time)
        save_ecg_to_csv(data=clinical_ecg[np.newaxis, :, earliest_activation_time:], filename=preprocessed_clinical_ecg_file_name)
        print('Saved preprocessed clinical ECG at ', preprocessed_clinical_ecg_file_name)
        save_ecg_to_csv(data=lat_corrected_uncertainty_population_ecg, filename=inference_ecg_uncertainty_population_filename)
        print('Saved selected inference ECGs at ', inference_ecg_uncertainty_population_filename)
        # Clear Arguments to prevent Argument recycling.
        inference_ecg_uncertainty_population_filename = None
        preprocessed_clinical_ecg_file_name = None
        ####################################################################################################################
        # Step 18: Visualise ECGs and their metrics for the final population.
        print('Step 19: Visualise ECGs and their metrics for the final population.')
        # Initialise arguments for plotting
        axes = None
        fig = None
        # Plot the ECG inference population
        if uncertainty_population_ecg.shape[0] > 1:
            axes, fig = visualise_ecg(ecg_list=uncertainty_population_ecg[1:, :, :], lead_name_list=lead_names, axes=axes,
                                      ecg_color='gray', fig=fig, label_list=None,
                                      linewidth=1.)
        axes, fig = visualise_ecg(ecg_list=uncertainty_population_ecg[0:1, :, :], lead_name_list=lead_names, axes=axes,
                                  ecg_color='k', fig=fig, label_list=['Best'],
                                  linewidth=2.)
        # Plot the clinical trace after the last iteration
        axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                                  ecg_color='lime', fig=fig, label_list=['Clinical'],
                                  linewidth=2.)
        for ax in axes:
            ax.hlines(0, 0, 800, 'r')
            ax.vlines(450, -1, 1, 'r')
            ax.vlines(500, -1, 1, 'r')
        axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.show(block=False)
        fig.savefig(for_monodomain_figure_result_file_name)
        print('Saved ecg figure: ', for_monodomain_figure_result_file_name)
        # Clear Arguments to prevent Argument recycling.
        axes = None
        fig = None
        for_monodomain_figure_result_file_name = None
        # population_biomarker = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)
    print('')

    #EOF




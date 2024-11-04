"""Generate history figure after Twave personalisation"""
import os
import sys
from warnings import warn

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    anatomy_subject_name_list = ['DTI024', 'DTI004', 'DTI032']
    for anatomy_subject_name in anatomy_subject_name_list:
        if len(sys.argv) < 2:
            anatomy_subject_name = anatomy_subject_name
            ecg_subject_name = anatomy_subject_name  # Allows using a different ECG for the personalisation than for the anatomy
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
        convert_from_monoalg3D_to_cm_and_translate, get_apd90_biomarker_name, get_sf_iks_biomarker_name
        from postprocess_functions import generate_repolarisation_map, visualise_ecg, visualise_biomarker

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
        verbose = True
        # Input Paths:
        data_dir = path_dict["data_path"]
        cellular_data_dir = data_dir + 'cellular_data/'
        geometric_data_dir = data_dir + 'geometric_data/'
        results_dir_root = path_dict["results_path"]
        # Intermediate Paths: # e.g., results from the QRS inference
        experiment_type = 'personalisation'
        # TODO have a single definition of the heart rate or the cycle length for the whole pipeline
        if anatomy_subject_name == 'DTI024':  # Subject 1
            heart_rate = 66
        elif anatomy_subject_name == 'DTI004':  # Subject 2
            heart_rate = 48
        elif anatomy_subject_name == 'DTI032':  # Subject 3
            heart_rate = 74
        cycle_length = get_cycle_length(heart_rate=heart_rate)
        cycle_length_str = str(int(cycle_length))
        ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_' + cycle_length_str
        print('ep_model_twave_name ', ep_model_twave_name)
        # if anatomy_subject_name == 'DTI024':
        #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_909'
        # elif anatomy_subject_name == 'DTI032':
        #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_810'
        # elif anatomy_subject_name == 'DTI004':
        #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_1250'
        # else:
        #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_'
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
        current_month_text = 'Jun'  # datetime.now().strftime('%h')  # e.g., Feb
        current_year_full = datetime.now().strftime('%Y')  # e.g., 2024
        date_str = current_month_text + '_' + current_year_full
        results_dir_twave = results_dir_part_twave + date_str + '_fixed_filter/'
        assert os.path.exists(results_dir_twave)  # Path should already exist from running the Twave inference
        results_dir_part_twave = None  # Clear Arguments to prevent Argument recycling
        # Read hyperparamter dictionary
        hyperparameter_result_file_name = results_dir_twave + anatomy_subject_name + '_' + inference_resolution + '_hyperparameter.txt'
        hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
        # Check that the heart rate is the same than used during the inference
        # assert heart_rate == hyperparameter_dict['heart_rate'] # TODO uncomment
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
        # History dir
        inference_history_dir = results_dir_twave + 'inference_history/'
        if not os.path.exists(inference_history_dir):
            print('inference_history_dir ', inference_history_dir)
            raise Exception("There is no inference history directory! "
                            "You should re-run the inference with the lines for saving the history uncommented.")
        history_theta_name_tag = 'population_theta_'
        history_theta_name_end = '.csv'
        # Output Paths:
        # Precomputed subfolder
        history_precomputed_dir = inference_history_dir + 'history_precomputed/'
        if not os.path.exists(history_precomputed_dir):
            os.mkdir(history_precomputed_dir)
        history_ecg_name_tag = anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_history_ecg'
        history_biomarker_name_tag = anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_history_biomarker'
        history_separator_tag = '_'
        # Visualisation
        visualisation_dir = inference_history_dir + 'history_figure/'
        if not os.path.exists(visualisation_dir):
            os.mkdir(visualisation_dir)
        figure_ecg_history_file_name = visualisation_dir + history_ecg_name_tag + '.png'
        figure_biomarker_history_file_name = visualisation_dir + history_biomarker_name_tag + '.png'
        figure_theta_history_file_name = visualisation_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_history_theta' + '.png'
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
        vc_tv_name = hyperparameter_dict['vc_aprt_name']
        celltype_vc_info = hyperparameter_dict['celltype_vc_info']
        vc_name_list = hyperparameter_dict['vc_name_list']
        # Create geometry with a dummy conduction system to allow initialising the geometry.
        geometry = SimulationGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                   conduction_system=EmptyConductionSystem(verbose=verbose),
                                   geometric_data_dir=geometric_data_dir, resolution=inference_resolution,
                                   subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        geometric_data_dir = None
        list_celltype_name = None
        inference_resolution = None
        vc_name_list = None
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
        smoothing_ghost_distance_to_self = hyperparameter_dict[
            'smoothing_ghost_distance_to_self']  # cm # This parameter enables to control how much spatial smoothing happens and
        print('smoothing_ghost_distance_to_self ', smoothing_ghost_distance_to_self)
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
        max_lat = np.amax(lat_prescribed)
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
        electrophysiology_parameter_name_list_in_order = hyperparameter_dict[
            'electrophysiology_parameter_name_list_in_order']
        # Spatial and temporal smoothing parameters:
        smoothing_dt = hyperparameter_dict['smoothing_dt']
        print('smoothing_dt ', smoothing_dt)
        # smoothing_past_present_window = hyperparameter_dict['smoothing_past_present_window']
        # print('smoothing_past_present_window ', smoothing_past_present_window)
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
        high_freq_cut = hyperparameter_dict['high_freq_cut']
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
        # Clear Arguments to prevent Argument recycling
        clinical_data_filename_path = None
        clinical_ecg_raw = None
        filtering = None
        max_len_ecg = None
        max_len_qrs = None
        normalise = None
        preprocessed_clinical_ecg_file_name = None
        v3_name = None
        v5_name = None
        zero_align = None
        ####################################################################################################################
        # Step 8: Define instance of the simulation method.
        print('Step 8: Define instance of the simulation method.')
        simulator_ecg = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
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
        # theta_name_list_in_order = None
        ####################################################################################################################
        # Step 10: Create evaluators for the ECG, LAT and VM.
        print('Step 10: Create evaluators for the ECG.')
        evaluator_ecg = ParameterSimulator(adapter=adapter, simulator=simulator_ecg, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        adapter = None
        simulator_ecg = None
        ####################################################################################################################
        # Step 11: Read the history of theta values.
        print('Step 11: Read the history of theta values.')
        theta_file_name_list = os.listdir(inference_history_dir)
        theta_file_path_list = [inference_history_dir + theta_file_name for theta_file_name in theta_file_name_list if
                                os.path.isfile(inference_history_dir + theta_file_name)]  # Filtering only the files.
        # print(*theta_file_path_list, sep="\n")
        max_iteration_count = 0
        for theta_file_path in theta_file_path_list:
            inference_iteration_count = int(theta_file_path.split('_')[-1][:-4])   # Take the last part minus ".csv"
            max_iteration_count = max(max_iteration_count, inference_iteration_count)
        population_theta_history = []
        for iteration_count in range(max_iteration_count):
            population_theta_iteration = np.loadtxt(inference_history_dir + history_theta_name_tag + str(iteration_count) + history_theta_name_end, delimiter=',')
            population_theta_history.append(np.unique(population_theta_iteration, axis=0))
        print(' max_iteration_count == len(population_theta_history) ',  max_iteration_count == len(population_theta_history))
        assert max_iteration_count == len(population_theta_history)
        ####################################################################################################################
        # Step 12: Simulate history of theta values.
        print('Step 12: define biomarker ECG metric.')
        # Arguments for history simulation and biomarkers calculation:
        # heart_rate = hyperparameter_dict['heart_rate']  # TODO uncomment
        # TODO Remove the following section and use dictionary value instead
        if anatomy_subject_name == 'DTI024':
            heart_rate = 66
        elif anatomy_subject_name == 'DTI032':
            heart_rate = 74
        elif anatomy_subject_name == 'DTI004':
            heart_rate = 48
        # Biomarker names and initialisation
        qtc_dur_name = get_qtc_dur_name()
        t_pe_name = get_t_pe_name()
        t_peak_name = get_t_peak_name()
        tpeak_dispersion_name = get_tpeak_dispersion_name()
        biomarker_name_list = [qtc_dur_name, t_pe_name, t_peak_name, tpeak_dispersion_name]
        metric = BiomarkerFromOnlyECG(biomarker_name_list=biomarker_name_list, heart_rate=heart_rate, lead_v3_i=lead_v3_i,
                                      lead_v5_i=lead_v5_i,
                                      qtc_dur_name=qtc_dur_name, qtpeak_dur_name=get_qtpeak_dur_name(),
                                      t_pe_name=t_pe_name, t_peak_name=t_peak_name, t_polarity_name=get_t_polarity_name(),
                                      tpeak_dispersion_name=tpeak_dispersion_name)
        ####################################################################################################################
        # Step 13: Simulate history of theta values.
        print('Step 13: Evaluate history ECGs.')
        # Arguments for history simulation and biomarkers calculation:
        # Initialisation of data structures
        population_ecg_history = []
        population_biomarker_history = []
        history_jump_size = 1
        iteration_history_list = list(range(len(population_theta_history)))[::history_jump_size]  # Only every n-th iteration of the inference
        max_iteration_number = max_iteration_count - 1
        if max_iteration_number not in iteration_history_list:   # Make sure that the last iteration is included
            iteration_history_list.append(max_iteration_number)
        print('len(population_theta_history) ', len(population_theta_history))
        print('iteration_history_list ', iteration_history_list)
        # History main loop for simulation and biomarker calculations
        for population_theta_i in iteration_history_list:
            # Check if the ecg simulation for this iteration has already been saved
            iteration_history_ecg_filename = history_precomputed_dir + history_ecg_name_tag + history_separator_tag \
                                             + str(population_theta_i) + '.csv'
            population_ecg_past = np.array([])  # Initialise the population of ECGs to an empty array
            # If the ECGs have already been simulated and there are exactly as many as parameter-sets, then load them and don't recompute them
            if os.path.isfile(iteration_history_ecg_filename):
                print('reading ecgs for ', population_theta_i)
                population_ecg_past = read_ecg_from_csv(filename=iteration_history_ecg_filename, nb_leads=nb_leads)
            # If the data for this iteration has not been saved or does not match theta in shape: Generate the data and save it.
            population_theta_past = population_theta_history[population_theta_i]
            if population_theta_past.shape[0] != population_ecg_past.shape[0]:
                print('generating ecgs for ', population_theta_i)
                population_ecg_past = evaluator_ecg.simulate_theta_population(theta_population=population_theta_past)
                save_ecg_to_csv(data=population_ecg_past, filename=iteration_history_ecg_filename)  # Save the ecgs for next time
            population_ecg_history.append(population_ecg_past)
            # Check if the biomarker data for this iteration has already been saved
            iteration_history_biomarker_filename = history_precomputed_dir + history_biomarker_name_tag \
                                                   + history_separator_tag + str(population_theta_i) + '.csv'
            if os.path.isfile(iteration_history_biomarker_filename):
                population_biomarker_past = read_csv_file(filename=iteration_history_biomarker_filename)
            # qt_dur, qt_dur_lead, t_pe, t_pe_lead, t_peak, t_peak_lead, qtpeak_dur, qtpeak_dur_lead, t_polarity, \
            #     t_polarity_lead, tpeak_dispersion = calculate_ecg_augmented_biomarker_from_only_ecg(
            #     max_lat=max_lat, predicted_ecg_list=population_ecg_past, lead_v3_i=lead_v3_i, lead_v5_i=lead_v5_i)
            # # Biomarker aggregation
            # population_biomarker_past = np.concatenate(([qt_dur[:, np.newaxis],
            #                                              t_pe[:, np.newaxis],
            #                                              t_peak[:, np.newaxis],
            #                                              tpeak_dispersion[:, np.newaxis]]), axis=1)
            # population_biomarker_history.append(population_biomarker_past)
            else:   # If the data for this iteration has not been saved: Generate the data and save it.
                max_lat_population = np.zeros((population_ecg_past.shape[0])) + max_lat  # Only valid for the T wave inference with prescribed LATs
                population_biomarker_past = metric.evaluate_metric_population(max_lat_population=max_lat_population, predicted_data_population=population_ecg_past)
                save_csv_file(data=population_biomarker_past, filename=iteration_history_biomarker_filename) # Save the biomarkers for next time
            population_biomarker_history.append(population_biomarker_past)
        # print('population_biomarker_history ', population_biomarker_history)
        print('History of populations of ecgs and biomarkers has been simulated and calcualted.')
        # Clear Arguments to prevent Argument recycling.
        qtc_dur_name = None
        population_qt_dur_history = None
        population_t_pe_history = None
        population_t_peak_history = None
        population_tpeak_dispersion_history = None
        t_pe_name = None
        t_peak_name = None
        tpeak_dispersion_name = None
        ####################################################################################################################
        # Step 14: Consistency check.
        print('Step 14: Consistency check.')
        print('len(population_ecg_history) == len(population_biomarker_history) ',
              len(population_ecg_history) == len(population_biomarker_history))
        print('len(iteration_history_list) ', len(iteration_history_list))
        print('len(population_ecg_history) ', len(population_ecg_history))
        print('len(population_biomarker_history) ', len(population_biomarker_history))
        assert len(iteration_history_list) == len(population_ecg_history)
        assert len(population_ecg_history) == len(population_biomarker_history)
        history_colour_list = np.linspace(0.9, 0., num=len(iteration_history_list))
        ####################################################################################################################
        # Step 15: Define the discrepancy metric.
        print('Step 15: Define the discrepancy metric.')
        # Arguments for discrepancy metrics:
        # Read hyperparameters
        error_method_name_inference_metric = hyperparameter_dict['error_method_name']
        # Create discrepancy metric instance using the inference metric:
        discrepancy_metric_inference = DiscrepancyECG(
            error_method_name=error_method_name_inference_metric)
        # Evaluate discrepancy of the final ECG iteration
        discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
            predicted_data_population=population_ecg_history[-1], target_data=clinical_ecg)
        best_discrepancy_index = np.argmin(discrepancy_population_inference)
        # Clear Arguments to prevent Argument recycling.
        discrepancy_metric_inference = None
        discrepancy_population_inference = None
        ####################################################################################################################
        # Step 16: Plotting of the history of Theta, ECGs and Biomarkers.
        print('Step 16: Plotting of the history of Theta, ECGs and Biomarkers.')
        # Plot theta history
        # Initialise arguments for plotting
        axes = None
        fig = None
        # Plot the theta inference History
        print('theta_name_list_in_order ', theta_name_list_in_order)
        print('population_theta_history[population_theta_i] ', population_theta_history[0].shape)
        for iteration_history_i in range(len(iteration_history_list)):
            population_theta_i = iteration_history_list[iteration_history_i]
            axes, fig = visualise_biomarker(biomarker_list=population_theta_history[population_theta_i],
                                            biomarker_name_list=theta_name_list_in_order, axes=axes,
                                            biomerker_color=str(history_colour_list[iteration_history_i]), fig=fig,
                                            label_list=None, x_axis_value=population_theta_i)
        print('population_theta_i ', population_theta_i)
        best_theta = population_theta_history[-1][best_discrepancy_index, :][np.newaxis, :]
        print('best_theta ', best_theta)
        axes, fig = visualise_biomarker(biomarker_list=best_theta,
                                        biomarker_name_list=theta_name_list_in_order, axes=axes,
                                        biomerker_color='red', biomarker_marker='*', biomarker_size=20.,
                                        fig=fig, label_list=None, x_axis_value=population_theta_i)
        theta_range_list = hyperparameter_dict['boundaries_continuous_theta']
        print('theta_range_list ', theta_range_list)
        print('apd_max ', hyperparameter_dict['apd_max_max'])
        print('apd_min ', hyperparameter_dict['apd_min_min'])
        # for ax_i in range(len(axes)):
        #     axes[ax_i].set_ylim(theta_range_list[ax_i])
        # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.show(block=False)
        fig.savefig(figure_theta_history_file_name)
        print('Saved theta figure: ', figure_theta_history_file_name)
        ####################################################################################################################
        # Step 17: Plotting of the history of ECGs.
        print('Step 17: Plotting of the history of ECGs.')
        # Plotting ECG history
        # Initialise arguments for plotting
        axes = None
        fig = None
        # Plot the ECG inference History
        for iteration_history_i in range(len(iteration_history_list)):
            ecg_list = population_ecg_history[iteration_history_i]
            axes, fig = visualise_ecg(ecg_list=ecg_list, lead_name_list=lead_names, axes=axes,
                                      ecg_color=str(history_colour_list[iteration_history_i]), fig=fig, label_list=None,
                                      linewidth=1.)
        # # Plot the best discreancy ECG trace after the last iteration
        # axes, fig = visualise_ecg(ecg_list=[ecg_list[best_discrepancy_index, :, :]], lead_name_list=lead_names, axes=axes,
        #                           ecg_color='cyan', fig=fig, label_list=['Best'],
        #                           linewidth=2.)
        # Plot the clinical trace after the last iteration
        axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                                  ecg_color='lime', fig=fig, label_list=['Clinical'],
                                  linewidth=2.)
        # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.show(block=False)
        fig.savefig(figure_ecg_history_file_name)
        print('Saved ecg figure: ', figure_ecg_history_file_name)
        ####################################################################################################################
        # Step 18: Plotting of the history of Biomarkers.
        print('Step 18: Plotting of the history of Biomarkers.')
        # Plot the Biomarker inference History
        # # Calculate ground truth values - These have been manually calculated
        # clinical_qt_interval = [345, 349, 318, 342, 343, 343, 347, 355]
        # clinical_qt_interval_mean = np.mean(clinical_qt_interval)
        # clinical_qtpeak_interval = [280, 284, 257, 269, 276, 280, 288, 293]
        # clinical_tpeak_dispersion = clinical_qtpeak_interval[lead_v5_i] - clinical_qtpeak_interval[lead_v3_i]
        # clinical_tpeak = [0.138, 0.292, 0.155, 0.471, 0.340, 0.246, 0.187, 0.101]
        # clinical_tpeak_mean = np.mean(clinical_tpeak)
        # clinical_t_pe_mean = np.mean(clinical_qt_interval_mean-clinical_qtpeak_interval)
        # # clinical_t_polarity = [1., 1., 1., 1., 1., 1., 1., 1.]
        # ground_truth_biomarker_manual = [clinical_qt_interval_mean, clinical_t_pe_mean, clinical_tpeak_mean, clinical_tpeak_dispersion]
        # print('ground_truth_biomarker ', ground_truth_biomarker_manual)
        # Calculate ground truth values - These have been automatically calculated assuming that the inferred LATs were ground truth
        ground_truth_biomarker_automatic = metric.evaluate_metric_population(max_lat_population=np.array([[max_lat]]), predicted_data_population=clinical_ecg[np.newaxis, :, :])[0]
        print('ground_truth_biomarker_automatic ', ground_truth_biomarker_automatic)
        # Initialise arguments for plotting
        axes = None
        fig = None
        # Plot the ECG Biomarker inference History
        print('biomarker_name_list ', biomarker_name_list)
        for iteration_history_i in range(len(iteration_history_list)):
            population_theta_i = iteration_history_list[iteration_history_i]
            axes, fig = visualise_biomarker(biomarker_list=population_biomarker_history[iteration_history_i],
                                            biomarker_name_list=biomarker_name_list, axes=axes,
                                            biomerker_color=str(history_colour_list[iteration_history_i]), fig=fig,
                                            ground_truth_biomarker=ground_truth_biomarker_automatic, ground_truth_color='lime',
                                            label_list=None, x_axis_value=population_theta_i)
        # axes, fig = visualise_biomarker(biomarker_list=[],
        #                                 biomarker_name_list=biomarker_name_list, axes=axes,
        #                                 biomerker_color=None, fig=fig,
        #                                 ground_truth_biomarker=ground_truth_biomarker_manual, ground_truth_color='red',
        #                                 label_list=None, x_axis_value=None)
        print('population_theta_i ', population_theta_i)
        best_biomarker = population_biomarker_history[-1][best_discrepancy_index, :][np.newaxis, :]
        print('best_biomarker ', best_biomarker)
        axes, fig = visualise_biomarker(biomarker_list=best_biomarker,
                                        biomarker_name_list=biomarker_name_list, axes=axes,
                                        biomerker_color='red', biomarker_marker='*', biomarker_size=20.,
                                        fig=fig, label_list=None, x_axis_value=population_theta_i)
        # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.show(block=False)
        fig.savefig(figure_biomarker_history_file_name)
        print('Saved biomarker figure: ', figure_biomarker_history_file_name)
        # Clear Arguments to prevent Argument recycling.
        clinical_ecg = None
        evaluator_ecg = None
        max_lat = None
        parameter_name_list_in_order = None
        parameter_population = None
        population_theta = None
        anatomy_subject_name = None
        best_theta = None
        best_parameter = None
        evaluator_ep = None
        figure_ecg_history_file_name = None
        frequency = None
        geometry = None
        inferred_theta_population = None
        raw_geometry = None
        results_dir = None
        theta_file_name_list = None
        unprocessed_node_mapping_index = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

    # EOF



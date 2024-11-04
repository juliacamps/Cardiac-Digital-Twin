"""This script visualises the results from the inference of repolarisation properties from the T wave"""
import os
import sys
from warnings import warn
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) < 2:
        anatomy_subject_name = 'rodero_13'  # 'rodero_13' # 'rodero_13'  # 'DTI004'  # 'UKB_1000532' #'UKB_1000268'
        ecg_subject_name = 'DTI004'  # 'DTI004'  # 'UKB_1000532' # 'UKB_1000268'  # Allows using a different ECG for the personalisation than for the anatomy
        # anatomy_subject_name = 'UKB_1008115'  # 'DTI004'  # 'UKB_1000532' #'UKB_1000268'
        # ecg_subject_name = 'UKB_1008115'  # 'DTI004'  # 'UKB_1000532' # 'UKB_1000268'  # Allows using a different ECG for the personalisation than for the anatomy
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
    from conduction_system import DjikstraConductionSystemVC, EmptyConductionSystem
    from ecg_functions import PseudoEcgTetFromVM
    from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet
    from propagation_models import EikonalDjikstraTet
    from simulator_functions import SimulateECG, SimulateEP
    from adapter_theta_params import AdapterThetaParams, RoundTheta
    from discrepancy_functions import DiscrepancyECG, BiomarkerFromOnlyECG
    from evaluation_functions import DiscrepancyEvaluator, ParameterSimulator
    from cellular_models import CellularModelBiomarkerDictionary
    from electrophysiology_functions import ElectrophysiologyAPDmap
    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields, read_dictionary, save_ecg_to_csv, \
    export_ensight_timeseries_case, save_pandas, save_csv_file, read_ecg_from_csv, read_csv_file
    from utils import map_indexes, remap_pandas_from_row_index, get_qt_dur_name, \
    get_t_pe_name, get_t_peak_name, get_tpeak_dispersion_name, get_qtpeak_dur_name, \
    get_t_polarity_name
    from postprocess_functions import generate_repolarisation_map, visualise_ecg

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
    source_resolution = 'coarse'
    target_resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    cellular_data_dir = data_dir + 'cellular_data/'
    geometric_data_dir = data_dir + 'geometric_data/'
    results_dir_root = path_dict["results_path"]
    # Intermediate Paths: # e.g., results from the QRS inference
    experiment_type = 'personalisation'
    ep_model = 'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    # Use date to name the result folder to preserve some history of results
    current_month_text = datetime.now().strftime('%h')  # Feb
    current_year_full = datetime.now().strftime('%Y')  # 2018
    results_dir = results_dir_root + experiment_type + '_data/' + anatomy_subject_name + '/qt_' \
                  + gradient_ion_channel_str + '_' + ep_model + '/' + current_month_text + '_' + current_year_full + '/'
    # Read hyperparamter dictionary
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_hyperparameter.txt'
    hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
    result_tag = hyperparameter_dict['result_tag']
    parameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_parameter_population.csv'
    # Output Paths:
    population_metric_dir = results_dir + 'inference_population_metrics/'
    if not os.path.exists(population_metric_dir):
        os.mkdir(population_metric_dir)
    visualisation_dir = results_dir + 'ensight/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    figure_result_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_population.png'
    # Uncertainty
    uncertainty_dir = results_dir + 'drug_testing_population/'
    if not os.path.exists(uncertainty_dir):
        os.mkdir(uncertainty_dir)
    uncertainty_biomarker_result_file_name_start = uncertainty_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-biomarker_'
    uncertainty_biomarker_result_file_name_end = '.csv'
    uncertainty_parameter_population_file_name = uncertainty_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_drug_parameter_population.csv'
    # Best discrepancy
    translation_dir = results_dir + 'best_discrepancy/'
    if not os.path.exists(translation_dir):
        os.mkdir(translation_dir)
    lat_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-lat.csv'
    vm_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-vm.csv'
    best_parameter_result_file_name = translation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '-best-parameter.csv'
    biomarker_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-biomarker.csv'
    # Precomputed
    precomputed_dir = results_dir + 'precomputed/'
    if not os.path.exists(precomputed_dir):
        os.mkdir(precomputed_dir)
    ecg_population_file_name = precomputed_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_ecg_population.csv'
    max_lat_population_file_name = precomputed_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_max_lat_population.csv'
    preprocessed_clinical_ecg_file_name = precomputed_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_ecg_clinical.csv'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Read hyperparameters
    clinical_data_filename = hyperparameter_dict['clinical_data_filename']
    clinical_data_filename_path = data_dir + clinical_data_filename
    clinical_qrs_offset = hyperparameter_dict['clinical_qrs_offset']
    # qrs_lat_prescribed_filename = hyperparameter_dict['qrs_lat_prescribed_filename']
    # qrs_lat_prescribed_filename_path = results_dir_root + qrs_lat_prescribed_filename
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
    data_dir = None
    ecg_subject_name = None
    # qrs_lat_prescribed_filename = None
    results_dir_root = None
    ####################################################################################################################
    ########## TODO THE FOLLOWING CODE AVOIDS RECALCULATING EVERYTHING EVERY TIME THIS SCRIPT IS RUN, BUT ITS
    ########## TODO NOT CONSISTENT IN STYLE WITH THE REST OF THE REPOSITORY
    # TODO THESE RESULTS SHOULD NOT BE SAVED IN THE MIDDLE OF THE RESULTS REPOSITORY, THAT FOLDER IS RESERVED FOR THE INFERENCE PROCESS ONLY
    # raise ()  # TODO Remove this line after defining an alternative path for saving precomputed results
    lead_names = hyperparameter_dict['lead_names']
    nb_leads = hyperparameter_dict['nb_leads']
    v3_name = hyperparameter_dict['v3_name']
    v5_name = hyperparameter_dict['v5_name']
    lead_v3_i = lead_names.index(v3_name)
    lead_v5_i = lead_names.index(v5_name)
    # lat_prescribed = (np.loadtxt(qrs_lat_prescribed_filename_path, delimiter=',')).astype(int)
    # max_lat = np.amax(lat_prescribed)
    assert nb_leads == len(lead_names)
    # Clear Arguments to prevent Argument recycling
    v3_name = None
    v5_name = None
    # TODO take out that False!!
    if not(#os.path.exists(population_metric_dir + 'discrepancy_population_inference_metric.npy') and
           os.path.exists(ecg_population_file_name) and os.path.exists(max_lat_population_file_name) and
           os.path.exists(preprocessed_clinical_ecg_file_name)):
        print('os.path.exists(population_metric_dir + discrepancy_population_inference_metric.npy) ', os.path.exists(population_metric_dir + 'discrepancy_population_inference_metric.npy'))
        print('os.path.exists(ecg_population_file_name) ', os.path.exists(ecg_population_file_name))
        print('os.path.exists(max_lat_population_file_name) ', os.path.exists(max_lat_population_file_name))
        print('os.path.exists(preprocessed_clinical_ecg_file_name) ', os.path.exists(preprocessed_clinical_ecg_file_name))
        print('preprocessed_clinical_ecg_file_name ', preprocessed_clinical_ecg_file_name)
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
        ep_model = hyperparameter_dict['ep_model']
        list_celltype_name = hyperparameter_dict['list_celltype_name']
        stimulation_protocol = hyperparameter_dict['stimulation_protocol']
        cellular_data_dir_complete = cellular_data_dir + cellular_model_convergence + '_' + stimulation_protocol + '_' + str(
            cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model + '/'
        # Create cellular model instance.
        cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                          biomarker_apd90_name=biomarker_apd90_name,
                                                          biomarker_celltype_name=biomarker_celltype_name,
                                                          cellular_data_dir=cellular_data_dir_complete,
                                                          cellular_model_name=cellular_model_name,
                                                          list_celltype_name=list_celltype_name, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        # biomarker_apd90_name = None
        biomarker_upstroke_name = None
        cellular_data_dir = None
        cellular_data_dir_complete = None
        cellular_model_name = None
        cellular_stim_amp = None
        cellular_model_convergence = None
        ep_model = None
        stimulation_protocol = None
        ####################################################################################################################
        # Step 3: Generate a cardiac geometry.
        print('Step 3: Generate a cardiac geometry.')
        # Argument setup: (in Alphabetical order)
        # Read hyperparameters
        vc_ab_name = hyperparameter_dict['vc_ab_name']
        vc_ab_cut_name = hyperparameter_dict['vc_ab_cut_name']
        vc_aprt_name = hyperparameter_dict['vc_aprt_name']
        vc_rt_name = hyperparameter_dict['vc_rt_name']
        vc_rvlv_name = hyperparameter_dict['vc_rvlv_name']
        # vc_sep_name = hyperparameter_dict['vc_sep_name']
        vc_tm_name = hyperparameter_dict['vc_tm_name']
        # vc_tv_name = hyperparameter_dict['vc_aprt_name']
        endo_celltype_name = hyperparameter_dict['endo_celltype_name']  # TODO is this necessary?
        epi_celltype_name = hyperparameter_dict['epi_celltype_name']  # TODO is this necessary?
        celltype_vc_info = hyperparameter_dict['celltype_vc_info']
        # print('celltype_vc_info ', celltype_vc_info)
        vc_name_list = hyperparameter_dict['vc_name_list']
        # print('vc_name_list ', vc_name_list)
        # Create geometry with a dummy conduction system to allow initialising the geometry.
        geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                   conduction_system=EmptyConductionSystem(verbose=verbose),
                                   geometric_data_dir=geometric_data_dir, resolution=source_resolution,
                                   subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
        raw_geometry = RawEmptyCardiacGeoTet(conduction_system=EmptyConductionSystem(verbose=verbose),
                                             geometric_data_dir=geometric_data_dir, resolution=target_resolution,
                                             subject_name=anatomy_subject_name, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        geometric_data_dir = None
        list_celltype_name = None
        source_resolution = None
        vc_name_list = None
        ####################################################################################################################
        # Step 4: Create conduction system for the propagation model to be initialised.
        # TODO load this properties from hyperparameters and inference results!!!
        print('Step 4: Create rule-based Purkinje network using ventricular coordinates.')
        # Arguments for Conduction system:
        approx_djikstra_purkinje_max_path_len = hyperparameter_dict['approx_djikstra_purkinje_max_path_len']
        lv_inter_root_node_distance = hyperparameter_dict['lv_inter_root_node_distance']
        rv_inter_root_node_distance = hyperparameter_dict['rv_inter_root_node_distance']
        # Create conduction system
        conduction_system = DjikstraConductionSystemVC(
            approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, geometry=geometry,
            lv_candidate_root_node_meta_index=, rv_candidate_root_node_meta_index=, purkinje_max_ab_cut_threshold=,
            vc_ab_cut_name=vc_ab_cut_name, vc_rt_name=vc_rt_name, verbose=verbose)
        # Assign conduction_system to its geometry
        geometry.set_conduction_system(conduction_system)
        # Clear Arguments to prevent Argument recycling
        approx_djikstra_purkinje_max_path_len = None
        conduction_system = None
        lv_inter_root_node_distance = None
        rv_inter_root_node_distance = None
        ####################################################################################################################
        # Step 5: Prepare smoothing configuration to resemble diffusion effects
        print('Step 5: Prepare smoothing configuration to resemble diffusion effects.')
        # Define the speeds used during the fibre-based smoothing
        warn(
            'Inference from QT can, but does NOT, update the speeds in the smoothing function!\nAlso, it requires some initial fixed values!')
        fibre_speed_name = hyperparameter_dict['fibre_speed_name']
        transmural_speed_name = hyperparameter_dict['sheet_speed_name']
        normal_speed_name = hyperparameter_dict['normal_speed_name']
        fibre_speed = hyperparameter_dict[fibre_speed_name]
        sheet_speed = hyperparameter_dict[transmural_speed_name]
        normal_speed = hyperparameter_dict[normal_speed_name]

        # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
        smoothing_ghost_distance_to_self = hyperparameter_dict['smoothing_ghost_distance_to_self']  # cm # This parameter enables to control how much spatial smoothing happens and
        # smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 5% and the present as 95%
        # full_smoothing_time_index = 400  # (ms) assumming 1000Hz
        warn('Precompuing the smoothing, change this please!')  # TODO refactor
        geometry.precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(
            fibre_speed=fibre_speed, sheet_speed=sheet_speed, normal_speed=normal_speed,
            ghost_distance_to_self=smoothing_ghost_distance_to_self)
        ####################################################################################################################
        # Step 6: Create Eikonal instance. Eikonal will require a conduction and an Eikonal-friendly mesh on creation.
        print('Step 6: Create propagation model instance.')
        # Arguments for propagation model:
        fibre_speed_name = hyperparameter_dict['fibre_speed_name']
        transmural_speed_name = hyperparameter_dict['sheet_speed_name']
        normal_speed_name = hyperparameter_dict['normal_speed_name']
        endo_dense_speed_name = hyperparameter_dict['endo_dense_speed_name']
        endo_sparse_speed_name = hyperparameter_dict['endo_sparse_speed_name']
        purkinje_speed_name = hyperparameter_dict['purkinje_speed_name']
        nb_speed_parameters = hyperparameter_dict['nb_speed_parameters']
        nb_candidate_root_nodes = geometry.get_nb_candidate_root_node()
        candidate_root_node_names = ['r' + str(root_i) for root_i in range(nb_candidate_root_nodes)]
        propagation_parameter_name_list_in_order = hyperparameter_dict['propagation_parameter_name_list_in_order']
        propagation_model = EikonalDjikstraTet(
            endo_dense_speed_name=endo_dense_speed_name, endo_sparse_speed_name=endo_sparse_speed_name,
            fibre_speed_name=fibre_speed_name, geometry=geometry, module_name=propagation_module_name,
            nb_speed_parameters=nb_speed_parameters, normal_speed_name=normal_speed_name,
            parameter_name_list_in_order=propagation_parameter_name_list_in_order,
            purkinje_speed_name=purkinje_speed_name,
            transmural_speed_name=transmural_speed_name, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        nb_speed_parameters = None
        ####################################################################################################################
        # Step 7: Create Whole organ Electrophysiology model.
        print('Step 7: Create Whole organ Electrophysiology model.')
        # Read hyperparameters
        apd_max_name = hyperparameter_dict['apd_max_name']
        apd_min_name = hyperparameter_dict['apd_min_name']
        g_vc_ab_name = hyperparameter_dict['g_vc_ab_name']
        g_vc_aprt_name = hyperparameter_dict['g_vc_aprt_name']
        g_vc_rvlv_name = hyperparameter_dict['g_vc_rvlv_name']
        g_vc_tm_name = hyperparameter_dict['g_vc_tm_name']
        electrophysiology_parameter_name_list_in_order = hyperparameter_dict['electrophysiology_parameter_name_list_in_order']
        # Spatial and temporal smoothing parameters:
        smoothing_count = hyperparameter_dict['smoothing_count']
        smoothing_past_present_window = hyperparameter_dict['smoothing_past_present_window']
        full_smoothing_time_index = hyperparameter_dict['full_smoothing_time_index']
        electrophysiology_model = ElectrophysiologyAPDmap(apd_max_name=apd_max_name, apd_min_name=apd_min_name,
                                                      cellular_model=cellular_model,
                                                      fibre_speed_name=fibre_speed_name,
                                                      full_smoothing_time_index=full_smoothing_time_index,
                                                      module_name=electrophysiology_module_name,
                                                      normal_speed_name=normal_speed_name,
                                                      parameter_name_list_in_order=electrophysiology_parameter_name_list_in_order,
                                                      propagation_model=propagation_model,
                                                      sheet_speed_name=transmural_speed_name,
                                                      smoothing_count=smoothing_count,
                                                      smoothing_ghost_distance_to_self=smoothing_ghost_distance_to_self,
                                                      smoothing_past_present_window=np.asarray(
                                                          smoothing_past_present_window),
                                                      verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        cellular_model = None
        propagation_model = None
        smoothing_count = None
        smoothing_ghost_distance_to_self = None
        smoothing_past_present_window = None
        ####################################################################################################################
        # Step 8: Create ECG calculation method.
        print('Step 8: Create ECG calculation method.')
        # Arguments for ECG calculation:
        # Read hyperparameters
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
        untrimmed_clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
        clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
        untrimmed_clinical_ecg_raw = None   # Clear Arguments to prevent Argument recycling
        # Create ECG model
        ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                       frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                       low_freq_cut=low_freq_cut,
                                       max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                       nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                       reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                       tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
        clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
        save_ecg_to_csv(data=clinical_ecg, filename=preprocessed_clinical_ecg_file_name)
        # Clear Arguments to prevent Argument recycling
        clinical_data_filename_path = None
        clinical_ecg_raw = None
        filtering = None
        high_freq_cut = None
        low_freq_cut = None
        max_len_ecg = None
        max_len_qrs = None
        normalise = None
        zero_align = None
        ####################################################################################################################
        # Step 9: Define instance of the simulation method.
        print('Step 9: Define instance of the simulation method.')
        simulator_ecg = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
        simulator_ep = SimulateEP(electrophysiology_model=electrophysiology_model, verbose=verbose)    # Clear Arguments to prevent Argument recycling
        electrophysiology_model = None
        ecg_model = None
        ####################################################################################################################
        # Step 10: Define Adapter to translate between theta and parameters.
        print('Step 10: Define Adapter to translate between theta and parameters.')
        # Read hyperparameters
        # TODO make the following code into a for loop!!
        # Theta resolutions
        endo_dense_speed_resolution = hyperparameter_dict['endo_dense_speed_resolution']
        endo_sparse_speed_resolution = hyperparameter_dict['endo_sparse_speed_resolution']
        transmural_speed_resolution = hyperparameter_dict['transmural_speed_resolution']
        apd_max_resolution = hyperparameter_dict['apd_max_resolution']
        apd_min_resolution = hyperparameter_dict['apd_min_resolution']
        g_vc_ab_resolution = hyperparameter_dict['g_vc_ab_resolution']
        g_vc_aprt_resolution = hyperparameter_dict['g_vc_aprt_resolution']
        g_vc_rvlv_resolution = hyperparameter_dict['g_vc_rvlv_resolution']
        g_vc_tm_resolution = hyperparameter_dict['g_vc_tm_resolution']
        theta_adjust_function_list_in_order = [RoundTheta(resolution=transmural_speed_resolution),
                                           RoundTheta(resolution=endo_dense_speed_resolution),
                                           RoundTheta(resolution=endo_sparse_speed_resolution),
                                           RoundTheta(resolution=apd_max_resolution),
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
        transmural_speed_name = None
        normal_speed_name = None
        endo_dense_speed_name = None
        endo_sparse_speed_name = None
        parameter_fixed_value_dict = None
        theta_name_list_in_order = None
        ####################################################################################################################
        # Step 11: Create evaluators for the ECG, LAT and VM.
        print('Step 11: Create evaluators for the ECG, LAT and VM.')
        evaluator_ecg = ParameterSimulator(adapter=adapter, simulator=simulator_ecg, verbose=verbose)
        evaluator_ep = ParameterSimulator(adapter=adapter, simulator=simulator_ep, verbose=verbose)
        ####################################################################################################################
        # Step 12: Read the values inferred for parameters and evaluate their ECG.
        print('Step 12: Read the values inferred for parameters and evaluate their ECG.')
        # TODO save candidate root nodes and their times so that the meta-indexes can be used to point at them.
        pandas_parameter_population = pd.read_csv(parameter_result_file_name, delimiter=',')
        parameter_population = evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
        # Simulate the ECGs from the parameter population from the inference
        population_ecg = evaluator_ecg.simulate_parameter_population(parameter_population=parameter_population)
        save_ecg_to_csv(data=population_ecg, filename=ecg_population_file_name)
        # Simulate the LATs from the parameter population from the inference
        # TODO The LAT population is needed because there is no function to compute the QRS offset, so the max(LAT) is used instead!
        population_lat, population_vm = evaluator_ep.simulate_parameter_population(parameter_population=parameter_population)
        print('population_lat ', population_lat.shape)
        population_max_lat = np.amax(population_lat, axis=1)
        save_csv_file(data=population_max_lat, filename=max_lat_population_file_name)
        # Clear Arguments to prevent Argument recycling.
        evaluator_ecg = None
        pandas_parameter_population = None
        parameter_result_file_name = None
        population_lat = None
        population_vm = None
        ####################################################################################################################
        # Step 13: Define the discrepancy metric and make sure that the result is the same when calling the evaluator.
        print('Step 13: Define the discrepancy metric.')
        # Arguments for discrepancy metrics:
        # Read hyperparameters
        error_method_name_inference_metric = hyperparameter_dict['error_method_name']
        print('error_method_name_inference_metric ', error_method_name_inference_metric)
        # Create discrepancy metric instance using the inference metric:
        discrepancy_metric_inference = DiscrepancyECG(
            error_method_name=error_method_name_inference_metric)
        # Evaluate discrepancy:
        discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
            predicted_data_population=population_ecg, target_data=clinical_ecg)
        # Create discrepancy evaluator to assess code correctness!!!
        evaluator_ecg_inference_metric = DiscrepancyEvaluator(
            adapter=adapter, discrepancy_metric=discrepancy_metric_inference, simulator=simulator_ecg,
            target_data=clinical_ecg, verbose=verbose)
        discrepancy_population_inference_from_evaluator = evaluator_ecg_inference_metric.evaluate_parameter_population(
            parameter_population=parameter_population)
        if not(np.all(discrepancy_population_inference == discrepancy_population_inference_from_evaluator)):
            warn('These should be identical: discrepancy_population_inference '
                 + str(discrepancy_population_inference.shape)
                 + ' discrepancy_population_inference_from_evaluator '
                 + str(discrepancy_population_inference_from_evaluator.shape))
        # Clear Arguments to prevent Argument recycling.
        discrepancy_metric_inference = None
        error_method_name_inference_metric = None
        evaluator_ecg_inference_metric = None
        population_discrepancy_inference_from_evaluator = None
        ####################################################################################################################
        # Step 14: Select best discrepancy particle.
        print('Step 14: Select best discrepancy particle.')
        best_parameter = parameter_population[np.argmin(discrepancy_population_inference)]
        np.savetxt(best_parameter_result_file_name, best_parameter[np.newaxis, :], delimiter=',',
                   header=','.join(parameter_name_list_in_order), comments='')
        print('Saved best parameter: ', best_parameter_result_file_name)
        # Clear Arguments to prevent Argument recycling.
        best_parameter_result_file_name = None
        discrepancy_population_inference = None
        ####################################################################################################################
        # Step 15: Interpolate simulation results to have the same indexing that the input data files.
        print('15: Interpolate simulation results to have the same indexing that the input data files.')
        unprocessed_node_best_lat, unprocessed_node_best_vm = evaluator_ep.simulate_parameter_particle(
            parameter_particle=best_parameter)
        # Interpolate nodefield
        unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=raw_geometry.get_node_xyz(),
                                                     reference_points_xyz=geometry.get_node_xyz())
        best_vm = unprocessed_node_best_vm[unprocessed_node_mapping_index, :]
        # np.savetxt(lat_result_file_name, best_lat, delimiter=',')
        # print('Saved best lat: ', lat_result_file_name)
        np.savetxt(vm_result_file_name, best_vm, delimiter=',')
        print('Saved best vm: ', vm_result_file_name)
        export_ensight_timeseries_case(dir=visualisation_dir,
                                       casename=anatomy_subject_name + '_' + target_resolution + '_RE',
                                       dataname_list=['INTRA'],
                                       vm_list=[best_vm], dt=1. / frequency, nodesxyz=raw_geometry.get_node_xyz(),
                                       tetrahedrons=raw_geometry.get_tetra())
        print('Saved best ensight vm: ', visualisation_dir)
        # Clear Arguments to prevent Argument recycling.
        adapter = None
        simulator_ep = None
        ####################################################################################################################
        # Step 16: Save EP configuration for translation to Monodomain.
        print('Step 16: Save EP configuration for translation to Monodomain.')
        # Arguments for discrepancy metric:
        activation_time_map_biomarker_name = 'lat'  # TODO make these names globally defined in utils.py
        repolarisation_time_map_biomarker_name = 'repol'  # TODO make these names globally defined in utils.py
        # Calculate nodewise biomarkers for translation to Alya:
        best_lat = unprocessed_node_best_lat[unprocessed_node_mapping_index]
        best_repol = generate_repolarisation_map(best_vm)
        # TODO this requires all biomarkers to be numerical values!! Should this be checked? Or throw a warning or print?
        unprocessed_node_biomarker = evaluator_ep.biomarker_parameter_particle(parameter_particle=best_parameter)
        node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker,
                                                     row_index=unprocessed_node_mapping_index)
        node_biomarker[activation_time_map_biomarker_name] = best_lat
        node_biomarker[repolarisation_time_map_biomarker_name] = best_repol
        save_pandas(df=node_biomarker, filename=biomarker_result_file_name)
        print('Saved biomarkers that allow translation to Alya ', biomarker_result_file_name)
        node_sf_list = []
        for ionic_scaling_name in gradient_ion_channel_list:
            node_sf_list.append(node_biomarker[ionic_scaling_name])
        node_apd90 = node_biomarker[biomarker_apd90_name]
        node_celltype_str = node_biomarker[biomarker_celltype_name]
        node_celltype = np.zeros(node_apd90.shape)
        node_celltype[
            node_celltype_str == endo_celltype_name] = 1  # TODO Is this something that gets used by Alya? if Yes, then define in utils.py
        node_celltype[
            node_celltype_str == epi_celltype_name] = 3  # TODO Is this something that gets used by Alya? if Yes, then define in utils.py
        node_transmural = geometry.node_vc[vc_tm_name]
        node_rvlv = geometry.node_vc[vc_rvlv_name]
        write_geometry_to_ensight_with_fields(
            geometry=raw_geometry,
            node_field_list=[best_lat, node_apd90, node_celltype, best_repol, node_transmural,
                             node_rvlv] + node_sf_list,
            node_field_name_list=[activation_time_map_biomarker_name, biomarker_apd90_name, biomarker_celltype_name,
                                  repolarisation_time_map_biomarker_name, vc_tm_name,
                                  vc_rvlv_name] + gradient_ion_channel_list,
            subject_name=anatomy_subject_name + '_' + target_resolution + '_sf', verbose=verbose,
            visualisation_dir=visualisation_dir)
        # Clear Arguments to prevent Argument recycling.
        visualisation_dir = None
        ####################################################################################################################
        # Step 17: Randomly select some % of the particles in the final population and save their biomarkers.
        # TODO move this to another script because it's not related to visualising the results
        # if False:
        print('Step 17: Randomly select some % of the particles in the final population and save their biomarkers.')
        uncertainty_proportion = 0.05   # 5% of the population size
        population_size = parameter_population.shape[
            0]  # The population size is computed with respect to the initial population
        unique_parameter_population = np.unique(parameter_population, axis=0)
        unique_population_size = unique_parameter_population.shape[0]
        nb_uncertainty_particles = math.ceil(uncertainty_proportion * population_size)
        assert nb_uncertainty_particles < unique_population_size, 'We cannot sample more particles than the unique ones available in the final population!'
        uncertainty_index = np.random.permutation(unique_population_size)[:nb_uncertainty_particles]
        uncertainty_parameter_population = unique_parameter_population[uncertainty_index, :]
        print('uncertainty_index ', uncertainty_index)
        print('uncertainty_parameter_population ', uncertainty_parameter_population.shape)
        save_csv_file(data=uncertainty_parameter_population, filename=uncertainty_parameter_population_file_name,
                      column_name_list=parameter_name_list_in_order)
        # Calculate the effect of uncertainty in the biomarkers and save them
        for uncertainty_i in range(uncertainty_parameter_population.shape[0]):
            uncertainty_biomarker_result_file_name = uncertainty_biomarker_result_file_name_start + str(
                uncertainty_i) + uncertainty_biomarker_result_file_name_end
            # TODO comment the following line
            if not os.path.exists(uncertainty_biomarker_result_file_name):  # TODO 2024/01/26 why is this here and why it should be commented out?
                uncertainty_parameter_particle = uncertainty_parameter_population[uncertainty_i, :]
                unprocessed_node_biomarker = evaluator_ep.biomarker_parameter_particle(
                    parameter_particle=uncertainty_parameter_particle)
                node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker,
                                                             row_index=unprocessed_node_mapping_index)
                # Save biomarkers to allow translation to Alya
                save_pandas(df=node_biomarker, filename=uncertainty_biomarker_result_file_name)
                print('Saved: ', uncertainty_biomarker_result_file_name)

        print('Save results.')
        # TODO it's a bit inconvenient that its using numpy saving functions instead of csv
        np.save(population_metric_dir + 'parameter_name_list_in_order.npy', parameter_name_list_in_order)
        np.save(population_metric_dir + 'best_parameter.npy', best_parameter)
        np.save(population_metric_dir + 'uncertainty_parameter_population.npy', uncertainty_parameter_population)
        # Clear Arguments to prevent Argument recycling.
        anatomy_subject_name = None
        best_theta = None
        ecg_population_file_name = None
        evaluator_ep = None
        figure_result_file_name = None
        frequency = None
        geometry = None
        inferred_theta_population = None
        max_lat_population_file_name = None
        population_metric_dir = None
        preprocessed_clinical_ecg_file_name = None
        raw_geometry = None
        uncertainty_parameter_population_file_name = None
        unprocessed_node_mapping_index = None
    else:   # This means that the postprocessing from the inference has been previously precomputed
        # Step 18: Read precomputed results.
        print('Step 18: Read precomputed results.')
        population_ecg = read_ecg_from_csv(filename=ecg_population_file_name, nb_leads=nb_leads)
        population_max_lat = read_csv_file(filename=max_lat_population_file_name)
        clinical_ecg = np.squeeze(read_ecg_from_csv(filename=preprocessed_clinical_ecg_file_name, nb_leads=nb_leads))
        # TODO it's a bit inconvenient that its using numpy saving functions instead of csv
        unique_parameter_population = np.load(population_metric_dir + 'unique_parameter_population.npy')
        parameter_name_list_in_order = np.load(population_metric_dir + 'parameter_name_list_in_order.npy')
        best_parameter = np.load(population_metric_dir + 'best_parameter.npy')
        # Clear Arguments to prevent Argument recycling.
        ecg_population_file_name = None
        max_lat_population_file_name = None
        population_metric_dir = None
        preprocessed_clinical_ecg_file_name = None
    ####################################################################################################################
    # Step 19: Calcualte ECG metrics for the final population.
    print('Step 19: Calcualte ECG metrics for the final population.')
    # Arguments for history simulation and biomarkers calculation:
    # Biomarker names and initialisation
    qt_dur_name = get_qt_dur_name()
    t_pe_name = get_t_pe_name()
    t_peak_name = get_t_peak_name()
    tpeak_dispersion_name = get_tpeak_dispersion_name()
    biomarker_name_list = [qt_dur_name, t_pe_name, t_peak_name, tpeak_dispersion_name]

    print('population_max_lat ', population_max_lat.shape)
    print('population_ecg ', population_ecg.shape)
    biomarker_metric = BiomarkerFromOnlyECG(biomarker_name_list=biomarker_name_list, lead_v3_i=lead_v3_i,
                                            lead_v5_i=lead_v5_i, qt_dur_name=qt_dur_name,
                                            qtpeak_dur_name=get_qtpeak_dur_name(),
                                            t_pe_name=t_pe_name, t_peak_name=t_peak_name,
                                            t_polarity_name=get_t_polarity_name(),
                                            tpeak_dispersion_name=tpeak_dispersion_name)
    population_biomarker = biomarker_metric.evaluate_metric_population(
        max_lat_population=population_max_lat, predicted_data_population=population_ecg)
    # Discrepancy used during the inference process:
    error_method_name_inference_metric = hyperparameter_dict['error_method_name']
    # Create discrepancy metric instance using the inference metric:
    discrepancy_metric_inference = DiscrepancyECG(error_method_name=error_method_name_inference_metric)
    # Evaluate discrepancy:
    discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
        predicted_data_population=population_ecg, target_data=clinical_ecg)
    # save_csv_file(data=population_biomarker, filename=biomarker_population_file_name)
    # RMSE between predicted and clinical ECGs:
    error_method_name_rmse = 'rmse'
    discrepancy_metric_rmse = DiscrepancyECG(error_method_name=error_method_name_rmse)
    discrepancy_population_rmse = discrepancy_metric_rmse.evaluate_metric_population(
        predicted_data_population=population_ecg, target_data=clinical_ecg)
    # PCC between predicted and clinical ECGs:
    error_method_name_pcc = 'pcc'
    discrepancy_metric_pcc = DiscrepancyECG(error_method_name=error_method_name_pcc)
    discrepancy_population_pcc = discrepancy_metric_pcc.evaluate_metric_population(
        predicted_data_population=population_ecg, target_data=clinical_ecg)
    # TODO only for Maxx Holmes 2024
    # TODO only for Maxx Holmes 2024
    # TODO only for Maxx Holmes 2024
    # PCC between predicted and clinical ECGs:     # TODO only for Maxx Holmes 2024
    error_method_name_pcc = 'pcc'    # TODO only for Maxx Holmes 2024
    discrepancy_metric_pcc = DiscrepancyECG(error_method_name=error_method_name_pcc)    # TODO only for Maxx Holmes 2024
    discrepancy_population_maxx = discrepancy_metric_pcc.evaluate_metric_population(    # TODO only for Maxx Holmes 2024
        predicted_data_population=population_ecg[:, 2:, 100:], target_data=clinical_ecg[2:, 100:])    # TODO only for Maxx Holmes 2024
    selected_i = np.argmax(discrepancy_population_maxx)     # TODO only for Maxx Holmes 2024
    print('selected_i d ', discrepancy_population_maxx[selected_i])     # TODO only for Maxx Holmes 2024
    print('selected_i ', selected_i)     # TODO only for Maxx Holmes 2024
    selected_ecg = population_ecg[selected_i, :, :]   # TODO only for Maxx Holmes 2024
    # TODO only for Maxx Holmes 2024
    # TODO only for Maxx Holmes 2024
    # TODO only for Maxx Holmes 2024
    # Clear Arguments to prevent Argument recycling.
    biomarker_metric = None
    error_method_name_inference_metric = None
    error_method_name_pcc = None
    error_method_name_rmse = None
    discrepancy_metric_inference = None
    discrepancy_metric_pcc = None
    discrepancy_metric_rmse = None
    population_max_lat = None
    ####################################################################################################################
    # Step 20: Visualise ECGs and their metrics for the final population.
    print('Step 20: Visualise ECGs and their metrics for the final population.')
    # Initialise arguments for plotting
    axes = None
    fig = None
    # Plot the ECG inference population
    axes, fig = visualise_ecg(ecg_list=population_ecg, lead_name_list=lead_names, axes=axes,
                              ecg_color='k', fig=fig, label_list=None,
                              linewidth=0.1)
    # Plot the clinical trace after the last iteration
    axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                              ecg_color='lime', fig=fig, label_list=['Clinical'],
                              linewidth=2.)
    # TODO only for Maxx Holmes 2024
    # Plot the clinical trace after the last iteration  # TODO only for Maxx Holmes 2024
    axes, fig = visualise_ecg(ecg_list=[selected_ecg], lead_name_list=lead_names, axes=axes,  # TODO only for Maxx Holmes 2024
                              ecg_color='red', fig=fig, label_list=['Selected'],  # TODO only for Maxx Holmes 2024
                              linewidth=3.)  # TODO only for Maxx Holmes 2024
    # TODO only for Maxx Holmes 2024
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show(block=False)
    # fig.savefig(figure_result_file_name) # TODO uncomment 2024
    print('Saved ecg figure: ', figure_result_file_name)
    print('population_biomarker ', population_biomarker.shape)
    for biomarker_i in range(len(biomarker_name_list)):
        plt.hist(population_biomarker[biomarker_i, :])
        plt.title(biomarker_name_list[biomarker_i])
        plt.show(block=False)
    # Clear Arguments to prevent Argument recycling.
    axes = None
    fig = None
    figure_result_file_name = None
    population_biomarker = None
    ####################################################################################################################
    # Step 21: Report final population metrics.
    print('Step 21: Report final population metrics.')
    print('##Final population report: ')
    print('Parameter names: ', parameter_name_list_in_order)
    print('(mean +- std): ', np.mean(unique_parameter_population, axis=0), ' +- ',
          np.std(unique_parameter_population, axis=0))
    print('Inference metric (mean +- std): ', np.mean(discrepancy_population_inference), '+-',
          np.std(discrepancy_population_inference))
    print('discrepancy_population_rmse shape ', discrepancy_population_rmse.shape)
    print('RMSE (mean +- std): ', np.mean(discrepancy_population_rmse), ' +- ',
          np.std(discrepancy_population_rmse))
    print('PCC (mean +- std): ', np.mean(discrepancy_population_pcc), ' +- ',
          np.std(discrepancy_population_pcc))
    print('## Best parameters report: ')
    print('Parameter names : ', parameter_name_list_in_order)
    print('best_parameter ', best_parameter)
    print('Inference metric: ', np.amin(discrepancy_population_inference))
    print('RMSE: ', discrepancy_population_rmse[np.argmin(discrepancy_population_inference)])
    print('PCC: ', discrepancy_population_pcc[np.argmin(discrepancy_population_inference)])
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)
    print('')

    #EOF




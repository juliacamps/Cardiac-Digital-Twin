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
        anatomy_subject_name = 'DTI032'
        ecg_subject_name = 'DTI032'   # Allows using a different ECG for the personalisation than for the anatomy
    else:
        anatomy_subject_name = sys.argv[1]
        ecg_subject_name = sys.argv[1]
    print('anatomy_subject_name: ', anatomy_subject_name)
    print('ecg_subject_name: ', ecg_subject_name)
    ecg_subject_name = None  # Clear Arguments to prevent Argument recycling
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
    from conduction_system import EmptyConductionSystem
    from ecg_functions import PseudoEcgTetFromVM, get_cycle_length
    from geometry_functions import RawEmptyCardiacGeoPointCloud, \
        SimulationGeometry
    from propagation_models import EmptyPropagation
    from simulator_functions import SimulateECG
    from adapter_theta_params import AdapterThetaParams, RoundTheta
    from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
    from electrophysiology_functions import PrescribedVM
    from evaluation_functions import ParameterEvaluator
    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields, read_dictionary, save_ecg_to_csv, \
    save_csv_file, read_ecg_from_csv, read_csv_file, \
    read_pandas, read_monoalg_vm_ensight, export_ensight_timeseries_case, read_monoalg_geo_ensight
    from utils import map_indexes, remap_pandas_from_row_index, \
    get_repol_biomarker_name, get_lat_biomarker_name, get_best_str, \
    convert_from_monoalg3D_to_cm_and_translate, get_apd90_biomarker_name, get_sf_iks_biomarker_name
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
    for_monodomain_translation_resolution = 'hex500'
    monodomain_simulation_resolution = 'hex500'
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
    # ep_model_twave = 'GKs5_GKr0.6_tjca60'  # 'MitchellSchaefferEP' #'no_rescale' #'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = [get_sf_iks_biomarker_name()]
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    # Build results folder structure
    results_dir_part = results_dir_root + experiment_type + '_data/'
    assert os.path.exists(results_dir_part)  # Path should already exist from running the Twave inference
    results_dir_root = None  # Clear Arguments to prevent Argument recycling
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
    # Continue defining results paths and configuration
    result_tag = hyperparameter_dict['result_tag']
    # Uncertainty for Translation to Monodomain
    for_monodomain_dir = results_dir_twave + 'for_translation_to_monodomain/'
    assert os.path.exists(for_monodomain_dir)  # Path should exist from running translate_*_personalisation_to_MonoAlg3D.py
    for_monodomain_parameter_population_file_name = for_monodomain_dir + anatomy_subject_name + '_' \
                                                    + inference_resolution + '_' + result_tag + '_selected_parameter_population.csv'
    for_monodomain_biomarker_result_file_name_start = anatomy_subject_name + '_' \
                                                      + for_monodomain_translation_resolution + '_nodefield_' + result_tag + '-biomarker_'
    for_monodomain_biomarker_result_file_name_end = '.csv'
    # Precomputed subfolder specific for translation to monodomain
    for_monodomain_precomputed_dir = for_monodomain_dir + 'precomputed/'
    assert os.path.exists(for_monodomain_precomputed_dir)  # Path should exist from running translate_*_personalisation_to_MonoAlg3D.py
    inference_ecg_population_filename = for_monodomain_precomputed_dir + anatomy_subject_name + '_' + inference_resolution \
                                        + '_' + result_tag + '_selected_pseudo_ecg_population.csv'
    preprocessed_clinical_ecg_file_name = for_monodomain_precomputed_dir + anatomy_subject_name + '_' + inference_resolution \
                                          + '_' + result_tag + '_ecg_clinical.csv'
    for_monodomain_precomputed_dir = None  # Clear Arguments to prevent Argument recycling
    # Monodomain simulations folder
    monodomain_simulation_dir = for_monodomain_dir + 'monoalg_simulation/' + monodomain_simulation_resolution + '/'
    print('monodomain_simulation_dir ', monodomain_simulation_dir)
    assert os.path.exists(monodomain_simulation_dir)   # Path should already exist from running the monodomain simulations
    translation_tag = 'translation_'
    monodomain_simulation_vm_ensight_folder_name = 'vm_ensight/'
    # Output Paths:
    comparison_dir = monodomain_simulation_dir + 'comparison_re_mono/'
    if not os.path.exists(comparison_dir):
        os.mkdir(comparison_dir)
    comparison_dir_tag = comparison_dir + translation_tag
    ## ECG
    ecg_comparison_figure_result_file_name_start = anatomy_subject_name + '_ecg_translation_re_mono_'
    ecg_comparison_figure_result_file_name_end = '.png'
    ## LAT and REPOL - Ensight
    ensight_comparison_figure_result_file_name_start = anatomy_subject_name + '_translation_re_mono_'
    # Clear Arguments to prevent Argument recycling
    comparison_dir = None
    # Precomputed subfolder
    monodomain_results_precomputed_dir = monodomain_simulation_dir + 'precomputed/'
    if not os.path.exists(monodomain_results_precomputed_dir):
        os.mkdir(monodomain_results_precomputed_dir)
    ## Reaction-Eikonal
    eikonal_ecg_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                      + inference_resolution + '_' \
                                      + result_tag + '_eikonal_pseudo_ecg_population.csv'
    eikonal_lat_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                      + inference_resolution + '_' \
                                      + result_tag + '_eikonal_lat_population.csv'
    eikonal_repol_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                        + inference_resolution + '_' \
                                        + result_tag + '_eikonal_repol_population.csv'
    eikonal_apd90_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                        + inference_resolution + '_' \
                                        + result_tag + '_eikonal_apd90_population.csv'
    eikonal_GKs_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                        + inference_resolution + '_' \
                                        + result_tag + '_eikonal_GKs_population.csv'
    ## Monodomain
    monodomain_ecg_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                         + monodomain_simulation_resolution + '_' \
                                         + result_tag + '_monodomain_pseudo_ecg_population.csv'
    monodomain_lat_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                         + monodomain_simulation_resolution + '_' \
                                         + result_tag + '_monodomain_lat_population.csv'
    monodomain_repol_population_filename = monodomain_results_precomputed_dir + anatomy_subject_name + '_' \
                                           + monodomain_simulation_resolution + '_' \
                                           + result_tag + '_monodomain_repol_population.csv'
    monodomain_results_precomputed_dir = None  # Clear Arguments to prevent Argument recycling
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Clear Arguments to prevent Argument recycling
    data_dir = None
    ####################################################################################################################
    # Step 2: Load precomputed results from the inference process
    print('Step 2: Read precomputed results from the inference process.')
    # Read hyperparameters for ECG processing
    lead_names = hyperparameter_dict['lead_names']
    nb_leads = hyperparameter_dict['nb_leads']
    v3_name = hyperparameter_dict['v3_name']
    v5_name = hyperparameter_dict['v5_name']
    lead_v3_i = lead_names.index(v3_name)
    lead_v5_i = lead_names.index(v5_name)
    assert nb_leads == len(lead_names)
    # Clear Arguments to prevent Argument recycling
    v3_name = None
    v5_name = None
    # Load preprocessed clinical ECG
    clinical_ecg = read_ecg_from_csv(filename=preprocessed_clinical_ecg_file_name, nb_leads=nb_leads)
    clinical_ecg = clinical_ecg[0, :, :]
    print('clinical_ecg ', clinical_ecg.shape)
    preprocessed_clinical_ecg_file_name = None  # Clear Arguments to prevent Argument recycling
    ####################################################################################################################
    # Step 3: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
    print('Step 3: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
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
    # print('ep_model ', ep_model_twave_name)
    if ep_model_twave_name == 'MitchellSchaefferEP':
        apd_resolution = hyperparameter_dict['apd_resolution']
        cycle_length = hyperparameter_dict['cycle_length']
        vm_max = hyperparameter_dict['vm_max']
        vm_min = hyperparameter_dict['vm_min']
        cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                        apd_resolution=apd_resolution, cycle_length=cycle_length,
                                                        list_celltype_name=list_celltype_name, verbose=verbose,
                                                        vm_max=vm_max, vm_min=vm_min)
        # Clear Arguments to prevent Argument recycling
        apd_resolution = None
        cycle_length = None
        vm_max = None
        vm_min = None
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
    biomarker_upstroke_name = None
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_model_name = None
    cellular_stim_amp = None
    cellular_model_convergence = None
    ep_model_twave_name = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 4: Generate a cardiac geometry.
    print('Step 4: Generate a cardiac geometry.')
    # Argument setup: (in Alphabetical order)
    # Read hyperparameters
    vc_ab_cut_name = hyperparameter_dict['vc_ab_cut_name']
    vc_aprt_name = hyperparameter_dict['vc_aprt_name']
    vc_rvlv_name = hyperparameter_dict['vc_rvlv_name']
    vc_tm_name = hyperparameter_dict['vc_tm_name']
    celltype_vc_info = hyperparameter_dict['celltype_vc_info']
    vc_name_list = hyperparameter_dict['vc_name_list']
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    eikonal_geometry = SimulationGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                       conduction_system=EmptyConductionSystem(verbose=verbose),
                                       geometric_data_dir=geometric_data_dir, resolution=inference_resolution,
                                       subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    for_monodomain_translation_geometry = RawEmptyCardiacGeoPointCloud(
        conduction_system=EmptyConductionSystem(verbose=verbose),
                                                            geometric_data_dir=geometric_data_dir,
                                                            resolution=for_monodomain_translation_resolution,
                                                            subject_name=anatomy_subject_name, verbose=verbose)
    warn(
        'This should not be done in here!'
        'This should be done before calling this script and all meshse should be consistent in scale and location.')
    print('for_monodomain_translation_geometry')
    print('min max ', np.amin(for_monodomain_translation_geometry.get_node_xyz()),
          np.amax(for_monodomain_translation_geometry.get_node_xyz()))
    # TODO create function set_node_xyz that handles which attribute to use
    for_monodomain_translation_scale = np.array([1., 1., 1.])
    for_monodomain_translation_geometry.unprocessed_node_xyz = convert_from_monoalg3D_to_cm_and_translate(
        monoalg3D_xyz=for_monodomain_translation_geometry.get_node_xyz(), inference_xyz=eikonal_geometry.get_node_xyz(),
        scale=for_monodomain_translation_scale)
    print('min max ', np.amin(for_monodomain_translation_geometry.get_node_xyz()),
          np.amax(for_monodomain_translation_geometry.get_node_xyz()))
    print('geometry min max ', np.amin(eikonal_geometry.get_node_xyz()),
          np.amax(eikonal_geometry.get_node_xyz()))
    # Clear Arguments to prevent Argument recycling
    for_monodomain_translation_scale = None
    geometric_data_dir = None
    list_celltype_name = None
    monodomain_simulation_scale = None
    # ####################################################################################################################
    # Step 5: Prepare all files and modules to minimise repeating the processing of MONODOMAIN simualtions.
    print('5: Prepare all files and modules to minimise repeating the processing of monodomain simualtions.')
    # Check how many simulations were tranlsated to monodomain
    print(os.listdir(monodomain_simulation_dir))
    monodomain_dir_list = [dir_name for dir_name in os.listdir(monodomain_simulation_dir) if
                           os.path.isdir(os.path.join(monodomain_simulation_dir, dir_name))
                           and (translation_tag in dir_name)]
    # print('monodomain_dir_list ', monodomain_dir_list)
    monodomain_translation_tag_list = [dir_name.replace(translation_tag, '') for dir_name in monodomain_dir_list]
    # Sort Monodomain tags
    monodomain_uncertainty_i_list = []
    for translation_i in range(len(monodomain_translation_tag_list)):
        iteration_str_tag = monodomain_translation_tag_list[translation_i]
        if iteration_str_tag == get_best_str():
            uncertainty_i = 0
        else:
            uncertainty_i = int(iteration_str_tag)
        monodomain_uncertainty_i_list.append(uncertainty_i)
    monodomain_uncertainty_sort_index = np.argsort(monodomain_uncertainty_i_list)
    monodomain_dir_list = [monodomain_dir_list[monodomain_uncertainty_sort_index[i]] for i in range(len(monodomain_uncertainty_sort_index))]
    monodomain_translation_tag_list = [monodomain_translation_tag_list[monodomain_uncertainty_sort_index[i]] for i in range(len(monodomain_uncertainty_sort_index))]
    monodomain_uncertainty_i_list = [monodomain_uncertainty_i_list[monodomain_uncertainty_sort_index[i]] for i in range(len(monodomain_uncertainty_sort_index))]
    # CHECK IF MONODOMAIN SIMULATION RESULTS HAVE ALREADY BEEN PROCESSED
    if os.path.isfile(monodomain_ecg_population_filename) and os.path.isfile(monodomain_lat_population_filename) \
            and os.path.isfile(monodomain_repol_population_filename):
        print('5.1: Read all processed monodomain simualtions.')
        process_monodomain_results = False
        monodomain_ecg_population = read_ecg_from_csv(filename=monodomain_ecg_population_filename,
                                                      nb_leads=nb_leads)
        print('read monodomain_ecg_population ', monodomain_ecg_population.shape)
        monodomain_node_lat_population = read_csv_file(filename=monodomain_lat_population_filename)
        if len(monodomain_node_lat_population.shape) <2:
            monodomain_node_lat_population = monodomain_node_lat_population[np.newaxis, :]
        print('read mono_lat_population ', monodomain_node_lat_population.shape)
        print('dim ', monodomain_node_lat_population.ndim)
        monodomain_node_repol_population = read_csv_file(filename=monodomain_repol_population_filename)
        if len(monodomain_node_repol_population.shape) <2:
            monodomain_node_repol_population = monodomain_node_repol_population[np.newaxis, :]
        print('read monodomain_repol_population ', monodomain_node_repol_population.shape)
    else:
        print('5.2: Prepare for processing all monodomain simualtions.')
        process_monodomain_results = True
        # Initialise the result data structures to prevent processing the monodomain simulations next time
        monodomain_ecg_population = []
        monodomain_node_lat_population = []
        monodomain_node_repol_population = []
        # BUILD the NECESSARY MODULES to process the monodomain VMs
        ####################################################################################################################
        # Step 4: Create propagation model instance, this will be a static dummy propagation model.
        print('Step 3.3: Create propagation model instance, this will be a static dummy propagation model.')
        # Arguments for propagation model:
        # Read hyperparameters
        # Create propagation model
        propagation_model = EmptyPropagation(module_name=propagation_module_name, verbose=verbose)
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
        # Create ECG model
        ecg_model = PseudoEcgTetFromVM(electrode_positions=eikonal_geometry.get_electrode_xyz(), filtering=filtering,
                                       frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                       low_freq_cut=low_freq_cut,
                                       max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                       nodes_xyz=eikonal_geometry.get_node_xyz(), normalise=normalise,
                                       reference_ecg=clinical_ecg, tetra=eikonal_geometry.get_tetra(),
                                       tetra_centre=eikonal_geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
        # Clear Arguments to prevent Argument recycling
        filtering = None
        max_len_ecg = None
        max_len_qrs = None
        normalise = None
        v3_name = None
        v5_name = None
        zero_align = None
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
        theta_name_list_in_order = None
    # ####################################################################################################################
    # Step 6: Prepare all files and modules to minimise repeating the processing of Eikonal simualtions.
    print('6: Prepare all files and modules to minimise repeating the processing of Eikonal simualtions.')
    # CHECK IF EIKONAL SIMULATION RESULTS HAVE ALREADY BEEN PROCESSED
    if os.path.isfile(eikonal_ecg_population_filename) and os.path.isfile(eikonal_lat_population_filename) \
            and os.path.isfile(eikonal_repol_population_filename) and os.path.isfile(eikonal_apd90_population_filename) \
            and os.path.isfile(eikonal_GKs_population_filename):
        print('6.1: Read all processed Eikonal simualtions.')
        process_eikonal_results = False
        eikonal_ecg_population = read_ecg_from_csv(filename=eikonal_ecg_population_filename,
                                                      nb_leads=nb_leads)
        # print('read eikonal_ecg_population ', eikonal_ecg_population.shape)
        eikonal_node_lat_population = read_csv_file(filename=eikonal_lat_population_filename)
        if len(eikonal_node_lat_population.shape) <2:
            eikonal_node_lat_population = eikonal_node_lat_population[np.newaxis, :]
        # print('read eikonal_lat_population ', eikonal_node_lat_population.shape)
        eikonal_node_repol_population = read_csv_file(filename=eikonal_repol_population_filename)
        if len(eikonal_node_repol_population.shape) <2:
            eikonal_node_repol_population = eikonal_node_repol_population[np.newaxis, :]
        # print('read eikonal_repol_population ', eikonal_node_repol_population.shape)
        eikonal_node_apd90_population = read_csv_file(filename=eikonal_apd90_population_filename)
        if len(eikonal_node_apd90_population.shape) <2:
            eikonal_node_apd90_population = eikonal_node_apd90_population[np.newaxis, :]
        # print('read eikonal_node_apd90_population ', eikonal_node_apd90_population.shape)
        eikonal_node_GKs_population = read_csv_file(filename=eikonal_GKs_population_filename)
        if len(eikonal_node_GKs_population.shape) <2:
            eikonal_node_GKs_population = eikonal_node_GKs_population[np.newaxis, :]
        # print('read eikonal_node_GKs_population ', eikonal_node_GKs_population.shape)
    else:
        print('6.2: Get all Eikonal translation files.')
        # Get all translation files
        eikonal_filename_list = [filename for filename in os.listdir(for_monodomain_dir) if
                                 os.path.isfile(os.path.join(for_monodomain_dir, filename))
                                 and (for_monodomain_biomarker_result_file_name_start in filename)]
        # print('eikonal_filename_list ', eikonal_filename_list)
        # print('for_monodomain_dir ', for_monodomain_dir)
        eikonal_translation_tag_list = [filename.replace(for_monodomain_biomarker_result_file_name_start, '').replace(
            for_monodomain_biomarker_result_file_name_end, '')
                                        for filename in eikonal_filename_list]
        # print('eikonal_translation_tag_list ', eikonal_translation_tag_list)
        # Only keep those that actually have a monodomain simulation
        eikonal_translation_tag_i = np.asarray([tag_i for tag_i in range(len(eikonal_translation_tag_list)) if
                                                eikonal_translation_tag_list[tag_i] in monodomain_translation_tag_list])
        eikonal_filename_list = [eikonal_filename_list[eikonal_translation_tag_i[i]] for i in range(len(eikonal_translation_tag_i))]
        eikonal_translation_tag_list = [eikonal_translation_tag_list[eikonal_translation_tag_i[i]] for i in range(len(eikonal_translation_tag_i))]
        # Sort Eikonal tags
        eikonal_uncertainty_i_list = []
        for translation_i in range(len(eikonal_translation_tag_list)):
            iteration_str_tag = eikonal_translation_tag_list[translation_i]
            # print('translation ', translation_i)
            if iteration_str_tag == get_best_str():
                uncertainty_i = 0
            else:
                uncertainty_i = int(iteration_str_tag)
            # print('uncertainty_i ', uncertainty_i)
            eikonal_uncertainty_i_list.append(uncertainty_i)
        # print('eikonal_uncertainty_i_list ', eikonal_uncertainty_i_list)
        eikonal_uncertainty_sort_index = np.argsort(eikonal_uncertainty_i_list)
        # print('eikonal_uncertainty_sort_index ', eikonal_uncertainty_sort_index)
        eikonal_filename_list = [eikonal_filename_list[eikonal_uncertainty_sort_index[i]] for i in range(len(eikonal_uncertainty_sort_index))]
        eikonal_translation_tag_list = [eikonal_translation_tag_list[eikonal_uncertainty_sort_index[i]] for i in range(len(eikonal_uncertainty_sort_index))]
        # print('eikonal_filename_list ', eikonal_filename_list)

        print('6.3: Prepare for processing all Eikonal simualtions.')
        process_eikonal_results = True
        # Initialise the result data structures to prevent processing the monodomain simulations next time
        eikonal_ecg_population = []
        eikonal_node_lat_population = []
        eikonal_node_repol_population = []
        eikonal_node_apd90_population = []
        eikonal_node_GKs_population = []
        # Load selected Eikonal ECGs for translation to monodomain
        inference_ecg_population = read_ecg_from_csv(filename=inference_ecg_population_filename, nb_leads=nb_leads)
        # print('inference_ecg_population ', inference_ecg_population.shape)
        inference_ecg_population_filename = None  # Clear Arguments to prevent Argument recycling
        # # Create new interpolation indexes for simulated monodmain VMs
        print('6.4: Create interpolation indexes between saved results for translating to monodomain simulation and inference mesh.')
        for_monodomain_translation_node_mapping_index = map_indexes(
            points_to_map_xyz=eikonal_geometry.get_node_xyz(),
            reference_points_xyz=for_monodomain_translation_geometry.get_node_xyz())
    # ####################################################################################################################
    # Step 7: Iterate for all particles chosen to represent the uncertainty of the inference.
    print('7: Iterate for all particles chosen to represent the uncertainty of the inference.')
    # Initialise variables
    activation_time_map_biomarker_name = get_lat_biomarker_name()
    repolarisation_time_map_biomarker_name = get_repol_biomarker_name()
    apd90_biomarker_name = get_apd90_biomarker_name()
    sf_iks_biomarker_name = get_sf_iks_biomarker_name()
    if process_monodomain_results:
        # Read parameter values
        pandas_parameter_population = pd.read_csv(for_monodomain_parameter_population_file_name, delimiter=',')
    # ITERATE for all particles chosen to represent the uncertainty of the inference
    for translation_i in range(len(monodomain_translation_tag_list)):
        iteration_str_tag = monodomain_translation_tag_list[translation_i]
        # print('translation ', translation_i)
        uncertainty_i = monodomain_uncertainty_i_list[translation_i]
        # print('uncertainty_i ', uncertainty_i)

        # CREATE RESULT DIRECTORY
        current_comparison_dir = comparison_dir_tag + iteration_str_tag + '/'
        if not os.path.exists(current_comparison_dir):
            os.mkdir(current_comparison_dir)
        eikonal_field_tag = 'RE_'
        monodomain_field_tag = 'MONO_'

        # EIKONAL
        if process_eikonal_results:
            print('process_eikonal_results ', iteration_str_tag)
            # LOAD RESULTS FROM THE INFERENCE (EIKONAL) SIMULATIONS
            eikonal_pseudo_ecg = inference_ecg_population[uncertainty_i, :, :]
            eikonal_ecg_population.append(eikonal_pseudo_ecg)
            # BIOMARKERS
            # Inference (Eikonal) biomarkers filename
            inference_biomarker_result_file_name = for_monodomain_dir + for_monodomain_biomarker_result_file_name_start + iteration_str_tag + for_monodomain_biomarker_result_file_name_end
            # Load Biomarkers from inference selected particles for translation to monodomain
            unprocessed_node_biomarker = read_pandas(filename=inference_biomarker_result_file_name)
            inference_node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker,
                                                                   row_index=for_monodomain_translation_node_mapping_index)
            # Clear Arguments to prevent Argument recycling
            # for_monodomain_translation_node_mapping_index = None
            unprocessed_node_biomarker = None
            # LAT AND REPOL MAPS
            eikonal_node_lat = inference_node_biomarker[activation_time_map_biomarker_name]
            eikonal_node_lat_population.append(eikonal_node_lat)
            eikonal_node_repol = inference_node_biomarker[repolarisation_time_map_biomarker_name]
            eikonal_node_repol_population.append(eikonal_node_repol)
            # APD and GKs MAPS
            eikonal_node_apd90 = inference_node_biomarker[apd90_biomarker_name]
            eikonal_node_apd90_population.append(eikonal_node_apd90)
            eikonal_node_GKs = inference_node_biomarker[sf_iks_biomarker_name]
            eikonal_node_GKs_population.append(eikonal_node_GKs)
        else:
            print('skip eikonal ', iteration_str_tag)
            eikonal_pseudo_ecg = eikonal_ecg_population[translation_i, :, :]
            eikonal_node_lat = eikonal_node_lat_population[translation_i, :]
            eikonal_node_repol = eikonal_node_repol_population[translation_i, :]
            eikonal_node_apd90 = eikonal_node_apd90_population[translation_i, :]
            eikonal_node_GKs = eikonal_node_GKs_population[translation_i, :]
        # print('eikonal_pseudo_ecg ', eikonal_pseudo_ecg.shape)

        # MONODOMAIN
        if process_monodomain_results:
            print('process monodomain ', iteration_str_tag)
            # LOAD VM RESULTS FROM THE MONODOMAIN SIMULATIONS
            # VM MAP
            monodomain_simulation_ensight_dir = monodomain_simulation_dir + translation_tag + iteration_str_tag + '/' \
                                                + monodomain_simulation_vm_ensight_folder_name
            # print('monodomain_simulation_ensight_dir ', monodomain_simulation_ensight_dir)
            assert os.path.exists(monodomain_simulation_ensight_dir)
            unordered_unprocessed_monodomain_xyz = read_monoalg_geo_ensight(
                ensight_dir=monodomain_simulation_ensight_dir)
            nb_node = unordered_unprocessed_monodomain_xyz.shape[0]
            unordered_unprocessed_monodomain_vm = read_monoalg_vm_ensight(
                ensight_dir=monodomain_simulation_ensight_dir, nb_node=nb_node)#,
                # nb_node=for_monodomain_translation_nb_node)

            warn(
                'This should not be done in here!'
                'This should be done before calling this script and all meshse should be consistent in scale and location.')
            print('unprocessed_monodomain_xyz')
            print('min max ', np.amin(unordered_unprocessed_monodomain_xyz),
                  np.amax(unordered_unprocessed_monodomain_xyz))
            # TODO create function set_node_xyz that handles which attribute to use
            monodomain_simulation_translation_scale = np.array([1e+4, 1e+4, 1e+4])
            unordered_unprocessed_monodomain_xyz = convert_from_monoalg3D_to_cm_and_translate(
                monoalg3D_xyz=unordered_unprocessed_monodomain_xyz,
                inference_xyz=eikonal_geometry.get_node_xyz(),
                scale=monodomain_simulation_translation_scale)
            print('min max ', np.amin(unordered_unprocessed_monodomain_xyz),
                  np.amax(unordered_unprocessed_monodomain_xyz))
            print('geometry min max ', np.amin(eikonal_geometry.get_node_xyz()),
                  np.amax(eikonal_geometry.get_node_xyz()))

            # It takes too long to do the mapping between two fine resolutions
            # So, we direclty interpolate to the Eikonal geomery, assuming that reading only
            # the number of values in the .alg file (for_translation_geometry...) has made
            # the trick of removing the Purkinje vm values
            monodomain_simulation_mapping_index = map_indexes(
                points_to_map_xyz=eikonal_geometry.get_node_xyz(),
                reference_points_xyz=unordered_unprocessed_monodomain_xyz)
            print('monodomain_simulation_mapping_index ', monodomain_simulation_mapping_index.shape)

            # print('unordered_unprocessed_monodomain_vm ', unordered_unprocessed_monodomain_vm.shape)
            warn('Monodomain simulations may have used a different xyz that includes a Purkinje network!')
            # print('monodomain_simulation_mapping_index ', monodomain_simulation_mapping_index.shape)
            # It takes too long to do the mapping between two fine resolutions
            # First we inerpolate to the hexahedral mesh without Purkinje nodes
            monodomain_vm = unordered_unprocessed_monodomain_vm[monodomain_simulation_mapping_index, :]
            print('monodomain_vm ', monodomain_vm.shape)
            print('eikonal_geometry.get_node_xyz() ', eikonal_geometry.get_node_xyz().shape)


            # # Then we interpolate to the Eikonal's tetrahedral mesh
            # monodomain_vm = unprocessed_monodomain_vm[for_monodomain_translation_node_mapping_index, :]
            # print('monodomain_vm ', monodomain_vm.shape)
            # Clear Arguments to prevent Argument recycling
            unordered_unprocessed_monodomain_xyz = None
            unordered_unprocessed_monodomain_vm = None
            monodomain_simulation_mapping_index = None

            # CALCULATE MONODOMAIN LAT and REPOL MAPS
            ## Calculate LATs
            # TODO make the percentage for claculating the LATs into a global varibale to be consistent
            monodomain_node_lat = generate_activation_map(vm=monodomain_vm, percentage=70)
            # We want the frist LAT value to be 1 ms
            monodomain_earliest_activation_time = int(max(np.amin(monodomain_node_lat) - 1, 0))
            ## Correct LATs
            monodomain_node_lat = monodomain_node_lat - monodomain_earliest_activation_time
            print('monodomain_node_lat ', monodomain_node_lat.shape)
            ## Save LATs
            monodomain_node_lat_population.append(monodomain_node_lat)
            ## Calculate REPOLs
            monodomain_node_repol = generate_repolarisation_map(vm=monodomain_vm)
            ## Correct REPOLs - we want to aling the repols according to the new LATs
            monodomain_node_repol = monodomain_node_repol - monodomain_earliest_activation_time
            print('monodomain_node_repol ', monodomain_node_repol.shape)
            ## Save REPOLs
            monodomain_node_repol_population.append(monodomain_node_repol)

            # SIMULATE MONODOMAIN ECG
            # Create monodomain ep model:
            monodomain_electrophysiology_model = PrescribedVM(cellular_model=cellular_model,
                                                              module_name=electrophysiology_module_name,
                                                              propagation_model=propagation_model,
                                                              verbose=verbose, vm_prescribed=monodomain_vm)
            # Clear Arguments to prevent Argument recycling
            # Simulate ECGs
            monodomain_simulator_ecg = SimulateECG(ecg_model=ecg_model,
                                                   electrophysiology_model=monodomain_electrophysiology_model,
                                                   verbose=verbose)
            monodomain_evaluator_ecg = ParameterEvaluator(adapter=adapter,
                                                          simulator=monodomain_simulator_ecg,
                                                          verbose=verbose)
            uncertainty_parameter_population = monodomain_evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
            parameter_particle = uncertainty_parameter_population[uncertainty_i, :]
            monodomain_pseudo_ecg = monodomain_evaluator_ecg.simulate_parameter_particle(
                parameter_particle=parameter_particle)
            ## Correct ECG using the new LAT start
            print('monodomain_pseudo_ecg ', monodomain_pseudo_ecg.shape)
            monodomain_pseudo_ecg = monodomain_pseudo_ecg[:, monodomain_earliest_activation_time:]
            ## Save ECGs
            monodomain_ecg_population.append(monodomain_pseudo_ecg)
            # Clear Arguments to prevent Argument recycling
            monodomain_electrophysiology_model = None
            monodomain_evaluator_ecg = None

            # SAVE monodomain resutls
            ## Correct the VMs using the new LAT start - This will enable generating aligned ECGs and REPOLs
            monodomain_vm = monodomain_vm[:, monodomain_earliest_activation_time:]
            print('monodomain_vm ', monodomain_vm.shape)
            # VM Monodomain
            export_ensight_timeseries_case(dir=current_comparison_dir,
                                           casename=anatomy_subject_name + '_' + inference_resolution + '_simulation',
                                           dataname_list=[monodomain_field_tag + 'VM'],
                                           vm_list=[monodomain_vm], dt=1. / frequency,
                                           nodesxyz=eikonal_geometry.get_node_xyz(),
                                           tetrahedrons=eikonal_geometry.get_tetra())
        else:
            print('skip monodomain ', iteration_str_tag)
            monodomain_pseudo_ecg = monodomain_ecg_population[translation_i, :, :]
            monodomain_node_lat = monodomain_node_lat_population[translation_i, :]
            monodomain_node_repol = monodomain_node_repol_population[translation_i, :]

        # SAVE COMPARISON results
        # LAT and REPOL comparison
        # Save non-time depenedent fields for comparison
        write_geometry_to_ensight_with_fields(geometry=eikonal_geometry, node_field_list=[
            eikonal_node_repol-eikonal_node_lat,
            monodomain_node_repol-monodomain_node_lat,
            eikonal_node_lat,
            monodomain_node_lat,
            eikonal_node_repol,
            monodomain_node_repol,
            # lat_prescribed,
            eikonal_node_apd90,
            eikonal_node_GKs
        ],
                                              node_field_name_list=[
                                                  eikonal_field_tag + 'ARI',
                                                  monodomain_field_tag + 'ARI',
                                                  eikonal_field_tag + activation_time_map_biomarker_name,
                                                  monodomain_field_tag + activation_time_map_biomarker_name,
                                                  eikonal_field_tag + repolarisation_time_map_biomarker_name,
                                                  monodomain_field_tag + repolarisation_time_map_biomarker_name,
                                                  # 'read_' + activation_time_map_biomarker_name,
                                                  eikonal_field_tag + apd90_biomarker_name,
                                                  eikonal_field_tag + sf_iks_biomarker_name
                                              ],
                                              subject_name=anatomy_subject_name + '_' + inference_resolution + '_COMPARISON',
                                              verbose=verbose,
                                              visualisation_dir=current_comparison_dir)

        # # VM comparison
        # export_ensight_timeseries_case(dir=current_comparison_dir, casename=anatomy_subject_name + '_' + inference_resolution + '_simulation',
        #                                dataname_list=[eikonal_field_tag + 'VM', monodomain_field_tag + 'VM'],
        #                                vm_list=[eikonal_vm, monodomain_vm], dt=1. / frequency, nodesxyz=eikonal_geometry.get_node_xyz(),
        #                                tetrahedrons=eikonal_geometry.get_tetra())
        # ECG comparison
        print('Visualise ECGs and their metrics for the final population.')
        ecg_comparison_figure_result_file_name = current_comparison_dir + ecg_comparison_figure_result_file_name_start + iteration_str_tag + ecg_comparison_figure_result_file_name_end
        # Initialise arguments for plotting
        axes = None
        fig = None
        # Plot the clinical trace after the last iteration
        axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                                  ecg_color='lime', fig=fig, label_list=['Clinical'],
                                  linewidth=2.)
        # Plot the Eikonal ECG
        axes, fig = visualise_ecg(ecg_list=[eikonal_pseudo_ecg], lead_name_list=lead_names,
                                  axes=axes,
                                  ecg_color='k', fig=fig, label_list=['Eikonal'],
                                  linewidth=2.)
        # Plot the Eikonal ECG
        # axes, fig = visualise_ecg(ecg_list=[monodomain_pseudo_ecg[:, 37:]], lead_name_list=lead_names,
        axes, fig=visualise_ecg(ecg_list=[monodomain_pseudo_ecg], lead_name_list=lead_names,
                                                    axes=axes,
                                  ecg_color='m', fig=fig, label_list=['Monodomain'],
                                  linewidth=2.)
        axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.show(block=False)
        fig.savefig(ecg_comparison_figure_result_file_name)
        print('Saved ecg figure: ', ecg_comparison_figure_result_file_name)
        # Clear Arguments to prevent Argument recycling.
        axes = None
        fig = None
        ecg_comparison_figure_result_file_name = None
        eikonal_pseudo_ecg = None
        monodomain_pseudo_ecg = None
    # Clear Arguments to prevent Argument recycling.
    # for_monodomain_translation_nb_node = None
    for_monodomain_translation_node_mapping_index = None

    # Save monodomain results first in case the saving of the Eikonal fails and causes the process to terminate
    if process_monodomain_results:
        # Save precomputed ECGs, LAT, and REPOL for the selected particles to translate to monodomain
        monodomain_ecg_population = np.stack(monodomain_ecg_population)
        monodomain_node_lat_population = np.stack(monodomain_node_lat_population)
        monodomain_node_repol_population = np.stack(monodomain_node_repol_population)
        # print('monodomain_ecg_population ', monodomain_ecg_population.shape)
        # print('monodomain_lat_population ', monodomain_node_lat_population.shape)
        # print('inference_repol_population ', monodomain_node_repol_population.shape)
        save_ecg_to_csv(data=monodomain_ecg_population, filename=monodomain_ecg_population_filename)
        print('Saved ECGs for each selected particle for translation to monodomain at ',
              monodomain_ecg_population_filename)
        save_csv_file(data=monodomain_node_lat_population, filename=monodomain_lat_population_filename)
        print('Saved LAT for each selected particle for translation to monodomain at ',
              monodomain_lat_population_filename)
        save_csv_file(data=monodomain_node_repol_population, filename=monodomain_repol_population_filename)
        print('Saved REPOL for each selected particle for translation to monodomain at ',
              monodomain_repol_population_filename)

    if process_eikonal_results:
        # Save precomputed ECGs, LAT, and REPOL for the selected particles to translate to monodomain
        eikonal_ecg_population = np.stack(eikonal_ecg_population)
        eikonal_node_lat_population = np.stack(eikonal_node_lat_population)
        eikonal_node_repol_population = np.stack(eikonal_node_repol_population)
        # print('eikonal_ecg_population ', eikonal_ecg_population.shape)
        # print('eikonal_lat_population ', eikonal_node_lat_population.shape)
        # print('inference_repol_population ', eikonal_node_repol_population.shape)
        save_ecg_to_csv(data=eikonal_ecg_population, filename=eikonal_ecg_population_filename)
        print('Saved ECGs for each selected particle for translation to eikonal at ',
              eikonal_ecg_population_filename)
        save_csv_file(data=eikonal_node_lat_population, filename=eikonal_lat_population_filename)
        print('Saved LAT for each selected particle for translation to eikonal at ',
              eikonal_lat_population_filename)
        save_csv_file(data=eikonal_node_repol_population, filename=eikonal_repol_population_filename)
        print('Saved REPOL for each selected particle for translation to eikonal at ',
              eikonal_repol_population_filename)
        save_csv_file(data=eikonal_node_apd90_population, filename=eikonal_apd90_population_filename)
        print('Saved REPOL for each selected particle for translation to eikonal at ',
              eikonal_apd90_population_filename)
        save_csv_file(data=eikonal_node_GKs_population, filename=eikonal_GKs_population_filename)
        print('Saved REPOL for each selected particle for translation to eikonal at ',
              eikonal_GKs_population_filename)


    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

    #EOF
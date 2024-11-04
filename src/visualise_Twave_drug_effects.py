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
        anatomy_subject_name = 'DTI004'
        ecg_subject_name = 'DTI004'   # Allows using a different ECG for the personalisation than for the anatomy
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
    convert_from_monoalg3D_to_cm_and_translate, get_apd90_biomarker_name, get_sf_iks_biomarker_name, \
    get_monoalg_drug_folder_name_tag, get_monoalg_drug_scaling, get_qtc_dur_name, get_t_pe_name, get_t_peak_name, \
    get_tpeak_dispersion_name, get_qtpeak_dur_name, get_t_polarity_name, get_nan_value
    from postprocess_functions import generate_repolarisation_map, visualise_ecg, generate_activation_map
    from discrepancy_functions import BiomarkerFromOnlyECG

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
    # for_monodomain_translation_resolution = 'hex500'
    monodomain_simulation_resolution = 'hex500'
    experiment_drug_name = 'dofetilide'
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
    # Precomputed subfolder specific for translation to monodomain
    for_monodomain_precomputed_dir = for_monodomain_dir + 'precomputed/'
    assert os.path.exists(
        for_monodomain_precomputed_dir)  # Path should exist from running translate_*_personalisation_to_MonoAlg3D.py
    preprocessed_clinical_ecg_file_name = for_monodomain_precomputed_dir + anatomy_subject_name + '_' + inference_resolution \
                                          + '_' + result_tag + '_ecg_clinical.csv'
    for_monodomain_precomputed_dir = None  # Clear Arguments to prevent Argument recycling
    # Monodomain simulations folder
    monodomain_simulation_dir = for_monodomain_dir + 'monoalg_simulation/' + monodomain_simulation_resolution + '/'
    print('monodomain_simulation_dir ', monodomain_simulation_dir)
    assert os.path.exists(monodomain_simulation_dir)  # Path should already exist from running the monodomain simulations
    translation_tag = 'translation_'
    experiment_drug_subfolder_tag = get_monoalg_drug_folder_name_tag()
    # monodomain_simulation_vm_ensight_folder_name = 'vm_ensight/'
    # Output Paths:
    drug_analysis_dir = monodomain_simulation_dir + 'analysis_mono_' + experiment_drug_name +'/'
    if not os.path.exists(drug_analysis_dir):
        os.mkdir(drug_analysis_dir)
    drug_analysis_dir_tag = drug_analysis_dir + translation_tag
    # Precomputed subfolder
    drug_precomputed_dir = drug_analysis_dir + 'precomputed/'
    if not os.path.exists(drug_precomputed_dir):
        os.mkdir(drug_precomputed_dir)
    history_ecg_name_tag = anatomy_subject_name + '_' + monodomain_simulation_resolution + '_' + result_tag + '_monodomain_ecg'
    history_biomarker_name_tag = anatomy_subject_name + '_' + monodomain_simulation_resolution + '_' + result_tag + '_monodomain_biomarker'
    history_separator_tag = '_'
    # Visualisation
    visualisation_dir = drug_analysis_dir +'figure/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    figure_ecg_history_file_name = visualisation_dir + history_ecg_name_tag + '.png'
    figure_biomarker_history_file_name = visualisation_dir + history_biomarker_name_tag + '.png'
    figure_theta_history_file_name = visualisation_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_uncertainty_theta' + '.png'
    drug_analysis_dir = None  # Clear Arguments to prevent Argument recycling
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
    # Clear Arguments to prevent Argument recycling
    for_monodomain_translation_scale = None
    geometric_data_dir = None
    list_celltype_name = None
    monodomain_simulation_scale = None
    # ####################################################################################################################
    # Step 5: Prepare all files and modules to minimise repeating the processing of MONODOMAIN simualtions.
    print('5: Prepare all files and modules to minimise repeating the processing of monodomain simualtions.')
    # Check how many simulations were tranlsated to monodomain
    print('monodomain_simulation_dir ', monodomain_simulation_dir)
    print(os.listdir(monodomain_simulation_dir))
    monodomain_dir_list = [dir_name for dir_name in os.listdir(monodomain_simulation_dir) if
                           os.path.isdir(os.path.join(monodomain_simulation_dir, dir_name))
                           and (translation_tag in dir_name)    # Get translation folders
                           and get_best_str() not in dir_name]  # Remove best from the list of translations
    print('monodomain_dir_list ', monodomain_dir_list)
    monodomain_translation_tag_list = [dir_name.replace(translation_tag, '') for dir_name in monodomain_dir_list]
    # Sort Monodomain Translation tags
    monodomain_uncertainty_i_list = []
    for translation_i in range(len(monodomain_translation_tag_list)):
        iteration_str_tag = monodomain_translation_tag_list[translation_i]
        uncertainty_i = int(iteration_str_tag)
        monodomain_uncertainty_i_list.append(uncertainty_i)
    monodomain_uncertainty_sort_index = np.argsort(monodomain_uncertainty_i_list)
    monodomain_dir_list = [monodomain_dir_list[monodomain_uncertainty_sort_index[i]] for i in range(len(monodomain_uncertainty_sort_index))]
    monodomain_translation_tag_list = [monodomain_translation_tag_list[monodomain_uncertainty_sort_index[i]] for i in range(len(monodomain_uncertainty_sort_index))]
    monodomain_uncertainty_i_list = [monodomain_uncertainty_i_list[monodomain_uncertainty_sort_index[i]] for i in range(len(monodomain_uncertainty_sort_index))]
    # Check how many translations have simulations for the current drug
    print('5.1: Check how many translations have simulations for the current drug ', experiment_drug_name, '.')
    has_drug_index_list = []
    drug_sim_dir_list = []
    for translation_i_i in range(len(monodomain_dir_list)):
        drug_sim_dir = monodomain_simulation_dir + monodomain_dir_list[translation_i_i] + '/' + experiment_drug_name + '/'
        if os.path.isdir(drug_sim_dir):
            has_drug_index_list.append(translation_i_i)
            drug_sim_dir_list.append(drug_sim_dir)
    has_drug_index_list = np.asarray(has_drug_index_list)
    monodomain_dir_list = np.array(monodomain_dir_list)[has_drug_index_list]
    monodomain_translation_tag_list = np.array(monodomain_translation_tag_list)[has_drug_index_list]
    monodomain_uncertainty_i_list = np.array(monodomain_uncertainty_i_list)[has_drug_index_list]
    # Clear Arguments to prevent Argument recycling
    has_drug_index_list = None
    # Check how many dosages have been simulated of the drug
    print('5.2: Check how many dosages have been simulated of the drug ', experiment_drug_name, '.')
    translation_drug_dir_list_list = []
    translation_drug_dosage_list_list = []
    for translation_i_i in range(len(monodomain_dir_list)):
        drug_sim_dir = drug_sim_dir_list[translation_i_i]
        drug_dir_list = [dir_name for dir_name in os.listdir(drug_sim_dir) if
                         os.path.isdir(os.path.join(drug_sim_dir, dir_name))
                         and (experiment_drug_subfolder_tag in dir_name)]  # Get drug evaluation folders
        drug_scaling_list = [get_monoalg_drug_scaling(folder_name=dir_name) for dir_name in drug_dir_list]
        # Remove duplicate results
        drug_scaling_list, unique_index = np.unique(drug_scaling_list, return_index=True)
        translation_drug_dosage_list_list.append(drug_scaling_list)
        drug_dir_list = np.array(drug_dir_list)[unique_index]
        translation_drug_dir_list_list.append(drug_dir_list)
    # Check that all translations were simulated with the same dosages
    print('5.3: Check that all translations were simulated with the same dosages.')
    translation_1_dosage_list = translation_drug_dosage_list_list[0]
    print('translation_1_dosage_list ', translation_1_dosage_list)
    for translation_i_i in range(1, len(monodomain_dir_list)):
        if not np.all(translation_1_dosage_list == translation_drug_dosage_list_list[translation_i_i]):
            translation_i = np.array(monodomain_translation_tag_list)[translation_i_i]
            warn('Translation ' + translation_i + ' , had different dosages than translation 1')
            # raise Exception('Remove or re-run any translations that do not have the exact same dosages before proceeding.')
    dosage_list = translation_1_dosage_list
    print('Translation list ', monodomain_dir_list)
    print('Dosage list ', dosage_list)
    # Clear Arguments to prevent Argument recycling

    ####################################################################################################################
    # BUILD the NECESSARY MODULES to process the monodomain VMs and ECGs.
    print('BUILD the NECESSARY MODULES to process the monodomain VMs and ECGs.')
    # monodomain_node_lat_population = []
    # monodomain_node_repol_population = []
    # BUILD the NECESSARY MODULES to process the monodomain VMs
    # Step 6: Create propagation model instance, this will be a static dummy propagation model.
    print('Step 6: Create propagation model instance, this will be a static dummy propagation model.')
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
    ####################################################################################################################
    # Step 10: Define biomarker ECG metric.
    print('Step 10: Define biomarker ECG metric.')
    # Arguments for history simulation and biomarkers calculation:
    # heart_rate = hyperparameter_dict['heart_rate']  # TODO uncomment
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
    # ####################################################################################################################
    # Step 11: Iterate for all particles chosen to represent the uncertainty of the inference.
    print('11: Iterate for all particles chosen to represent the uncertainty of the inference.')
    # Initialise variables
    drug_ecg_list_list = []
    drug_biomarker_list_list = []
    # activation_time_map_biomarker_name = get_lat_biomarker_name()
    # repolarisation_time_map_biomarker_name = get_repol_biomarker_name()
    # apd90_biomarker_name = get_apd90_biomarker_name()
    # sf_iks_biomarker_name = get_sf_iks_biomarker_name()
    # Initialise variables to reduce computation
    monodomain_evaluator_ecg = None
    monodomain_simulation_mapping_index = None
    monodomain_simulation_mapping_index_ant = None  # Check that the interpolation is always the same
    nb_node = None
    parameter_particle = None
    # Read parameter values
    pandas_parameter_population = pd.read_csv(for_monodomain_parameter_population_file_name, delimiter=',')
    print('monodomain_translation_tag_list ', monodomain_translation_tag_list)
    # ITERATE for all particles chosen to represent the uncertainty of the inference
    for translation_i_i in range(len(monodomain_translation_tag_list)):
        translation_i = monodomain_translation_tag_list[translation_i_i]
        # print('translation ', translation_i)
        # uncertainty_i = monodomain_uncertainty_i_list[translation_i_i]
        print('translation ', translation_i)
        # print('uncertainty_i ', uncertainty_i)

        # Check if the ecg simulations for this translation have already been saved
        drug_ecg_filename = drug_precomputed_dir + history_ecg_name_tag + history_separator_tag \
                            + str(translation_i) + '.csv'
        drug_ecg_list = []  # Initialise the population of ECGs to an empty array
        # If the ECGs have already been simulated and there are exactly as many as parameter-sets, then load them and don't recompute them
        if os.path.isfile(drug_ecg_filename):
            print('Reading ECGs for ', translation_i)
            drug_ecg_list = read_ecg_from_csv(filename=drug_ecg_filename, nb_leads=nb_leads)

        # Check if the biomarker simulations for this translation have already been saved
        drug_biomarker_filename = drug_precomputed_dir + history_biomarker_name_tag \
                                  + history_separator_tag + str(translation_i) + '.csv'
        drug_biomarker_list = []  # Initialise the population of ECGs to an empty array
        if os.path.isfile(drug_biomarker_filename):
            drug_biomarker_list = read_csv_file(filename=drug_biomarker_filename)

        # If the data for this iteration has not been saved or does not match dosage in shape:
        # print('drug_ecg_list ', drug_ecg_list)
        ## Generate all the data and save it.
        if len(dosage_list) == len(drug_ecg_list) and len(dosage_list) == len(drug_biomarker_list):
            print('Skip ECG and Biomarker generation for ', translation_i)
        else:
            print('Generate all ECGs and Biomarkers for ', translation_i)
            # Initialise the result data structures to prevent processing the monodomain simulations next time
            drug_ecg_list = []
            drug_biomarker_list = []
            max_lat_list = []
            # Iterate over drug dosages and generate the Monodomain ECGs
            translation_drug_dir_list = translation_drug_dir_list_list[translation_i_i]
            for translation_drug_dir_i in range(len(translation_drug_dir_list)):
                translation_drug_dir = drug_sim_dir + translation_drug_dir_list[translation_drug_dir_i] + '/'
                drug_dosage = dosage_list[translation_drug_dir_i]
                # LOAD VM RESULTS FROM THE MONODOMAIN SIMULATIONS
                # SPEED UP TRICK - CHECK IF THE ECG GEOMETRY ALREADY EXISTS AND ASSUME IT WILL BE IDENTICAL IF IT DOES.
                # TODO WARNING: THIS COULD YIELD WRONG INTERPOLATIONS IF THE .GEO FILES ARE DIFFERENT IN DIFFERENT
                #  SIMULATION RESULTS FOR THE SAME SUBJECT
                if monodomain_simulation_mapping_index is None:
                    unordered_unprocessed_monodomain_xyz = read_monoalg_geo_ensight(
                        ensight_dir=translation_drug_dir)
                    nb_node = unordered_unprocessed_monodomain_xyz.shape[0]

                    warn(
                        'This should not be done in here!'
                        'This should be done before calling this script and all meshes should be consistent in scale and location.')
                    # print('unprocessed_monodomain_xyz')
                    # print('min max ', np.amin(unordered_unprocessed_monodomain_xyz),
                    #       np.amax(unordered_unprocessed_monodomain_xyz))
                    # TODO create function set_node_xyz that handles which attribute to use
                    monodomain_simulation_translation_scale = np.array([1e+4, 1e+4, 1e+4])
                    # TODO currently, this is done inside all for loops to make sure that everything is correct even if the
                    #  geometries in MonoAlg3D are saved differently across different simulations, however, this could be
                    #  speed up drastically, if the geometry was saved externally to the simulation results and only
                    #  processed once, as in the Eikonal pipeline
                    unordered_unprocessed_monodomain_xyz = convert_from_monoalg3D_to_cm_and_translate(
                        monoalg3D_xyz=unordered_unprocessed_monodomain_xyz,
                        inference_xyz=eikonal_geometry.get_node_xyz(),
                        scale=monodomain_simulation_translation_scale)
                    # print('min max ', np.amin(unordered_unprocessed_monodomain_xyz),
                    #       np.amax(unordered_unprocessed_monodomain_xyz))
                    # print('geometry min max ', np.amin(eikonal_geometry.get_node_xyz()),
                    #       np.amax(eikonal_geometry.get_node_xyz()))

                    # It takes too long to do the mapping between two fine resolutions
                    # So, we direclty interpolate to the Eikonal geomery, assuming that reading only
                    # the number of values in the .alg file (for_translation_geometry...) has made
                    # the trick of removing the Purkinje vm values
                    monodomain_simulation_mapping_index = map_indexes(
                        points_to_map_xyz=eikonal_geometry.get_node_xyz(),
                        reference_points_xyz=unordered_unprocessed_monodomain_xyz)
                    if monodomain_simulation_mapping_index_ant is not None:
                        print('VERIFY INTERPOLATION ', translation_i, '  ',  np.all(monodomain_simulation_mapping_index_ant==monodomain_simulation_mapping_index))
                    else:
                        monodomain_simulation_mapping_index_ant = monodomain_simulation_mapping_index

                # READ MONODOMAIN VMS
                unordered_unprocessed_monodomain_vm = read_monoalg_vm_ensight(
                    ensight_dir=translation_drug_dir, nb_node=nb_node)

                # print('monodomain_simulation_mapping_index ', monodomain_simulation_mapping_index.shape)

                # print('unordered_unprocessed_monodomain_vm ', unordered_unprocessed_monodomain_vm.shape)
                # warn('Monodomain simulations may have used a different xyz that includes a Purkinje network!')
                # print('monodomain_simulation_mapping_index ', monodomain_simulation_mapping_index.shape)
                # It takes too long to do the mapping between two fine resolutions
                # First we inerpolate to the hexahedral mesh without Purkinje nodes
                monodomain_vm = unordered_unprocessed_monodomain_vm[monodomain_simulation_mapping_index, :]
                # print('monodomain_vm ', monodomain_vm.shape)
                # print('eikonal_geometry.get_node_xyz() ', eikonal_geometry.get_node_xyz().shape)

                # # Then we interpolate to the Eikonal's tetrahedral mesh
                # monodomain_vm = unprocessed_monodomain_vm[for_monodomain_translation_node_mapping_index, :]
                # print('monodomain_vm ', monodomain_vm.shape)
                # Clear Arguments to prevent Argument recycling
                unordered_unprocessed_monodomain_xyz = None
                unordered_unprocessed_monodomain_vm = None
                # monodomain_simulation_mapping_index = None
                # CALCULATE MONODOMAIN LAT and REPOL MAPS
                ## Calculate LATs
                # TODO make the percentage for claculating the LATs into a global varibale to be consistent
                monodomain_node_lat = generate_activation_map(vm=monodomain_vm, percentage=70)
                # We want the frist LAT value to be 1 ms
                monodomain_earliest_activation_time = int(max(np.amin(monodomain_node_lat) - 1, 0))
                ## Correct LATs
                monodomain_node_lat = monodomain_node_lat - monodomain_earliest_activation_time
                max_lat_list.append(np.amax(monodomain_node_lat))
                monodomain_node_lat = None  # Clear Arguments to prevent Argument recycling


                # SPEED UP TRICK - CHECK IF THE ECG EVALUATOR ALREADY EXISTS AND ONLY SWAP THE VMS IF IT DOES.
                if monodomain_evaluator_ecg is None:
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
                else:
                    monodomain_evaluator_ecg.simulator.electrophysiology_model.prescribe_vm(vm_prescribed=monodomain_vm)
                # SPEED UP TRICK - CHECK IF THERE IS A PARAMETER PARTICLE ALREADY TO USE, THEY WON'T AFFECT THE SIMULATION BECAUSE IT LOADS THE VMS DIRECTLY ANYWAYS
                # TODO HOWEVER!!! WARNING!! IF IN THE FUTURE THE PARAMETERS WERE TO ACTUALLY GET USED, THIS WOULD YIELD WRONG RESULTS!!!!
                if parameter_particle is None:
                    # TODO this call could be moved to outside the for loops if we pre-defined an evaluator
                    uncertainty_parameter_population = monodomain_evaluator_ecg.translate_from_pandas_to_parameter(
                        pandas_parameter_population)
                    parameter_particle = uncertainty_parameter_population[uncertainty_i, :]

                # SIMULATE MONODOMAIN ECG
                monodomain_pseudo_ecg = monodomain_evaluator_ecg.simulate_parameter_particle(
                    parameter_particle=parameter_particle)
                ## Correct ECG using the new LAT start
                # # print('monodomain_pseudo_ecg_untrimmed ', monodomain_pseudo_ecg_untrimmed.shape)
                # monodomain_pseudo_ecg = np.zeros(monodomain_pseudo_ecg_untrimmed.shape, dtype=float) + get_nan_value()
                # # monodomain_pseudo_ecg_trimmed = monodomain_pseudo_ecg_untrimmed[:, monodomain_earliest_activation_time:]
                # monodomain_pseudo_ecg[:, :-monodomain_earliest_activation_time] = monodomain_pseudo_ecg_untrimmed[:, monodomain_earliest_activation_time:]
                # monodomain_pseudo_ecg[:, -monodomain_earliest_activation_time:] = monodomain_pseudo_ecg_untrimmed[:, -1:]

                ## Save ECGs
                drug_ecg_list.append(monodomain_pseudo_ecg[:, monodomain_earliest_activation_time:])
                # Clear Arguments to prevent Argument recycling
                monodomain_electrophysiology_model = None
                monodomain_evaluator_ecg = None

            # Stretch and Save ECGs
            ecg_len = 0
            for ecg in drug_ecg_list:
                print('ecg ', ecg.shape)
                ecg_len = max(ecg_len, ecg.shape[1])
            drug_ecg_array = np.zeros((len(drug_ecg_list), ecg.shape[0], ecg_len), dtype=float) + get_nan_value()
            for ecg_i in range(len(drug_ecg_list)):
                ecg = drug_ecg_list[ecg_i]
                drug_ecg_array[ecg_i, :, :ecg.shape[1]] = ecg
                drug_ecg_array[ecg_i, :, ecg.shape[1]:] = ecg[:, -1:]
            # monodomain_pseudo_ecg = np.zeros(monodomain_pseudo_ecg_untrimmed.shape, dtype=float) + get_nan_value()
            # # monodomain_pseudo_ecg_trimmed = monodomain_pseudo_ecg_untrimmed[:, monodomain_earliest_activation_time:]
            # monodomain_pseudo_ecg[:, :-monodomain_earliest_activation_time] = monodomain_pseudo_ecg_untrimmed[:,
            #                                                                   monodomain_earliest_activation_time:]
            # monodomain_pseudo_ecg[:, -monodomain_earliest_activation_time:] = monodomain_pseudo_ecg_untrimmed[:, -1:]
            # # max_len_simulated_ecg =
            # # drug_ecg = np.zeros((len(drug_ecg_list), monodomain_pseudo_ecg.shape[0], ))

            # Save ECGs for current simulation
            # drug_ecg_list = np.array(drug_ecg_list)
            save_ecg_to_csv(data=drug_ecg_array, filename=drug_ecg_filename)  # Save the ecgs for next time
            print('Saved ECGs from simulated translation ', translation_i, ' at ', drug_ecg_filename)
            # Calculate Biomarkers - Always recalculate the biomarker data if the ECGs are being recalculated
            ## Only valid for the T wave inference with prescribed LATs
            # max_lat_list = np.array(max_lat_list)
            drug_biomarker_list = metric.evaluate_metric_population(max_lat_population=np.array(max_lat_list),
                                                                    predicted_data_population=drug_ecg_array)
            # Save Biomarkers for current simulation
            save_csv_file(data=drug_biomarker_list, filename=drug_biomarker_filename)  # Save the biomarkers for next time
            print('Saved Biomarkers from simulated translation ', translation_i, ' at ', drug_biomarker_filename)
        # Save ECGs and Biomarkers for figure generation
        drug_ecg_list_list.append(drug_ecg_list)
        drug_biomarker_list_list.append(drug_biomarker_list)

    ####################################################################################################################
    # Step 14: Consistency check.
    print('Step 14: Consistency check.')
    print('len(drug_ecg_list_list) == len(drug_biomarker_list_list) ',
          len(drug_ecg_list_list) == len(drug_biomarker_list_list))
    print('len(drug_ecg_list_list) ', len(drug_ecg_list_list))
    print('len(drug_biomarker_list_list) ', len(drug_biomarker_list_list))
    assert len(drug_ecg_list_list) == len(drug_biomarker_list_list)
    history_colour_list = np.linspace(0.9, 0., num=len(dosage_list))
    print('drug_dosage ', dosage_list)
    print('history_colour_list ', history_colour_list)

    # # ECG Figures
    # print('Visualise ECGs and their metrics for the final population.')
    # ecg_comparison_figure_result_file_name = current_comparison_dir + ecg_comparison_figure_result_file_name_start + iteration_str_tag + ecg_comparison_figure_result_file_name_end
    # # Initialise arguments for plotting
    # axes = None
    # fig = None
    # # Plot the clinical trace after the last iteration
    # axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
    #                           ecg_color='lime', fig=fig, label_list=['Clinical'],
    #                           linewidth=2.)
    # # Plot the Eikonal ECG
    # axes, fig = visualise_ecg(ecg_list=[eikonal_pseudo_ecg], lead_name_list=lead_names,
    #                           axes=axes,
    #                           ecg_color='k', fig=fig, label_list=['Eikonal'],
    #                           linewidth=2.)
    # # Plot the Eikonal ECG
    # # axes, fig = visualise_ecg(ecg_list=[monodomain_pseudo_ecg[:, 37:]], lead_name_list=lead_names,
    # axes, fig=visualise_ecg(ecg_list=[monodomain_pseudo_ecg], lead_name_list=lead_names,
    #                                             axes=axes,
    #                           ecg_color='m', fig=fig, label_list=['Monodomain'],
    #                           linewidth=2.)
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # plt.show(block=False)
    # fig.savefig(ecg_comparison_figure_result_file_name)
    # print('Saved ecg figure: ', ecg_comparison_figure_result_file_name)
    # # Clear Arguments to prevent Argument recycling.
    # axes = None
    # fig = None
    # ecg_comparison_figure_result_file_name = None
    # eikonal_pseudo_ecg = None
    # monodomain_pseudo_ecg = None
    # # Clear Arguments to prevent Argument recycling.


    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

    #EOF

"""This script visualises the results from the inference of repolarisation properties from the T wave"""
import os
import sys
from warnings import warn
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

from translate_QT_personalisation_to_MonoAlg3D import convert_from_monoalg3D_to_cm_and_translate

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
    from conduction_system import DjikstraConductionSystemVC, EmptyConductionSystem
    from ecg_functions import PseudoEcgTetFromVM
    from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet, RawEmptyCardiacGeoPointCloud
    from propagation_models import EikonalDjikstraTet, PrescribedLAT
    from simulator_functions import SimulateECG, SimulateEP
    from adapter_theta_params import AdapterThetaParams, RoundTheta
    from discrepancy_functions import DiscrepancyECG, BiomarkerFromOnlyECG
    from evaluation_functions import DiscrepancyEvaluator, ParameterSimulator, ParameterEvaluator
    from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
    from electrophysiology_functions import ElectrophysiologyAPDmap, PrescribedVM
    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields, read_dictionary, save_ecg_to_csv, \
    export_ensight_timeseries_case, save_pandas, save_csv_file, read_ecg_from_csv, read_csv_file, read_time_csv_fields, \
    read_pandas, read_monoalg_vm_ensight
    from utils import map_indexes, remap_pandas_from_row_index, get_qt_dur_name, \
    get_t_pe_name, get_t_peak_name, get_tpeak_dispersion_name, get_qtpeak_dur_name, \
    get_t_polarity_name, get_repol_biomarker_name, get_lat_biomarker_name, get_best_str
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
    monodomain_simulation_resolution = 'hex500pk'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    cellular_data_dir = data_dir + 'cellular_data/'
    geometric_data_dir = data_dir + 'geometric_data/'
    results_dir_root = path_dict["results_path"]
    # Intermediate Paths: # e.g., results from the QRS inference
    experiment_type = 'personalisation'
    if anatomy_subject_name == 'DTI024':
        ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_909'
    elif anatomy_subject_name == 'DTI032':
        ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_810'
    elif anatomy_subject_name == 'DTI004':
        ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_1250'
    else:
        ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_'
    # ep_model_twave = 'GKs5_GKr0.6_tjca60'  # 'MitchellSchaefferEP' #'no_rescale' #'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
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
    results_dir_twave = results_dir_part_twave + date_str + '/'
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
    results_dir_qrs = results_dir_part_qrs + date_str + '/'
    assert os.path.exists(results_dir_qrs)  # Path should already exist from running the QRS inference
    results_dir_part_qrs = None  # Clear Arguments to prevent Argument recycling
    qrs_lat_prescribed_filename = hyperparameter_dict['qrs_lat_prescribed_filename']
    qrs_lat_prescribed_filename_path = results_dir_qrs + qrs_lat_prescribed_filename
    # Clear Arguments to prevent Argument recycling
    results_dir_qrs = None
    qrs_lat_prescribed_filename = None
    if not os.path.isfile(qrs_lat_prescribed_filename_path):
        print('qrs_lat_prescribed_filename_path: ', qrs_lat_prescribed_filename_path)
        raise Exception(
            "This inference needs to be run after the QRS inference and need the correct path with those results.")
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
    monodomain_simulation_dir = for_monodomain_dir + 'monoalg_simulation/'
    assert os.path.exists(monodomain_simulation_dir)   # Path should already exist from running the monodomain simulations
    monodomain_simulation_vm_ensight_dir_tag = 'vm_ensight_'
    # Output Paths:
    comparison_dir = monodomain_simulation_dir + 'comparison_re_mono/'
    if not os.path.exists(comparison_dir):
        os.mkdir(comparison_dir)
    comparison_dir_tag = comparison_dir + 'translation_'
    ## ECG
    # ecg_population_translation_figure_result_file_name = comparison_dir + anatomy_subject_name + '_ecg_selected_translation_re_mono.png'
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
    # Read hyperparameters
    clinical_data_filename = hyperparameter_dict['clinical_data_filename']
    clinical_data_filename_path = data_dir + clinical_data_filename
    clinical_qrs_offset = hyperparameter_dict['clinical_qrs_offset']
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
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
    print('clinical_ecg ', clinical_ecg.shape)
    preprocessed_clinical_ecg_file_name = None  # Clear Arguments to prevent Argument recycling
    # Load inference LAT
    # lat_prescribed = (np.loadtxt(qrs_lat_prescribed_filename_path, delimiter=',')).astype(int)
    # max_lat = np.amax(lat_prescribed)
    # qrs_lat_prescribed_filename_path = None  # Clear Arguments to prevent Argument recycling
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
    apd_resolution = hyperparameter_dict['apd_resolution']
    cycle_length = hyperparameter_dict['cycle_length']
    vm_max = hyperparameter_dict['vm_max']
    vm_min = hyperparameter_dict['vm_min']
    # Create cellular model instance.
    print('ep_model ', ep_model_twave_name)
    if ep_model_twave_name == 'MitchellSchaefferEP':
        cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                        apd_resolution=apd_resolution, cycle_length=500,
                                                        list_celltype_name=list_celltype_name, verbose=verbose,
                                                        vm_max=1., vm_min=0.)
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
    # Step 4: Generate a cardiac geometry.
    print('Step 4: Generate a cardiac geometry.')
    # Argument setup: (in Alphabetical order)
    # Read hyperparameters
    vc_ab_cut_name = hyperparameter_dict['vc_ab_cut_name']
    vc_aprt_name = hyperparameter_dict['vc_aprt_name']
    # vc_rt_name = hyperparameter_dict['vc_rt_name']
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
    inference_geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=inference_resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    for_monodomain_translation_geometry = RawEmptyCardiacGeoPointCloud(
        conduction_system=EmptyConductionSystem(verbose=verbose),
                                                            geometric_data_dir=geometric_data_dir,
                                                            resolution=for_monodomain_translation_resolution,
                                                            subject_name=anatomy_subject_name, verbose=verbose)
    monodomain_simulation_geometry = RawEmptyCardiacGeoPointCloud(
        conduction_system=EmptyConductionSystem(verbose=verbose),
        geometric_data_dir=geometric_data_dir,
        resolution=monodomain_simulation_resolution,
        subject_name=anatomy_subject_name, verbose=verbose)
    # TODO DELETE THE FOLLOWING CODE
    warn(
        'This should not be done in here!\nThis hack will only work for DTI... meshes, and should be done before calling the script in the futrure.')
    print('min max ', np.amin(for_monodomain_translation_geometry.get_node_xyz()),
          np.amax(for_monodomain_translation_geometry.get_node_xyz()))
    # TODO create function set_node_xyz that handles which attribute to use
    for_monodomain_translation_geometry.unprocessed_node_xyz = convert_from_monoalg3D_to_cm_and_translate(
        monoalg3D_xyz=for_monodomain_translation_geometry.get_node_xyz(), inference_xyz=inference_geometry.get_node_xyz())
    print('min max ', np.amin(for_monodomain_translation_geometry.get_node_xyz()),
          np.amax(for_monodomain_translation_geometry.get_node_xyz()))
    print('geometry min max ', np.amin(inference_geometry.get_node_xyz()),
          np.amax(inference_geometry.get_node_xyz()))
    print('Now the second one!')
    warn(
        'This should not be done in here!\nThis hack will only work for DTI... meshes, and should be done before calling the script in the futrure.')
    print('min max ', np.amin(monodomain_simulation_geometry.get_node_xyz()),
          np.amax(monodomain_simulation_geometry.get_node_xyz()))
    # TODO create function set_node_xyz that handles which attribute to use
    monodomain_simulation_geometry.unprocessed_node_xyz = convert_from_monoalg3D_to_cm_and_translate(
        monoalg3D_xyz=monodomain_simulation_geometry.get_node_xyz(),
        inference_xyz=inference_geometry.get_node_xyz())
    print('min max ', np.amin(monodomain_simulation_geometry.get_node_xyz()),
          np.amax(monodomain_simulation_geometry.get_node_xyz()))
    print('geometry min max ', np.amin(inference_geometry.get_node_xyz()),
          np.amax(inference_geometry.get_node_xyz()))
    # TODO DELETE THE ABOVE CODE
    # Clear Arguments to prevent Argument recycling
    geometric_data_dir = None
    list_celltype_name = None
    inference_resolution = None
    # vc_name_list = None
    # ####################################################################################################################
    # Step 5: Prepare all files and modules to minimise repeating the processing of monodomain simualtions.
    print('5: Prepare all files and modules to minimise repeating the processing of monodomain simualtions.')
    # Check how many simulations were tranlsated to monodomain
    monodomain_dir_list = [dir_name for dir_name in os.listdir(monodomain_simulation_dir) if
                           os.path.isdir(os.path.join(monodomain_simulation_dir, dir_name))
                           and (monodomain_simulation_vm_ensight_dir_tag in dir_name)]
    monodomain_translation_tag_list = [dir_name.replace(monodomain_simulation_vm_ensight_dir_tag, '') for dir_name in monodomain_dir_list]
    # Sort Monodomain tags
    monodomain_uncertainty_i_list = []
    for translation_i in range(len(monodomain_translation_tag_list)):
        iteration_str_tag = monodomain_translation_tag_list[translation_i]
        print('translation ', translation_i)
        if iteration_str_tag == get_best_str():
            uncertainty_i = 0
        else:
            uncertainty_i = int(iteration_str_tag)
        print('uncertainty_i ', uncertainty_i)
        monodomain_uncertainty_i_list.append(uncertainty_i)
    print('monodomain_uncertainty_i_list ', monodomain_uncertainty_i_list)
    monodomain_uncertainty_sort_index = np.argsort(monodomain_uncertainty_i_list)
    print('monodomain_uncertainty_sort_index ', monodomain_uncertainty_sort_index)
    monodomain_dir_list = monodomain_dir_list[monodomain_uncertainty_sort_index]
    monodomain_translation_tag_list = monodomain_translation_tag_list[monodomain_uncertainty_sort_index]
    monodomain_uncertainty_i_list = monodomain_uncertainty_i_list[monodomain_uncertainty_sort_index]
    print('monodomain_uncertainty_i_list ', monodomain_uncertainty_i_list)
    # CHECK IF MONODOMAIN SIMULATION RESULTS HAVE ALREADY BEEN PROCESSED
    if os.path.isfile(monodomain_ecg_population_filename) and os.path.isfile(monodomain_lat_population_filename) \
            and os.path.isfile(monodomain_repol_population_filename):
        print('5.1: Read all processed monodomain simualtions.')
        process_monodomain_results = False
        monodomain_ecg_population = read_ecg_from_csv(filename=monodomain_ecg_population_filename,
                                                      nb_leads=nb_leads)
        print('read monodomain_ecg_population ', monodomain_ecg_population.shape)
        monodomain_node_lat_population = read_csv_file(filename=monodomain_lat_population_filename)
        print('read mono_lat_population ', monodomain_node_lat_population.shape)
        monodomain_node_repol_population = read_csv_file(filename=monodomain_repol_population_filename)
        print('read monodomain_repol_population ', monodomain_node_repol_population.shape)
    else:
        print('5.2: Prepare for processing all monodomain simualtions.')
        process_monodomain_results = True
        # Initialise the result data structures to prevent processing the monodomain simulations next time
        monodomain_ecg_population = []
        monodomain_node_lat_population = []
        monodomain_node_repol_population = []
        # BUILD the NECESSARY MODULES to process the monodomain VMs


        # Create new interpolation indexes for simulated monodmain VMs
        print('5.: Create interpolation indexes between monodomain simulation and inference mesh.')
        monodomain_simulation_node_mapping_index = map_indexes(points_to_map_xyz=inference_geometry.get_node_xyz(),
                                                                    reference_points_xyz=monodomain_simulation_geometry.get_node_xyz())
    # ####################################################################################################################
    # Step 6: Prepare all files and modules to minimise repeating the processing of Eikonal simualtions.
    print('6: Prepare all files and modules to minimise repeating the processing of Eikonal simualtions.')
    # CHECK IF EIKONAL SIMULATION RESULTS HAVE ALREADY BEEN PROCESSED
    if os.path.isfile(eikonal_ecg_population_filename) and os.path.isfile(eikonal_lat_population_filename) \
            and os.path.isfile(eikonal_repol_population_filename):
        print('6.1: Read all processed Eikonal simualtions.')
        process_eikonal_results = False
        eikonal_ecg_population = read_ecg_from_csv(filename=eikonal_ecg_population_filename,
                                                      nb_leads=nb_leads)
        print('read eikonal_ecg_population ', eikonal_ecg_population.shape)
        eikonal_node_lat_population = read_csv_file(filename=eikonal_lat_population_filename)
        print('read eikonal_lat_population ', eikonal_node_lat_population.shape)
        eikonal_node_repol_population = read_csv_file(filename=eikonal_repol_population_filename)
        print('read eikonal_repol_population ', eikonal_node_repol_population.shape)
    else:
        print('6.2: Get all Eikonal translation files.')
        # Get all translation files
        eikonal_filename_list = [filename for filename in os.listdir(for_monodomain_dir) if
                                 os.path.isfile(os.path.join(for_monodomain_dir, filename))
                                 and (for_monodomain_biomarker_result_file_name_start in filename)]
        eikonal_translation_tag_list = [filename.replace(for_monodomain_biomarker_result_file_name_start, '').replace(
            for_monodomain_biomarker_result_file_name_end, '')
                                        for filename in eikonal_filename_list]
        # Only keep those that actually have a monodomain simulation
        eikonal_translation_tag_i = np.asarray([tag_i for tag_i in range(len(eikonal_translation_tag_list)) if
                                                eikonal_translation_tag_list[tag_i] in monodomain_translation_tag_list])
        eikonal_filename_list = eikonal_filename_list[eikonal_translation_tag_i]
        eikonal_translation_tag_list = eikonal_translation_tag_list[eikonal_translation_tag_i]
        print('eikonal_filename_list ', eikonal_filename_list)
        print('monodomain_dir_list ', monodomain_dir_list)
        # Sort Eikonal tags
        eikonal_uncertainty_i_list = []
        for translation_i in range(len(eikonal_translation_tag_list)):
            iteration_str_tag = eikonal_translation_tag_list[translation_i]
            print('translation ', translation_i)
            if iteration_str_tag == get_best_str():
                uncertainty_i = 0
            else:
                uncertainty_i = int(iteration_str_tag)
            print('uncertainty_i ', uncertainty_i)
            eikonal_uncertainty_i_list.append(uncertainty_i)
        print('eikonal_uncertainty_i_list ', eikonal_uncertainty_i_list)
        eikonal_uncertainty_sort_index = np.argsort(eikonal_uncertainty_i_list)
        print('eikonal_uncertainty_sort_index ', eikonal_uncertainty_sort_index)
        eikonal_dir_list = eikonal_filename_list[eikonal_uncertainty_sort_index]
        eikonal_translation_tag_list = eikonal_translation_tag_list[eikonal_uncertainty_sort_index]
        print('eikonal_filename_list ', eikonal_filename_list)

        print('6.3: Prepare for processing all Eikonal simualtions.')
        process_eikonal_results = True
        # Initialise the result data structures to prevent processing the monodomain simulations next time
        eikonal_ecg_population = []
        eikonal_node_lat_population = []
        eikonal_node_repol_population = []
        # Load selected Eikonal ECGs for translation to monodomain
        inference_ecg_population = read_ecg_from_csv(filename=inference_ecg_population_filename, nb_leads=nb_leads)
        print('inference_ecg_population ', inference_ecg_population.shape)
        inference_ecg_population_filename = None  # Clear Arguments to prevent Argument recycling
        # Create new interpolation indexes for simulated monodmain VMs
        print('6.4: Create interpolation indexes between saved results for translating to monodomain simulation and inference mesh.')
        for_monodomain_translation_node_mapping_index = map_indexes(
            points_to_map_xyz=inference_geometry.get_node_xyz(),
            reference_points_xyz=for_monodomain_translation_geometry.get_node_xyz())

    # ####################################################################################################################
    # Step 7: Iterate for all particles chosen to represent the uncertainty of the inference.
    print('7: Iterate for all particles chosen to represent the uncertainty of the inference.')
    # Initialise variables
    activation_time_map_biomarker_name = get_lat_biomarker_name()
    repolarisation_time_map_biomarker_name = get_repol_biomarker_name()
    # ITERATE for all particles chosen to represent the uncertainty of the inference
    for translation_i in range(len(monodomain_translation_tag_list)):
        iteration_str_tag = monodomain_translation_tag_list[translation_i]
        print('translation ', translation_i)
        uncertainty_i = monodomain_uncertainty_i_list[translation_i]
        print('uncertainty_i ', uncertainty_i)

        if process_eikonal_results:
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
            for_monodomain_translation_node_mapping_index = None
            unprocessed_node_biomarker = None
            # LAT AND REPOL MAPS
            eikonal_node_lat = inference_node_biomarker[activation_time_map_biomarker_name]
            eikonal_node_lat_population.append(eikonal_node_lat)
            eikonal_node_repol = inference_node_biomarker[repolarisation_time_map_biomarker_name]
            eikonal_node_repol_population.append(eikonal_node_repol)
        else:
            eikonal_pseudo_ecg = eikonal_ecg_population[translation_i, :, :]
            eikonal_node_lat = eikonal_node_lat_population[translation_i, :]
            eikonal_node_repol = eikonal_node_repol_population[translation_i, :]

        if process_monodomain_results:
            # LOAD VM RESULTS FROM THE MONODOMAIN SIMULATIONS
            # VM MAP
            monodomain_simulation_ensight_dir = monodomain_simulation_vm_ensight_dir_tag + iteration_str_tag
            print('monodomain_simulation_ensight_dir ', monodomain_simulation_ensight_dir)
            assert os.path.exists(monodomain_simulation_ensight_dir)
            unprocessed_monodomain_vm = read_monoalg_vm_ensight(ensight_dir=monodomain_simulation_ensight_dir)
            print('unprocessed_monodomain_vm ', unprocessed_monodomain_vm.shape)
            warn('Monodomain simulations may have used a different xyz that includes a Purkinje network!')
            monodomain_vm = unprocessed_monodomain_vm[monodomain_simulation_node_mapping_index, :]
            print('monodomain_vm ', monodomain_vm.shape)
            # Clear Arguments to prevent Argument recycling
            monodomain_unprocessed_vm = None

            # CALCULATE MONODOMAIN LAT and REPOL MAPS
            monodomain_node_lat = generate_activation_map(vm=monodomain_vm, percentage=70)
            monodomain_node_lat_population.append(monodomain_node_lat)
            monodomain_node_repol = generate_repolarisation_map(vm=monodomain_vm)
            monodomain_node_repol_population.append(monodomain_node_repol)

            # SIMULATE MONODOMAIN ECG
            # Create monodomain ep model:
            monodomain_electrophysiology_model = PrescribedVM(cellular_model=cellular_model,
                                                              module_name=electrophysiology_module_name,
                                                              propagation_model=propagation_model,
                                                              verbose=verbose, vm_prescribed=monodomain_vm)
            # Clear Arguments to prevent Argument recycling
            monodomain_vm = None
            # Simulate ECGs
            monodomain_simulator_ecg = SimulateECG(ecg_model=ecg_model,
                                                   electrophysiology_model=monodomain_electrophysiology_model,
                                                   verbose=verbose)
            monodomain_evaluator_ecg = ParameterEvaluator(adapter=adapter,
                                                          simulator=monodomain_simulator_ecg,
                                                          verbose=verbose)
            parameter_particle = parameter_population[translation_i, :]
            monodomain_pseudo_ecg = monodomain_evaluator_ecg.simulate_parameter_particle(
                parameter_particle=parameter_particle)
            monodomain_ecg_population.append(monodomain_pseudo_ecg)
            # Clear Arguments to prevent Argument recycling
            monodomain_electrophysiology_model = None
            monodomain_evaluator_ecg = None
            monodomain_pseudo_ecg = None
        else:
            monodomain_node_lat =
            monodomain_node_repol =
            monodomain_pseudo_ecg =

        # CREATE RESULT DIRECTORY
        current_comparison_dir = comparison_dir_tag + iteration_str_tag
        if not os.path.exists(current_comparison_dir):
            os.mkdir(current_comparison_dir)


        # Read all monodomain vm maps
        # TODO write a new function that can read MonoAlg3D ensight simulation files (vm)
        monodomain_unprocessed_vm, monodomain_xyz = read_time_csv_fields(anatomy_subject_name=anatomy_subject_name,
                                                                         csv_dir=monodomain_simulation_dir,
                                                                         file_name_tag='scalar.INTRA',
                                                                         node_xyz_filename=monodomain_xyz_filename)
        # Interpolate nodefield from the monodomain simulations to the Eikonal geometry
        for_monodomain_translation_node_mapping_index = map_indexes(points_to_map_xyz=geometry.get_node_xyz(),
                                                                    reference_points_xyz=raw_geometry_point_cloud.get_node_xyz())
        monodomain_vm = monodomain_unprocessed_vm[for_monodomain_translation_node_mapping_index, :]
        print('monodomain_vm ', monodomain_vm.shape)
        # Clear Arguments to prevent Argument recycling
        monodomain_unprocessed_vm = None
        monodomain_xyz = None
        raw_geometry_point_cloud = None
        for_monodomain_translation_node_mapping_index = None
        # Calculate LAT and REPOL
        monodomain_repol = generate_repolarisation_map(vm=monodomain_vm)
        monodomain_node_repol_population.append(monodomain_repol)
        print('monodomain_repol ', np.amin(monodomain_repol), ' ', np.amax(monodomain_repol))
        monodomain_repol = None  # Clear Arguments to prevent Argument recycling
        mono_lat_hack = generate_activation_map(vm=monodomain_vm, percentage=70)  # np.argmax(monodomain_vm, axis=1)
        monodomain_lat_population.append(mono_lat_hack)
        print('mono_lat_hack ', np.amin(mono_lat_hack), ' ', np.amax(mono_lat_hack))
        mono_lat_hack = None  # Clear Arguments to prevent Argument recycling
        # Create monodomain ep model:
        monodomain_electrophysiology_model = PrescribedVM(cellular_model=cellular_model,
                                                          module_name=electrophysiology_module_name,
                                                          propagation_model=propagation_model,
                                                          verbose=verbose, vm_prescribed=monodomain_vm)
        monodomain_vm = None  # Clear Arguments to prevent Argument recycling
        # # Simulate LAT and VMs
        # monodomain_simulator_ep = SimulateEP(electrophysiology_model=monodomain_electrophysiology_model,
        #                                      verbose=verbose)
        # monodomain_evaluator_ep = ParameterSimulator(adapter=adapter, simulator=monodomain_simulator_ep,
        #                                              verbose=verbose)
        # _, monodomain_vm = evaluator_ep.simulate_parameter_particle(
        #     parameter_particle=parameter_particle)
        # # Clear Arguments to prevent Argument recycling
        # evaluator_ep = None
        # monodomain_electrophysiology_model = None
        # Simulate ECGs
        monodomain_simulator_ecg = SimulateECG(ecg_model=ecg_model,
                                               electrophysiology_model=monodomain_electrophysiology_model,
                                               verbose=verbose)
        monodomain_evaluator_ecg = ParameterEvaluator(adapter=adapter,
                                                      simulator=monodomain_simulator_ecg,
                                                      verbose=verbose)
        parameter_particle = parameter_population[translation_i, :]
        monodomain_pseudo_ecg = monodomain_evaluator_ecg.simulate_parameter_particle(
            parameter_particle=parameter_particle)
        monodomain_ecg_population.append(monodomain_pseudo_ecg)
        # Clear Arguments to prevent Argument recycling
        monodomain_electrophysiology_model = None
        monodomain_evaluator_ecg = None
        monodomain_pseudo_ecg = None

    # Save precomputed ECGs, LAT, and REPOL for the selected particles to translate to monodomain
    monodomain_ecg_population = np.stack(monodomain_ecg_population)
    monodomain_node_lat_population = np.stack(monodomain_node_lat_population)
    monodomain_node_repol_population = np.stack(monodomain_node_repol_population)
    print('monodomain_ecg_population ', monodomain_ecg_population.shape)
    print('monodomain_lat_population ', monodomain_node_lat_population.shape)
    print('inference_repol_population ', monodomain_node_repol_population.shape)
    save_ecg_to_csv(data=monodomain_ecg_population, filename=monodomain_ecg_population_filename)
    print('Saved ECGs for each selected particle for translation to monodomain at ',
          monodomain_ecg_population_filename)
    save_csv_file(data=monodomain_node_lat_population, filename=monodomain_lat_population_filename)
    print('Saved LAT for each selected particle for translation to monodomain at ',
          monodomain_lat_population_filename)
    save_csv_file(data=monodomain_node_repol_population, filename=monodomain_repol_population_filename)
    print('Saved REPOL for each selected particle for translation to monodomain at ',
          monodomain_repol_population_filename)















    node_repol = generate_repolarisation_map(vm=uncertainty_population_vm[translation_i, :, :])
        unprocessed_node_biomarker[repolarisation_time_map_biomarker_name] = node_repol
        # inference_repol_population.append(node_repol)
        # Save biomarkers to allow translation to MonoAlg3D and Alya
        print('Saving biomarkers for uncertainty_i ', translation_i)
        inference_node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker,
                                                               row_index=for_monodomain_translation_node_mapping_index)
        save_pandas(df=inference_node_biomarker, filename=inference_biomarker_result_file_name)
        print('Saved: ', inference_biomarker_result_file_name)
    ####################################################################################################################
    # Step 3: Load precomputed results from this script or run the full script
    if os.path.exists(monodomain_ecg_population_filename):
        print('Step 3: Read precomputed results from previous execution of this script.')
        monodomain_ecg_population = read_ecg_from_csv(filename=monodomain_ecg_population_filename, nb_leads=nb_leads)
        print('monodomain_ecg_population ', monodomain_ecg_population.shape)
        mono_lat_hack_population = read_csv_file(filename=monodomain_lat_population_filename)
        print('mono_lat_hack_population ', mono_lat_hack_population.shape)
        monodomain_node_repol_population = read_csv_file(filename=monodomain_repol_population_filename)
        print('monodomain_repol_population ', monodomain_node_repol_population.shape)
    else:
        print('Step 3: Calculate and save all results for future runs of the script.')
        ####################################################################################################################
        # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
        print('Step 3.1: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
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
        apd_resolution = hyperparameter_dict['apd_resolution']
        cycle_length = hyperparameter_dict['cycle_length']
        vm_max = hyperparameter_dict['vm_max']
        vm_min = hyperparameter_dict['vm_min']
        # Create cellular model instance.
        print('ep_model ', ep_model_twave_name)
        if ep_model_twave_name == 'MitchellSchaefferEP':
            cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                            apd_resolution=apd_resolution, cycle_length=500,
                                                            list_celltype_name=list_celltype_name, verbose=verbose,
                                                            vm_max=1., vm_min=0.)
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
        print('Step 3.2: Generate a cardiac geometry that cannot run the Eikonal.')
        # Argument setup: (in Alphabetical order)
        # Read hyperparameters
        vc_ab_name = hyperparameter_dict['vc_ab_name']
        vc_aprt_name = hyperparameter_dict['vc_aprt_name']
        vc_rt_name = hyperparameter_dict['vc_rt_name']
        vc_rvlv_name = hyperparameter_dict['vc_rvlv_name']
        vc_tm_name = hyperparameter_dict['vc_tm_name']
        vc_tv_name = hyperparameter_dict['vc_aprt_name']
        endo_celltype_name = hyperparameter_dict['endo_celltype_name']
        epi_celltype_name = hyperparameter_dict['epi_celltype_name']
        celltype_vc_info = hyperparameter_dict['celltype_vc_info']
        vc_name_list = hyperparameter_dict['vc_name_list']
        # Create geometry with a dummy conduction system to allow initialising the geometry.
        geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                   conduction_system=EmptyConductionSystem(verbose=verbose),
                                   geometric_data_dir=geometric_data_dir, resolution=inference_resolution,
                                   subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
        raw_geometry_point_cloud = RawEmptyCardiacGeoPointCloud(conduction_system=EmptyConductionSystem(verbose=verbose),
                                                                geometric_data_dir=geometric_data_dir,
                                                                resolution=for_monodomain_translation_resolution,
                                                                subject_name=anatomy_subject_name, verbose=verbose)
        # TODO DELETE THE FOLLOWING CODE
        warn(
            'This should not be done in here!\nThis hack will only work for DTI... meshes, and should be done before calling the script in the futrure.')
        print('min max ', np.amin(raw_geometry_point_cloud.unprocessed_node_xyz),
              np.amax(raw_geometry_point_cloud.unprocessed_node_xyz))
        raw_geometry_point_cloud.unprocessed_node_xyz = convert_from_monoalg3D_to_cm_and_translate(
            raw_geometry_point_cloud.get_node_xyz(), meshname=anatomy_subject_name)
        print('min max ', np.amin(raw_geometry_point_cloud.unprocessed_node_xyz),
              np.amax(raw_geometry_point_cloud.unprocessed_node_xyz))
        # TODO DELETE THE ABOVE CODE
        # Clear Arguments to prevent Argument recycling
        geometric_data_dir = None
        list_celltype_name = None
        # inference_resolution = None
        monodomain_unprocessed_vm = None
        for_monodomain_translation_node_mapping_index = None
        vc_name_list = None
        ####################################################################################################################
        # Step 4: Create propagation model instance, this will be a static dummy propagation model.
        print('Step 3.3: Create propagation model instance, this will be a static dummy propagation model.')
        # Arguments for propagation model:
        # Read hyperparameters
        propagation_parameter_name_list_in_order = hyperparameter_dict['propagation_parameter_name_list_in_order']
        propagation_model = PrescribedLAT(geometry=geometry, lat_prescribed=lat_prescribed,
                                          module_name=propagation_module_name, verbose=verbose)
        # Clear Arguments to prevent Argument recycling
        qrs_lat_prescribed_file_name = None
        lat_prescribed = None
        ####################################################################################################################
        # Step 6: Create ECG calculation method.
        # Arguments for ECG calculation:
        # Read hyperparameters
        clinical_qrs_offset = hyperparameter_dict['clinical_qrs_offset']
        filtering = hyperparameter_dict['filtering']
        # freq_cut = hyperparameter_dict['freq_cut']
        low_freq_cut = hyperparameter_dict['low_freq_cut']
        high_freq_cut = hyperparameter_dict['high_freq_cut']
        max_len_qrs = hyperparameter_dict['max_len_qrs']
        max_len_ecg = hyperparameter_dict['max_len_ecg']
        normalise = hyperparameter_dict['normalise']
        zero_align = hyperparameter_dict['zero_align']
        frequency = hyperparameter_dict['frequency']
        if frequency != 1000:
            warn(
                'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
        # freq_cut = hyperparameter_dict['freq_cut']
        lead_names = hyperparameter_dict['lead_names']
        nb_leads = hyperparameter_dict['nb_leads']
        # Read clinical data
        untrimmed_clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
        clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
        untrimmed_clinical_ecg_raw = None   # Clear Arguments to prevent Argument recycling
        # Create ECG model
        ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                       frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names, low_freq_cut=low_freq_cut,
                                       max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                       nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                       reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                       tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
        # clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
        # Read monodomain ecg
        # untrimmed_monodomain_ecg_raw = read_alya_ecg_mat(file_path=monodomain_ecg_result_file_path)
        # untrimmed_monodomain_ecg = ecg_model.preprocess_ecg(untrimmed_monodomain_ecg_raw)
        # Print out the PCC between monodomain and clinical ECGs
        # untrimmed_monodomain_ecg_pcc = np.mean(calculate_ecg_pcc(ecg_1=clinical_ecg, ecg_2=untrimmed_monodomain_ecg))
        # print('untrimmed_monodomain_ecg_pcc ', untrimmed_monodomain_ecg_pcc)
        # Clear Arguments to prevent Argument recycling
        clinical_data_filename_path = None
        clinical_ecg_raw = None
        filtering = None
        freq_cut = None
        # lead_names = None
        max_len_ecg = None
        max_len_qrs = None
        max_len_st = None
        # nb_leads = None
        normalise = None
        untrimmed_monodomain_ecg_pcc = None
        zero_align = None
    # ####################################################################################################################
    # # Step 7: Define instance of the simulation method.
    # inference_simulator_ecg = SimulateECG(ecg_model=ecg_model, electrophysiology_model=inference_electrophysiology_model, verbose=verbose)
    # inference_simulator_ep = SimulateEP(electrophysiology_model=inference_electrophysiology_model, verbose=verbose)
    # # Clear Arguments to prevent Argument recycling
    # inference_electrophysiology_model = None
    # # ecg_model = None
        ####################################################################################################################
        # Step 8: Define Adapter to translate between theta and parameters.
        # Read hyperparameters
        apd_max_resolution = hyperparameter_dict['apd_max_resolution']
        apd_min_resolution = hyperparameter_dict['apd_min_resolution']
        destination_module_name_list_in_order = hyperparameter_dict['destination_module_name_list_in_order']
        print('destination_module_name_list_in_order ', destination_module_name_list_in_order)
        g_vc_ab_resolution = hyperparameter_dict['g_vc_ab_resolution']
        g_vc_aprt_resolution = hyperparameter_dict['g_vc_aprt_resolution']
        g_vc_rvlv_resolution = hyperparameter_dict['g_vc_rvlv_resolution']
        g_vc_tm_resolution = hyperparameter_dict['g_vc_tm_resolution']
        parameter_destination_module_dict = hyperparameter_dict['parameter_destination_module_dict']
        # TODO Fix this
        # parameter_destination_module_dict['electrophysiology_module'] = electrophysiology_parameter_name_list_in_order
        print('parameter_destination_module_dict ', parameter_destination_module_dict)
        parameter_fixed_value_dict = hyperparameter_dict['parameter_fixed_value_dict']
        print('parameter_fixed_value_dict ', parameter_fixed_value_dict) # TODO add conduction speeds as fixed values?
        parameter_name_list_in_order = hyperparameter_dict['parameter_name_list_in_order']
        physiological_rules_larger_than_dict = hyperparameter_dict['physiological_rules_larger_than_dict']
        theta_name_list_in_order = hyperparameter_dict['theta_name_list_in_order']
        theta_adjust_function_list_in_order = [RoundTheta(resolution=apd_max_resolution),
                                               RoundTheta(resolution=apd_min_resolution),
                                               RoundTheta(resolution=g_vc_ab_resolution),
                                               RoundTheta(resolution=g_vc_aprt_resolution),
                                               RoundTheta(resolution=g_vc_rvlv_resolution),
                                               RoundTheta(resolution=g_vc_tm_resolution)]
        if len(theta_adjust_function_list_in_order) != len(theta_name_list_in_order):
            print('theta_name_list_in_order ', len(theta_name_list_in_order))
            print('theta_adjust_function_list_in_order ', len(theta_adjust_function_list_in_order))
            raise Exception('Different number of adjusting functions and theta for the inference')
        # Create an adapter that can translate between theta and parameters
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
        # Step 12: Iterate over the monodomain vm maps and evaluate their ECGs, LATs and REPOLs.
        # Read parameter population selected for translation to monodomain simulations
        parameter_population = read_csv_file(filename=for_monodomain_parameter_population_file_name, skiprows=1)
        print('parameter_population ', parameter_population.shape)
        # Iterate for all particles chosen to represent the uncertainty of the inference
        monodomain_ecg_population = []
        monodomain_lat_population = []
        monodomain_node_repol_population = []
        for translation_i in range(parameter_population.shape[0]):
            print('uncertainty_i ', translation_i)
            # Read all monodomain vm maps
            # TODO write a new function that can read MonoAlg3D ensight simulation files (vm)
            monodomain_unprocessed_vm, monodomain_xyz = read_time_csv_fields(anatomy_subject_name=anatomy_subject_name,
                                                                             csv_dir=monodomain_simulation_dir,
                                                                             file_name_tag='scalar.INTRA',
                                                                             node_xyz_filename=monodomain_xyz_filename)
            # Interpolate nodefield from the monodomain simulations to the Eikonal geometry
            for_monodomain_translation_node_mapping_index = map_indexes(points_to_map_xyz=geometry.get_node_xyz(),
                                                                        reference_points_xyz=raw_geometry_point_cloud.get_node_xyz())
            monodomain_vm = monodomain_unprocessed_vm[for_monodomain_translation_node_mapping_index, :]
            print('monodomain_vm ', monodomain_vm.shape)
            # Clear Arguments to prevent Argument recycling
            monodomain_unprocessed_vm = None
            monodomain_xyz = None
            raw_geometry_point_cloud = None
            for_monodomain_translation_node_mapping_index = None
            # Calculate LAT and REPOL
            monodomain_repol = generate_repolarisation_map(vm=monodomain_vm)
            monodomain_node_repol_population.append(monodomain_repol)
            print('monodomain_repol ', np.amin(monodomain_repol), ' ', np.amax(monodomain_repol))
            monodomain_repol = None  # Clear Arguments to prevent Argument recycling
            mono_lat_hack = generate_activation_map(vm=monodomain_vm, percentage=70)  # np.argmax(monodomain_vm, axis=1)
            monodomain_lat_population.append(mono_lat_hack)
            print('mono_lat_hack ', np.amin(mono_lat_hack), ' ', np.amax(mono_lat_hack))
            mono_lat_hack = None  # Clear Arguments to prevent Argument recycling
            # Create monodomain ep model:
            monodomain_electrophysiology_model = PrescribedVM(cellular_model=cellular_model,
                                                              module_name=electrophysiology_module_name,
                                                              propagation_model=propagation_model,
                                                              verbose=verbose, vm_prescribed=monodomain_vm)
            monodomain_vm = None  # Clear Arguments to prevent Argument recycling
            # # Simulate LAT and VMs
            # monodomain_simulator_ep = SimulateEP(electrophysiology_model=monodomain_electrophysiology_model,
            #                                      verbose=verbose)
            # monodomain_evaluator_ep = ParameterSimulator(adapter=adapter, simulator=monodomain_simulator_ep,
            #                                              verbose=verbose)
            # _, monodomain_vm = evaluator_ep.simulate_parameter_particle(
            #     parameter_particle=parameter_particle)
            # # Clear Arguments to prevent Argument recycling
            # evaluator_ep = None
            # monodomain_electrophysiology_model = None
            # Simulate ECGs
            monodomain_simulator_ecg = SimulateECG(ecg_model=ecg_model,
                                                   electrophysiology_model=monodomain_electrophysiology_model,
                                                   verbose=verbose)
            monodomain_evaluator_ecg = ParameterEvaluator(adapter=adapter,
                                                          simulator=monodomain_simulator_ecg,
                                                          verbose=verbose)
            parameter_particle = parameter_population[translation_i, :]
            monodomain_pseudo_ecg = monodomain_evaluator_ecg.simulate_parameter_particle(
                parameter_particle=parameter_particle)
            monodomain_ecg_population.append(monodomain_pseudo_ecg)
            # Clear Arguments to prevent Argument recycling
            monodomain_electrophysiology_model = None
            monodomain_evaluator_ecg = None
            monodomain_pseudo_ecg = None

        # Save precomputed ECGs, LAT, and REPOL for the selected particles to translate to monodomain
        monodomain_ecg_population = np.stack(monodomain_ecg_population)
        monodomain_lat_population = np.stack(monodomain_lat_population)
        inference_repol_population = np.stack(inference_repol_population)
        print('monodomain_ecg_population ', monodomain_ecg_population.shape)
        print('monodomain_lat_population ', monodomain_lat_population.shape)
        print('inference_repol_population ', inference_repol_population.shape)
        save_ecg_to_csv(data=monodomain_ecg_population, filename=monodomain_ecg_population_filename)
        print('Saved ECGs for each selected particle for translation to monodomain at ',
              monodomain_ecg_population_filename)
        save_csv_file(data=monodomain_lat_population, filename=monodomain_lat_population_filename)
        print('Saved LAT for each selected particle for translation to monodomain at ',
              monodomain_lat_population_filename)
        save_csv_file(data=monodomain_node_repol_population, filename=monodomain_repol_population_filename)
        print('Saved REPOL for each selected particle for translation to monodomain at ',
              inference_repol_population_filename)

    # Print out the PCC between inference and clinical ECGs
    # inference_ecg_pcc = np.mean(calculate_ecg_pcc(ecg_1=clinical_ecg, ecg_2=inference_ecg))
    # print('inference_ecg_pcc ', inference_ecg_pcc)

    # # Remove first axis when equal to 1:
    # if monodomain_pseudo_ecg.shape[0] == 1:
    #     monodomain_pseudo_ecg = np.squeeze(monodomain_pseudo_ecg, axis=0)
    # print('after monodomain_ecg ', monodomain_pseudo_ecg.shape)
    # Clear Arguments to prevent Argument recycling.
    monodomain_ecg_population_filename = None
    monodomain_lat_population_filename = None
    monodomain_repol_population_filename = None
    ####################################################################################################################



    # # Step 13: Plotting of the ECGs.
    # # Initialise arguments for plotting
    # axes = None
    # fig = None
    # # # Plot the inference ECGs
    # # axes, fig = visualise_ecg(ecg_list=[inference_ecg], lead_name_list=lead_names, axes=axes,
    # #                           ecg_color='black', fig=fig, label_list=['RE'],
    # #                           linewidth=1.)
    # # # Plot the monodomain ECGs
    # # axes, fig = visualise_ecg(ecg_list=[untrimmed_monodomain_ecg], lead_name_list=lead_names, axes=axes,
    # #                           ecg_color='blue', fig=fig, label_list=['Monodomain'],
    # #                           linewidth=1.)
    # # Plot the clinical trace
    # axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
    #                           ecg_color='lime', fig=fig, label_list=['Clinical'],
    #                           linewidth=2.)
    # # Plot the inference ECGs on top of the clinical one
    # axes, fig = visualise_ecg(ecg_list=[inference_ecg], lead_name_list=lead_names, axes=axes,
    #                           ecg_color='black', fig=fig, label_list=['RE'],
    #                           linewidth=1.)
    # # Plot the monodomain ECGs
    # axes, fig = visualise_ecg(ecg_list=[untrimmed_monodomain_ecg], lead_name_list=lead_names, axes=axes,
    #                           ecg_color='blue', fig=fig, label_list=['Monodomain'],
    #                           linewidth=1.)
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # fig.suptitle('Alya ecg vs RE ecg', fontsize=14)
    # plt.show(block=False)
    # fig.savefig(ecg_translation_figure_result_file_name)
    # ## Pseudo ECG comparison in monodomain simulation
    # # Initialise arguments for plotting
    # axes = None
    # fig = None
    # # Plot the inference ECGs
    # axes, fig = visualise_ecg(ecg_list=[monodomain_pseudo_ecg], lead_name_list=lead_names, axes=axes,
    #                           ecg_color='red', fig=fig, label_list=['simplified'],
    #                           linewidth=1.)
    # # Plot the monodomain ECGs
    # axes, fig = visualise_ecg(ecg_list=[untrimmed_monodomain_ecg], lead_name_list=lead_names, axes=axes,
    #                           ecg_color='blue', fig=fig, label_list=['alya'],
    #                           linewidth=1.)
    # # TODO ADD BACK
    # # # Plot the clinical trace
    # # axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
    # #                           ecg_color='lime', fig=fig, label_list=['Clinical'],
    # #                           linewidth=2.)
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # fig.suptitle('ECG calculation comparison inference pipeline vs Alya', fontsize=14)
    # plt.show(block=False)
    # fig.savefig(ecg_calculation_figure_result_file_name)
    # # Clear Arguments to prevent Argument recycling.
    # axes = None
    # fig = None
    # ecg_translation_figure_result_file_name = None
    # ecg_calculation_figure_result_file_name = None
    # inference_ecg_population = None
    # # monodomain_ecg_list = None
    # clinical_ecg = None
    # ####################################################################################################################
    # # Step 14: Generating VM, LAT and REPOL ensights.
    # # Generate coarse VM map for RE and Monodomain comparison
    # # TODO Fix: the monodomain_vm is not saved, so this can only run if it has been generated in this call of the code!!
    # min_len_vm = min(inference_vm.shape[1], monodomain_vm.shape[1])
    # export_ensight_timeseries_case(dir=visualisation_dir, casename=anatomy_subject_name + '_' + inference_resolution
    #                                                                + '_simulation',
    #                                dataname_list=['INTRA_RE', 'INTRA_Alya', 'INTRA_diff'],
    #                                vm_list=[inference_vm, monodomain_vm,
    #                                         monodomain_vm[:, :min_len_vm]-inference_vm[:, :min_len_vm]], dt=1. / frequency,
    #                                nodesxyz=geometry.get_node_xyz(),
    #                                tetrahedrons=geometry.get_tetra())
    # # Generate coarse LAT and REPOL comparison
    # inference_repol = generate_repolarisation_map(inference_vm)
    # print('inference_repol ', np.amin(inference_repol), ' ', np.amax(inference_repol))
    # print('inference_lat ', np.amin(inference_lat), ' ', np.amax(inference_lat))
    # inf_lat_hack = generate_activation_map(inference_vm, 70)
    # print('inference_vm ', inference_vm.shape)
    # print('inf_lat_hack ', inf_lat_hack.shape)
    # print('inf_lat_hack ', np.amin(inf_lat_hack), ' ', np.amax(inf_lat_hack))
    # # Generate node-wise biomarker fields for visualisation
    # node_sf_list = []
    # for ionic_scaling_name in gradient_ion_channel_list:
    #     node_sf_list.append(inference_node_biomarker[ionic_scaling_name])
    # node_apd90 = inference_node_biomarker[biomarker_apd90_name]
    # # Save translation between RE and Monodomain
    # write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=[inference_repol, monodomain_repol,
    #                                                                           inf_lat_hack, inference_lat,
    #                                                                           mono_lat_hack, node_apd90]
    #                                                                          + node_sf_list,
    #                                       node_field_name_list=['inf_repol', 'mono_repol', 'inf_lat_hack', 'inf_lat',
    #                                                             'mono_lat_hack', biomarker_apd90_name]
    #                                                            + gradient_ion_channel_list,
    #                                       subject_name=anatomy_subject_name + '_LAT_REPOL',
    #                                       verbose=verbose,
    #                                       visualisation_dir=visualisation_dir)
    # # Clear Arguments to prevent Argument recycling.
    # anatomy_subject_name = None
    # best_theta = None
    # best_parameter = None
    # # inference_evaluator_ep = None
    # # figure_result_file_name = None
    # frequency = None
    # geometry = None
    # inferred_theta_population = None
    # raw_geometry = None
    # results_dir = None
    # unprocessed_node_mapping_index = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

    #EOF
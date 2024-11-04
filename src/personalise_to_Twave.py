import multiprocessing
import os
import sys
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) < 2:
        anatomy_subject_name = 'DTI032'
        ecg_subject_name = 'DTI032'  # Allows using a different ECG for the personalisation than for the anatomy
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
    from conduction_system import EmptyConductionSystem
    from ecg_functions import PseudoEcgTetFromVM, get_cycle_length
    from geometry_functions import EikonalGeometry, SimulationGeometry
    from simulator_functions import SimulateECG
    from adapter_theta_params import AdapterThetaParams, RoundTheta
    from discrepancy_functions import DiscrepancyECG
    from evaluation_functions import DiscrepancyEvaluator
    from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
    from electrophysiology_functions import ElectrophysiologyAPDmap
    from inference_functions import sample_theta_uniform, ContinuousSMCABC
    from propagation_models import PrescribedLAT
    from path_config import get_path_mapping
    from io_functions import save_dictionary, write_geometry_to_ensight_with_fields, read_csv_file
    from utils import get_vc_ab_name, get_vc_aprt_name, get_vc_rt_name, get_vc_rvlv_name, get_vc_tm_name, \
    get_fibre_speed_name, get_sheet_speed_name, get_normal_speed_name, get_vc_ab_cut_name, get_apd90_biomarker_name, \
    get_sf_iks_biomarker_name

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
    hyperparameter_dict = {}  # Save hyperparameters for reproducibility
    ####################################################################################################################
    # Step 1: Define paths and other environment variables.
    # General settings:
    resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    clinical_data_filename = 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    clinical_data_filename_path = data_dir + clinical_data_filename
    # if ecg_subject_name == 'DTI004':
    #     clinical_qrs_offset = 100
    # else:
    #     clinical_qrs_offset = 0 # 100  # ms TODO This could be calculated automatically and potentially, the clinical ECG could be trimmed to start with the QRS at time zero
    # print('clinical_qrs_offset ', clinical_qrs_offset)
    geometric_data_dir = data_dir + 'geometric_data/'
    # Output Paths:
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
    #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_909'
    # elif anatomy_subject_name == 'DTI032':
    #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_810'
    # elif anatomy_subject_name == 'DTI004':
    #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_1250'
    # else:
    #     ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_'
    # ep_model_twave_name = 'GKs5_GKr0.5_tjca60_CL_'#'GKs5_GKr0.5_tjca60_CL_909'#'GKs5_GKr0.6_tjca60' #'GKs5_GKr0.6_tjca60'  # 'MitchellSchaefferEP' #'no_rescale' #'GKs5_GKr0.6_tjca60'
    ep_model_qrs_name = 'stepFunction'
    gradient_ion_channel_list = [get_sf_iks_biomarker_name()]
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    results_dir_root = path_dict["results_path"]
    # Build results folder structure
    results_dir_part = results_dir_root + experiment_type + '_data/'
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
    results_dir_part = results_dir_part + anatomy_subject_name + '/'
    assert os.path.exists(results_dir_part) # Path should already exist from running the QRS inference
    results_dir_part_twave = results_dir_part + 'twave_' + gradient_ion_channel_str + '_' + ep_model_twave_name + '/'
    if not os.path.exists(results_dir_part_twave):
        os.mkdir(results_dir_part_twave)
    results_dir_part_qrs = results_dir_part + 'qrs_' + ep_model_qrs_name + '/'
    assert os.path.exists(results_dir_part_qrs)  # Path should already exist from running the QRS inference
    results_dir_part = None     # Clear Arguments to prevent Argument recycling
    # Use date to name the result folder to preserve some history of results
    current_month_text = 'Jun'#datetime.now().strftime('%h')  # e.g., Feb
    current_year_full = datetime.now().strftime('%Y')  # e.g., 2024
    date_str = current_month_text + '_' + current_year_full
    results_dir_twave = results_dir_part_twave + date_str + '_fixed_filter/'
    # Create results directory
    if not os.path.exists(results_dir_twave):
        os.mkdir(results_dir_twave)
    results_dir_part_twave = None  # Clear Arguments to prevent Argument recycling
    results_dir_qrs = results_dir_part_qrs + date_str + '/best_discrepancy/'
    assert os.path.exists(results_dir_qrs)  # Path should already exist from running the QRS inference
    results_dir_part_qrs = None  # Clear Arguments to prevent Argument recycling
    # Intermediate Paths: # e.g., results from the QRS inference
    result_tag = experiment_type
    qrs_lat_prescribed_filename = anatomy_subject_name + '_' + resolution + '_nodefield_' + result_tag + '-lat.csv'
    qrs_lat_prescribed_filename_path = results_dir_qrs + qrs_lat_prescribed_filename
    results_dir_qrs = None  # Clear Arguments to prevent Argument recycling
    if not os.path.isfile(qrs_lat_prescribed_filename_path):
        raise Exception(
            "This inference needs to be run after the QRS inference and need the correct path with those results at\n" + qrs_lat_prescribed_filename_path)
    # Continue defining results paths and configuration
    hyperparameter_result_file_name = results_dir_twave + anatomy_subject_name + '_' + resolution + '_hyperparameter.txt'
    theta_result_file_name = results_dir_twave + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_theta_population.csv'
    parameter_result_file_name = results_dir_twave + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_parameter_population.csv'
    # Enable saving partial inference results and restarting from where it was left
    unfinished_process_dir = results_dir_twave + 'unfinished_process/'
    if not os.path.exists(unfinished_process_dir):
        os.mkdir(unfinished_process_dir)
    unfinished_theta_result_file_name = unfinished_process_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_theta_population.csv'
    unfinished_parameter_result_file_name = unfinished_process_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_parameter_population.csv'
    # Inference history
    inference_history_dir = results_dir_twave + 'inference_history/'
    if not os.path.exists(inference_history_dir):
        os.mkdir(inference_history_dir)
    # # Uncomment these lines to make sure that the process starts from scratch
    # print('Inference process will start from scratch!')
    # if os.path.isfile(hyperparameter_result_file_name):
    #     os.remove(hyperparameter_result_file_name)
    # if os.path.isfile(theta_result_file_name):
    #     os.remove(theta_result_file_name)
    # if os.path.isfile(parameter_result_file_name):
    #     os.remove(parameter_result_file_name)
    # if os.path.isfile(unfinished_theta_result_file_name):
    #     os.remove(unfinished_theta_result_file_name)
    # if os.path.isfile(unfinished_parameter_result_file_name):
    #     os.remove(unfinished_parameter_result_file_name)
    # APD dictionary configuration:
    cellular_stim_amp = 11
    cellular_model_convergence = 'not_converged'
    stimulation_protocol = 'diffusion'
    cellular_data_relative_path = 'cellular_data/' + cellular_model_convergence + '_' + stimulation_protocol + '_' + str(
        cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model_twave_name + '/'
    cellular_data_dir_complete = data_dir + cellular_data_relative_path
    # Directory to save the configuration of the inference before it runs to allow manual inspection:
    visualisation_dir = results_dir_twave + 'checkpoint/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Save hyperparameters for reproducibility
    hyperparameter_dict['cellular_data_relative_path'] = cellular_data_relative_path
    hyperparameter_dict['cellular_stim_amp'] = cellular_stim_amp
    hyperparameter_dict['clinical_data_filename'] = clinical_data_filename
    # hyperparameter_dict['clinical_qrs_offset'] = clinical_qrs_offset
    hyperparameter_dict['cellular_model_convergence'] = cellular_model_convergence
    hyperparameter_dict['experiment_type'] = experiment_type  # This will tell in the future if this was sa or personalisation
    hyperparameter_dict['ep_model_qrs'] = ep_model_qrs_name
    hyperparameter_dict['ep_model_twave'] = ep_model_twave_name
    hyperparameter_dict['gradient_ion_channel_list'] = gradient_ion_channel_list
    hyperparameter_dict['heart_rate'] = heart_rate
    hyperparameter_dict['qrs_lat_prescribed_filename'] = qrs_lat_prescribed_filename
    hyperparameter_dict['result_tag'] = result_tag
    hyperparameter_dict['stimulation_protocol'] = stimulation_protocol
    # Clear Arguments to prevent Argument recycling
    cellular_data_relative_path = None
    cellular_stim_amp = None
    clinical_data_dir_tag = None
    clinical_data_filename = None
    cellular_model_convergence = None
    cycle_length = None
    data_dir = None
    ecg_subject_name = None
    ep_model_qrs_name = None
    experiment_type = None
    gradient_ion_channel_list = None
    heart_rate = None
    qrs_lat_prescribed_filename = None
    intermediate_dir = None
    results_dir_twave = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
    # Arguments for cellular model:
    print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
    # TODO link the cycle lenght with the heart rates and read them from somewhere
    # if anatomy_subject_name == 'DTI024':
    #     # cellular_model_name = 'torord_calibrated_pom_CL909'
    #     heart_rate = 66
    # elif anatomy_subject_name == 'DTI032':
    #     heart_rate = 74
    #     # cellular_model_name = 'torord_calibrated_pom_CL810'
    # elif anatomy_subject_name == 'DTI004':
    #     heart_rate = 48
    #     # cellular_model_name = 'torord_calibrated_pom_CL1250'
    # else:
    #     cellular_model_name = 'torord_calibrated_pom_CL'
    # cycle_length = get_cycle_length(heart_rate=heart_rate)
    cellular_model_name = 'torord_calibrated_pom_CL' + cycle_length_str
    print('cellular_model_name ', cellular_model_name)
    # cellular_model_name = 'torord_calibrated_pom_CL909'#'torord_calibrated_pom_1000Hz'
    endo_celltype_name = 'endo'
    # epi_celltype_name = 'epi'
    list_celltype_name = [endo_celltype_name] # , epi_celltype_name]
    biomarker_upstroke_name = 'activation_time'  # TODO consider chaning to something different with the name upstroke in it
    biomarker_apd90_name = get_apd90_biomarker_name()
    biomarker_celltype_name = 'celltype'
    # Create cellular model instance.
    if ep_model_twave_name == 'MitchellSchaefferEP':
        print('Using MS cellular model!')
        # TODO The APD ranges could be set automatically from the ST duration
        apd_max_max = 400
        apd_min_min = 200
        apd_resolution = 1
        # cycle_length = 800  # TODO This should be personalised to the subject's heart rate
        vm_max = 1.
        vm_min = 0.
        cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                        apd_resolution=apd_resolution, cycle_length=cycle_length,
                                                        list_celltype_name=list_celltype_name, verbose=verbose,
                                                        vm_max=vm_max, vm_min=vm_min)
        # Save hyperparameters for reproducibility
        hyperparameter_dict['apd_resolution'] = apd_resolution
        # hyperparameter_dict['cycle_length'] = cycle_length
        hyperparameter_dict['vm_max'] = vm_max
        hyperparameter_dict['vm_min'] = vm_min
        # Clear Arguments to prevent Argument recycling
        apd_resolution = None
        cycle_length = None
        vm_max = None
        vm_min = None
    else:
        print('Using ToR-ORd cellular model!')
        cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                          biomarker_apd90_name=biomarker_apd90_name,
                                                          biomarker_celltype_name=biomarker_celltype_name,
                                                          cellular_data_dir=cellular_data_dir_complete,
                                                          cellular_model_name=cellular_model_name,
                                                          list_celltype_name=list_celltype_name, verbose=verbose)
        # TODO The APD ranges could be set automatically from the ST duration
        apd_min_min, apd_max_max = cellular_model.get_biomarker_range(biomarker_name=biomarker_apd90_name)

    print('apd_min_min ', apd_min_min)
    print('apd_max_max ', apd_max_max)
    assert apd_max_max > apd_min_min
    # Save hyperparameters for reproducibility
    hyperparameter_dict['apd_max_max'] = apd_max_max
    hyperparameter_dict['apd_min_min'] = apd_min_min
    hyperparameter_dict['biomarker_apd90_name'] = biomarker_apd90_name
    hyperparameter_dict['biomarker_celltype_name'] = biomarker_celltype_name
    hyperparameter_dict['biomarker_upstroke_name'] = biomarker_upstroke_name
    hyperparameter_dict['cellular_model_name'] = cellular_model_name
    # hyperparameter_dict['endo_celltype_name'] = endo_celltype_name
    # hyperparameter_dict['epi_celltype_name'] = epi_celltype_name
    hyperparameter_dict['list_celltype_name'] = list_celltype_name
    # Clear Arguments to prevent Argument recycling
    biomarker_apd90_name = None
    biomarker_celltype_name = None
    biomarker_upstroke_name = None
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_model_name = None
    cycle_length_str = None
    ep_model_twave_name = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 3: Generate a cardiac geometry that cannot run the Eikonal.
    # Argument setup: (in Alphabetical order)
    print('Step 3: Generate a cardiac geometry that cannot run the Eikonal.')
    vc_ab_cut_name = get_vc_ab_cut_name()
    vc_aprt_name = get_vc_aprt_name()
    # vc_rt_name = get_vc_rt_name()  # TODO this should not be required! __define_fast_endocardial_layer_dense_sparse_regions_vc should not run in this script!
    vc_rvlv_name = get_vc_rvlv_name()
    vc_tm_name = get_vc_tm_name()
    # vc_name_list = [vc_ab_cut_name, vc_aprt_name, vc_rt_name, vc_rvlv_name, vc_tm_name]
    vc_name_list = [vc_ab_cut_name, vc_aprt_name, vc_rvlv_name, vc_tm_name]
    # Pre-assign celltype spatial correspondence.
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 1.]}}
    # celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0.3, 1.]}, epi_celltype_name: {vc_tm_name: [0., 0.3]}}
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = SimulationGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['celltype_vc_info'] = celltype_vc_info
    hyperparameter_dict['vc_name_list'] = vc_name_list
    hyperparameter_dict['vc_ab_cut_name'] = vc_ab_cut_name
    hyperparameter_dict['vc_aprt_name'] = vc_aprt_name
    # hyperparameter_dict['vc_rt_name'] = vc_rt_name
    hyperparameter_dict['vc_rvlv_name'] = vc_rvlv_name
    hyperparameter_dict['vc_tm_name'] = vc_tm_name
    # Clear Arguments to prevent Argument recycling
    geometric_data_dir = None
    # vc_name_list = None
    ####################################################################################################################
    # Step 4: Prepare smoothing configuration to resemble diffusion effects
    print('Step 4: Prepare smoothing configuration to resemble diffusion effects.')
    # Define the speeds used during the fibre-based smoothing
    warn(
        'Inference from QT can, but does NOT, update the speeds in the smoothing function!\nAlso, it requires some initial fixed values!')
    # TODO in the case of doing the inference sequentially, first the QRS and later the T wave, we could use the inferred speed values in here!!
    fibre_speed = 0.065  # Taggart et al. (2000) https://doi.org/10.1006/jmcc.2000.1105
    #TODO What should be transmural and what should be sheet? These two words mean different things?
    warn('What should be transmural and what should be sheet?')
    # sheet_speed = 0.051  # Taggart et al. (2000) https://doi.org/10.1006/jmcc.2000.1105
    # TODO make code to automatically read these values from the inference results
    # Mannually taken from the inference results
    if anatomy_subject_name == 'DTI024':
        sheet_speed = 0.036
    elif anatomy_subject_name == 'DTI004':
        sheet_speed = 0.027
    elif anatomy_subject_name == 'DTI032':
        sheet_speed = 0.04
    normal_speed = 0.048  # Taggart et al. (2000) https://doi.org/10.1006/jmcc.2000.1105
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    # TODO the following value is the strenght of the smoothing and it depends on the resolution of the monodomain simulation?
    # TODO this distance scaling should be directly proportional to dt_smoothing, right?
    smoothing_ghost_distance_to_self = 0.01 #0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    print('Precompuing the smoothing, change this please!')
    geometry.precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(
        fibre_speed=fibre_speed, sheet_speed=sheet_speed, normal_speed=normal_speed,
        ghost_distance_to_self=smoothing_ghost_distance_to_self)
    # Save hyperparameters for reproducibility
    fibre_speed_name = get_fibre_speed_name()
    sheet_speed_name = get_sheet_speed_name()
    normal_speed_name = get_normal_speed_name()
    hyperparameter_dict['fibre_speed_name'] = fibre_speed_name
    hyperparameter_dict['sheet_speed_name'] = sheet_speed_name
    hyperparameter_dict['normal_speed_name'] = normal_speed_name
    hyperparameter_dict[fibre_speed_name] = fibre_speed
    hyperparameter_dict[sheet_speed_name] = sheet_speed
    hyperparameter_dict[normal_speed_name] = normal_speed
    hyperparameter_dict['smoothing_ghost_distance_to_self'] = smoothing_ghost_distance_to_self
    ####################################################################################################################
    # Step 4: Create propagation model instance, this will be a static dummy propagation model.
    print('Step 4: Create propagation model instance, this will be a static dummy propagation model.')
    propagation_parameter_name_list_in_order = []
    lat_prescribed = (read_csv_file(filename=qrs_lat_prescribed_filename_path)).astype(int)
    propagation_model = PrescribedLAT(geometry=geometry, lat_prescribed=lat_prescribed,
                                      module_name=propagation_module_name, verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['propagation_parameter_name_list_in_order'] = propagation_parameter_name_list_in_order
    # Clear Arguments to prevent Argument recycling
    qrs_lat_prescribed_filename_path = None
    lat_prescribed = None
    # celltype = None
    # node_vc = None
    ####################################################################################################################
    # Step 5: Create Whole organ Electrophysiology model.
    print('Step 5: Create Whole organ Electrophysiology model.')
    # Arguments for Electrophysiology model:
    apd_max_name = 'apd_max'
    apd_min_name = 'apd_min'
    g_vc_ab_name = vc_ab_cut_name
    g_vc_aprt_name = vc_aprt_name
    g_vc_rvlv_name = vc_rvlv_name
    g_vc_tm_name = vc_tm_name
    # TODO should the conduction speed be included here?
    electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name, g_vc_rvlv_name, g_vc_tm_name]
    # electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_tm_name]#, , g_vc_rvlv_name, g_vc_tm_name]
    # Spatial and temporal smoothing parameters:
    # smoothing_count = 5 # is 5 enough?
    # TODO the following line on Friday 01/09/2023 it was set to 0.2!!! If the hump persists, perhaps increase the smoothing again!!
    # smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 10% and the present as 90%
    # smoothing_count = 22#45#40
    smoothing_dt = 20  # ms #TODO this should be multiplying the strength of the smoothing
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    # smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    # TODO WE ARE NO LONGER USING TEMPORAL SMOOTHING
    # smoothing_past_present_window = [0.0, 1.0]#[0.05, 0.95]    # Weight the past as 5% and the present as 95%
    start_smoothing_time_index = 100  # (ms) assumming 1000Hz
    end_smoothing_time_index = 450#400  # (ms) assumming 1000Hz
    # fibre_speed_name = 'fibre_speed'
    # sheet_speed_name = 'sheet_speed'
    # normal_speed_name = 'normal_speed'
    electrophysiology_model = ElectrophysiologyAPDmap(apd_max_name=apd_max_name, apd_min_name=apd_min_name,
                                                                cellular_model=cellular_model,
                                                                end_smoothing_time_index=end_smoothing_time_index,
                                                                fibre_speed_name=fibre_speed_name,
                                                                module_name=electrophysiology_module_name,
                                                                normal_speed_name=normal_speed_name,
                                                                parameter_name_list_in_order=electrophysiology_parameter_name_list_in_order,
                                                                propagation_model=propagation_model,
                                                                sheet_speed_name=sheet_speed_name,
                                                                smoothing_dt=smoothing_dt,
                                                                smoothing_ghost_distance_to_self=smoothing_ghost_distance_to_self,
                                                                start_smoothing_time_index=start_smoothing_time_index,
                                                                verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['apd_max_name'] = apd_max_name
    hyperparameter_dict['apd_min_name'] = apd_min_name
    hyperparameter_dict['g_vc_ab_name'] = g_vc_ab_name
    hyperparameter_dict['g_vc_aprt_name'] = g_vc_aprt_name
    hyperparameter_dict['g_vc_rvlv_name'] = g_vc_rvlv_name
    hyperparameter_dict['g_vc_tm_name'] = g_vc_tm_name
    hyperparameter_dict['electrophysiology_parameter_name_list_in_order'] = electrophysiology_parameter_name_list_in_order
    hyperparameter_dict['end_smoothing_time_index'] = end_smoothing_time_index
    hyperparameter_dict['start_smoothing_time_index'] = start_smoothing_time_index
    # hyperparameter_dict['smoothing_count'] = smoothing_count
    hyperparameter_dict['smoothing_dt'] = smoothing_dt
    # hyperparameter_dict['smoothing_past_present_window'] = smoothing_past_present_window
    # Clear Arguments to prevent Argument recycling
    cellular_model = None
    propagation_model = None
    smoothing_count = None
    smoothing_ghost_distance_to_self = None
    smoothing_past_present_window = None
    vc_ab_cut_name = None
    vc_aprt_name = None
    vc_rvlv_name = None
    vc_tm_name = None
    ####################################################################################################################
    # Step 6: Create ECG calculation method.
    print('Step 6: Create ECG calculation method.')
    # Arguments for ECG calculation:
    filtering = True
    max_len_qrs = 200#256  # can use 200 to save memory space # This hyper-paramter is used when paralelising the ecg computation, because it needs a structure to synchronise the results from the multiple threads.
    max_len_st = 300#512  # can use 200 to save memory space
    max_len_ecg = max_len_qrs + max_len_st
    normalise = True
    zero_align = True
    frequency = 1000  # Hz
    if frequency != 1000:
        warn(
            'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
    low_freq_cut = 0.001#0.5
    high_freq_cut = 100#150
    I_name = 'I'
    II_name = 'II'
    v3_name = 'V3'
    v5_name = 'V5'
    lead_names = [I_name, II_name, 'V1', 'V2', v3_name, 'V4', v5_name, 'V6']
    nb_leads = len(lead_names)
    # Read clinical data
    # TODO This code may not work well for an ECG with only one lead!!
    clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
    # Create ECG model
    ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                   frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                   low_freq_cut=low_freq_cut, max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs,
                                   nb_leads=nb_leads, nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                   reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                   tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
    clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['filtering'] = filtering
    hyperparameter_dict['frequency'] = frequency
    hyperparameter_dict['high_freq_cut'] = high_freq_cut
    hyperparameter_dict['lead_names'] = lead_names
    hyperparameter_dict['low_freq_cut'] = low_freq_cut
    hyperparameter_dict['max_len_qrs'] = max_len_qrs
    hyperparameter_dict['max_len_ecg'] = max_len_ecg
    hyperparameter_dict['nb_leads'] = nb_leads
    hyperparameter_dict['normalise'] = normalise
    hyperparameter_dict['I_name'] = I_name
    hyperparameter_dict['II_name'] = II_name
    hyperparameter_dict['v3_name'] = v3_name
    hyperparameter_dict['v5_name'] = v5_name
    hyperparameter_dict['zero_align'] = zero_align
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename_path = None
    clinical_ecg_raw = None
    filtering = None
    frequency = None
    high_freq_cut = None
    geometry = None  # Clear Geometry
    lead_names = None
    max_len_ecg = None
    max_len_qrs = None
    max_len_st = None
    nb_leads = None
    normalise = None
    v3_name = None
    v5_name = None
    untrimmed_clinical_ecg_raw = None
    zero_align = None
    ####################################################################################################################
    # Step 7: Define instance of the simulation method.
    print('Step 7: Define instance of the simulation method.')
    simulator = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    electrophysiology_model = None
    ecg_model = None
    ####################################################################################################################
    # Step 8: Define Adapter to translate between theta and parameters.
    print('Step 8: Define Adapter to translate between theta and parameters.')
    parameter_name_list_in_order = propagation_parameter_name_list_in_order + electrophysiology_parameter_name_list_in_order
    theta_name_list_in_order = [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name, g_vc_rvlv_name, g_vc_tm_name]
    continuous_theta_name_list_in_order = theta_name_list_in_order
    nb_discrete_theta = len(theta_name_list_in_order) - len(continuous_theta_name_list_in_order)
    parameter_fixed_value_dict = {}     # Define values for non-theta parameters.
    # parameter_fixed_value_dict[apd_max_name] = 268 # TODO
    # parameter_fixed_value_dict[apd_min_name] = 180 # TODO
    physiological_rules_larger_than_dict = {}   # Define custom rules to constrain which parameters must be larger than others.
    physiological_rules_larger_than_dict[apd_max_name] = [apd_min_name]  # TODO Check that this rule is being used!
    # [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name, g_vc_rvlv_name, g_vc_tm_name]
    apd_max_resolution = 2.
    apd_min_resolution = 2.
    g_vc_ab_resolution = 0.1 # used to be 0.001
    g_vc_aprt_resolution = 0.1
    g_vc_rvlv_resolution = 0.1
    g_vc_tm_resolution = 0.1
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
    # Distribute parameters into modules
    destination_module_name_list_in_order = [propagation_module_name, electrophysiology_module_name]
    parameter_destination_module_dict = {}
    parameter_destination_module_dict[propagation_module_name] = propagation_parameter_name_list_in_order
    parameter_destination_module_dict[electrophysiology_module_name] = electrophysiology_parameter_name_list_in_order
    print(
        'Caution: these rules have only been enabled for the inferred parameters!')  # TODO: modify this to also enable rules for fixed parameters (e.g., fibre_speed >= sheet_speed)
    # Create an adapter that can translate between theta and parameters
    adapter = AdapterThetaParams(destination_module_name_list_in_order=destination_module_name_list_in_order,
                                 parameter_fixed_value_dict=parameter_fixed_value_dict,
                                 parameter_name_list_in_order=parameter_name_list_in_order,
                                 parameter_destination_module_dict=parameter_destination_module_dict,
                                 theta_adjust_function_list_in_order=theta_adjust_function_list_in_order,
                                 theta_name_list_in_order=theta_name_list_in_order,
                                 physiological_rules_larger_than_dict=physiological_rules_larger_than_dict,
                                 verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['apd_max_resolution'] = apd_max_resolution
    hyperparameter_dict['apd_min_resolution'] = apd_min_resolution
    hyperparameter_dict['continuous_theta_name_list_in_order'] = continuous_theta_name_list_in_order
    hyperparameter_dict['destination_module_name_list_in_order'] = destination_module_name_list_in_order
    hyperparameter_dict['g_vc_ab_resolution'] = g_vc_ab_resolution
    hyperparameter_dict['g_vc_aprt_resolution'] = g_vc_aprt_resolution
    hyperparameter_dict['g_vc_rvlv_resolution'] = g_vc_rvlv_resolution
    hyperparameter_dict['g_vc_tm_resolution'] = g_vc_tm_resolution
    hyperparameter_dict['nb_discrete_theta'] = nb_discrete_theta
    hyperparameter_dict['parameter_destination_module_dict'] = parameter_destination_module_dict
    hyperparameter_dict['parameter_fixed_value_dict'] = parameter_fixed_value_dict
    hyperparameter_dict['parameter_name_list_in_order'] = parameter_name_list_in_order
    hyperparameter_dict['physiological_rules_larger_than_dict'] = physiological_rules_larger_than_dict
    hyperparameter_dict['theta_name_list_in_order'] = theta_name_list_in_order
    # Clear Arguments to prevent Argument recycling
    apd_max_name = None
    apd_max_resolution = None
    apd_min_name = None
    apd_min_resolution = None
    candidate_root_node_names = None
    # continuous_theta_name_list_in_order = None
    g_vc_ab_name = None
    g_vc_aprt_name = None
    g_vc_rvlv_name = None
    g_vc_tm_name = None
    g_vc_ab_resolution = None
    g_vc_aprt_resolution = None
    g_vc_rvlv_resolution = None
    g_vc_sep_resolution = None
    g_vc_tm_resolution = None
    nb_discrete_theta = None
    normal_speed_name = None
    parameter_fixed_value_dict = None
    speed_parameter_name_list_in_order = None
    theta_adjust_function_list_in_order = None
    sheet_speed_name = None
    ####################################################################################################################
    # Step 9: Define the discrepancy metric.
    print('Step 9: Define the discrepancy metric.')
    # Arguments for discrepancy metric:
    error_method_name = 'rmse_pcc_cubic'
    # Create discrepancy metric instance.
    discrepancy_metric = DiscrepancyECG(error_method_name=error_method_name)  # TODO: add weighting control between PCC and RMSE
    # Save hyperparameters for reproducibility
    hyperparameter_dict['error_method_name'] = error_method_name
    # Clear Arguments to prevent Argument recycling
    error_method_name = None
    ####################################################################################################################
    # Step 10: Create evaluator_ecg.
    print('Step 10: Create evaluator_ecg.')
    evaluator = DiscrepancyEvaluator(adapter=adapter, discrepancy_metric=discrepancy_metric, simulator=simulator,
                                     target_data=clinical_ecg, verbose=verbose)
    # Save hyperparameters for reproducibility
    # Clear Arguments to prevent Argument recycling.
    adapter = None
    discrepancy_metric = None
    simulator = None
    clinical_ecg = None
    ####################################################################################################################
    # Step 11: Create instance of inference method.
    print('Step 11: Create instance of inference method.')
    # Arguments for Bayesian Inference method:
    # Population ranges and priors
    '''apd'''
    apd_exploration_margin = 80   # ms     # TODO Could be informed using ECG metrics
    apd_max_range = [apd_max_max - apd_exploration_margin, apd_max_max]  # cm/ms
    apd_max_prior = None  # [mean, std]
    apd_min_range = [apd_min_min, apd_min_min + apd_exploration_margin]  # cm/ms
    apd_min_prior = None  # [mean, std]
    '''ab'''
    gab_max = 1
    gab_min = -1
    g_vc_ab_range = [gab_min, gab_max]  # cm/ms
    g_vc_ab_prior = None  # [mean, std]
    '''aprt'''
    gaprt_max = 1
    gaprt_min = -1
    g_vc_aprt_range = [gaprt_min, gaprt_max]  # cm/ms
    g_vc_aprt_prior = None  # [mean, std]
    '''rvlv'''
    grvlv_max = 1
    grvlv_min = -1
    g_vc_rvlv_range = [grvlv_min, grvlv_max]  # cm/ms
    g_vc_rvlv_prior = None  # [mean, std]
    '''tm'''
    gtm_max = 1
    gtm_min = -1    # the findings in the lit review suggest that it can go both ways
    g_vc_tm_range = [gtm_min, gtm_max]  # cm/ms
    g_vc_tm_prior = None  # [mean, std]
    # Aggregate ranges and priors
    boundaries_continuous_theta = [apd_max_range, apd_min_range, g_vc_ab_range, g_vc_aprt_range, g_vc_rvlv_range,
                                   g_vc_tm_range]
    continuous_theta_prior_list = [apd_max_prior, apd_min_prior, g_vc_ab_prior, g_vc_aprt_prior, g_vc_rvlv_prior,
                                   g_vc_tm_prior]
    nb_continuous_theta = len(continuous_theta_name_list_in_order)
    # Check consistency of sizes
    if nb_continuous_theta != len(continuous_theta_prior_list) or nb_continuous_theta != len(
            boundaries_continuous_theta):
        raise Exception("Not a consistent number of parameters for the inference.")
    ### Define SMC-ABC configuration
    # TODO use a max_memory_population_size parameter that the machine can handle to enable larger population sizes to be split internally!!
    population_size = 120  # 512   # Rule of thumb number (at least x2 number of processes)    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    max_mcmc_steps = 50  # This number allows for extensive exploration
    unique_stopping_ratio = 0.5  # if only 50% of the population is unique, then terminate the inference and consider that it has converged.
    # Specify the "retain ratio". This is the proportion of samples that would match the current data in the case of N_on = 1 and all particles having the same variable switched on. That is to say,
    # it is an approximate chance of choosing "random updates" over the particle information
    retain_ratio = 0.5  # original value in Brodie's code
    max_root_node_jiggle_rate = 0.1
    keep_fraction = max((population_size - 2 * multiprocessing.cpu_count()) / population_size,  0.5)  # 0.75)   # without the max() function it can go negative when the population size is smaller than the number of threads
    if verbose:
        print('multiprocessing.cpu_count() ', multiprocessing.cpu_count())
        print('population_size ', population_size)
        print('keep_fraction ', keep_fraction)
        print('worst_keep ', int(np.round(population_size * keep_fraction)))
        print('jiggle number of samples ', population_size - int(np.round(population_size * keep_fraction)))

    ### Create instance of the inference method.
    # Define initialisation function for theta
    ini_population_continuous_theta = sample_theta_uniform  # In some cases it may be easier for the inference to start from a grid search instead of LHS
    inference_method = ContinuousSMCABC(boundaries_theta=boundaries_continuous_theta,
                                        theta_prior_list=continuous_theta_prior_list,
                                        evaluator=evaluator,
                                        ini_population_theta=ini_population_continuous_theta,
                                        keep_fraction=keep_fraction,
                                        max_mcmc_steps=max_mcmc_steps,
                                        population_size=population_size, retain_ratio=retain_ratio,
                                        verbose=verbose)

    # Save hyperparameters for reproducibility
    hyperparameter_dict['population_size'] = population_size  # Hyperparameter
    hyperparameter_dict['max_mcmc_steps'] = max_mcmc_steps  # Hyperparameter
    hyperparameter_dict['retain_ratio'] = retain_ratio  # Hyperparameter
    hyperparameter_dict['keep_fraction'] = keep_fraction  # Hyperparameter
    hyperparameter_dict['boundaries_continuous_theta'] = boundaries_continuous_theta  # Hyperparameter
    hyperparameter_dict['continuous_theta_prior_list'] = continuous_theta_prior_list  # Hyperparameter
    # Clear Arguments to prevent Argument recycling.
    evaluator = None
    boundaries_theta = None
    ini_population_theta = None
    max_mcmc_steps = None
    nb_root_node_prior = None
    nb_candidate_root_nodes = None
    nb_root_nodes_range = None
    nb_theta = None
    population_size = None
    retain_ratio = None
    theta_prior_list = None
    verbose = None
    ####################################################################################################################
    # Step 12: Run the inference process.
    print('Step 12: Run the inference process.')
    desired_discrepancy = 0.5 #0.5  # used to be 0.1 # This value needs to be changed with respect of what discrepancy metric you want to use.  # this value is for the DTW metric was 0.35  # After several tests was found good with the latest discrepancy metric strategy
    max_process_alive_time = 20.  # hours, in Supercomputers, usually there is a maximum 24 hour limit on any job that you submit.
    visualisation_count = 5     # Minimum of 1 to avoid division by zero
    # Save geometry as a check point
    geometry = inference_method.evaluator.simulator.electrophysiology_model.propagation_model.geometry
    # geometry.node_xyz = geometry.get_node_xyz() - np.amin(geometry.get_node_xyz(), axis=0)
    vc_node_field_list = []
    for vc_name in vc_name_list:
        vc_node_field_list.append(geometry.node_vc[vc_name])
    write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=vc_node_field_list,
                                          node_field_name_list=vc_name_list,
                                          subject_name=anatomy_subject_name + '_' + resolution + '_checkpoint',
                                          verbose=verbose,
                                          visualisation_dir=visualisation_dir)
    print('Saved geometry before inference in ', visualisation_dir)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['desired_discrepancy'] = desired_discrepancy  # Hyperparameter
    hyperparameter_dict['max_process_alive_time'] = max_process_alive_time  # Hyperparameter
    hyperparameter_dict['unique_stopping_ratio'] = unique_stopping_ratio  # Hyperparameter
    save_dictionary(dictionary=hyperparameter_dict, filename=hyperparameter_result_file_name)
    print('Saved hyperparameter: ', hyperparameter_result_file_name)
    # Initialise population of parameter sets
    # TODO check the following code
    if os.path.isfile(unfinished_theta_result_file_name):
        print('Continuing previous inference using population in ', unfinished_theta_result_file_name)
        pandas_previous_population_theta = pd.read_csv(unfinished_theta_result_file_name, delimiter=',')
        previous_population_theta = inference_method.evaluator.translate_from_pandas_to_theta(
            pandas_theta=pandas_previous_population_theta)
        inference_method.set_population_theta(population_theta=previous_population_theta)
    # Run inference process
    population_theta, inference_success = inference_method.sample(desired_discrepancy=desired_discrepancy,
                                                                  max_sampling_time=max_process_alive_time,
                                                                  unique_stopping_ratio=unique_stopping_ratio,
                                                                  visualisation_count=visualisation_count,
                                                                  inference_history_dir=inference_history_dir)
    population_parameter = inference_method.get_parameter_from_theta(population_theta)
    # Clear Arguments to prevent Argument recycling.
    anatomy_subject_name = None
    desired_discrepancy = None
    inference_method = None
    max_process_alive_time = None
    resolution = None
    unique_stopping_ratio = None
    vc_name_list = None
    ####################################################################################################################
    # Step 14: Save the inference results.
    print('Step 14: Save the inference results.')
    # TODO save the root node locations and their activation times, so that the simulation can be replicated with future versions
    # TODO of the code.
    # TODO save candidate root nodes so that the meta-indexes can be used to point at them.
    if not inference_success:
        warn('Saving partial results to the inference process')
        theta_result_file_name = unfinished_theta_result_file_name
        parameter_result_file_name = unfinished_parameter_result_file_name
    elif os.path.isfile(unfinished_theta_result_file_name):
        os.remove(unfinished_theta_result_file_name)
    # Save inference results
    np.savetxt(theta_result_file_name, population_theta, delimiter=',', header=','.join(theta_name_list_in_order),
               comments='')
    print('Saved inferred population theta: ', theta_result_file_name)
    np.savetxt(parameter_result_file_name, population_parameter, delimiter=',',
               header=','.join(parameter_name_list_in_order), comments='')
    print('Saved inferred population parameter: ', parameter_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    hyperparameter_dict = None
    parameter_name_list_in_order = None
    population_parameter = None
    population_theta = None
    theta_name_list_in_order = None
    theta_result_file_name = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)


# EOF



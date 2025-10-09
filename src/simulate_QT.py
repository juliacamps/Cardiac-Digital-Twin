import multiprocessing
import os
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conduction_system import DjikstraConductionSystemVC, EmptyConductionSystem, PurkinjeSystemVC
from ecg_functions import PseudoQRSTetFromStepFunction, PseudoEcgTetFromVM
from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet, RawVCFibreCardiacGeoTet
from propagation_models import EikonalDjikstraTet
from simulator_functions import SimulateECG, SimulateEP
from adapter_theta_params import AdapterThetaParams,RoundTheta
from discrepancy_functions import DiscrepancyECG
from evaluation_functions import DiscrepancyEvaluator
from cellular_models import StepFunctionUpstrokeEP, CellularModelBiomarkerDictionary
from electrophysiology_functions import ElectrophysiologyUpstrokeStepFunction, ElectrophysiologyAPDmap
from path_config import get_path_mapping
from io_functions import write_geometry_to_ensight, export_ensight_timeseries_case, read_dictionary,save_dictionary, write_geometry_to_ensight_with_fields, write_purkinje_vtk, \
        write_root_node_csv
from evaluation_functions import ParameterSimulator
from postprocess_functions import scatter_visualise_field
from utils import map_indexes, get_vc_aprt_name, get_vc_rt_name, get_vc_rvlv_name, get_vc_tm_name, \
    get_vc_ab_cut_name, get_fibre_speed_name, get_sheet_speed_name, \
    get_normal_speed_name, get_endo_dense_speed_name, get_endo_sparse_speed_name, \
    get_purkinje_speed_name, get_xyz_name_list
from datetime import datetime

if __name__ == '__main__':
    save_results = False
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
    anatomy_subject_name = 'DTI032'# 'UKB_1002379'#'UKB_1000268' #'UKB_1000532'#'UKB_1000268' #'DTI004'  #'rodero_13' # 'rodero_13'  # 'DTI004'  # 'UKB_1000532' #'UKB_1000268'
    ecg_subject_name = 'DTI032'# 'UKB_1002379'#'UKB_1000268' #'UKB_1000532'#'UKB_1000268' #'DTI004'  # 'DTI004'  # 'UKB_1000532' # 'UKB_1000268'  # Allows using a different ECG for the personalisation than for the anatomy
        
    resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    # cellular_data_dir = data_dir + 'cellular_data/'
    clinical_data_filename = 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    clinical_data_filename_path = data_dir + clinical_data_filename
    # if ecg_subject_name == 'DTI004':
    #     clinical_qrs_offset = 100
    # else:
    #     clinical_qrs_offset = 0 # 100  # ms TODO This could be calculated automatically and potentially, the clinical ECG could be trimmed to start with the QRS at time zero
    # print('clinical_qrs_offset ', clinical_qrs_offset)
    geometric_data_dir = data_dir + 'geometric_data/'
    # Output Paths:
    experiment_type = 'simulation'
    if anatomy_subject_name == 'DTI024':
        ep_model = 'GKs5_GKr0.5_tjca60_CL_909'
    elif anatomy_subject_name == 'DTI032':
        ep_model = 'GKs5_GKr0.5_tjca60_CL_810'
    elif anatomy_subject_name == 'DTI004':
        ep_model = 'GKs5_GKr0.5_tjca60_CL_1250'
    else:
        ep_model = 'GKs5_GKr0.5_tjca60_CL_'
    # ep_model = 'GKs5_GKr0.6_tjca60' #'MitchellSchaefferEP' #'no_rescale' #'GKs5_GKr0.6_tjca60' #TODO revert
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    results_dir_root = path_dict["results_path"]
    # Build results folder structure
    results_dir_part = results_dir_root + experiment_type + '_data/'
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
    results_dir_part = results_dir_part + anatomy_subject_name + '/'
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
    results_dir_part = results_dir_part + 'qt_' + gradient_ion_channel_str + '_' + ep_model + '/'
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
    # Use date to name the result folder to preserve some history of results
    current_month_text = datetime.now().strftime('%h')  # Feb
    current_year_full = datetime.now().strftime('%Y')  # 2018
    # results_dir = results_dir_part + 'qt_' + gradient_ion_channel_str + '_' + ep_model + '/smoothing_fibre_128_64_05/' #+ '/smoothing_fibre_256_64_05/'
    results_dir = results_dir_part + current_month_text + '_' + current_year_full + '/'
    results_dir_part = None  # Clear Arguments to prevent Argument recycling
    # Create results directory
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    result_tag = experiment_type
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_hyperparameter.txt'
    theta_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_theta_population.csv'
    parameter_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_parameter_population.csv'
    # Enable saving partial inference results and restarting from where it was left
    unfinished_process_dir = results_dir + 'unfinished_process/'
    if not os.path.exists(unfinished_process_dir):
        os.mkdir(unfinished_process_dir)
    unfinished_theta_result_file_name = unfinished_process_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_theta_population.csv'
    unfinished_parameter_result_file_name = unfinished_process_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_parameter_population.csv'
    # Inference history
    inference_history_dir = results_dir + 'inference_history/'
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
        cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model + '/'
    cellular_data_dir_complete = data_dir + cellular_data_relative_path
    # Directory to save the configuration of the inference before it runs to allow manual inspection:
    visualisation_dir = results_dir + 'checkpoint/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    # save ecg
    figure_result_file_name = visualisation_dir + anatomy_subject_name + '_' + resolution + '_ecg.png'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Save hyperparameters for reproducibility
    hyperparameter_dict['cellular_data_relative_path'] = cellular_data_relative_path
    hyperparameter_dict['cellular_stim_amp'] = cellular_stim_amp
    hyperparameter_dict['clinical_data_filename'] = clinical_data_filename  # Hyperparameter
    # hyperparameter_dict['clinical_qrs_offset'] = clinical_qrs_offset  # Hyperparameter
    hyperparameter_dict['cellular_model_convergence'] = cellular_model_convergence
    hyperparameter_dict[
        'experiment_type'] = experiment_type  # This will tell in the future if this was sa or personalisation
    hyperparameter_dict['ep_model'] = ep_model
    hyperparameter_dict['gradient_ion_channel_list'] = gradient_ion_channel_list
    hyperparameter_dict['result_tag'] = result_tag
    hyperparameter_dict['stimulation_protocol'] = stimulation_protocol
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
    intermediate_dir = None
    results_dir = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
    # Arguments for cellular model:
    print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
    if anatomy_subject_name == 'DTI024':
        cellular_model_name = 'torord_calibrated_pom_CL909'
    elif anatomy_subject_name == 'DTI032':
        cellular_model_name = 'torord_calibrated_pom_CL810'
    elif anatomy_subject_name == 'DTI004':
        cellular_model_name = 'torord_calibrated_pom_CL1250'
    else:
        cellular_model_name = 'torord_calibrated_pom_CL'
    endo_celltype_name = 'endo'
    # epi_celltype_name = 'epi'
    list_celltype_name = [endo_celltype_name]
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
    # Save hyperparameters for reproducibility
    hyperparameter_dict['biomarker_apd90_name'] = biomarker_apd90_name
    hyperparameter_dict['biomarker_celltype_name'] = biomarker_celltype_name
    hyperparameter_dict['biomarker_upstroke_name'] = biomarker_upstroke_name
    hyperparameter_dict['cellular_model_name'] = cellular_model_name
    hyperparameter_dict['endo_celltype_name'] = endo_celltype_name
    # hyperparameter_dict['epi_celltype_name'] = epi_celltype_name
    hyperparameter_dict['list_celltype_name'] = list_celltype_name
    # Clear Arguments to prevent Argument recycling
    biomarker_apd90_name = None
    biomarker_celltype_name = None
    biomarker_upstroke_name = None
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_model_name = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 3: Generate a cardiac geometry that can run the Eikonal.
    # Argument setup: (in Alphabetical order)
    print('Step 3: Generate a cardiac geometry that can run the Eikonal.')
    # vc_ab_name = get_vc_ab_name()
    vc_ab_cut_name = get_vc_ab_cut_name()
    vc_aprt_name = get_vc_aprt_name()
    vc_rt_name = get_vc_rt_name()
    vc_rvlv_name = get_vc_rvlv_name()
    vc_tm_name = get_vc_tm_name()
    # vc_tv_name = get_vc_tv_name()
    # vc_rvlv_binary_name = get_vc_rvlv_binary_name()
    vc_name_list = [vc_ab_cut_name, vc_aprt_name, vc_rt_name, vc_rvlv_name, vc_tm_name]#, vc_rvlv_binary_name]
    # Pre-assign celltype spatial correspondence.
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 1.]}}#, epi_celltype_name: {vc_tm_name: [0., 0.3]}}
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['celltype_vc_info'] = celltype_vc_info
    hyperparameter_dict['vc_name_list'] = vc_name_list
    # hyperparameter_dict['vc_ab_name'] = vc_ab_name
    hyperparameter_dict['vc_ab_cut_name'] = vc_ab_cut_name
    hyperparameter_dict['vc_aprt_name'] = vc_aprt_name
    hyperparameter_dict['vc_rt_name'] = vc_rt_name
    hyperparameter_dict['vc_rvlv_name'] = vc_rvlv_name
    hyperparameter_dict['vc_tm_name'] = vc_tm_name
    # hyperparameter_dict['vc_tv_name'] = vc_tv_name
    # hyperparameter_dict['vc_rvlv_binary_name'] = vc_rvlv_binary_name
    # Clear Arguments to prevent Argument recycling
    geometric_data_dir = None
    # resolution = None
    # anatomy_subject_name = None
    # vc_name_list = None
    ####################################################################################################################
    # Step 4: Create conduction system for the propagation model to be initialised.
    print('Step 4: Create rule-based Purkinje network using ventricular coordinates.')
    # Arguments for Conduction system:
    approx_djikstra_purkinje_max_path_len = 200
    lv_inter_root_node_distance = 1.5  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    rv_inter_root_node_distance = 1.5  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    # Create conduction system
    conduction_system = PurkinjeSystemVC(
        approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, geometry=geometry,
        lv_inter_root_node_distance=lv_inter_root_node_distance,
        rv_inter_root_node_distance=rv_inter_root_node_distance,
        verbose=verbose)
    # Assign conduction_system to its geometry
    geometry.set_conduction_system(conduction_system)
    conduction_system = None  # Clear Arguments to prevent Argument recycling
    # Save hyperparameters for reproducibility
    hyperparameter_dict['approx_djikstra_purkinje_max_path_len'] = approx_djikstra_purkinje_max_path_len
    hyperparameter_dict['lv_inter_root_node_distance'] = lv_inter_root_node_distance
    hyperparameter_dict['rv_inter_root_node_distance'] = rv_inter_root_node_distance
    # Clear Arguments to prevent Argument recycling
    approx_djikstra_purkinje_max_path_len = None
    lv_inter_root_node_distance = None
    rv_inter_root_node_distance = None
    vc_rt_name = None
    # Save candidate Purkinje system as .vtk file
    lv_pk_edge, rv_pk_edge = geometry.get_lv_rv_candidate_purkinje_edge()
    node_xyz = geometry.get_node_xyz()
    node_vc_list = [geometry.get_node_vc_field(vc_name=vc_name) for vc_name in vc_name_list]
    # LV
    write_purkinje_vtk(edge_list=lv_pk_edge, filename=anatomy_subject_name + '_LV_Purkinje', node_xyz=node_xyz,
                       verbose=verbose, visualisation_dir=visualisation_dir)
    lv_candidate_root_node_index, rv_candidate_root_node_index = geometry.get_lv_rv_candidate_root_node_index()
    write_root_node_csv(filename=anatomy_subject_name + '_LV_root_nodes.csv', node_vc_list=node_vc_list,
                        node_xyz=node_xyz,
                        root_node_index_list=lv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                        visualisation_dir=visualisation_dir, xyz_name_list=get_xyz_name_list())
    # RV
    write_purkinje_vtk(edge_list=rv_pk_edge, filename=anatomy_subject_name + '_RV_Purkinje', node_xyz=node_xyz,
                       verbose=verbose, visualisation_dir=visualisation_dir)
    write_root_node_csv(filename=anatomy_subject_name + '_RV_root_nodes.csv', node_vc_list=node_vc_list,
                        node_xyz=node_xyz,
                        root_node_index_list=rv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                        visualisation_dir=visualisation_dir, xyz_name_list=get_xyz_name_list())
    # Clear Arguments to prevent Argument recycling
    lv_candidate_root_node_index = None
    lv_pk_edge = None
    node_xyz = None
    rv_candidate_root_node_index = None
    rv_pk_edge = None
    ####################################################################################################################
    # Step 5: Prepare smoothing configuration to resemble diffusion effects
    print('Step 5: Prepare smoothing configuration to resemble diffusion effects.')
    # Define the speeds used during the fibre-based smoothing
    warn('Inference from QT can, but does NOT, update the speeds in the smoothing function!\nAlso, it requires some initial fixed values!')
    fibre_speed = 0.065     # Taggart et al. (2000) https://doi.org/10.1006/jmcc.2000.1105
    sheet_speed = 0.051     # Taggart et al. (2000) https://doi.org/10.1006/jmcc.2000.1105
    normal_speed = 0.048    # Taggart et al. (2000) https://doi.org/10.1006/jmcc.2000.1105
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    warn('Precompuing the smoothing, change this please!')  # TODO refactor
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
    # Step 6: Create Eikonal instance. Eikonal will require a conduction and an Eikonal-friendly mesh on creation.
    print('Step 6: Create propagation model instance.')
    # Arguments for propagation model:
    endo_dense_speed_name = get_endo_dense_speed_name()
    endo_sparse_speed_name = get_endo_sparse_speed_name()
    purkinje_speed_name = get_purkinje_speed_name()
    speed_parameter_name_list_in_order = [fibre_speed_name, sheet_speed_name, normal_speed_name, endo_dense_speed_name,
                                          endo_sparse_speed_name, purkinje_speed_name]
    nb_speed_parameters = len(speed_parameter_name_list_in_order)
    nb_candidate_root_nodes = geometry.get_nb_candidate_root_node()
    candidate_root_node_names = ['r' + str(root_i) for root_i in range(nb_candidate_root_nodes)]
    propagation_parameter_name_list_in_order = speed_parameter_name_list_in_order + candidate_root_node_names
    propagation_model = EikonalDjikstraTet(
        endo_dense_speed_name=endo_dense_speed_name, endo_sparse_speed_name=endo_sparse_speed_name,
        fibre_speed_name=fibre_speed_name, geometry=geometry, module_name=propagation_module_name,
        nb_speed_parameters=nb_speed_parameters, normal_speed_name=normal_speed_name,
        parameter_name_list_in_order=propagation_parameter_name_list_in_order, purkinje_speed_name=purkinje_speed_name,
        sheet_speed_name=sheet_speed_name, verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['endo_dense_speed_name'] = endo_dense_speed_name
    hyperparameter_dict['endo_sparse_speed_name'] = endo_sparse_speed_name
    hyperparameter_dict['purkinje_speed_name'] = purkinje_speed_name
    hyperparameter_dict['nb_speed_parameters'] = nb_speed_parameters
    hyperparameter_dict['propagation_parameter_name_list_in_order'] = propagation_parameter_name_list_in_order
    # Clear Arguments to prevent Argument recycling
    # nb_speed_parameters = None
    ####################################################################################################################
    # Step 7: Create Whole organ Electrophysiology model.
    print('Step 7: Create Whole organ Electrophysiology model.')
    # Arguments for Electrophysiology model:
    apd_max_name = 'apd_max'
    apd_min_name = 'apd_min'
    g_vc_ab_name = vc_ab_cut_name
    g_vc_aprt_name = vc_aprt_name
    g_vc_rvlv_name = vc_rvlv_name
    g_vc_tm_name = vc_tm_name
    electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name,
                                                      g_vc_rvlv_name, g_vc_tm_name]
    # Spatial and temporal smoothing parameters:
    smoothing_dt = 20
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    # smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    # smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 5% and the present as 95%
    start_smoothing_time_index = 100  # (ms) assumming 1000Hz
    end_smoothing_time_index = 450  # 400  # (ms) assumming 1000Hz
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
    # Save hyperparameters for reproducibility
    hyperparameter_dict['apd_max_name'] = apd_max_name
    hyperparameter_dict['apd_min_name'] = apd_min_name
    hyperparameter_dict['g_vc_ab_name'] = g_vc_ab_name
    hyperparameter_dict['g_vc_aprt_name'] = g_vc_aprt_name
    hyperparameter_dict['g_vc_rvlv_name'] = g_vc_rvlv_name
    hyperparameter_dict['g_vc_tm_name'] = g_vc_tm_name
    hyperparameter_dict['electrophysiology_parameter_name_list_in_order'] = electrophysiology_parameter_name_list_in_order
    hyperparameter_dict['end_smoothing_time_index'] = end_smoothing_time_index
    hyperparameter_dict['smoothing_dt'] = smoothing_dt
    hyperparameter_dict['start_smoothing_time_index'] = start_smoothing_time_index
    # Clear Arguments to prevent Argument recycling
    cellular_model = None
    propagation_model = None
    smoothing_count = None
    smoothing_ghost_distance_to_self = None
    smoothing_past_present_window = None
    vc_ab_name = None
    vc_aprt_name = None
    vc_rvlv_name = None
    vc_tm_name = None
    # ####################################################################################################################
     # Step 8: Create ECG calculation method.
    print('Step 8: Create ECG calculation method.')
    # Arguments for ECG calculation:
    filtering = True
    max_len_qrs = 200  # can use 200 to save memory space # This hyper-paramter is used when paralelising the ecg computation, because it needs a structure to synchronise the results from the multiple threads.
    max_len_st = 250    # can use 200 to save memory space
    max_len_ecg = max_len_qrs + max_len_st
    normalise = True
    zero_align = True
    frequency = 1000  # Hz
    if frequency != 1000:
        warn('The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
    low_freq_cut = 0.001  # 0.5
    high_freq_cut = 100  # 150
    I_name = 'I'
    II_name = 'II'
    v3_name = 'V3'
    v5_name = 'V5'
    lead_names = [I_name, II_name, 'V1', 'V2', v3_name, 'V4', v5_name, 'V6']
    nb_leads = len(lead_names)
    # Read clinical data
    # TODO This code may not work well for an ECG with only one lead!!
    clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
    print('clinical_ecg_raw ', clinical_ecg_raw.shape)
    # clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
    # Create ECG model
    ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                   frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                   low_freq_cut=low_freq_cut, max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs,
                                   nb_leads=nb_leads, nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                   reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                   tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
    clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    # # TODO revert to above 2024/01/23!
    # if max_len_ecg > clinical_ecg_raw.shape[1]:    # TODO revert to above 2024/01/23!
    #     clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)[:, :max_len_ecg]    # TODO revert to above 2024/01/23!
    # else:    # TODO revert to above 2024/01/23!
    #     clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)    # TODO revert to above 2024/01/23!
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
    geometry = None
    lead_names = None
    max_len_ecg = None
    max_len_qrs = None
    max_len_st = None
    nb_leads = None
    normalise = None
    untrimmed_clinical_ecg_raw = None
    zero_align = None
    ####################################################################################################################
    # Step 9: Define instance of the simulation method.
    print('Step 9: Define instance of the simulation method.')
    simulator = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    electrophysiology_model = None
    ecg_model = None
    ####################################################################################################################
    # Step 10: Define Adapter to translate between theta and parameters.
    print('Step 10: Define Adapter to translate between theta and parameters.')
    # Arguments for Adapter:
    parameter_name_list_in_order = propagation_parameter_name_list_in_order + electrophysiology_parameter_name_list_in_order
    continuous_theta_name_list_in_order = [sheet_speed_name, endo_dense_speed_name, endo_sparse_speed_name,
                                           apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name, g_vc_rvlv_name,
                                           g_vc_tm_name]
    theta_name_list_in_order = continuous_theta_name_list_in_order + candidate_root_node_names
    parameter_fixed_value_dict = {}
    parameter_fixed_value_dict[fibre_speed_name] = 0.065  # Taggart et al. (2000)
    parameter_fixed_value_dict[normal_speed_name] = 0.048  # Taggart et al. (2000)
    parameter_fixed_value_dict[purkinje_speed_name] = 0.3  # (cm/ms), consistent with literature

    physiological_rules_larger_than_dict = {}
    physiological_rules_larger_than_dict[endo_dense_speed_name] = [endo_sparse_speed_name]  # Define custom rules to constrain which parameters must be larger than others.
    physiological_rules_larger_than_dict[apd_max_name] = [apd_min_name]  # Define custom rules to constrain which parameters must be larger than others.
    # [sheet_speed_name, endo_dense_speed_name, endo_sparse_speed_name, g_vc_tm_name]
    endo_dense_speed_resolution = 0.001
    endo_sparse_speed_resolution = 0.001
    transmural_speed_resolution = 0.001
    apd_max_resolution = 2.
    apd_min_resolution = 2.
    g_vc_ab_resolution = 0.1
    g_vc_aprt_resolution = 0.1
    g_vc_rvlv_resolution = 0.1
    g_vc_tm_resolution = 0.1
    nb_discrete_theta = len(candidate_root_node_names)
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
    for root_i in range(nb_discrete_theta):
        theta_adjust_function_list_in_order.append(None)
    if len(theta_adjust_function_list_in_order) != len(theta_name_list_in_order):
        print('theta_name_list_in_order ', len(theta_name_list_in_order))
        print('theta_adjust_function_list_in_order ', len(theta_adjust_function_list_in_order))
        raise Exception('Different number of adjusting functions and theta for the inference')
    # Distribute parameters into modules
    destination_module_name_list_in_order = [propagation_module_name, electrophysiology_module_name]
    parameter_destination_module_dict = {}
    parameter_destination_module_dict[propagation_module_name] = propagation_parameter_name_list_in_order
    parameter_destination_module_dict[electrophysiology_module_name] = electrophysiology_parameter_name_list_in_order
    print('Caution: these rules have only been enabled for the inferred parameters!')   # TODO: modify this to also enable rules for fixed parameters (e.g., fibre_speed >= transmural_speed)
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
    hyperparameter_dict['endo_dense_speed_resolution'] = endo_dense_speed_resolution
    hyperparameter_dict['endo_sparse_speed_resolution'] = endo_sparse_speed_resolution
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
    hyperparameter_dict['transmural_speed_resolution'] = transmural_speed_resolution
    # Clear Arguments to prevent Argument recycling
    apd_max_name = None
    apd_max_resolution = None
    apd_min_name = None
    apd_min_resolution = None
    candidate_root_node_names = None
    # continuous_theta_name_list_in_order = None
    destination_module_name_list_in_order = None
    endo_dense_speed_name = None
    endo_dense_speed_resolution = None
    endo_sparse_speed_name = None
    endo_sparse_speed_resolution = None
    g_vc_ab_name = None
    g_vc_aprt_name = None
    g_vc_rvlv_name = None
    g_vc_tm_name = None
    g_vc_ab_resolution = None
    g_vc_aprt_resolution = None
    g_vc_rvlv_resolution = None
    g_vc_tm_resolution = None
    nb_discrete_theta = None
    normal_speed_name = None
    parameter_fixed_value_dict = None
    speed_parameter_name_list_in_order = None
    theta_adjust_function_list_in_order = None
    sheet_speed_name = None
    fibre_speed_name = None
    transmural_speed_resolution = None
    ####################################################################################################################
    # Step 11: Define the discrepancy metric.
    print('Step 11: Define the discrepancy metric.')
    # Arguments for DTW discrepancy metric:
    # max_slope = 1.5
    # w_max = 10.
    error_method_name = 'rmse_pcc_cubic' #"rmse_pcc" #
    # Create discrepancy metric instance.
    discrepancy_metric = DiscrepancyECG(error_method_name=error_method_name)    # TODO: add weighting control between PCC and RMSE
    # Save hyperparameters for reproducibility
    hyperparameter_dict['error_method_name'] = error_method_name  # Hyperparameter
    # Clear Arguments to prevent Argument recycling
    error_method_name = None
    ####################################################################################################################
    # Step 12: Create evaluator_ecg.
    print('Step 12: Create evaluator_ecg.')
    evaluator = DiscrepancyEvaluator(adapter=adapter, discrepancy_metric=discrepancy_metric, simulator=simulator,
                                     target_data=clinical_ecg, verbose=verbose)
    # Save hyperparameters for reproducibility
    # Clear Arguments to prevent Argument recycling.
    adapter = None
    discrepancy_metric = None
    simulator = None
    clinical_ecg = None
    ####################################################################################################################
    # Step 13: Define simulation parameter values
    # Propagation parameters
    print('This is the sequence of propagation parameters in order expected by the method: ', propagation_parameter_name_list_in_order)
    propagation_parameter_example = np.zeros((len(propagation_parameter_name_list_in_order)), dtype=float)
    fibre_speed_value = 0.065  # Taggart et al. (2000)
    transmural_speed_value = 0.052  # Taggart et al. (2000)
    normal_speed_value = 0.048  # Taggart et al. (2000)
    endo_dense_speed_value = 0.180
    endo_sparse_speed_value = 0.120
    purkinje_speed_name = 0.300
    speed_parameter_value_list_in_order = np.array([fibre_speed_value, transmural_speed_value, normal_speed_value,
                                          endo_dense_speed_value,
                                          endo_sparse_speed_value, purkinje_speed_name])
    propagation_parameter_example[:nb_speed_parameters] = speed_parameter_value_list_in_order
    root_node_parameter_value_list_in_order = np.ones((nb_candidate_root_nodes))    # Select here which root nodes you would like to use
    propagation_parameter_example[nb_speed_parameters:] = root_node_parameter_value_list_in_order
    # Electrophysiology parameters
    print('This is the sequence of electrophysiology parameters in order expected by the method: ', electrophysiology_parameter_name_list_in_order)
    nb_ep_parameters = len(electrophysiology_parameter_name_list_in_order)
    apd_max_value = 278
    apd_min_value = 190
    g_vc_ab_value = 0.
    g_vc_aprt_value = 0.
    g_vc_rvlv_value = 0.
    g_vc_tm_value = 0.
    electrophysiology_parameter_example = np.array([apd_max_value, apd_min_value, g_vc_ab_value, g_vc_aprt_value,
                                                    g_vc_rvlv_value, g_vc_tm_value])
    # Agregate parameters into particle
    parameter_particle_example = np.concatenate((propagation_parameter_example, electrophysiology_parameter_example))
    # Clear Arguments to prevent Argument recycling
    nb_candidate_root_nodes = None
    nb_speed_parameters = None
    propagation_parameter_example = None
    electrophysiology_parameter_example = None
    # ####################################################################################################################
    # Step 14: Simulate ECG  using example parameters.
    print('parameter_particle_example ', parameter_particle_example.shape)
    print('parameter_particle_example ', parameter_particle_example)
    discrepancy_particle = evaluator.evaluate_parameter(parameter_particle=parameter_particle_example)
    fig = evaluator.visualise_parameter_population(discrepancy_population=np.array([discrepancy_particle, discrepancy_particle]),
                                                       parameter_population=np.concatenate(
        (parameter_particle_example[np.newaxis, :], parameter_particle_example[np.newaxis, :]), axis=0))
    fig.savefig(figure_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    evaluator_ecg = None
    parameter_name_list_in_order = None
    parameter_population = None
    population_theta = None
    ####################################################################################################################
    # # Step 13: Simulate LAT and VM, and Interpolate simulation results to have the same indexing than the input data files.
    # simulated_lat, simulated_vm = evaluator_ep.simulate_parameter_particle(parameter_particle=parameter_particle_example)
    # # Interpolate nodefield
    # unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=raw_geometry.get_node_xyz(),
    #                                              reference_points_xyz=geometry.get_node_xyz())
    # simulated_lat = simulated_lat[unprocessed_node_mapping_index]
    # simulated_vm = simulated_vm[unprocessed_node_mapping_index, :]
    # np.savetxt(lat_result_file_name, simulated_lat, delimiter=',')
    # print('Saved lat: ', lat_result_file_name)
    # np.savetxt(vm_result_file_name, simulated_vm, delimiter=',')
    # print('Saved vm: ', vm_result_file_name)
    # write_geometry_to_ensight(geometry=raw_geometry, subject_name=anatomy_subject_name, resolution=resolution,
    #                           visualisation_dir=visualisation_dir, verbose=verbose)
    # print('Saved ensight geometry: ', visualisation_dir)
    # export_ensight_timeseries_case(dir=visualisation_dir, casename=anatomy_subject_name + '_simulation', dataname_list=['INTRA_RE'],
    #                                vm_list=[simulated_vm], dt=1. / frequency, nodesxyz=raw_geometry.get_node_xyz(),
    #                                tetrahedrons=raw_geometry.get_tetra())
    # print('Saved ensight vm: ', visualisation_dir)
    # # Clear Arguments to prevent Argument recycling.
    # anatomy_subject_name = None
    # best_theta = None
    # best_parameter = None
    # evaluator_ep = None
    # figure_result_file_name = None
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

# EOF

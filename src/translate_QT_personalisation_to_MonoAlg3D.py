"""Interpolate GKs to cube centres for MonoAlg3D simulations"""
import os
import sys
from warnings import warn

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime





# def convert_from_monoalg3D_to_cm_and_translate(nodesXYZ, meshname):
#     warn('Hi Julia and Jenny, be super careful with this, has Ruben already corroborated these values?')
#     scale = [1e+4, 1e+4, 1e+4]
#     translate = [0, 0, 0]
#
#     if (meshname == "DTI003"):
#         translate = [40830.0, 117250.0, 111850.0]
#     elif (meshname == "DTI004"):
#         translate = [25720.0, 142000.0, 71290.0]
#     elif (meshname == "DTI024"):
#         translate = [12513.7, 59959.1, 46972.9]
#     elif (meshname == "DTI032"):
#         translate = [44728.1, 78681.1, 45941.4]
#     elif (meshname == "DTI124"):
#         translate = [-12550.0, -106550.0, -43030.0]
#     elif (meshname == "DTI4586"):
#         translate = [-38930.0, -93350.0, 23490.0]
#     else:
#         print("[-] ERROR! Invalid meshname '%s'!" % (meshname))
#         sys.exit(1)
#
#     num_nodes = len(nodesXYZ)
#     for i in range(num_nodes):
#         for j in range(3):
#             nodesXYZ[i][j] = (nodesXYZ[i][j] - translate[j])/ scale[j]   # Note that this is substracting the translations and dividing
#     return nodesXYZ


if __name__ == '__main__':
    if len(sys.argv) < 2:
        anatomy_subject_name = 'DTI004'  # 'rodero_13' # 'rodero_13'  # 'DTI004'  # 'UKB_1000532' #'UKB_1000268'
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
    from conduction_system import DjikstraConductionSystemVC, EmptyConductionSystem, PurkinjeSystemVC
    from ecg_functions import PseudoEcgTetFromVM
    from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet, RawEmptyCardiacGeoPointCloud
    from propagation_models import EikonalDjikstraTet
    from simulator_functions import SimulateECG, SimulateEP
    from adapter_theta_params import AdapterThetaParams, RoundTheta
    from discrepancy_functions import DiscrepancyECG, BiomarkerFromOnlyECG
    from evaluation_functions import DiscrepancyEvaluator, ParameterSimulator
    from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
    from electrophysiology_functions import ElectrophysiologyAPDmap
    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields, read_dictionary, save_ecg_to_csv, \
    export_ensight_timeseries_case, save_pandas, save_csv_file, read_ecg_from_csv, read_csv_file, write_purkinje_vtk, \
    write_root_node_csv
    from utils import map_indexes, remap_pandas_from_row_index, get_qt_dur_name, \
    get_t_pe_name, get_t_peak_name, get_tpeak_dispersion_name, get_qtpeak_dur_name, \
    get_t_polarity_name, get_root_node_meta_index_population_from_pandas, translate_from_pandas_to_array, \
    get_purkinje_speed_name, get_lat_biomarker_name, get_repol_biomarker_name
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
    ep_model = 'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    # Use date to name the result folder to preserve some history of results
    current_month_text = datetime.now().strftime('%h')  # Feb
    current_year_full = datetime.now().strftime('%Y')  # 2018
    results_dir = results_dir_root + experiment_type + '_data/' + anatomy_subject_name + '/qt_' \
                  + gradient_ion_channel_str + '_' + ep_model + '/' + current_month_text + '_' + current_year_full + '/'
    # Read hyperparamter dictionary
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + inference_resolution + '_hyperparameter.txt'
    hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
    result_tag = hyperparameter_dict['result_tag']
    parameter_result_file_name = results_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_parameter_population.csv'
    # Output Paths:
    # Translation to Monodomain
    for_monodomain_dir = results_dir + 'for_translation_to_monodomain/'
    if not os.path.exists(for_monodomain_dir):
        os.mkdir(for_monodomain_dir)
    for_monodomain_biomarker_result_file_name_start = for_monodomain_dir + anatomy_subject_name + '_' + monodomain_resolution + '_nodefield_' + result_tag + '-biomarker_'
    for_monodomain_Purkinje_result_file_name_start = for_monodomain_dir + anatomy_subject_name + '_' + monodomain_resolution + '_nodefield_' + result_tag + '-Purkinje_'
    for_monodomain_root_node_result_file_name_start = for_monodomain_dir + anatomy_subject_name + '_' + monodomain_resolution + '_nodefield_' + result_tag + '-Purkinje_'
    for_monodomain_biomarker_result_file_name_end = '.csv'
    for_monodomain_parameter_population_file_name = for_monodomain_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_selected_parameter_population.csv'
    for_monodomain_figure_result_file_name = for_monodomain_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_population.png'
    # Precomputed subfolder
    inference_precomputed_dir = for_monodomain_dir + 'precomputed/'
    if not os.path.exists(inference_precomputed_dir):
        os.mkdir(inference_precomputed_dir)
    inference_ecg_uncertainty_population_filename = inference_precomputed_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_selected_pseudo_ecg_population.csv'
    inference_ecg_inferred_population_filename = inference_precomputed_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_inferred_pseudo_ecg_population.csv'
    inference_repol_uncertainty_population_filename = inference_precomputed_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_selected_repol_population.csv'
    preprocessed_clinical_ecg_file_name = inference_precomputed_dir + anatomy_subject_name + '_' + inference_resolution + '_' + result_tag + '_ecg_clinical.csv'
    inference_precomputed_dir = None  # Clear Arguments to prevent Argument recycling
    # # Best discrepancy
    # translation_dir = results_dir + 'best_discrepancy/'
    # if not os.path.exists(translation_dir):
    #     os.mkdir(translation_dir)
    # lat_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-lat.csv'
    # vm_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-vm.csv'
    # best_parameter_result_file_name = translation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '-best-parameter.csv'
    # biomarker_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-biomarker.csv'
    # Precomputed
    # precomputed_dir = results_dir + 'precomputed/'
    # if not os.path.exists(precomputed_dir):
    #     os.mkdir(precomputed_dir)
    # ecg_population_file_name = precomputed_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_ecg_population.csv'
    # max_lat_population_file_name = precomputed_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_max_lat_population.csv'
    # preprocessed_clinical_ecg_file_name = precomputed_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_ecg_clinical.csv'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Read hyperparameters
    clinical_data_filename = hyperparameter_dict['clinical_data_filename']
    clinical_data_filename_path = data_dir + clinical_data_filename
    # clinical_qrs_offset = hyperparameter_dict['clinical_qrs_offset']
    # qrs_lat_prescribed_filename = hyperparameter_dict['qrs_lat_prescribed_filename']
    # qrs_lat_prescribed_filename_path = results_dir_root + qrs_lat_prescribed_filename
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
    data_dir = None
    ecg_subject_name = None
    # qrs_lat_prescribed_filename = None
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
    raw_geometry_point_cloud.unprocessed_node_xyz = convert_from_monoalg3D_to_cm_and_translate(
        monoalg3D_xyz=raw_geometry_point_cloud.get_node_xyz(), inference_xyz=geometry.get_node_xyz())
    print('min max ', np.amin(raw_geometry_point_cloud.unprocessed_node_xyz),
          np.amax(raw_geometry_point_cloud.unprocessed_node_xyz))
    print('geometry min max ', np.amin(geometry.get_node_xyz()),
          np.amax(geometry.get_node_xyz()))
    # TODO DELETE THE ABOVE CODE

    # Clear Arguments to prevent Argument recycling
    geometric_data_dir = None
    list_celltype_name = None
    inference_resolution = None
    # vc_name_list = None
    ####################################################################################################################
    # Step 4: Create conduction system for the propagation model to be initialised.
    print('Step 4: Create rule-based Purkinje network using ventricular coordinates.')
    # Arguments for Conduction system:
    approx_djikstra_purkinje_max_path_len = hyperparameter_dict['approx_djikstra_purkinje_max_path_len']
    lv_inter_root_node_distance = hyperparameter_dict['lv_inter_root_node_distance']
    rv_inter_root_node_distance = hyperparameter_dict['rv_inter_root_node_distance']
    # Create conduction system
    conduction_system = PurkinjeSystemVC(
        approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, geometry=geometry,
        lv_inter_root_node_distance=lv_inter_root_node_distance, rv_inter_root_node_distance=rv_inter_root_node_distance,
        verbose=verbose)
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
    sheet_speed_name = hyperparameter_dict['sheet_speed_name']
    normal_speed_name = hyperparameter_dict['normal_speed_name']
    fibre_speed = hyperparameter_dict[fibre_speed_name]
    sheet_speed = hyperparameter_dict[sheet_speed_name]
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
    sheet_speed_name = hyperparameter_dict['sheet_speed_name']
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
        sheet_speed_name=sheet_speed_name, verbose=verbose)
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
                                                      sheet_speed_name=sheet_speed_name,
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
    save_ecg_to_csv(data=clinical_ecg[np.newaxis, :, :], filename=preprocessed_clinical_ecg_file_name)
    print('Saved preprocessed clinical ECG at ', preprocessed_clinical_ecg_file_name)
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
    # Step 9: Define instance of the simulation method.
    print('Step 9: Define instance of the simulation method.')
    simulator_ecg = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    simulator_ep = SimulateEP(electrophysiology_model=electrophysiology_model, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
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
    sheet_speed_resolution = hyperparameter_dict['sheet_speed_resolution']
    apd_max_resolution = hyperparameter_dict['apd_max_resolution']
    apd_min_resolution = hyperparameter_dict['apd_min_resolution']
    g_vc_ab_resolution = hyperparameter_dict['g_vc_ab_resolution']
    g_vc_aprt_resolution = hyperparameter_dict['g_vc_aprt_resolution']
    g_vc_rvlv_resolution = hyperparameter_dict['g_vc_rvlv_resolution']
    g_vc_tm_resolution = hyperparameter_dict['g_vc_tm_resolution']
    theta_adjust_function_list_in_order = [RoundTheta(resolution=sheet_speed_resolution),
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
    sheet_speed_name = None
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
    # Clear Arguments to prevent Argument recycling
    adapter = None
    simulator_ecg = None
    simulator_ep = None
    ####################################################################################################################
    # Step 12: Read the values inferred for parameters.
    print('Step 12: Read the values inferred for parameters.')
    # TODO save candidate root nodes and their times so that the meta-indexes can be used to point at them.
    pandas_parameter_population = pd.read_csv(parameter_result_file_name, delimiter=',')
    print('pandas_parameter_population ', pandas_parameter_population)
    root_node_meta_index_population = get_root_node_meta_index_population_from_pandas(pandas_parameter_population=pandas_parameter_population)
    purkinje_speed_population = translate_from_pandas_to_array(name_list_in_order=[get_purkinje_speed_name()], pandas_data=pandas_parameter_population)
    print('root_node_meta_index_population ', root_node_meta_index_population.shape)
    parameter_population = evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
    unique_parameter_population, unique_index = np.unique(parameter_population, axis=0, return_index=True)
    unique_root_node_meta_index_population = root_node_meta_index_population[unique_index, :]
    unique_purkinje_speed_population = purkinje_speed_population[unique_index]
    # Clear Arguments to prevent Argument recycling.
    pandas_parameter_population = None
    parameter_result_file_name = None
    purkinje_speed_population = None
    root_node_meta_index_population = None
    unique_index = None
    ####################################################################################################################
    # # Step 13: Evaluate their ECG.
    print('Step 13: Evaluate their ECG.')
    # Simulate the parameter population from the inference
    unique_population_ecg = evaluator_ecg.simulate_parameter_population(parameter_population=unique_parameter_population)
    save_ecg_to_csv(data=unique_population_ecg, filename=inference_ecg_inferred_population_filename)
    print('Saved inferred ECGs at ', inference_ecg_inferred_population_filename)
    # Clear Arguments to prevent Argument recycling.
    inference_ecg_inferred_population_filename = None
    ####################################################################################################################
    # Step 14: Define the discrepancy metric and make sure that the result is the same when calling the evaluator.
    print('Step 14: Define the discrepancy metric.')
    # Arguments for discrepancy metrics:
    # Read hyperparameters
    error_method_name_inference_metric = hyperparameter_dict['error_method_name']
    # Create discrepancy metric instance using the inference metric:
    discrepancy_metric_inference = DiscrepancyECG(
        error_method_name=error_method_name_inference_metric)
    # Evaluate discrepancy:
    unique_discrepancy_population_inference = discrepancy_metric_inference.evaluate_metric_population(
        predicted_data_population=unique_population_ecg, target_data=clinical_ecg)
    # Create discrepancy evaluator to assess code correctness!!!
    evaluator_ecg_inference_metric = DiscrepancyEvaluator(
        adapter=adapter, discrepancy_metric=discrepancy_metric_inference, simulator=simulator_ecg,
        target_data=clinical_ecg, verbose=verbose)
    discrepancy_population_inference_from_evaluator = evaluator_ecg_inference_metric.evaluate_parameter_population(
        parameter_population=unique_parameter_population)
    if not (np.all(unique_discrepancy_population_inference == discrepancy_population_inference_from_evaluator)):
        warn('These should be identical: discrepancy_population_inference '
             + str(unique_discrepancy_population_inference.shape)
             + ' discrepancy_population_inference_from_evaluator '
             + str(discrepancy_population_inference_from_evaluator.shape))
    # Clear Arguments to prevent Argument recycling.
    discrepancy_metric_inference = None
    discrepancy_population_inference = None
    error_method_name_inference_metric = None
    evaluator_ecg_inference_metric = None
    population_discrepancy_inference_from_evaluator = None
    ####################################################################################################################
    # Step 15: Select best discrepancy particle and save best parameter.
    print('Step 15: Select best discrepancy particle.')
    unique_best_index = np.argmin(unique_discrepancy_population_inference)
    print('Best discrepancy ', unique_discrepancy_population_inference[unique_best_index])
    # best_parameter = unique_parameter_population[unique_best_index]
    # print('Best parameter ', best_parameter)
    # np.savetxt(best_parameter_result_file_name, best_parameter[np.newaxis, :], delimiter=',',
    #            header=','.join(parameter_name_list_in_order), comments='')
    # print('Saved best parameter: ', best_parameter_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    # best_parameter_result_file_name = None
    unique_discrepancy_population_inference = None
    ####################################################################################################################
    # Step 16: Randomly select some % of the particles in the final population and save their biomarkers.
    print('Step 16: Randomly select some % of the particles in the final population and save their biomarkers.')
    # Biomarker names
    activation_time_map_biomarker_name = get_lat_biomarker_name()  # TODO make these names globally defined in utils.py
    repolarisation_time_map_biomarker_name = get_repol_biomarker_name()  # TODO make these names globally defined in utils.py
    # Arguments for uncertainty quantification:
    uncertainty_proportion = 0.01   # 5% of the population size
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
    # Index parameter population
    uncertainty_parameter_population = unique_parameter_population[uncertainty_index, :]    # Parameter values
    # Print out indexes
    print('uncertainty_parameter_population ', uncertainty_parameter_population.shape)
    # print('unique_population_lat ', unique_population_lat.shape)
    # uncertainty_population_lat = unique_population_lat[uncertainty_index, :]  # LAT values
    # print('uncertainty_population_lat ', uncertainty_population_lat.shape)
    # uncertainty_population_vm = unique_population_vm[uncertainty_index, :, :]  # VM values
    # print('uncertainty_population_vm ', uncertainty_population_vm.shape)

    # Save parameter values for the translation to monodomain simulations
    save_csv_file(data=uncertainty_parameter_population, filename=for_monodomain_parameter_population_file_name,
                  column_name_list=parameter_name_list_in_order)
    print('uncertainty_parameter_population ', uncertainty_parameter_population.shape)
    # Simulate ECGs from uncertainty selection of parameters
    uncertainty_population_ecg = evaluator_ecg.simulate_parameter_population(
        parameter_population=uncertainty_parameter_population)
    save_ecg_to_csv(data=uncertainty_population_ecg, filename=inference_ecg_uncertainty_population_filename)
    print('Saved selected inference ECGs at ', inference_ecg_uncertainty_population_filename)
    inference_ecg_uncertainty_population_filename = None  # Clear Arguments to prevent Argument recycling.
    # Simulate LAT and VMs form uncertainty selection of parameters
    uncertainty_population_lat, uncertainty_population_vm = evaluator_ep.simulate_parameter_population(
        parameter_population=uncertainty_parameter_population)
    print('uncertainty_population_lat ', uncertainty_population_lat.shape)
    print('uncertainty_population_vm ', uncertainty_population_vm.shape)
    # population_max_lat = np.amax(uncertainty_population_lat, axis=1)  # TODO The LAT population is needed because there is no function to compute the QRS offset, so the max(LAT) is used instead!
    # Save parameter configurations for translation to monodomain (MonoAlg3D and Alya) simulations
    uncertainty_root_node_meta_index_population = unique_root_node_meta_index_population[uncertainty_index, :]
    uncertainty_purkinje_speed_population = unique_purkinje_speed_population[uncertainty_index]
    # Clear Arguments to prevent Argument recycling.
    unique_root_node_meta_index_population = None
    unique_purkinje_speed_population = None
    uncertainty_index = None
    # ####################################################################################################################
    # Step 17: Interpolate simulation results to have the same indexing that the input data files.
    print('17: Interpolate simulation results to have the same indexing that the input data files.')
    # Interpolate nodefield
    unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=raw_geometry_point_cloud.get_node_xyz(),
                                                 reference_points_xyz=geometry.get_node_xyz())
    # ####################################################################################################################
    # Step 18: Iterate for all particles chosen to represent the uncertainty of the inference.
    print('18: Iterate for all particles chosen to represent the uncertainty of the inference.')
    # Iterate for all particles chosen to represent the uncertainty of the inference
    # inference_repol_population = []
    for uncertainty_i in range(uncertainty_parameter_population.shape[0]):
        print('uncertainty_i ', uncertainty_i)
        if uncertainty_i == 0:
            iteration_str_tag = 'best'
        else:
            iteration_str_tag = str(uncertainty_i)
        # BIOMARKERS
        # Calculate the effect of uncertainty in the biomarkers and save them
        uncertainty_biomarker_result_file_name = for_monodomain_biomarker_result_file_name_start + iteration_str_tag + for_monodomain_biomarker_result_file_name_end
        uncertainty_parameter_particle = uncertainty_parameter_population[uncertainty_i, :]
        unprocessed_node_biomarker = evaluator_ep.biomarker_parameter_particle(
            parameter_particle=uncertainty_parameter_particle)
        # LAT AND REPOL MAPS
        unprocessed_node_biomarker[activation_time_map_biomarker_name] = uncertainty_population_lat[uncertainty_i, :]
        node_repol = generate_repolarisation_map(vm=uncertainty_population_vm[uncertainty_i, :, :])
        unprocessed_node_biomarker[repolarisation_time_map_biomarker_name] = node_repol
        # inference_repol_population.append(node_repol)
        # Save biomarkers to allow translation to MonoAlg3D and Alya
        print('Saving biomarkers for uncertainty_i ', uncertainty_i)
        node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker,
                                                     row_index=unprocessed_node_mapping_index)
        save_pandas(df=node_biomarker, filename=uncertainty_biomarker_result_file_name)
        print('Saved: ', uncertainty_biomarker_result_file_name)
        # PURKINJE
        uncertainty_Purkinje_result_file_name = for_monodomain_Purkinje_result_file_name_start + iteration_str_tag + for_monodomain_biomarker_result_file_name_end
        root_node_meta_index = uncertainty_root_node_meta_index_population[uncertainty_i, :]
        purkinje_speed = uncertainty_purkinje_speed_population[uncertainty_i]
        print('root_node_meta_index ', root_node_meta_index)
        # Save inferred Purkinje system as .vtk file
        # PURKINJE VTK
        lv_pk_edge, rv_pk_edge = geometry.get_lv_rv_selected_purkinje_edge(root_node_meta_index=root_node_meta_index)
        node_xyz = geometry.get_node_xyz()
        # LV
        write_purkinje_vtk(edge_list=lv_pk_edge,
                           filename=anatomy_subject_name + '_' + iteration_str_tag + '_LV_Purkinje', node_xyz=node_xyz,
                           verbose=verbose, visualisation_dir=for_monodomain_dir)
        # RV
        write_purkinje_vtk(edge_list=rv_pk_edge,
                           filename=anatomy_subject_name + '_' + iteration_str_tag + '_RV_Purkinje', node_xyz=node_xyz,
                           verbose=verbose, visualisation_dir=for_monodomain_dir)
        print('Saved Purkinje as .vtk files')
        # ROOT NODES AND THEIR PROPERTIES
        lv_root_node_meta_bool_index, rv_root_node_meta_bool_index = geometry.get_lv_rv_selected_root_node_meta_index(
            root_node_meta_index=root_node_meta_index)
        lv_selected_root_node_index, rv_selected_root_node_index = geometry.get_lv_rv_selected_root_node_index(
            root_node_meta_index=root_node_meta_index)
        print('lv_candidate_root_node_index ', lv_selected_root_node_index)
        print('rv_candidate_root_node_index ', rv_selected_root_node_index)
        root_node_field_name_list = ['x', 'y', 'z', 'd', 't'] + vc_name_list
        # LV
        lv_root_node_filename = for_monodomain_root_node_result_file_name_start + iteration_str_tag + '_LV_root_nodes' + for_monodomain_biomarker_result_file_name_end
        lv_candidate_root_node_xyz = geometry.get_selected_root_node_xyz(root_node_index=lv_selected_root_node_index)
        print('lv_candidate_root_node_xyz ', lv_candidate_root_node_xyz.shape)
        lv_root_node_distance = geometry.get_selected_root_node_distance(
            root_node_meta_index=lv_root_node_meta_bool_index)
        lv_root_node_time = geometry.get_selected_root_node_time(root_node_meta_index=lv_root_node_meta_bool_index,
                                                                 purkinje_speed=purkinje_speed)
        print('lv_root_node_distance ', lv_root_node_distance.shape)
        print('lv_root_node_time ', lv_root_node_time.shape)
        lv_root_node_data = np.concatenate(
            (np.concatenate((lv_candidate_root_node_xyz, lv_root_node_distance[:, np.newaxis]), axis=1),
             lv_root_node_time[:, np.newaxis]), axis=1
        )
        for vc_i in range(len(vc_name_list)):
            vc_name = vc_name_list[vc_i]
            node_vc = geometry.get_selected_root_node_vc_field(root_node_index=lv_selected_root_node_index,
                                                               vc_name=vc_name)
            lv_root_node_data = np.concatenate((lv_root_node_data, node_vc[:, np.newaxis]))
        print('lv_root_node_data ', lv_root_node_data.shape)
        # Save LV root nodes
        save_csv_file(data=lv_root_node_data, filename=lv_root_node_filename,
                      column_name_list=root_node_field_name_list)
        print('Saved lv_root_node_filename ', lv_root_node_filename)
        # RV
        rv_root_node_filename = for_monodomain_root_node_result_file_name_start + iteration_str_tag + '_RV_root_nodes' + for_monodomain_biomarker_result_file_name_end
        rv_candidate_root_node_xyz = geometry.get_selected_root_node_xyz(root_node_index=rv_selected_root_node_index)
        rv_root_node_distance = geometry.get_selected_root_node_distance(
            root_node_meta_index=rv_root_node_meta_bool_index)
        rv_root_node_time = geometry.get_selected_root_node_time(root_node_meta_index=rv_root_node_meta_bool_index,
                                                                 purkinje_speed=purkinje_speed)
        rv_root_node_data = np.concatenate(
            (np.concatenate((rv_candidate_root_node_xyz, rv_root_node_distance[:, np.newaxis]), axis=1),
             rv_root_node_time[:, np.newaxis]), axis=1)
        for vc_i in range(len(vc_name_list)):
            vc_name = vc_name_list[vc_i]
            node_vc = geometry.get_selected_root_node_vc_field(root_node_index=rv_selected_root_node_index,
                                                               vc_name=vc_name)
            rv_root_node_data = np.concatenate((rv_root_node_data, node_vc[:, np.newaxis]))
        # Save RV root nodes
        save_csv_file(data=rv_root_node_data, filename=rv_root_node_filename,
                      column_name_list=root_node_field_name_list)
        print('Saved rv_root_node_filename ', rv_root_node_filename)
    # # Save precomputed REPOL for the selected particles to translate to monodomain
    # inference_repol_population = np.stack(inference_repol_population)
    # print('inference_repol_population ', inference_repol_population.shape)
    # save_csv_file(data=inference_repol_population, filename=inference_repol_uncertainty_population_filename)
    # print('Saved REPOL for each selected particle for translation to monodomain at ', inference_repol_uncertainty_population_filename)
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
    preprocessed_clinical_ecg_file_name = None
    raw_geometry_point_cloud = None
    for_monodomain_parameter_population_file_name = None
    unique_population_lat = None
    unique_population_vm = None
    uncertainty_population_lat = None
    uncertainty_population_vm = None
    unprocessed_node_mapping_index = None
    ####################################################################################################################
    # Step 19: Visualise ECGs and their metrics for the final population.
    print('Step 19: Visualise ECGs and their metrics for the final population.')
    # Initialise arguments for plotting
    axes = None
    fig = None
    # Plot the ECG inference population
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
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show(block=False)
    fig.savefig(for_monodomain_figure_result_file_name)
    print('Saved ecg figure: ', for_monodomain_figure_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    axes = None
    fig = None
    for_monodomain_figure_result_file_name = None
    population_biomarker = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)
    print('')

    #EOF




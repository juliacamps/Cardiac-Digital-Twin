import os
import time
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing

from adapter_theta_params import AdapterThetaParams, RoundTheta
from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
from conduction_system import EmptyConductionSystem
from discrepancy_functions import BiomarkerFromOnlyECG
from evaluation_functions import MetricEvaluator
from ecg_functions import PseudoEcgTetFromVM
from geometry_functions import EikonalGeometry
from inference_functions import SaltelliSensitivityAnalysis
from propagation_models import PrescribedLAT
from simulator_functions import SimulateECG
from path_config import get_path_mapping
from electrophysiology_functions import ElectrophysiologyAPDmap
from io_functions import save_dictionary, read_pandas, read_dictionary
# from utils import TestFailed
from postprocess_functions import scatter_visualise_field
from utils import get_biomarker_lead_name, get_qt_dur_name, get_t_pe_name, get_t_peak_name, get_qtpeak_dur_name, \
    get_t_polarity_name, get_tpeak_dispersion_name, translate_from_pandas_to_array, get_vc_ab_name, get_vc_aprt_name, \
    get_vc_rt_name, get_vc_rvlv_name, get_vc_tm_name, get_vc_tv_name

if __name__ == '__main__':
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
    # Simulate and plot QRS
    if os.path.isfile('../.custom_config/.your_path_mapping.txt'):
        path_dict = get_path_mapping()
    else:
        raise 'Missing data and results configuration file at: ../.custom_config/.your_path_mapping.txt'
    ####################################################################################################################
    # Step 0: Reproducibility:
    random_seed_value = 7  # Ensures reproducibility and turns off stochasticity
    np.random.seed(seed=random_seed_value)  # Ensures reproducibility and turns off stochasticity
    hyperparameter_dict = {}    # Save hyperparameters for reproducibility
    ####################################################################################################################
    # Step 1: Define paths and other environment variables.
    # General settings:
    anatomy_subject_name = 'DTI004'
    print('anatomy_subject_name: ', anatomy_subject_name)
    ecg_subject_name = 'DTI004'  # Allows using a different ECG for the personalisation than for the anatomy
    print('ecg_subject_name: ', ecg_subject_name)
    resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    cellular_data_dir = data_dir + 'cellular_data/'
    clinical_data_filename = 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    clinical_data_filename_path = data_dir + clinical_data_filename
    clinical_qrs_offset = 100 # ms TODO This could be calculated automatically and potentially, the clinical ECG could be trimmed to start with the QRS at time zero
    geometric_data_dir = data_dir + 'geometric_data/'
    # Intermediate Paths: # e.g., results from the QRS inference
    results_dir_root = path_dict["results_path"]
    qrs_lat_prescribed_filename = 'personalisation_data/' + anatomy_subject_name + '/qrs/' + anatomy_subject_name \
                                  + '_' + resolution + '_nodefield_inferred-lat.csv'
    qrs_lat_prescribed_filename_path = results_dir_root + qrs_lat_prescribed_filename
    # qrs_lat_prescribed_file_name = path_dict["results_path"] + 'personalisation_data/' + anatomy_subject_name + '/qrs/' \
    #                                + anatomy_subject_name + '_' + resolution + '_nodefield_inferred-lat.csv'
    # Output Paths:
    experiment_type = 'sa'
    ep_model = 'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    results_dir = path_dict["results_path"] + experiment_type + '_data/' + anatomy_subject_name + '/twave_' \
                  + gradient_ion_channel_str + '_' + ep_model + '/smoothing_fibre/' #'/only_endo/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    result_tag = experiment_type
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_hyperparameter.txt'
    theta_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_theta_population.csv'
    parameter_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_parameter_population.csv'
    sobol_indicies_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_sobol_indicies.csv'
    qoi_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_' + result_tag + '_qoi_population.csv'
    visualisation_dir = results_dir + 'checkpoint/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Save hyperparameters for reproducibility
    hyperparameter_dict['clinical_data_filename'] = clinical_data_filename  # Hyperparameter
    hyperparameter_dict['clinical_qrs_offset'] = clinical_qrs_offset    # Hyperparameter
    hyperparameter_dict['experiment_type'] = experiment_type    # This will tell in the future if this was sa or personalisation
    hyperparameter_dict['ep_model'] = ep_model
    hyperparameter_dict['gradient_ion_channel_list'] = gradient_ion_channel_list
    hyperparameter_dict['qrs_lat_prescribed_filename'] = qrs_lat_prescribed_filename
    hyperparameter_dict['result_tag'] = result_tag
    # Clear Arguments to prevent Argument recycling
    clinical_data_dir_tag = None
    clinical_data_filename = None
    data_dir = None
    ecg_subject_name = None
    experiment_type = None
    qrs_lat_prescribed_filename = None
    intermediate_dir = None
    results_dir = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
    # Arguments for cellular model:
    print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
    cellular_stim_amp = 11
    convergence = 'not_converged'
    stimulation_protocol = 'diffusion'
    cellular_data_dir_complete = cellular_data_dir + convergence + '_' + stimulation_protocol + '_' + str(
        cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model + '/'
    print('cellular_data_dir_complete ', cellular_data_dir_complete)
    cellular_model_name = 'torord_calibrated_pom_1000Hz'
    endo_celltype_name = 'endo'
    epi_celltype_name = 'epi'
    list_celltype_name = [endo_celltype_name, epi_celltype_name]
    biomarker_upstroke_name = 'activation_time'
    biomarker_apd90_name = 'apd90'
    biomarker_celltype_name = 'celltype'
    # Create cellular model instance.
    print('ep_model ', ep_model)
    if ep_model == 'MitchellSchaefferEP':
        apd_min_min = 200
        apd_max_max = 400
        apd_resolution = 1
        cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                        apd_resolution=apd_resolution, cycle_length=500,
                                                        list_celltype_name=list_celltype_name, verbose=verbose,
                                                        vm_max=1., vm_min=0.)
    else:

        # TODO revert or parameterise
        # Create cellular model instance.
        cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                          biomarker_apd90_name=biomarker_apd90_name,
                                                          biomarker_celltype_name=biomarker_celltype_name,
                                                          cellular_data_dir=cellular_data_dir_complete,
                                                          cellular_model_name=cellular_model_name,
                                                          list_celltype_name=list_celltype_name, verbose=verbose)
        apd_resolution = None
        apd_min_min, apd_max_max = cellular_model.get_biomarker_range(biomarker_name=biomarker_apd90_name)
    print('apd_min_min ', apd_min_min)
    print('apd_max_max ', apd_max_max)
    assert apd_max_max > apd_min_min
    # Save hyperparameters for reproducibility
    hyperparameter_dict['apd_max_max'] = apd_max_max
    hyperparameter_dict['apd_min_min'] = apd_min_min
    hyperparameter_dict['apd_resolution'] = apd_resolution

    hyperparameter_dict['biomarker_apd90_name'] = biomarker_apd90_name
    hyperparameter_dict['biomarker_celltype_name'] = biomarker_celltype_name
    hyperparameter_dict['biomarker_upstroke_name'] = biomarker_upstroke_name
    hyperparameter_dict['cellular_model_name'] = cellular_model_name
    hyperparameter_dict['cellular_stim_amp'] = cellular_stim_amp
    hyperparameter_dict['convergence'] = convergence
    hyperparameter_dict['endo_celltype_name'] = endo_celltype_name
    hyperparameter_dict['epi_celltype_name'] = epi_celltype_name
    hyperparameter_dict['list_celltype_name'] = list_celltype_name
    hyperparameter_dict['stimulation_protocol'] = stimulation_protocol
    # Clear Arguments to prevent Argument recycling
    biomarker_apd90_name = None
    biomarker_celltype_name = None
    biomarker_upstroke_name = None
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_model_name = None
    cellular_stim_amp = None
    convergence = None
    ep_model = None
    gradient_ion_channel_str = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 3: Generate a cardiac geometry that cannot run the Eikonal.
    # Argument setup: (in Alphabetical order)
    print('Step 3: Generate a cardiac geometry that cannot run the Eikonal.')
    vc_ab_name = get_vc_ab_name()
    vc_aprt_name = get_vc_aprt_name()
    vc_rt_name = get_vc_rt_name()
    vc_rvlv_name = get_vc_rvlv_name()
    vc_tm_name = get_vc_tm_name()
    vc_tv_name = get_vc_tv_name()
    vc_name_list = [vc_ab_name, vc_tm_name, vc_rt_name, vc_tv_name, vc_aprt_name, vc_rvlv_name]
    # Pre-assign celltype spatial correspondence.
    # celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0.3, 1.]}, epi_celltype_name: {vc_tm_name: [0., 0.3]}}
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 1.]}}
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['celltype_vc_info'] = celltype_vc_info
    hyperparameter_dict['vc_name_list'] = vc_name_list
    hyperparameter_dict['vc_ab_name'] = vc_ab_name
    hyperparameter_dict['vc_aprt_name'] = vc_aprt_name
    hyperparameter_dict['vc_tm_name'] = vc_tm_name
    hyperparameter_dict['vc_rt_name'] = vc_rt_name
    hyperparameter_dict['vc_rvlv_name'] = vc_rvlv_name
    hyperparameter_dict['vc_tv_name'] = vc_tv_name
    # Clear Arguments to prevent Argument recycling
    anatomy_subject_name = None
    geometric_data_dir = None
    list_celltype_name = None
    resolution = None
    vc_name_list = None
    ####################################################################################################################
    # Step TODO NEW STEP FOR SMOOTHING
    # TODO Fix this hack!!!
    fibre_speed = 6.500000000000000222e-02  # param_dict[self.fibre_speed_name]
    sheet_speed = 2.900000000000000147e-02  # param_dict[self.fibre_speed_name]
    normal_speed = 4.800000000000000100e-02  # param_dict[self.fibre_speed_name]

    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    # smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 5% and the present as 95%
    # full_smoothing_time_index = 400  # (ms) assumming 1000Hz
    print('Precompuing the smoothing, change this please!')
    geometry.precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(
        fibre_speed=fibre_speed, sheet_speed=sheet_speed, normal_speed=normal_speed,
        ghost_distance_to_self=smoothing_ghost_distance_to_self)

    hyperparameter_dict['fibre_speed'] = fibre_speed
    hyperparameter_dict['sheet_speed'] = sheet_speed
    hyperparameter_dict['normal_speed'] = normal_speed
    hyperparameter_dict['smoothing_ghost_distance_to_self'] = smoothing_ghost_distance_to_self
    ####################################################################################################################
    # Step 4: Create propagation model instance, this will be a static dummy propagation model.
    print('Step 4: Create propagation model instance, this will be a static dummy propagation model.')
    propagation_parameter_name_list_in_order = []
    lat_prescribed = (np.loadtxt(qrs_lat_prescribed_filename_path, delimiter=',')).astype(int)
    propagation_model = PrescribedLAT(geometry=geometry, lat_prescribed=lat_prescribed,
                                      module_name=propagation_module_name, verbose=verbose)
    # Visualise LAT, celltype, VC.
    # lat = lat_prescribed
    # celltype = propagation_model.geometry.get_node_celltype()
    # node_vc = propagation_model.geometry.get_node_vc()
    # node_xyz = propagation_model.geometry.get_node_xyz()
    # if verbose:
    #     print(cellular_model.get_celltype_to_id_correspondence())
    #     list_field_name = [vc_tm_name, vc_ab_name, vc_rt_name, vc_tv_name, vc_aprt_name, vc_rvlv_name]
    #     fig = plt.figure(figsize=(12, 8))
    #     ax1 = fig.add_subplot(241, projection='3d')
    #     p1 = scatter_visualise(ax1, node_xyz, lat, 'lat')
    #     ax2 = fig.add_subplot(242, projection='3d')
    #     ax3 = fig.add_subplot(243, projection='3d')
    #     ax4 = fig.add_subplot(244, projection='3d')
    #     ax5 = fig.add_subplot(245, projection='3d')
    #     ax6 = fig.add_subplot(246, projection='3d')
    #     ax7 = fig.add_subplot(247, projection='3d')
    #     ax8 = fig.add_subplot(248, projection='3d')
    #
    #     p2 = scatter_visualise(ax2, node_xyz, celltype, 'celltype')
    #     p3 = scatter_visualise(fig.add_subplot(243, projection='3d'), node_xyz, node_vc[vc_tm_name], vc_tm_name)
    #     p4 = scatter_visualise(ax4, node_xyz, node_vc[vc_ab_name], vc_ab_name)
    #     p5 = scatter_visualise(ax5, node_xyz, node_vc[vc_rt_name], vc_rt_name)
    #     p6 = scatter_visualise(ax6, node_xyz, node_vc[vc_tv_name], vc_tv_name)
    #     p7 = scatter_visualise(ax7, node_xyz, node_vc[vc_aprt_name], vc_aprt_name)
    #     p8 = scatter_visualise(ax8, node_xyz, node_vc[vc_rvlv_name], vc_rvlv_name)
    #     fig.colorbar(p8, ax=ax8)
    #     plt.savefig(visualisation_dir + 'check_fields.png')
    #     plt.show()
    # Save hyperparameters for reproducibility
    hyperparameter_dict['propagation_parameter_name_list_in_order'] = propagation_parameter_name_list_in_order
    # hyperparameter_dict['qrs_lat_prescribed_file_name'] = qrs_lat_prescribed_file_name
    # Clear Arguments to prevent Argument recycling
    qrs_lat_prescribed_filename_path = None
    # celltype = None
    # node_vc = None
    ####################################################################################################################
    # Step 5: Create Whole organ Electrophysiology model.
    print('Step 5: Create Whole organ Electrophysiology model.')
    # Arguments for Electrophysiology model:
    apd_max_name = 'apd_max'
    apd_min_name = 'apd_min'
    g_vc_ab_name = vc_ab_name
    g_vc_aprt_name = vc_aprt_name
    g_vc_rvlv_name = vc_rvlv_name
    g_vc_tm_name = vc_tm_name
    electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name, g_vc_rvlv_name, g_vc_tm_name]
    # electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_tm_name]#, , g_vc_rvlv_name, g_vc_tm_name]
    # Spatial and temporal smoothing parameters:
    smoothing_count = 40
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    # smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 5% and the present as 95%
    full_smoothing_time_index = 400  # (ms) assumming 1000Hz
    fibre_speed_name = 'fibre_speed'
    sheet_speed_name = 'sheet_speed'
    normal_speed_name = 'normal_speed'
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

    # electrophysiology_model = ElectrophysiologyAPDmap(apd_max_name=apd_max_name, apd_min_name=apd_min_name,
    #                                                   cellular_model=cellular_model,
    #                                                   parameter_name_list_in_order=electrophysiology_parameter_name_list_in_order,
    #                                                   module_name=electrophysiology_module_name,
    #                                                   propagation_model=propagation_model,
    #                                                   smoothing_count=smoothing_count,
    #                                                   smoothing_ghost_distance_to_self=smoothing_ghost_distance_to_self,
    #                                                   smoothing_past_present_window=np.asarray(smoothing_past_present_window),
    #                                                   verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['fibre_speed_name'] = fibre_speed_name
    hyperparameter_dict['sheet_speed_name'] = sheet_speed_name
    hyperparameter_dict['normal_speed_name'] = normal_speed_name
    hyperparameter_dict['full_smoothing_time_index'] = full_smoothing_time_index

    hyperparameter_dict['apd_max_name'] = apd_max_name
    hyperparameter_dict['apd_min_name'] = apd_min_name
    hyperparameter_dict['g_vc_ab_name'] = g_vc_ab_name
    hyperparameter_dict['g_vc_aprt_name'] = g_vc_aprt_name
    hyperparameter_dict['g_vc_rvlv_name'] = g_vc_rvlv_name
    hyperparameter_dict['g_vc_tm_name'] = g_vc_tm_name
    hyperparameter_dict['electrophysiology_parameter_name_list_in_order'] = electrophysiology_parameter_name_list_in_order
    hyperparameter_dict['smoothing_count'] = smoothing_count
    hyperparameter_dict['smoothing_ghost_distance_to_self'] = smoothing_ghost_distance_to_self
    hyperparameter_dict['smoothing_past_present_window'] = smoothing_past_present_window
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
    ####################################################################################################################
    # Step 6: Create ECG calculation method.
    print('Step 6: Create ECG calculation method.')
    # Arguments for ECG calculation:
    filtering = True
    max_len_qrs = 256  # This hyper-paramter is used when paralelising the ecg computation, because it needs a structure to synchronise the results from the multiple threads.
    max_len_st = 512  # ms
    max_len_ecg = max_len_qrs + max_len_st
    normalise = True
    zero_align = True
    frequency = 1000  # Hz
    if frequency != 1000:
        warn(
            'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
    low_freq_cut = 0.5
    high_freq_cut = 150
    I_name = 'I'
    II_name = 'II'
    v3_name = 'V3'
    v5_name = 'V5'
    lead_names = [I_name, II_name, 'V1', 'V2', v3_name, 'V4', v5_name, 'V6']
    nb_leads = len(lead_names)
    # Read clinical data
    # TODO This code may not work well for an ECG with only one lead!!
    untrimmed_clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
    clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
    # Create ECG model
    ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                   frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                   low_freq_cut=low_freq_cut, max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs,
                                   nb_leads=nb_leads, nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                   reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                   tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
    clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    lead_v3_i = lead_names.index(v3_name)
    lead_v5_i = lead_names.index(v5_name)
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
    # nb_leads = None
    normalise = None
    v3_name = None
    v5_name = None
    zero_align = None
    ####################################################################################################################
    # Step 7: Define instance of the simulation method.
    print('Step 7: Define instance of the simulation method.')
    simulator = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    # Save hyperparameters for reproducibility
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
    # physiological_rules_larger_than_dict[apd_max_name] = [apd_min_name]  # TODO Check that this rule is being used!
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
        'Caution: these rules have only been enabled for the inferred parameters!')  # TODO: modify this to also enable rules for fixed parameters (e.g., fibre_speed >= transmural_speed)
    # Create an adapter that can translate between theta and parameters
    adapter = AdapterThetaParams(destination_module_name_list_in_order=destination_module_name_list_in_order,
                                 parameter_fixed_value_dict=parameter_fixed_value_dict,
                                 parameter_name_list_in_order=parameter_name_list_in_order,
                                 parameter_destination_module_dict=parameter_destination_module_dict,
                                 physiological_rules_larger_than_dict=physiological_rules_larger_than_dict,
                                 theta_adjust_function_list_in_order=theta_adjust_function_list_in_order,
                                 theta_name_list_in_order=theta_name_list_in_order, verbose=verbose)
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
    # hyperparameter_dict['theta_name_list_in_order'] = theta_name_list_in_order
    # Clear Arguments to prevent Argument recycling
    # apd_max_name = None
    apd_max_resolution = None
    # apd_min_name = None
    apd_min_resolution = None
    candidate_root_node_names = None
    continuous_theta_name_list_in_order = None
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
    transmural_speed_name = None
    ####################################################################################################################
    # Step 9: Define the discrepancy metric.
    print('Step 9: Define the discrepancy metric.')
    # Quantities of interest
    qt_dur_name = get_qt_dur_name()
    t_pe_name = get_t_pe_name()
    t_peak_name = get_t_peak_name()
    qtpeak_dur_name = get_qtpeak_dur_name()
    t_polarity_name = get_t_polarity_name()
    tpeak_dispersion_name = get_tpeak_dispersion_name()
    qoi_name_list_for_average = [qt_dur_name, t_pe_name, t_peak_name, qtpeak_dur_name, t_polarity_name,
                                 tpeak_dispersion_name]
    qt_dur_name_list = []
    t_pe_name_list = []
    t_peak_name_list = []
    qtpeak_dur_name_list = []
    t_polarity_name_list = []
    # qoi_name_list_per_lead = []
    for lead_i in range(nb_leads):
        # qoi_name_list_per_lead.append(qt_dur_name + '_' + str(lead_i))
        # qoi_name_list_per_lead.append(t_pe_name + '_' + str(lead_i))
        # qoi_name_list_per_lead.append(t_peak_name + '_' + str(lead_i))
        # qoi_name_list_per_lead.append(qtpeak_dur_name + '_' + str(lead_i))
        # qoi_name_list_per_lead.append(t_polarity_name + '_' + str(lead_i))
        qt_dur_name_list.append(get_biomarker_lead_name(biomarker_lead_name=qt_dur_name, lead_i=lead_i))
        t_pe_name_list.append(get_biomarker_lead_name(biomarker_lead_name=t_pe_name, lead_i=lead_i))
        t_peak_name_list.append(get_biomarker_lead_name(biomarker_lead_name=t_peak_name, lead_i=lead_i))
        qtpeak_dur_name_list.append(get_biomarker_lead_name(biomarker_lead_name=qtpeak_dur_name, lead_i=lead_i))
        t_polarity_name_list.append(get_biomarker_lead_name(biomarker_lead_name=t_polarity_name, lead_i=lead_i))
    qoi_name_list_per_lead = qt_dur_name_list + t_pe_name_list + t_peak_name_list + qtpeak_dur_name_list \
                             + t_polarity_name_list
    qoi_name_list = qoi_name_list_for_average + qoi_name_list_per_lead
    if verbose:
        print('qoi_name_list ', qoi_name_list)
    # Create metric instance.
    # TODO This function needs to ensure that it can generate all the biomarkers in qoi_name_list
    metric = BiomarkerFromOnlyECG(biomarker_name_list=qoi_name_list, lead_v3_i=lead_v3_i, lead_v5_i=lead_v5_i,
                                  max_lat=max(lat_prescribed), qt_dur_name=qt_dur_name, qtpeak_dur_name=qtpeak_dur_name,
                                  t_pe_name=t_pe_name, t_peak_name=t_peak_name, t_polarity_name=t_polarity_name,
                                  tpeak_dispersion_name=tpeak_dispersion_name)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['qoi_name_list_per_lead'] = qoi_name_list_per_lead
    hyperparameter_dict['qt_dur_name'] = qt_dur_name
    hyperparameter_dict['t_pe_name'] = t_pe_name
    hyperparameter_dict['t_peak_name'] = t_peak_name
    hyperparameter_dict['qtpeak_dur_name'] = qtpeak_dur_name
    hyperparameter_dict['t_polarity_name'] = t_polarity_name
    hyperparameter_dict['tpeak_dispersion_name'] = tpeak_dispersion_name
    # Clear Arguments to prevent Argument recycling
    lat_prescribed = None
    lead_v3_i = None
    lead_v5_i = None
    ####################################################################################################################
    # Step 10: Create evaluator_ecg.
    print('Step 10: Create evaluator_ecg.')
    evaluator = MetricEvaluator(adapter=adapter, metric=metric, simulator=simulator,
                                verbose=verbose)
    # Save hyperparameters for reproducibility
    # Clear Arguments to prevent Argument recycling.
    adapter = None
    metric = None
    simulator = None
    ####################################################################################################################
    # Step 11: Create instance of inference method.
    print('Step 11: Create instance of SA method.')
    # Arguments for Bayesian Inference method:
    # Population ranges and priors
    '''apd'''
    apd_exploration_margin = 60  # 80   # ms
    apd_max_range = [apd_max_max - apd_exploration_margin, apd_max_max]  # cm/ms
    apd_min_range = [apd_min_min, apd_min_min + apd_exploration_margin]  # cm/ms
    '''ab'''
    gab_max = 1
    gab_min = -1
    g_vc_ab_range = [gab_min, gab_max]  # cm/ms
    '''aprt'''
    gaprt_max = 1
    gaprt_min = -1
    g_vc_aprt_range = [gaprt_min, gaprt_max]  # cm/ms
    '''rvlv'''
    grvlv_max = 1
    grvlv_min = -1
    g_vc_rvlv_range = [grvlv_min, grvlv_max]  # cm/ms
    '''tm'''
    gtm_max = 1
    gtm_min = -1    # the findings in the lit review suggest that it can go both ways
    g_vc_tm_range = [gtm_min, gtm_max]  # cm/ms
    # SMC-ABC configuration
    population_size = 2**11 #2**11  # 512   # Rule of thumb number    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    boundaries_theta = [apd_max_range, apd_min_range, g_vc_ab_range, g_vc_aprt_range, g_vc_rvlv_range, g_vc_tm_range]
    if verbose:
        print('boundaries_theta ', boundaries_theta)
    if len(theta_name_list_in_order) != len(boundaries_theta):
        raise Exception("Not a consistent number of parameters for the inference.")
    time_start = time.time()
    # Create instance of the inference method.
    sa_method = SaltelliSensitivityAnalysis(boundaries_theta=boundaries_theta, evaluator=evaluator,
                                            qoi_name_list=qoi_name_list, population_size=population_size,
                                            verbose=verbose)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['boundaries_theta'] = boundaries_theta  # Hyperparameter
    # hyperparameter_dict['qoi_name_list_for_average'] = qoi_name_list_for_average  # Hyperparameter
    hyperparameter_dict['qoi_name_list_per_lead'] = qoi_name_list_per_lead  # Hyperparameter
    hyperparameter_dict['qt_dur_name'] = qt_dur_name
    hyperparameter_dict['t_pe_name'] = t_pe_name
    hyperparameter_dict['t_peak_name'] = t_peak_name
    hyperparameter_dict['qtpeak_dur_name'] = qtpeak_dur_name
    hyperparameter_dict['t_polarity_name'] = t_polarity_name
    hyperparameter_dict['tpeak_dispersion_name'] = tpeak_dispersion_name
    hyperparameter_dict['population_size'] = population_size  # Hyperparameter
    # Clear Arguments to prevent Argument recycling.
    evaluator = None
    boundaries_theta = None
    ini_population_theta = None
    max_mcmc_steps = None
    nb_root_node_prior = None
    nb_candidate_root_nodes = None
    nb_root_nodes_range = None
    population_size = None
    retain_ratio = None
    verbose = None
    ####################################################################################################################
    # Step 13: Run SA sampling process
    print('Step 13: Run the SA sampling process.')
    apd_range_name = 'apd_range'
    qoi_amplitude_name = 'amp'
    # qoi_t_peak_name = 't_peak'
    if False and os.path.isfile(qoi_result_file_name) and os.path.isfile(theta_result_file_name):
        print('Reading precomputed results')
        pandas_population_qoi = read_pandas(filename=qoi_result_file_name)
        pandas_population_qoi[qoi_amplitude_name] = np.abs(pandas_population_qoi[t_peak_name])
        qoi_name_list.append(qoi_amplitude_name)
        qoi_name_list_for_average.append(qoi_amplitude_name)
        population_qoi = translate_from_pandas_to_array(name_list_in_order=qoi_name_list, pandas_data=pandas_population_qoi)
        pandas_population_theta = read_pandas(filename=theta_result_file_name)
        pandas_population_theta[apd_range_name] = pandas_population_theta[apd_max_name] - pandas_population_theta[apd_min_name]
        theta_name_list_in_order.append(apd_range_name)
        population_theta = translate_from_pandas_to_array(name_list_in_order=theta_name_list_in_order,
                                                          pandas_data=pandas_population_theta)
        # sa_method.problem.sample_sobol(N=sa_method.population_size, calc_second_order=True)
        # aux_theta = sa_method.problem.samples
        # print('aux_theta ', aux_theta.shape)
        # print('aux_theta ', aux_theta)
        # print()
        # print('Diff ', np.sum(np.abs(population_theta-aux_theta)))
        # print('population_theta ', population_theta)
        # print()
        # print('population_qoi ', population_qoi)
        # print()
        print('population_theta ', population_theta.shape)
        print('population_qoi ', population_qoi.shape)
        # raise()
    else:
        print('Running sampling porcess')
        # Arguments for parallelisation size:
        max_theta_per_iter = multiprocessing.cpu_count() * 4  # Here it uses x4 because in Heartsrv this is a good limit considering the memory available
        population_qoi, population_theta = sa_method.sample(max_theta_per_iter=max_theta_per_iter)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['apd_range_name'] = apd_range_name
    hyperparameter_dict['qoi_name_list'] = qoi_name_list
    hyperparameter_dict['qoi_name_list_for_average'] = qoi_name_list_for_average
    hyperparameter_dict['theta_name_list_in_order'] = theta_name_list_in_order
    # Clear Arguments to prevent Argument recycling.
    max_theta_per_iter = None
    pandas_population_qoi = None
    pandas_population_theta = None
    ####################################################################################################################
    # Step 14: Run SA analysis process
    print('Step 14: Run the SA analysis process.')
    sobol_list_list_df = sa_method.analyse_sa(qoi_name_list=qoi_name_list, population_qoi=population_qoi, population_theta=population_theta, theta_name_list=theta_name_list_in_order)
    # Format the SA result into a Pandas Dataframe of shape:
    sobol_indecies_df, sobol_indecies_name_list, value_column_name, conf_column_name = sa_method.convert_sobol_list_to_df(sobol_list_list_df)
    print('sobol_indecies_df ', sobol_indecies_df)
    sobol_indecies_df_keys = list(sobol_indecies_df.keys())
    print('sobol_indecies_df_keys ', sobol_indecies_df_keys)
    sobol_indecies_df_index = list(sobol_indecies_df.index)
    print('sobol_indecies_df_index ', sobol_indecies_df_index)
    nb_index_columns = len(sobol_indecies_df_index[0])
    print('nb_index_columns ', nb_index_columns)
    # Save hyperparameters for reproducibility
    hyperparameter_dict['sobol_name_list_in_order'] = sobol_indecies_name_list  # Hyperparameter
    hyperparameter_dict['value_column_name'] = value_column_name  # Hyperparameter
    hyperparameter_dict['conf_column_name'] = conf_column_name  # Hyperparameter
    hyperparameter_dict['nb_index_columns'] = nb_index_columns  # Hyperparameter
    # Clear Arguments to prevent Argument recycling.
    desired_discrepancy = None
    max_process_alive_time = None
    sobol_list_list_df = None
    unique_stopping_ratio = None
    ####################################################################################################################
    # Step 15: Save the SA results.
    print('Step 15: Save the SA results.')
    np.savetxt(theta_result_file_name, population_theta, delimiter=',', header=','.join(theta_name_list_in_order),
               comments='')
    print('Saved SA population theta: ', theta_result_file_name)
    population_parameter = sa_method.get_parameter_from_theta(population_theta)
    np.savetxt(parameter_result_file_name, population_parameter, delimiter=',',
               header=','.join(parameter_name_list_in_order), comments='')
    print('Saved SA population parameter: ', parameter_result_file_name)
    save_dictionary(dictionary=hyperparameter_dict, filename=hyperparameter_result_file_name)
    print('Saved hyperparameter: ', hyperparameter_result_file_name)
    sobol_indecies_df.to_csv(sobol_indicies_result_file_name, sep=',', index=True, header=True)     # Direclty save to also preserve the indexes of the dataframe
    # save_pandas(df=sobol_indecies_df, filename=sobol_indicies_result_file_name)
    print('Saved SA sobol indicies: ', sobol_indicies_result_file_name)
    np.savetxt(qoi_result_file_name, population_qoi, delimiter=',',
               header=','.join(qoi_name_list), comments='')
    print('Saved SA QOI values: ', qoi_result_file_name)
    print('Time spent in SA: ', round((time.time()-time_start)/3600., 1))
    # Clear Arguments to prevent Argument recycling.
    hyperparameter_dict = None
    parameter_name_list_in_order = None
    population_parameter = None
    population_theta = None
    sa_method = None
    sobol_indecies_df = None
    theta_name_list_in_order = None
    theta_result_file_name = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

# EOF




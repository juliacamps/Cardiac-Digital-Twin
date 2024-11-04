import multiprocessing
import os
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conduction_system import DjikstraConductionSystemVC, EmptyConductionSystem
from ecg_functions import PseudoQRSTetFromStepFunction, PseudoEcgTetFromVM
from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet, RawVCFibreCardiacGeoTet
from propagation_models import EikonalDjikstraTet
from simulator_functions import SimulateECG, SimulateEP
from adapter_theta_params import AdapterThetaParams
from discrepancy_functions import DiscrepancyECG
from evaluation_functions import DiscrepancyEvaluator
from cellular_models import StepFunctionUpstrokeEP, CellularModelBiomarkerDictionary
from electrophysiology_functions import ElectrophysiologyUpstrokeStepFunction, ElectrophysiologyAPDmap
from path_config import get_path_mapping
from io_functions import write_geometry_to_ensight, export_ensight_timeseries_case, read_dictionary
from evaluation_functions import ParameterSimulator
from postprocess_functions import scatter_visualise_field
from utils import map_indexes

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
    clinical_data_filename = data_dir + 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    clinical_qrs_offset = 100  # ms TODO This could be calculated automatically and potentially, the clinical ECG could be trimmed to start with the QRS at time zero
    geometric_data_dir = data_dir + 'geometric_data/'
    # Intermediate Paths: # e.g., results from the QRS inference
    results_dir = path_dict["results_path"] + 'ruben_tests_2/'
    # Output Paths:
    visualisation_dir = results_dir + 'ensight/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    figure_result_file_name = visualisation_dir + anatomy_subject_name + '_' + resolution + '_ecg.png'
    lat_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_nodefield_lat.csv'
    vm_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_nodefield_vm.csv'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Clear Arguments to prevent Argument recycling
    clinical_data_dir_tag = None
    data_dir = None
    ecg_subject_name = None
    intermediate_dir = None
    results_dir = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
    # Arguments for cellular model:
    print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
    cellular_stim_amp = 11
    convergence = 'not_converged'
    ion_channel = 'GKs_only'
    stimulation_protocol = 'diffusion'
    cellular_data_dir_complete = cellular_data_dir + convergence + '_' + stimulation_protocol + '_' + str(
        cellular_stim_amp) + '_' + ion_channel + '/'
    cellular_model_name = 'torord_calibrated_pom_1000Hz'
    endo_celltype_name = 'endo'
    epi_celltype_name = 'epi'
    list_celltype_name = [endo_celltype_name, epi_celltype_name]
    biomarker_upstroke_name = 'activation_time'
    biomarker_apd90_name = 'apd90'
    biomarker_celltype_name = 'celltype'
    # Create cellular model instance.
    cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                      biomarker_apd90_name=biomarker_apd90_name,
                                                      biomarker_celltype_name=biomarker_celltype_name,
                                                      cellular_data_dir=cellular_data_dir_complete,
                                                      cellular_model_name=cellular_model_name,
                                                      list_celltype_name=list_celltype_name, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    biomarker_apd90_name = None
    biomarker_celltype_name = None
    biomarker_upstroke_name = None
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_model_name = None
    cellular_stim_amp = None
    convergence = None
    ion_channel = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 3: Generate an Eikonal-friendly geometry.
    print('Step 3: Generate a cardiac geometry that can run the Eikonal.')
    # Argument setup: (in Alphabetical order)
    vc_ab_name = 'ab'
    vc_aprt_name = 'aprt'
    vc_tm_name = 'tm'
    vc_rt_name = 'rt'
    vc_rvlv_name = 'rvlv'
    vc_tv_name = 'tv'
    vc_name_list = [vc_ab_name, vc_tm_name, vc_rt_name, vc_tv_name, vc_aprt_name, vc_rvlv_name]
    # Only one celltype/no-celltype, because its using a step function as an action potential.
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0.3, 1.]}, epi_celltype_name: {vc_tm_name: [0., 0.3]}}
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    raw_geometry = RawVCFibreCardiacGeoTet(conduction_system=EmptyConductionSystem(verbose=verbose),
                                           geometric_data_dir=geometric_data_dir, resolution=resolution,
                                           subject_name=anatomy_subject_name, vc_name_list=vc_name_list,
                                           verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    geometric_data_dir = None
    # resolution = None
    # anatomy_subject_name = None
    vc_name_list = None
    ####################################################################################################################
    # Step 4: Create conduction system for the propagation model to be initialised.
    print('Step 4: Create rule-based Purkinje network using ventricular coordinates.')
    # Arguments for Conduction system:
    approx_djikstra_purkinje_max_path_len = 200 # this value was too small for the fine meshes from Ruben!
    lv_inter_root_node_distance = 2.5  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    rv_inter_root_node_distance = 2.5  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    # Create conduction system
    conduction_system = DjikstraConductionSystemVC(
        approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, geometry=geometry,
        lv_candidate_root_node_meta_index=, rv_candidate_root_node_meta_index=, purkinje_max_ab_cut_threshold=,
        vc_ab_cut_name=, vc_rt_name=, verbose=verbose)
    # Assign conduction_system to its geometry
    geometry.set_conduction_system(conduction_system)
    # Clear Arguments to prevent Argument recycling
    approx_djikstra_purkinje_max_path_len = None
    conduction_system = None
    lv_inter_root_node_distance = None
    rv_inter_root_node_distance = None
    ####################################################################################################################
    # Step 5: Create Eikonal instance. Eikonal will require a conduction and an Eikonal-friendly mesh on creation.
    print('Step 4: Create propagation model instance.')
    # Arguments for propagation model:
    fibre_speed_name = 'fibre_speed'
    transmural_speed_name = 'transmural_speed'
    normal_speed_name = 'normal_speed'
    endo_dense_speed_name = 'endo_dense_speed'
    endo_sparse_speed_name = 'endo_sparse_speed'
    purkinje_speed_name = 'purkinje_speed'
    speed_parameter_name_list_in_order = [fibre_speed_name, transmural_speed_name, normal_speed_name,
                                          endo_dense_speed_name,
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
        transmural_speed_name=transmural_speed_name, verbose=verbose)
    # Visualise LAT, celltype, VC.
    celltype = propagation_model.geometry.get_node_celltype()
    node_vc = propagation_model.geometry.get_node_vc()
    node_xyz = propagation_model.geometry.get_node_xyz()
    lat = np.zeros((node_xyz.shape[0]))
    if verbose:
        print(cellular_model.get_celltype_to_id_correspondence())
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(241, projection='3d')
        ax2 = fig.add_subplot(242, projection='3d')
        ax3 = fig.add_subplot(243, projection='3d')
        ax4 = fig.add_subplot(244, projection='3d')
        ax5 = fig.add_subplot(245, projection='3d')
        ax6 = fig.add_subplot(246, projection='3d')
        ax7 = fig.add_subplot(247, projection='3d')
        ax8 = fig.add_subplot(248, projection='3d')
        p1 = scatter_visualise_field(ax1, node_xyz, lat, 'lat')
        p2 = scatter_visualise_field(ax2, node_xyz, celltype, 'celltype')
        p3 = scatter_visualise_field(ax3, node_xyz, node_vc['tm'], 'tm_cobiveco')
        p4 = scatter_visualise_field(ax4, node_xyz, node_vc['ab'], 'ab_cobiveco')
        p5 = scatter_visualise_field(ax5, node_xyz, node_vc['rt'], 'rt_cobiveco')
        p6 = scatter_visualise_field(ax6, node_xyz, node_vc['tv'], 'tv_cobiveco')
        p7 = scatter_visualise_field(ax7, node_xyz, node_vc['aprt'], 'aprt_cobiveco')
        p8 = scatter_visualise_field(ax8, node_xyz, node_vc['rvlv'], 'rvlv_cobiveco')
        fig.colorbar(p8, ax=ax8)
        plt.savefig('check_fields.png')
        plt.show()
    ####################################################################################################################
    # Step 6: Create Whole organ Electrophysiology model.
    print('Step 6: Create Whole organ Electrophysiology model.')
    # Arguments for Electrophysiology model:
    apd_max_name = 'apd_max'
    apd_min_name = 'apd_min'
    g_vc_ab_name = vc_ab_name
    g_vc_aprt_name = vc_aprt_name
    g_vc_rvlv_name = vc_rvlv_name
    g_vc_tm_name = vc_tm_name
    electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name,
                                                      g_vc_rvlv_name, g_vc_tm_name]
    # Spatial and temporal smoothing parameters:
    smoothing_count = 5  # is 5 enough?
    smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 10% and the present as 90%
    electrophysiology_model = ElectrophysiologyAPDmap(apd_max_name=apd_max_name, apd_min_name=apd_min_name,
                                                      cellular_model=cellular_model,
                                                      parameter_name_list_in_order=electrophysiology_parameter_name_list_in_order,
                                                      module_name=electrophysiology_module_name,
                                                      propagation_model=propagation_model,
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
    vc_ab_name = None
    vc_aprt_name = None
    vc_rvlv_name = None
    vc_tm_name = None
    # ####################################################################################################################
    # Step 7: Create ECG calculation method.
    print('Step 7: Create ECG calculation method.')
    # Arguments for ECG calculation:
    filtering = True
    max_len_qrs = 256  # This hyper-paramter is used when paralelising the ecg computation, because it needs a structure to synchronise the results from the multiple threads.
    max_len_st = 512
    max_len_ecg = max_len_qrs + max_len_st
    normalise = True
    zero_align = True
    frequency = 1000  # Hz
    if frequency != 1000:
        warn(
            'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
    freq_cut = 150
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    nb_leads = 8
    # Read clinical data
    untrimmed_clinical_ecg_raw = np.genfromtxt(clinical_data_filename, delimiter=',')
    clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
    # Create ECG model
    ecg_model = PseudoEcgTetFromVM(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                   frequency=frequency, high_freq_cut=freq_cut, lead_names=lead_names,
                                   max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                   nodes_xyz=geometry.get_node_xyz(), normalise=normalise,
                                   reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                   tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
    clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
    clinical_ecg_raw = None
    filtering = None
    # frequency = None
    freq_cut = None
    lead_names = None
    max_len_ecg = None
    max_len_qrs = None
    max_len_st = None
    nb_leads = None
    normalise = None
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
    # Arguments for Adapter:
    parameter_name_list_in_order = propagation_parameter_name_list_in_order + electrophysiology_parameter_name_list_in_order
    continuous_theta_name_list_in_order = []
    theta_name_list_in_order = []
    parameter_fixed_value_dict = {}
    physiological_rules_larger_than_dict = {}
    # physiological_rules_larger_than_dict[endo_dense_speed_name] = [
    #     endo_sparse_speed_name]  # Define custom rules to constrain which parameters must be larger than others.
    nb_discrete_theta = len(candidate_root_node_names)
    theta_adjust_function_list_in_order = []
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
                                 theta_name_list_in_order=theta_name_list_in_order,
                                 theta_adjust_function_list_in_order=theta_adjust_function_list_in_order,
                                 physiological_rules_larger_than_dict=physiological_rules_larger_than_dict,
                                 verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    speed_parameter_name_list_in_order = None
    candidate_root_node_names = None
    fibre_speed_name = None
    transmural_speed_name = None
    normal_speed_name = None
    endo_dense_speed_name = None
    endo_sparse_speed_name = None
    theta_name_list_in_order = None
    parameter_fixed_value_dict = None
    # propagation_parameter_name_list_in_order = None
    ####################################################################################################################
    # Step 8: Define the discrepancy metric.
    # Arguments for DTW discrepancy metric:
    error_method_name = 'rmse_pcc_cubic'
    # Create discrepancy metric instance.
    discrepancy_metric = DiscrepancyECG(error_method_name=error_method_name)    # TODO: add weighting control between PCC and RMSE
    # Clear Arguments to prevent Argument recycling
    error_method_name = None
    ####################################################################################################################
    # Step 10: Create evaluator_ecg.
    # TODO Case01 has no electrodes!!!!!!!!!!!!!!!!!!!!!
    evaluator_ecg = DiscrepancyEvaluator(adapter=adapter, discrepancy_metric=discrepancy_metric,
                                         simulator=simulator_ecg,
                                         target_data=clinical_ecg, verbose=verbose)
    evaluator_ep = ParameterSimulator(adapter=adapter, simulator=simulator_ep, verbose=verbose)
    # Clear Arguments to prevent Argument recycling.
    adapter = None
    discrepancy_metric = None
    simulator_ecg = None
    simulator_ep = None
    clinical_ecg = None
    ####################################################################################################################
    # Step 11: Define simulation parameter values
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
    # Step 12: Simulate ECG  using example parameters.
    print('parameter_particle_example ', parameter_particle_example.shape)
    print('parameter_particle_example ', parameter_particle_example)
    discrepancy_particle = evaluator_ecg.evaluate_parameter(parameter_particle=parameter_particle_example)
    fig = evaluator_ecg.visualise_parameter_population(discrepancy_population=np.array([discrepancy_particle, discrepancy_particle]),
                                                       parameter_population=np.concatenate(
        (parameter_particle_example[np.newaxis, :], parameter_particle_example[np.newaxis, :]), axis=0))
    fig.savefig(figure_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    evaluator_ecg = None
    parameter_name_list_in_order = None
    parameter_population = None
    population_theta = None
    ####################################################################################################################
    # Step 13: Simulate LAT and VM, and Interpolate simulation results to have the same indexing than the input data files.
    simulated_lat, simulated_vm = evaluator_ep.simulate_parameter_particle(parameter_particle=parameter_particle_example)
    # Interpolate nodefield
    unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=raw_geometry.get_node_xyz(),
                                                 reference_points_xyz=geometry.get_node_xyz())
    simulated_lat = simulated_lat[unprocessed_node_mapping_index]
    simulated_vm = simulated_vm[unprocessed_node_mapping_index, :]
    np.savetxt(lat_result_file_name, simulated_lat, delimiter=',')
    print('Saved lat: ', lat_result_file_name)
    np.savetxt(vm_result_file_name, simulated_vm, delimiter=',')
    print('Saved vm: ', vm_result_file_name)
    write_geometry_to_ensight(geometry=raw_geometry, subject_name=anatomy_subject_name, resolution=resolution,
                              visualisation_dir=visualisation_dir, verbose=verbose)
    print('Saved ensight geometry: ', visualisation_dir)
    export_ensight_timeseries_case(dir=visualisation_dir, casename=anatomy_subject_name + '_simulation', dataname_list=['INTRA_RE'],
                                   vm_list=[simulated_vm], dt=1. / frequency, nodesxyz=raw_geometry.get_node_xyz(),
                                   tetrahedrons=raw_geometry.get_tetra())
    print('Saved ensight vm: ', visualisation_dir)
    # Clear Arguments to prevent Argument recycling.
    anatomy_subject_name = None
    best_theta = None
    best_parameter = None
    evaluator_ep = None
    figure_result_file_name = None
    frequency = None
    geometry = None
    inferred_theta_population = None
    raw_geometry = None
    results_dir = None
    unprocessed_node_mapping_index = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

# EOF

import os
from warnings import warn
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
import pymp
import pandas as pd
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.util import numpy_support as VN

from io_functions import export_ensight_initialise_case, export_ensight_add_case_node, export_ensight_geometry, \
    export_ensight_scalar_per_node, export_ensight_timeseries_case, read_dictionary, write_list_to_file, \
    save_vtk_to_csv, save_pandas, write_geometry_to_ensight, write_geometry_to_ensight_with_fields
from geometry_functions import RawEmptyCardiacGeoTet, RawVCFibreCardiacGeoTet
from postprocess_functions import scatter_visualise_field, generate_repolarisation_map, visualise_ecg
from utils import map_indexes, remap_pandas_from_row_index
from adapter_theta_params import AdapterThetaParams, RoundTheta
from cellular_models import CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary
from conduction_system import EmptyConductionSystem
from discrepancy_functions import DiscrepancyECG
from evaluation_functions import DiscrepancyEvaluator, ParameterSimulator
from ecg_functions import PseudoEcgTetFromVM
from geometry_functions import EikonalGeometry
from propagation_models import PrescribedLAT
from simulator_functions import SimulateECG, SimulateEP
from path_config import get_path_mapping
from electrophysiology_functions import ElectrophysiologyAPDmap


if __name__ == '__main__':
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
    data_dir = path_dict["data_path"]
    ecg_subject_name = 'DTI004'  # Allows using a different ECG for the personalisation than for the anatomy
    print('ecg_subject_name: ', ecg_subject_name)
    source_resolution = 'coarse'
    target_resolution = 'fine'
    verbose = True
    results_dir_root = path_dict["results_path"]
    # Input Paths:
    cellular_data_dir = data_dir + 'cellular_data/'
    geometric_data_dir = data_dir + 'geometric_data/'
    # Intermediate Paths: # e.g., results from the QRS inference
    experiment_type = 'personalisation'
    ep_model = 'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    results_dir = results_dir_root + experiment_type + '_data/' + anatomy_subject_name + '/twave_' \
                  + gradient_ion_channel_str + '_' + ep_model + '/smoothing_fibre_256_64_05/' #+ '/smoothing_fibre_256_64_05/' #+ '/smoothing_fibre/' #'/only_endo/'
    # Read hyperparamter dictionary
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_hyperparameter.txt'
    hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
    result_tag = hyperparameter_dict['result_tag']
    best_discrepancy_dir = results_dir + 'best_discrepancy/'
    best_parameter_result_file_name = best_discrepancy_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '-best-parameter.csv'
    # Output Paths:
    visualisation_dir_best = best_discrepancy_dir + 'best_ensight/'
    if not os.path.exists(visualisation_dir_best):
        os.mkdir(visualisation_dir_best)
    # visualisation_dir_fudged = best_discrepancy_dir + 'with_epi/'#'second_fudged_ensight/'
    visualisation_dir_fudged = best_discrepancy_dir + 'no_fudge/'#'second_fudged_ensight/'
    if not os.path.exists(visualisation_dir_fudged):
        os.mkdir(visualisation_dir_fudged)
    # translation_dir = best_discrepancy_dir + 'translation_to_monodomain_with_epi/' #'translation_to_monodomain_second_fudge/'
    translation_dir = best_discrepancy_dir + 'translation_to_monodomain/' #'translation_to_monodomain_second_fudge/'
    if not os.path.exists(translation_dir):
        os.mkdir(translation_dir)
    figure_best_fudged_file_name = translation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_best_fudged.png'
    figure_best_file_name = translation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_best.png'
    biomarker_result_file_name_best = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '_biomarker_best.csv'
    biomarker_result_file_name_fudged = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '_biomarker_fudged.csv'
    # lat_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-lat.csv'
    # vm_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-vm.csv'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Read hyperparameters
    clinical_data_filename = hyperparameter_dict['clinical_data_filename']
    qrs_lat_prescribed_filename = hyperparameter_dict['qrs_lat_prescribed_filename']
    print('qrs_lat_prescribed_filename ', qrs_lat_prescribed_filename)
    print('clinical_data_filename ', clinical_data_filename)
    # Read-in paths
    clinical_data_filename_path = data_dir + clinical_data_filename
    qrs_lat_prescribed_filename_path = results_dir_root + qrs_lat_prescribed_filename
    # Clear Arguments to prevent Argument recycling
    clinical_data_dir_tag = None
    clinical_data_filename = None
    data_dir = None
    ecg_subject_name = None
    lat_dir = None
    qrs_lat_prescribed_filename = None
    results_dir = None
    results_dir_root = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.
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
    # TODO uncomment
    # apd_max_max = hyperparameter_dict['apd_max_max']
    # apd_min_min = hyperparameter_dict['apd_max_max']
    # apd_resolution = hyperparameter_dict['apd_resolution']
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
    # TODO delete this line
    apd_min_min, apd_max_max = cellular_model.get_biomarker_range(biomarker_name=biomarker_apd90_name)
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
    # TODO delete this hack which was only to check if the reincorporation of the Epi celltype was possible Nov 2023
    # celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0.3, 1.]}, epi_celltype_name: {vc_tm_name: [0., 0.3]}}#hyperparameter_dict['celltype_vc_info']
    celltype_vc_info = hyperparameter_dict['celltype_vc_info']
    vc_name_list = hyperparameter_dict['vc_name_list']
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    source_geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                      conduction_system=EmptyConductionSystem(verbose=verbose),
                                      geometric_data_dir=geometric_data_dir, resolution=source_resolution,
                                      subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    target_geometry = RawEmptyCardiacGeoTet(conduction_system=EmptyConductionSystem(verbose=verbose),
                                            geometric_data_dir=geometric_data_dir, resolution=target_resolution,
                                            subject_name=anatomy_subject_name,  #vc_name_list=vc_name_list,
                                            verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    # geometric_data_dir = None
    list_celltype_name = None
    source_resolution = None
    # vc_name_list = None
    ####################################################################################################################
    # Step TODO NEW STEP FOR SMOOTHING
    # TODO Fix this hack!!!
    fibre_speed = 6.500000000000000222e-02  # param_dict[self.fibre_speed_name]
    sheet_speed = 2.900000000000000147e-02  # param_dict[self.fibre_speed_name]
    normal_speed = 4.800000000000000100e-02  # param_dict[self.fibre_speed_name]

    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    smoothing_ghost_distance_to_self = hyperparameter_dict[
        'smoothing_ghost_distance_to_self']  # cm # This parameter enables to control how much spatial smoothing happens and
    # smoothing_past_present_window = [0.05, 0.95]  # Weight the past as 5% and the present as 95%
    # full_smoothing_time_index = 400  # (ms) assumming 1000Hz
    print('Precompuing the smoothing, change this please!')
    source_geometry.precompute_spatial_smoothing_using_adjacentcies_orthotropic_fibres(
        fibre_speed=fibre_speed, sheet_speed=sheet_speed, normal_speed=normal_speed,
        ghost_distance_to_self=smoothing_ghost_distance_to_self)
    ####################################################################################################################
    # Step 4: Create propagation model instance, this will be a static dummy propagation model.
    # Arguments for propagation model:
    # Read hyperparameters
    propagation_parameter_name_list_in_order = hyperparameter_dict['propagation_parameter_name_list_in_order']
    lat_prescribed = (np.loadtxt(qrs_lat_prescribed_filename_path, delimiter=',')).astype(int)
    propagation_model = PrescribedLAT(geometry=source_geometry, lat_prescribed=lat_prescribed,
                                      module_name=propagation_module_name, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    qrs_lat_prescribed_file_name = None
    # lat_prescribed = None
    ####################################################################################################################
    # Step 5: Create Whole organ Electrophysiology model.
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
    # smoothing_ghost_distance_to_self = hyperparameter_dict['smoothing_ghost_distance_to_self']
    smoothing_past_present_window = hyperparameter_dict['smoothing_past_present_window']
    full_smoothing_time_index = hyperparameter_dict['full_smoothing_time_index']
    fibre_speed_name = hyperparameter_dict['fibre_speed_name']
    sheet_speed_name = hyperparameter_dict['sheet_speed_name']
    normal_speed_name = hyperparameter_dict['normal_speed_name']
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
    # Step 6: Create ECG calculation method.
    # Arguments for ECG calculation:
    # Read hyperparameters
    clinical_qrs_offset = hyperparameter_dict['clinical_qrs_offset']
    filtering = hyperparameter_dict['filtering']
    # freq_cut = hyperparameter_dict['freq_cut']
    low_freq_cut = hyperparameter_dict['low_freq_cut']
    high_freq_cut = hyperparameter_dict['high_freq_cut']
    lead_names = hyperparameter_dict['lead_names']
    max_len_qrs = hyperparameter_dict['max_len_qrs']
    max_len_ecg = hyperparameter_dict['max_len_ecg']
    nb_leads = hyperparameter_dict['nb_leads']
    assert nb_leads == len(lead_names)
    normalise = hyperparameter_dict['normalise']
    zero_align = hyperparameter_dict['zero_align']
    frequency = hyperparameter_dict['frequency']
    if frequency != 1000:
        warn(
            'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in '
            'the code, the other hyper-parameters will not give the expected outcome!')
    # Read clinical data
    untrimmed_clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
    clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
    untrimmed_clinical_ecg_raw = None   # Clear Arguments to prevent Argument recycling
    # Create ECG model
    ecg_model = PseudoEcgTetFromVM(electrode_positions=source_geometry.get_electrode_xyz(), filtering=filtering,
                                   frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                   low_freq_cut=low_freq_cut,
                                   max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                   nodes_xyz=source_geometry.get_node_xyz(), normalise=normalise,
                                   reference_ecg=clinical_ecg_raw, tetra=source_geometry.get_tetra(),
                                   tetra_centre=source_geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
    clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
    clinical_ecg_raw = None
    filtering = None
    freq_cut = None
    # lead_names = None
    max_len_ecg = None
    max_len_qrs = None
    max_len_st = None
    nb_leads = None
    normalise = None
    zero_align = None
    ####################################################################################################################
    # Step 7: Define instance of the simulation method.
    simulator_ecg = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    simulator_ep = SimulateEP(electrophysiology_model=electrophysiology_model, verbose=verbose)    # Clear Arguments to prevent Argument recycling
    electrophysiology_model = None
    ecg_model = None
    ####################################################################################################################
    # Step 8: Define Adapter to translate between theta and parameters.
    # Read hyperparameters
    apd_max_resolution = hyperparameter_dict['apd_max_resolution']
    apd_min_resolution = hyperparameter_dict['apd_min_resolution']
    destination_module_name_list_in_order = hyperparameter_dict['destination_module_name_list_in_order']
    g_vc_ab_resolution = hyperparameter_dict['g_vc_ab_resolution']
    g_vc_aprt_resolution = hyperparameter_dict['g_vc_aprt_resolution']
    g_vc_rvlv_resolution = hyperparameter_dict['g_vc_rvlv_resolution']
    g_vc_tm_resolution = hyperparameter_dict['g_vc_tm_resolution']
    parameter_destination_module_dict = hyperparameter_dict['parameter_destination_module_dict']
    parameter_fixed_value_dict = hyperparameter_dict['parameter_fixed_value_dict']
    parameter_name_list_in_order = hyperparameter_dict['parameter_name_list_in_order']
    physiological_rules_larger_than_dict = hyperparameter_dict['physiological_rules_larger_than_dict']
    theta_name_list_in_order = hyperparameter_dict['theta_name_list_in_order']
    theta_adjust_function_list_in_order = [RoundTheta(resolution=apd_max_resolution),
                                           RoundTheta(resolution=apd_min_resolution),
                                           RoundTheta(resolution=g_vc_ab_resolution),
                                           RoundTheta(resolution=g_vc_aprt_resolution),
                                           RoundTheta(resolution=g_vc_rvlv_resolution),
                                           # RoundTheta(resolution=g_vc_sep_resolution),
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
    # Step 9: Define the discrepancy metric.
    # Arguments for discrepancy metric:
    # Read hyperparameters
    error_method_name = hyperparameter_dict['error_method_name']
    # Create discrepancy metric instance.
    discrepancy_metric = DiscrepancyECG(error_method_name=error_method_name)  # TODO: add weighting control between PCC and RMSE
    # Clear Arguments to prevent Argument recycling
    error_method_name = None
    ####################################################################################################################
    # Step 10: Create evaluator_ecg.
    evaluator_ecg = DiscrepancyEvaluator(adapter=adapter, discrepancy_metric=discrepancy_metric, simulator=simulator_ecg,
                                         target_data=clinical_ecg, verbose=verbose)
    evaluator_ep = ParameterSimulator(adapter=adapter, simulator=simulator_ep, verbose=verbose)
    # Clear Arguments to prevent Argument recycling.
    adapter = None
    discrepancy_metric = None
    simulator_ecg = None
    simulator_ep = None
    ####################################################################################################################
    # Step 11: Read the values inferred for parameters and evaluate the ECGs.
    # TODO save candidate root nodes and their times so that the meta-indexes can be used to point at them.
    pandas_parameter_population = pd.read_csv(best_parameter_result_file_name, delimiter=',')
    parameter_population = evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
    # Simulate the loaded parameter
    best_ecg = evaluator_ecg.simulate_parameter_population(parameter_population=parameter_population)
    # Clear Arguments to prevent Argument recycling.
    best_parameter_result_file_name = None
    parameter_name_list_in_order = None
    ####################################################################################################################
    # Step 12: Apply fudge factor to improve translation between RE and monodomain ECG simulations.
    print('original values:')
    print('pandas_parameter_population[apd_min_name] ', pandas_parameter_population[apd_min_name].values)
    print('pandas_parameter_population[apd_max_name] ', pandas_parameter_population[apd_max_name].values)
    # Define fudge factor:
    # TODO delete this hack which was only to check if the reincorporation of the Epi celltype was possible Nov 2023
    apd_shift = 0#-30  # ms
    apd_range_padding = 0#20  # ms
    # Apply translation Fudge factor for monodomain simulations:
    pandas_parameter_population[apd_min_name] = pandas_parameter_population[
                                                    apd_min_name] + apd_shift - apd_range_padding
    print('apd_min_min ', apd_min_min)
    print('pandas_parameter_population[apd_min_name] ', pandas_parameter_population[apd_min_name].values)
    pandas_parameter_population[apd_max_name] = pandas_parameter_population[
                                                    apd_max_name] + apd_shift + apd_range_padding
    print('apd_max_max ', apd_max_max)
    print('pandas_parameter_population[apd_max_name] ', pandas_parameter_population[apd_max_name].values)
    fudged_parameter_population = evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
    print('fudged_parameter_population ', fudged_parameter_population)
    fudged_ecg = evaluator_ecg.simulate_parameter_population(parameter_population=fudged_parameter_population)
    # Clear Arguments to prevent Argument recycling.
    apd_max_name = None
    apd_min_name = None
    apd_shift = None
    apd_range_padding = None
    evaluator_ecg = None
    ####################################################################################################################
    # Step 13: Plotting and saving of the ECGs.
    # Initialise arguments for plotting best match to clinical
    axes = None
    fig = None
    # Plot the ECGs
    axes, fig = visualise_ecg(ecg_list=best_ecg, lead_name_list=lead_names, axes=axes,
                              ecg_color='black', fig=fig, label_list=['RE'],
                              linewidth=1.)
    axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                              ecg_color='lime', fig=fig, label_list=['Clinical'],
                              linewidth=1.)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show(block=False)
    fig.savefig(figure_best_file_name)
    # Initialise arguments for plotting best vs fudged match to clinical
    axes = None
    fig = None
    # Plot the ECGs
    axes, fig = visualise_ecg(ecg_list=best_ecg, lead_name_list=lead_names, axes=axes,
                              ecg_color='blue', fig=fig, label_list=['RE'],
                              linewidth=1.)
    axes, fig = visualise_ecg(ecg_list=fudged_ecg, lead_name_list=lead_names, axes=axes,
                              ecg_color='red', fig=fig, label_list=['Fudged'],
                              linewidth=1.)
    axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                              ecg_color='lime', fig=fig, label_list=['Clinical'],
                              linewidth=1.)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show(block=False)
    fig.savefig(figure_best_fudged_file_name)
    # Clear Arguments to prevent Argument recycling.
    axes = None
    best_ecg = None
    clinical_ecg = None
    fig = None
    fudged_ecg = None
    figure_best_fudged_file_name = None
    lead_names = None
    ####################################################################################################################
    # Step 14: Simulate vm maps and interpolate simulation results to have the same indexing that the input data files.
    # Best discrepancy parameter:
    best_parameter_particle = parameter_population[0, :]
    unprocessed_node_lat_best, unprocessed_node_vm_best = \
        evaluator_ep.simulate_parameter_particle(parameter_particle=best_parameter_particle)
    print('Is the LAT map the same after the inference? ', np.all(lat_prescribed == unprocessed_node_lat_best))
    print(np.sum(np.abs(lat_prescribed - unprocessed_node_lat_best)))
    assert np.all(lat_prescribed == unprocessed_node_lat_best)
    unprocessed_node_biomarker_best = evaluator_ep.biomarker_parameter_particle(parameter_particle=best_parameter_particle)
    # Fudged parameter:
    fudged_parameter_particle = fudged_parameter_population[0, :]
    unprocessed_node_lat_fudged, unprocessed_node_vm_fudged = \
        evaluator_ep.simulate_parameter_particle(parameter_particle=fudged_parameter_particle)
    unprocessed_node_biomarker_fudged = evaluator_ep.biomarker_parameter_particle(parameter_particle=fudged_parameter_particle)
    # Interpolate nodefield
    unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=target_geometry.get_node_xyz(),
                                                 reference_points_xyz=source_geometry.get_node_xyz())
    best_lat = unprocessed_node_lat_best[unprocessed_node_mapping_index]
    best_vm = unprocessed_node_vm_best[unprocessed_node_mapping_index, :]
    fudged_lat = unprocessed_node_lat_fudged[unprocessed_node_mapping_index]
    fudged_vm = unprocessed_node_vm_fudged[unprocessed_node_mapping_index, :]
    best_node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker_best,
                                                      row_index=unprocessed_node_mapping_index)
    fudged_node_biomarker = remap_pandas_from_row_index(df=unprocessed_node_biomarker_fudged,
                                                      row_index=unprocessed_node_mapping_index)
    # Clear Arguments to prevent Argument recycling.
    lat_prescribed = None
    unprocessed_node_lat_best = None
    unprocessed_node_lat_fudged = None
    unprocessed_node_vm_best = None
    unprocessed_node_vm_fudged = None
    ####################################################################################################################
    # Step 15: Save EP configuration for translation to Monodomain.
    warn('This requires all biomarkers to be numerical values.')
    # Arguments for discrepancy metric:
    lat_biomarker_name = 'lat'
    repol_biomarker_name = 'repol'
    # Calculate nodewise biomarkers for translation to Alya:
    ## Best discrepancy results:
    best_repol = generate_repolarisation_map(best_vm)
    best_node_biomarker[lat_biomarker_name] = best_lat
    best_node_biomarker[repol_biomarker_name] = best_repol
    save_pandas(df=best_node_biomarker, filename=biomarker_result_file_name_best)
    print('Saved best discrepancy biomarkers to allow translation to Alya')
    # Clear Arguments to prevent Argument recycling.
    biomarker_result_file_name_best = None
    ## Fudged for translation results:
    fudged_repol = generate_repolarisation_map(fudged_vm)
    fudged_node_biomarker[lat_biomarker_name] = fudged_lat
    fudged_node_biomarker[repol_biomarker_name] = fudged_repol
    save_pandas(df=fudged_node_biomarker, filename=biomarker_result_file_name_fudged)
    print('Saved fudged for translation biomarkers to allow translation to Alya')
    # Clear Arguments to prevent Argument recycling.
    biomarker_result_file_name_fudged = None
    ####################################################################################################################
    # Step 16: Save vm results as ensight files for both the best discrepancy and fudged results.
    # Save best discrepancy vm:
    export_ensight_timeseries_case(dir=visualisation_dir_best,
                                   casename=anatomy_subject_name + '_' + target_resolution + '_RE',
                                   dataname_list=['INTRA'],
                                   vm_list=[best_vm], dt=1. / frequency, nodesxyz=target_geometry.get_node_xyz(),
                                   tetrahedrons=target_geometry.get_tetra())
    print('Saved best ensight vm: ', visualisation_dir_best)
    # Save fudged vm:
    export_ensight_timeseries_case(dir=visualisation_dir_fudged,
                                   casename=anatomy_subject_name + '_' + target_resolution + '_RE',
                                   dataname_list=['INTRA'],
                                   vm_list=[fudged_vm], dt=1. / frequency, nodesxyz=target_geometry.get_node_xyz(),
                                   tetrahedrons=target_geometry.get_tetra())
    print('Saved fudged ensight vm: ', visualisation_dir_fudged)
    # Clear Arguments to prevent Argument recycling.
    best_vm = None
    frequency = None
    fudged_vm = None
    ####################################################################################################################
    # Step 17: Select few configuration and partial results as ensight files for both the best discrepancy and fudged.
    # Select which configuration and results to save:
    node_field_name_list = [lat_biomarker_name, repol_biomarker_name]
    best_node_field_list = [best_lat, best_repol]
    fudged_node_field_list = [fudged_lat, fudged_repol]
    # Clear Arguments to prevent Argument recycling.
    best_lat = None
    best_repol = None
    fudged_lat = None
    fudged_repol = None
    lat_biomarker_name = None
    repol_biomarker_name = None
    # VC fields
    node_field_name_list = node_field_name_list + vc_name_list
    for vc_name in vc_name_list:
        unprocessed_node_vc_field = source_geometry.get_node_vc_field(vc_name)
        node_vc_field = unprocessed_node_vc_field[unprocessed_node_mapping_index]
        best_node_field_list.append(node_vc_field)
        fudged_node_field_list.append(node_vc_field)
    # Clear Arguments to prevent Argument recycling.
    source_geometry = None
    vc_name_list = None
    # Continue selection:
    for ionic_scaling_name in gradient_ion_channel_list:
        node_field_name_list.append(ionic_scaling_name)
        best_node_field_list.append(best_node_biomarker[ionic_scaling_name])
        fudged_node_field_list.append(fudged_node_biomarker[ionic_scaling_name])
    node_field_name_list.append(biomarker_apd90_name)
    best_node_field_list.append(best_node_biomarker[biomarker_apd90_name])
    fudged_node_field_list.append(fudged_node_biomarker[biomarker_apd90_name])
    # Clear Arguments to prevent Argument recycling.
    biomarker_apd90_name = None
    gradient_ion_channel_list = None
    ionic_scaling_name = None
    # Celltype:
    endo_celltype_int = 1
    epi_celltype_int = 3
    node_field_name_list.append(biomarker_celltype_name)
    ## Best discrepancy:
    best_node_celltype_str = best_node_biomarker[biomarker_celltype_name]
    best_node_celltype = np.zeros((target_geometry.get_node_xyz().shape[0]))
    print('best_node_celltype_str ', best_node_celltype_str)
    best_node_celltype[best_node_celltype_str == endo_celltype_name] = endo_celltype_int
    print('Nb endo cells: ', np.sum(best_node_celltype_str == endo_celltype_name))
    best_node_celltype[best_node_celltype_str == epi_celltype_name] = epi_celltype_int
    print('Nb epi cells: ', np.sum(best_node_celltype_str == epi_celltype_name))
    best_node_field_list.append(best_node_celltype)
    ## Fudged fro translation:
    fudged_node_celltype_str = fudged_node_biomarker[biomarker_celltype_name]
    fudged_node_celltype = np.zeros((target_geometry.get_node_xyz().shape[0]))
    fudged_node_celltype[fudged_node_celltype_str == endo_celltype_name] = endo_celltype_int
    print('Nb endo cells: ', np.sum(fudged_node_celltype_str == endo_celltype_name))
    fudged_node_celltype[fudged_node_celltype_str == epi_celltype_name] = epi_celltype_int
    print('Nb epi cells: ', np.sum(fudged_node_celltype_str == epi_celltype_name))
    fudged_node_field_list.append(fudged_node_celltype)
    # Clear Arguments to prevent Argument recycling.
    best_node_celltype = None
    best_node_celltype_str = None
    endo_celltype_name = None
    endo_celltype_int = None
    epi_celltype_name = None
    epi_celltype_int = None
    fudged_node_celltype = None
    fudged_node_celltype_str = None
    ####################################################################################################################
    # Step 18: Save the selected results as ensight files for both the best discrepancy and fudged.
    # Best discrepancy:
    write_geometry_to_ensight_with_fields(geometry=target_geometry, node_field_list=best_node_field_list,
                                          node_field_name_list=node_field_name_list,
                                          subject_name=anatomy_subject_name + '_' + target_resolution + '_sf',
                                          verbose=verbose,
                                          visualisation_dir=visualisation_dir_best)
    # write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=[
    #                                                                                  best_lat, best_node_apd90,
    #                                                                                  best_node_celltype, best_repol,
    #                                                                                  node_transmural,
    #                                                                                  # node_sep,
    #                                                                                  node_rvlv] + best_node_field_list,
    #                                       node_field_name_list=[
    #                                                                'lat', biomarker_apd90_name, biomarker_celltype_name,
    #                                                                'repol', vc_tm_name,
    #                                                                # vc_sep_name,
    #                                                                vc_rvlv_name] + gradient_ion_channel_list,
    #                                       subject_name=anatomy_subject_name + '_' + target_resolution + '_sf',
    #                                       verbose=verbose,
    #                                       visualisation_dir=visualisation_dir)
    # Fudged for translation:

    ### TODO Delete
    write_geometry_to_ensight_with_fields(geometry=target_geometry, node_field_list=fudged_node_field_list,
                                          node_field_name_list=node_field_name_list,
                                          subject_name=anatomy_subject_name + '_' + target_resolution + '_sf',
                                          verbose=verbose,
                                          visualisation_dir=visualisation_dir_fudged)
    # Clear Arguments to prevent Argument recycling.
    anatomy_subject_name = None
    best_node_field_list = None
    target_geometry = None
    node_field_name_list = None
    target_resolution = None
    verbose = None
    visualisation_dir_best = None
    visualisation_dir_fudged = None
    # node_ab = geometry.get_node_vc_field(vc_name=vc_ab_name)
    # node_rt = geometry.get_node_vc_field(vc_name=vc_rt_name)
    # node_tm = geometry.get_node_vc_field(vc_name=vc_tm_name)
    # node_tv = geometry.get_node_vc_field(vc_name=vc_tv_name)
    # node_lvendo = geometry.get_node_lvendo()
    # node_rvendo = geometry.get_node_rvendo()
    # write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=[node_ab,
    #                                                                                 node_rt,
    #                                                                                 node_tm,
    #                                                                                 node_tv,
    #                                                                                 node_lvendo,
    #                                                                                 node_rvendo
    #                                                                                 ],
    #                                       node_field_name_list=[vc_ab_name,
    #                                                             vc_rt_name,
    #                                                             vc_tm_name,
    #                                                             vc_tv_name,
    #                                                             'lvendo',
    #                                                             'rvendo'],
    #                                       subject_name=anatomy_subject_name + '_' + target_resolution + '_ml',
    #                                       verbose=verbose,
    #                                       visualisation_dir=visualisation_dir)
    # Clear Arguments to prevent Argument recycling.
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

    #EOF



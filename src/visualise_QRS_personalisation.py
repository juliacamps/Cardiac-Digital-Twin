import os
import sys
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    if len(sys.argv) < 2:
        anatomy_subject_name = 'DTI004'
        ecg_subject_name = 'DTI004' # Allows using a different ECG for the personalisation than for the anatomy
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
    from ecg_functions import PseudoQRSTetFromStepFunction
    from geometry_functions import EikonalGeometry, RawEmptyCardiacGeoTet
    from propagation_models import EikonalDjikstraTet
    from simulator_functions import SimulateEP, SimulateECG
    from adapter_theta_params import AdapterThetaParams, RoundTheta
    from discrepancy_functions import DiscrepancyECG
    from evaluation_functions import ParameterSimulator, DiscrepancyEvaluator
    from electrophysiology_functions import ElectrophysiologyUpstrokeStepFunction
    from cellular_models import StepFunctionUpstrokeEP
    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields, \
        read_dictionary, export_ensight_timeseries_case
    from utils import map_indexes

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
    # hyperparameter_dict = {}  # Save hyperparameters for reproducibility
    ####################################################################################################################
    # Step 1: Define paths and other environment variables.
    # General settings:
    source_resolution = 'coarse'
    target_resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    geometric_data_dir = data_dir + 'geometric_data/'
    # Output Paths from Inference:
    experiment_type = 'personalisation'
    ep_model_qrs = 'stepFunction'  # TODO this cannot be read from hyperparam dictionary because it's part of the path
    results_dir_root = path_dict["results_path"]
    # Build results folder structure
    results_dir_part = results_dir_root + experiment_type + '_data/'
    assert os.path.exists(results_dir_part)
    results_dir_part = results_dir_part + anatomy_subject_name + '/'
    assert os.path.exists(results_dir_part)
    results_dir_part = results_dir_part + 'qrs_' + ep_model_qrs + '/'
    assert os.path.exists(results_dir_part)
    # Use date to name the result folder to preserve some history of results
    current_month_text = 'Jun'#datetime.now().strftime('%h')  # Feb
    current_year_full = datetime.now().strftime('%Y')  # 2024
    results_dir = results_dir_part + current_month_text + '_' + current_year_full + '/'
    assert os.path.exists(results_dir)
    results_dir_part = None  # Clear Arguments to prevent Argument recycling
    # Continue defining results paths and configuration
    # Read hyperparamter dictionary
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_hyperparameter.txt'
    hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
    hyperparameter_result_file_name = None  # Clear Arguments to prevent Argument recycling
    result_tag = hyperparameter_dict['result_tag']
    parameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_parameter_population.csv'
    # Output Paths:
    visualisation_dir = results_dir + 'ensight/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    figure_result_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_population.png'
    # Best discrepancy
    translation_dir = results_dir + 'best_discrepancy/'
    if not os.path.exists(translation_dir):
        os.mkdir(translation_dir)
    lat_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-lat.csv'
    vm_result_file_name = translation_dir + anatomy_subject_name + '_' + target_resolution + '_nodefield_' + result_tag + '-vm.csv'
    best_parameter_result_file_name = translation_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '-best-parameter.csv'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Read hyperparameters
    clinical_data_filename = hyperparameter_dict['clinical_data_filename']
    clinical_data_filename_path = data_dir + clinical_data_filename
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
    cellular_model_convergence = None
    data_dir = None
    ecg_subject_name = None
    experiment_type = None
    intermediate_dir = None
    results_dir = None
    ####################################################################################################################
    # Step 2: Create Cellular Electrophysiology model. In this case, it will use a step function as the AP's upstroke.
    print('Step 2: Create Cellular Electrophysiology model, using a step function as the APs upstroke.')
    # Arguments for cellular model:
    # Read hyperparameters
    resting_vm_value = hyperparameter_dict['resting_vm_value']
    upstroke_vm_value = hyperparameter_dict['upstroke_vm_value']
    cellular_model = StepFunctionUpstrokeEP(resting_vm_value=resting_vm_value, upstroke_vm_value=upstroke_vm_value,
                                            verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    resting_vm_value = None
    upstroke_vm_value = None
    ####################################################################################################################
    # Step 3: Generate a cardiac geometry.
    print('Step 3: Generate a cardiac geometry.')
    # Argument setup: (in Alphabetical order)
    # Read hyperparameters
    vc_ab_cut_name = hyperparameter_dict['vc_ab_cut_name']
    vc_rt_name = hyperparameter_dict['vc_rt_name']
    vc_name_list = hyperparameter_dict['vc_name_list']
    celltype_vc_info = hyperparameter_dict['celltype_vc_info']
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
    # Step 5: Create Eikonal instance. Eikonal will require a conduction and an Eikonal-friendly mesh on creation.
    print('Step 5: Create propagation model Eikonal instance.')
    # Arguments for propagation model:
    # Read hyperparameters
    fibre_speed_name = hyperparameter_dict['fibre_speed_name']
    sheet_speed_name = hyperparameter_dict['sheet_speed_name']
    normal_speed_name = hyperparameter_dict['normal_speed_name']
    endo_dense_speed_name = hyperparameter_dict['endo_dense_speed_name']
    endo_sparse_speed_name = hyperparameter_dict['endo_sparse_speed_name']
    purkinje_speed_name = hyperparameter_dict['purkinje_speed_name']
    nb_speed_parameters = hyperparameter_dict['nb_speed_parameters']
    propagation_parameter_name_list_in_order = hyperparameter_dict['propagation_parameter_name_list_in_order']
    propagation_model = EikonalDjikstraTet(
        endo_dense_speed_name=endo_dense_speed_name, endo_sparse_speed_name=endo_sparse_speed_name,
        fibre_speed_name=fibre_speed_name, geometry=geometry, module_name=propagation_module_name,
        nb_speed_parameters=nb_speed_parameters, normal_speed_name=normal_speed_name,
        parameter_name_list_in_order=propagation_parameter_name_list_in_order, purkinje_speed_name=purkinje_speed_name,
        sheet_speed_name=sheet_speed_name, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    nb_speed_parameters = None
    ####################################################################################################################
    # Step 6: Create Whole organ Electrophysiology model.
    electrophysiology_model = ElectrophysiologyUpstrokeStepFunction(
        cellular_model=cellular_model, module_name=electrophysiology_module_name, propagation_model=propagation_model,
        verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    cellular_model = None
    propagation_model = None
    ####################################################################################################################
    # Step 7: Create ECG calculation method. In this case, the ecg will calculate only the QRS and will use a step
    # function as the AP's upstroke.
    # Arguments for ECG calculation:
    # Read hyperparameters
    filtering = hyperparameter_dict['filtering']
    max_len_qrs = hyperparameter_dict['max_len_qrs']
    normalise = hyperparameter_dict['normalise']
    zero_align = hyperparameter_dict['zero_align']
    frequency = hyperparameter_dict['frequency']
    if frequency != 1000:
        warn('The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')

    low_freq_cut = hyperparameter_dict['low_freq_cut']
    high_freq_cut = hyperparameter_dict['high_freq_cut']
    lead_names = hyperparameter_dict['lead_names']
    nb_leads = hyperparameter_dict['nb_leads']
    # Read clinical data
    clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path, delimiter=',')
    # Create ECG model
    ecg_model = PseudoQRSTetFromStepFunction(electrode_positions=geometry.get_electrode_xyz(), filtering=filtering,
                                             frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                                             low_freq_cut=low_freq_cut,
                                             max_len_qrs=max_len_qrs, nb_leads=nb_leads, nodes_xyz=geometry.get_node_xyz(),
                                             normalise=normalise, reference_ecg=clinical_ecg_raw, tetra=geometry.get_tetra(),
                                             tetra_centre=geometry.get_tetra_centre(), verbose=verbose, zero_align=zero_align)
    clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    # Clear Arguments to prevent Argument recycling
    clinical_ecg_raw = None
    filtering = None
    normalise = None
    zero_align = None
    high_freq_cut = None
    low_freq_cut = None
    lead_names = None
    max_len_qrs = None
    nb_leads = None
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
    # Read hyperparameters
    # TODO make the following code into a for loop!!
    # Theta resolutions
    endo_dense_speed_resolution = hyperparameter_dict['endo_dense_speed_resolution']
    endo_sparse_speed_resolution = hyperparameter_dict['endo_sparse_speed_resolution']
    sheet_speed_resolution = hyperparameter_dict['sheet_speed_resolution']
    theta_adjust_function_list_in_order = [RoundTheta(resolution=sheet_speed_resolution),
                                           RoundTheta(resolution=endo_dense_speed_resolution),
                                           RoundTheta(resolution=endo_sparse_speed_resolution)
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

    # # Read hyperparameters
    # theta_name_list_in_order = hyperparameter_dict['theta_name_list_in_order']
    # continuous_theta_name_list_in_order = ['sheet_speed', 'endo_dense_speed', 'endo_sparse_speed']#hyperparameter_dict['continuous_theta_name_list_in_order']
    # destination_module_name_list_in_order = hyperparameter_dict['destination_module_name_list_in_order']
    # endo_dense_speed_resolution = hyperparameter_dict['endo_dense_speed_resolution']
    # endo_sparse_speed_resolution = hyperparameter_dict['endo_sparse_speed_resolution']
    # nb_discrete_theta = len(theta_name_list_in_order)-len(continuous_theta_name_list_in_order) #hyperparameter_dict['nb_discrete_theta']
    # parameter_destination_module_dict = hyperparameter_dict['parameter_destination_module_dict']
    # parameter_fixed_value_dict = hyperparameter_dict['parameter_fixed_value_dict']
    # parameter_name_list_in_order = hyperparameter_dict['parameter_name_list_in_order']
    # physiological_rules_larger_than_dict = hyperparameter_dict['physiological_rules_larger_than_dict']
    # print('physiological_rules_larger_than_dict')
    # print(physiological_rules_larger_than_dict)
    # theta_name_list_in_order = hyperparameter_dict['theta_name_list_in_order']
    # sheet_speed_resolution = hyperparameter_dict['sheet_speed_resolution']
    # theta_adjust_function_list_in_order = []
    # for continuous_i in range(len(continuous_theta_name_list_in_order)):
    #     continuous_i_name = continuous_theta_name_list_in_order[continuous_i]
    #     resolution_i_name = continuous_i_name + '_resolution'
    #     theta_i_resolution = hyperparameter_dict[resolution_i_name]
    #     theta_adjust_function_list_in_order.append(RoundTheta(resolution=theta_i_resolution))
    # for discrete_i in range(nb_discrete_theta):
    #     theta_adjust_function_list_in_order.append(None)
    # if len(theta_adjust_function_list_in_order) != len(theta_name_list_in_order):
    #     warn('different number of adjusting functions and theta for the inference')
    #     print('theta_name_list_in_order ', len(theta_name_list_in_order))
    #     print('theta_adjust_function_list_in_order ', len(theta_adjust_function_list_in_order))
    #     quit()
    # # Create an adapter that can translate between theta and parameters
    # adapter = AdapterThetaParams(destination_module_name_list_in_order=destination_module_name_list_in_order,
    #                              parameter_fixed_value_dict=parameter_fixed_value_dict,
    #                              parameter_name_list_in_order=parameter_name_list_in_order,
    #                              parameter_destination_module_dict=parameter_destination_module_dict,
    #                              theta_adjust_function_list_in_order=theta_adjust_function_list_in_order,
    #                              theta_name_list_in_order=theta_name_list_in_order,
    #                              physiological_rules_larger_than_dict=physiological_rules_larger_than_dict,
    #                              verbose=verbose)
    # # Clear Arguments to prevent Argument recycling
    # continuous_theta_name_list_in_order = None
    # candidate_root_node_names = None
    # endo_dense_speed_name = None
    # endo_sparse_speed_name = None
    # fibre_speed_name = None
    # normal_speed_name = None
    # parameter_fixed_value_dict = None
    # propagation_parameter_name_list_in_order = None
    # speed_parameter_name_list_in_order = None
    # theta_name_list_in_order = None
    # sheet_speed_name = None
    ####################################################################################################################
    # Step 10: Define the discrepancy metric.
    print('Step 10: Define the discrepancy metric.')
    # Arguments for DTW discrepancy metric:
    # Read hyperparameters
    error_method_name = hyperparameter_dict['error_method_name']
    # Create discrepancy metric instance.
    discrepancy_metric = DiscrepancyECG(error_method_name=error_method_name)    # TODO: add weighting control between PCC and RMSE
    # Clear Arguments to prevent Argument recycling
    error_method_name = None
    ####################################################################################################################
    # Step 10: Create evaluator_ecg.
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
    # Step 11: Read the values inferred for parameters and evaluate their ECG.
    print('Step 11: Read the values inferred for parameters and evaluate their ECG.')
    # TODO save candidate root nodes and their times so that the meta-indexes can be used to point at them.
    pandas_parameter_population = pd.read_csv(parameter_result_file_name, delimiter=',')
    parameter_population = evaluator_ecg.translate_from_pandas_to_parameter(pandas_parameter_population)
    # Simulate the parameter population from the inference
    discrepancy_population = evaluator_ecg.evaluate_parameter_population(parameter_population=parameter_population)
    fig = evaluator_ecg.visualise_parameter_population(discrepancy_population=discrepancy_population,
                                                       parameter_population=parameter_population)
    fig.savefig(figure_result_file_name)
    print('Saved ecg figure: ', figure_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    figure_result_file_name = None
    ####################################################################################################################
    # Step 13: Select best discrepancy particle.
    print('Step 13: Select best discrepancy particle.')
    best_parameter = parameter_population[np.argmin(discrepancy_population)]
    np.savetxt(best_parameter_result_file_name, best_parameter[np.newaxis, :], delimiter=',',
               header=','.join(parameter_name_list_in_order), comments='')
    print('Saved best parameter: ', best_parameter_result_file_name)
    # Clear Arguments to prevent Argument recycling.
    best_parameter_result_file_name = None
    discrepancy_population = None
    evaluator_ecg = None
    parameter_name_list_in_order = None
    parameter_population = None
    population_theta = None
    ####################################################################################################################
    # Step 11: Interpolate simulation results to have the same indexing than the input data files.
    best_lat, best_vm = evaluator_ep.simulate_parameter_particle(parameter_particle=best_parameter)
    # Interpolate nodefield
    unprocessed_node_mapping_index = map_indexes(points_to_map_xyz=raw_geometry.get_node_xyz(),
                                                 reference_points_xyz=geometry.get_node_xyz())
    best_lat = best_lat[unprocessed_node_mapping_index]
    best_vm = best_vm[unprocessed_node_mapping_index, :]
    np.savetxt(lat_result_file_name, best_lat, delimiter=',')
    print('Saved best lat: ', lat_result_file_name)
    np.savetxt(vm_result_file_name, best_vm, delimiter=',')
    print('Saved best vm: ', vm_result_file_name)
    # raw_geometry.lat = best_lat
    activation_time_map_biomarker_name = 'lat'  # TODO make these names globally defined in utils.py
    write_geometry_to_ensight_with_fields(
        geometry=raw_geometry,
        node_field_list=[best_lat],
        node_field_name_list=[activation_time_map_biomarker_name],
        subject_name=anatomy_subject_name + '_' + target_resolution + '_qrs', verbose=verbose,
        visualisation_dir=visualisation_dir)
    # print('Saved best ensight lat: ', visualisation_dir)
    export_ensight_timeseries_case(dir=visualisation_dir, casename=anatomy_subject_name + '_RE', dataname_list=['INTRA'],
                                   vm_list=[best_vm], dt=1. / frequency, nodesxyz=raw_geometry.get_node_xyz(),
                                   tetrahedrons=raw_geometry.get_tetra())
    print('Saved best ensight vm: ', visualisation_dir)
    # Clear Arguments to prevent Argument recycling.
    anatomy_subject_name = None
    best_theta = None
    best_parameter = None
    evaluator_ep = None
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

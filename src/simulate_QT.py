# Import libraries that get used in multiple cells
import time
import numpy as np
from warnings import warn
import os


# LOAD FUNCTIONS
from conduction_system import EmptyConductionSystem, PurkinjeSystemVC
from ecg_functions import PseudoQRSTetFromStepFunction,PseudoEcgTetFromVM # FIX
from geometry_functions import EikonalGeometry
from propagation_models import EikonalDjikstraTet
from simulator_functions import SimulateECGwithLATmax, SimulateEP,SimulateECG # FIX
from adapter_theta_params import AdapterThetaParams, RoundTheta
from discrepancy_functions import DiscrepancyHealthyQRS,DiscrepancyECG # FIX
from evaluation_functions import ParameterSimulator,DiscrepancyEvaluator # FIX
from electrophysiology_functions import ElectrophysiologyUpstrokeStepFunction,ElectrophysiologyAPDmap # FIX
from cellular_models import StepFunctionUpstrokeEP,CellularModelBiomarkerDictionary, MitchellSchaefferAPDdictionary  # FIX
from io_functions import save_dictionary, write_geometry_to_ensight_with_fields, write_purkinje_vtk, \
        write_root_node_csv
from utils import get_vc_rt_name, \
    get_vc_ab_cut_name, get_fibre_speed_name, get_sheet_speed_name, \
    get_normal_speed_name, get_endo_dense_speed_name, get_endo_sparse_speed_name, \
    get_purkinje_speed_name, get_unique_lead_name_list, get_xyz_name_list, unfold_ecg_matrix, \
    get_vc_aprt_name, get_vc_rvlv_name, get_vc_tm_name
from postprocess_functions import visualise_ecg
from datetime import datetime
# Define some new functions ot plot the activation maps on the 3D geometries
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

def list_faces(t):
  t.sort(axis=1)
  n_t, m_t= t.shape
  f = np.empty((4*n_t, 3) , dtype=int)
  i = 0
  for j in range(4):
    f[i:i+n_t,0:j] = t[:,0:j]
    f[i:i+n_t,j:3] = t[:,j+1:4]
    i=i+n_t
  return f

def extract_unique_triangles(t):
  _, indxs, count  = np.unique(t, axis=0, return_index=True, return_counts=True)
  return t[indxs[count==1]]

def extract_surface(t):
  f=list_faces(t)
  f=extract_unique_triangles(f)
  return f


def plot_geometry(xyz, surf, lat_simulation):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            colorbar=dict(title=dict(text='z')),
            colorscale=px.colors.sequential.Viridis,
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity=lat_simulation,
            # i, j and k give the vertices of triangles
            # here we represent the 4 triangles of the tetrahedron surface
            i=surf[:, 0],
            j=surf[:, 1],
            k=surf[:, 2],
            name='y',
            showscale=True
        )
    ])
    fig.update_layout( width=1000,
            height=1000)

    fig.show()

def plot_ecg(predicted_ecg, clinical_ecg, lead_names):
    # Initialise arguments for plotting
    axes = None
    fig = None
    # Plot the ECG inference population
    # print('population_ecg ', population_ecg.shape)
    # axes, fig = visualise_ecg(ecg_list=population_ecg, lead_name_list=lead_names, axes=axes,
    #                           ecg_color='k', fig=fig, label_list=None,
    #                           linewidth=0.1)
    axes, fig = visualise_ecg(ecg_list=[predicted_ecg], lead_name_list=lead_names, axes=axes,
                              ecg_color='magenta', fig=fig, label_list=['Simulated'],
                              linewidth=2.)
    # Plot the clinical trace after the last iteration
    axes, fig = visualise_ecg(ecg_list=[clinical_ecg], lead_name_list=lead_names, axes=axes,
                              ecg_color='lime', fig=fig, label_list=['Clinical'],
                              linewidth=2.)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # axes[0].set_title('12-lead ECGs')
    # plt.show(block=False)
    return plt

if __name__ == '__main__':
    # Step 0: Set the random seed for Reproducibility:
    random_seed_value = 7  # Ensures reproducibility and turns off stochasticity
    np.random.seed(seed=random_seed_value)  # Ensures reproducibility and turns off stochasticity

    # Step 1: Define paths and other environment variables.
    print('Step 1: Define paths and other environment variables.')
    # Define the subject name used to navigate the file system
    anatomy_subject_name = 'DTI004'
    ecg_subject_name = 'DTI004'
    # General settings:
    source_resolution = 'coarse'
    verbose = False
    # Input Paths:
    # data_dir = 'Cardiac-Digital-Twin/example_data/meta_data/'
    data_dir = 'data/meta_data/meta_data/'
    clinical_data_filename = 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    clinical_data_filename_path = data_dir + clinical_data_filename
    geometric_data_dir = data_dir + 'geometric_data/'


    # Output Paths:
    experiment_type = 'simulation_QT'
    AP_type = 'MitchellSchaefferEP'  # choose the action potential model
    # Define the electrophysiology model to use:
    if AP_type == 'MitchellSchaefferEP':
        ep_model = 'MitchellSchaefferEP'
    else:
        if anatomy_subject_name == 'DTI024':
            ep_model = 'GKs5_GKr0.5_tjca60_CL_909'
        elif anatomy_subject_name == 'DTI032':
            ep_model = 'GKs5_GKr0.5_tjca60_CL_810'
        elif anatomy_subject_name == 'DTI004':
            ep_model = 'GKs5_GKr0.5_tjca60_CL_1250'
        else:
            ep_model = 'GKs5_GKr0.5_tjca60_CL_'
        gradient_ion_channel_list = ['sf_IKs']
        gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
        # APD dictionary configuration:
        cellular_stim_amp = 11
        cellular_model_convergence = 'not_converged'
        stimulation_protocol = 'diffusion'


        cellular_data_relative_path = 'cellular_data/' + cellular_model_convergence + '_' + stimulation_protocol + '_' + str(
            cellular_stim_amp) + '_' + gradient_ion_channel_str + '_' + ep_model + '/'
        cellular_data_dir_complete = data_dir + cellular_data_relative_path
        print('Using cellular data from: ', cellular_data_dir_complete)
    # Build results folder structure
    results_dir_root = data_dir + 'results/'
    if not os.path.exists(results_dir_root):
        os.mkdir(results_dir_root)
    results_dir_part = results_dir_root + experiment_type + '_data/'
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
    results_dir_part = results_dir_part + anatomy_subject_name + '/'
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
        
    if AP_type != 'MitchellSchaefferEP':
        results_dir_part = results_dir_part + 'qt_' + gradient_ion_channel_str + '_' + ep_model + '/'
    else:
        results_dir_part = results_dir_part + 'ms_ep/'
        
    if not os.path.exists(results_dir_part):
        os.mkdir(results_dir_part)
    # Use date to name the result folder to preserve some history of results
    current_month_text = datetime.now().strftime('%h')  # Feb
    current_year_full = datetime.now().strftime('%Y')  # 2018
    # results_dir = results_dir_part + 'qt_' + gradient_ion_channel_str + '_' + ep_model + '/smoothing_fibre_128_64_05/' #+ '/smoothing_fibre_256_64_05/'
    results_dir = results_dir_part + current_month_text + '_' + current_year_full + '/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # save ecg
    visualisation_dir = results_dir + 'checkpoint/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    figure_result_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_ecg.png'
        
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'

    # Step 2: Create Cellular Electrophysiology model.
    if AP_type == 'MitchellSchaefferEP':
        apd_min_min = 180
        apd_max_max = 400
        # apd_max_value = 278
        # apd_min_value = 190
        apd_resolution = 1
        endo_celltype_name = 'endo'
        # epi_celltype_name = 'epi'
        list_celltype_name = [endo_celltype_name]
        cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
                                                        apd_resolution=apd_resolution, cycle_length=500,
                                                        list_celltype_name=list_celltype_name, verbose=verbose,
                                                        vm_max=1., vm_min=0.)
    else:
        print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
        # Arguments for cellular model:
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

    # Step 3: Generate an Eikonal-friendly geometry.
    t_start = time.time()
    print('Step 3: Generate a cardiac geometry that can run the Eikonal.')
    # Argument setup
    vc_ab_cut_name = get_vc_ab_cut_name()
    vc_aprt_name = get_vc_aprt_name()
    vc_rt_name = get_vc_rt_name()
    vc_rvlv_name = get_vc_rvlv_name()
    vc_tm_name = get_vc_tm_name()
    # vc_tv_name = get_vc_tv_name()
    # vc_rvlv_binary_name = get_vc_rvlv_binary_name()
    vc_name_list = [vc_ab_cut_name, vc_aprt_name, vc_rt_name, vc_rvlv_name, vc_tm_name]#, vc_rvlv_binary_name]

    # TODO: endo_celltype_name
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 1.]}}#, epi_celltype_name: {vc_tm_name: [0., 0.3]}}
        
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                            conduction_system=EmptyConductionSystem(verbose=verbose),
                            geometric_data_dir=geometric_data_dir, resolution=source_resolution,
                            subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)

    print('The loading of the geometry and generation of the cardiac conduction system took ', round((time.time()-t_start)/60., 2), ' min.')

    # Step 4: Create conduction system for the propagation model to be initialised.
    print('Step 4: Create rule-based Purkinje network using ventricular coordinates.')
    # Arguments for Conduction system:
    approx_djikstra_purkinje_max_path_len = 200
    lv_inter_root_node_distance = 3.  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    rv_inter_root_node_distance = 3.  # 1.5 cm    # TODO: Calibrate this hyper-parameter using sensitivity analysis
    # Create conduction system
    conduction_system = PurkinjeSystemVC(
        approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len,
        geometry=geometry, lv_inter_root_node_distance=lv_inter_root_node_distance,
        rv_inter_root_node_distance=rv_inter_root_node_distance,
        verbose=verbose)
    # Assign conduction_system to its geometry
    geometry.set_conduction_system(conduction_system)

    # (Optional) Save candidate Purkinje system as .vtk file
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

    # (Only for QT) Step 5: Prepare smoothing configuration to resemble diffusion effects
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

    # Step 6: Create Eikonal instance. Eikonal will require a conduction and an Eikonal-friendly mesh on creation.
    print('Step 6: Create propagation model instance.')
    # Arguments for propagation model:
    fibre_speed_name = get_fibre_speed_name()
    sheet_speed_name = get_sheet_speed_name()
    normal_speed_name = get_normal_speed_name()
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

    # Step 7: Create Whole organ Electrophysiology model.
    print('Step 6: Create ECG calculation method.')
    # Create electrophysiology instance
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
    low_freq_cut = 0.001  # 0.5
    high_freq_cut = 100  # 150
    lead_names = get_unique_lead_name_list()
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

    # Step 9: Define instance of the simulation method.
    print('Step 9: Define instance of the simulation method.')
    simulator_ecg = SimulateECGwithLATmax(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    simulator_ep = SimulateEP(electrophysiology_model=electrophysiology_model, verbose=verbose)

    # TODO:Step 10: Define Adapter to translate between theta and parameters.
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

    # Step 11: Create evaluator_ecg.
    print('Step 11: Create evaluator_ecg and evaluator_ep')
    evaluator_ecg = ParameterSimulator(adapter=adapter, simulator=simulator_ecg, verbose=verbose)
    evaluator_ep = ParameterSimulator(adapter=adapter, simulator=simulator_ep, verbose=verbose)

    # Step 12: Define simulation parameters
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
    parameter_particle = np.concatenate((propagation_parameter_example, electrophysiology_parameter_example))


    # Step 13: Simulate selected parameters.
    # print('Step 13: Simulate selected parameters.')
    # Simulate local activation times
    lat_simulation, vm_simulation = evaluator_ep.simulate_parameter_particle(
        parameter_particle=parameter_particle)
    # Simulate 12-lead ECGs
    healthy_predicted_ecg, max_lat = evaluator_ecg.simulate_parameter_particle(
        parameter_particle=parameter_particle)

    # Step 14: Visualise simulated activation map.
    # print('Step 14: Visualise simulated activation map.')
    plot_geometry(xyz=geometry.get_node_xyz(), surf=extract_surface(geometry.get_tetra()), lat_simulation=lat_simulation)

    # Step 15: Visualise ECGs for the final population.
    # print('Step 15: Visualise simulated ECG.')
    plot_ecg(predicted_ecg=healthy_predicted_ecg, clinical_ecg=clinical_ecg, lead_names=lead_names)
    plt.savefig(figure_result_file_name, dpi=200)
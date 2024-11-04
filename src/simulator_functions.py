import os
import time
import numpy as np

from geometry_functions import EikonalGeometry
from propagation_models import EikonalDjikstraTet, PrescribedLAT
from cellular_models import CellularModelBiomarkerDictionary
from ecg_functions import PseudoEcgTetFromVM
from electrophysiology_functions import ElectrophysiologyAPDmap
from path_config import get_path_mapping


class Simulator:
    def __init__(self, verbose):
        if verbose:
            print('Initialising Simulator')
        self.verbose = verbose

    def simulate_particle(self, parameter):
        raise NotImplementedError

    def simulate_population(self, population_parameter):
        raise NotImplementedError


class SimulateEP(Simulator):
    def __init__(self, electrophysiology_model, verbose):
        super().__init__(verbose=verbose)
        self.electrophysiology_model = electrophysiology_model

    def simulate_particle(self, parameter_particle_modules_dict):
        lat_simulation, vm_simulation = self.electrophysiology_model.simulate_electrophysiology(
            parameter_particle_modules_dict=parameter_particle_modules_dict)
        return lat_simulation, vm_simulation

    def simulate_population(self, parameter_population_modules_dict):
        lat_population, vm_population = self.electrophysiology_model.simulate_electrophysiology_population(
            parameter_population_modules_dict=parameter_population_modules_dict)
        return lat_population, vm_population

    def biomarker_particle(self, parameter_particle_modules_dict):
        return self.electrophysiology_model.biomarker_electrophysiology(parameter_particle_modules_dict)

    # def get_nb_candidate_root_nodes(self):
    #     return self.electrophysiology_model.get_nb_candidate_root_node()


class SimulateECG(SimulateEP):
    def __init__(self, ecg_model, electrophysiology_model, verbose):
        super().__init__(electrophysiology_model=electrophysiology_model, verbose=verbose)
        self.ecg_model = ecg_model

    def simulate_particle(self, parameter_particle_modules_dict):
        lat_simulation, vm_simulation = self.electrophysiology_model.simulate_electrophysiology(
            parameter_particle_modules_dict=parameter_particle_modules_dict)
        ecg_simulation = self.ecg_model.calculate_ecg(lat=lat_simulation, vm=vm_simulation)
        return ecg_simulation

    def simulate_population(self, parameter_population_modules_dict):
        lat_population, vm_population = self.electrophysiology_model.simulate_electrophysiology_population(
            parameter_population_modules_dict=parameter_population_modules_dict)
        ecg_population = self.ecg_model.calculate_ecg_population(lat_population=lat_population, vm_population=vm_population)
        return ecg_population

    def visualise_simulation_population(self, discrepancy_population, parameter_population_modules_dict):
        ecg_population = self.simulate_population(parameter_population_modules_dict=parameter_population_modules_dict)
        # TODO replace this function by the postporcessing visualisation of the ECG
        return self.ecg_model.visualise_ecg(discrepancy_population=discrepancy_population, ecg_population=ecg_population)


class SimulateECGwithLATmax(SimulateEP):
    def __init__(self, ecg_model, electrophysiology_model, verbose):
        super().__init__(electrophysiology_model=electrophysiology_model, verbose=verbose)
        self.ecg_model = ecg_model

    def simulate_particle(self, parameter_particle_modules_dict):
        lat_simulation, vm_simulation = self.electrophysiology_model.simulate_electrophysiology(
            parameter_particle_modules_dict=parameter_particle_modules_dict)
        ecg_simulation = self.ecg_model.calculate_ecg(lat=lat_simulation, vm=vm_simulation)
        return (ecg_simulation, np.amax(lat_simulation))

    def simulate_population(self, parameter_population_modules_dict):
        lat_population, vm_population = self.electrophysiology_model.simulate_electrophysiology_population(
            parameter_population_modules_dict=parameter_population_modules_dict)
        ecg_population = self.ecg_model.calculate_ecg_population(lat_population=lat_population, vm_population=vm_population)
        return (ecg_population, np.amax(lat_population, axis=1))

    def visualise_simulation_population(self, discrepancy_population, parameter_population_modules_dict):
        (ecg_population, _) = self.simulate_population(parameter_population_modules_dict=parameter_population_modules_dict)
        # TODO replace this function by the postporcessing visualisation of the ECG
        return self.ecg_model.visualise_ecg(discrepancy_population=discrepancy_population, ecg_population=ecg_population)


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
    ecg_subject_name = 'DTI004'  # Allows using a different ECG for the personalisation than for the anatomy
    print('ecg_subject_name: ', ecg_subject_name)
    resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    cellular_data_dir = data_dir + 'cellular_data/'
    clinical_data_filename = data_dir + 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    clinical_qrs_offset = 100 # ms TODO This could be calculated automatically and potentially, the clinical ECG could be trimmed to start with the QRS at time zero
    geometric_data_dir = data_dir + 'geometric_data/'
    # Intermediate Paths: # e.g., results from the QRS inference
    intermediate_dir = path_dict["results_path"] + 'personalisation_data/' + anatomy_subject_name + '/'
    lat_prescribed_file_name = intermediate_dir + anatomy_subject_name + '_' + resolution + '_nodefield_inferred-lat.csv'
    # Output Paths:
    results_dir = path_dict["results_path"] + 'personalisation_data/' + anatomy_subject_name + '/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    theta_result_file_name = results_dir + anatomy_subject_name + '_' + resolution + '_inferred_population.csv'
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
    stimulation_protocol = 'diffusion'
    cellular_stim_amp = 11
    cellular_data_dir_complete = cellular_data_dir + 'not_converged_' + stimulation_protocol + '_' + str(
        cellular_stim_amp) + '_GKs_only/'
    cellular_model_name = 'torord_calibrated_pom_1000Hz'
    cycle_length = 1000
    endo_celltype_name = 'endo'
    epi_celltype_name = 'epi'
    list_celltype_name = [endo_celltype_name, epi_celltype_name]
    # TODO Modifiy ToRORd code to save the following biomarkers and allow the action potentials to have pre-activation differences as well as to be trimmed to the essential minimum
    biomarker_upstroke_name = 'activation_time'
    apd_biomarker_name = 'apd90'
    # biomarker_len_name = 'signal_len'
    # Create cellular model instance.
    cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                      biomarker_apd90_name=apd_biomarker_name,
                                                      cellular_data_dir=cellular_data_dir_complete,
                                                      cellular_model_name=cellular_model_name,
                                                      list_celltype_name=list_celltype_name, verbose=verbose)
    # Pre-assign celltype spatial correspondence.
    list_transmural_celltype = list_celltype_name
    # Clear Arguments to prevent Argument recycling
    cellular_data_dir = None
    cellular_data_dir_complete = None
    cellular_stim_amp = None
    cycle_length = None
    stimulation_protocol = None
    ####################################################################################################################
    # Step 3: Generate a cardiac geometry that cannot run the Eikonal.
    # Argument setup: (in Alphabetical order)
    vc_ab_name = 'ab'
    vc_aprt_name = 'aprt'
    vc_tm_name = 'tm'
    vc_rt_name = 'rt'
    vc_rvlv_name = 'rvlv'
    vc_tv_name = 'tv'
    vc_name_list = [vc_ab_name, vc_tm_name, vc_rt_name, vc_tv_name]
    celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 0.3]}, epi_celltype_name: {vc_tm_name: [0.3, 1.]}}
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                               conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    anatomy_subject_name = None
    geometric_data_dir = None
    list_celltype_name = None
    resolution = None
    vc_name_list = None
    ####################################################################################################################
    # Step 4: Create propagation model instance, this will be a static dummy propagation model.
    propagation_parameter_name_list_in_order = []
    propagation_model = PrescribedLAT(geometry=geometry, lat_prescribed_file_name=lat_prescribed_file_name,
                                      module_name=propagation_module_name, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    lat_prescribed_file_name = None
    ####################################################################################################################
    # Step 5: Create Whole organ Electrophysiology model.
    apd_max_name = 'apd_max'
    apd_min_name = 'apd_min'
    g_vc_ab_name = vc_ab_name
    g_vc_aprt_name = vc_aprt_name
    g_vc_rvlv_name = vc_rvlv_name
    g_vc_tm_name = vc_tm_name
    electrophysiology_parameter_name_list_in_order = [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name,
                                                      g_vc_rvlv_name, g_vc_tm_name]
    # Spatial and temporal smoothing parameters:
    smoothing_count = 5
    smoothing_ghost_distance_to_self = 0.05  # cm # This parameter enables to control how much spatial smoothing happens and
    # makes sure that the spatial smoothing is based on distance instead of adjacentcies - smooth twice
    smoothing_past_present_window = np.asarray([0.05, 0.95])  # Weight the past as 10% and the present as 90%
    electrophysiology_model = ElectrophysiologyAPDmap(apd_max_name=apd_max_name, apd_min_name=apd_min_name,
                                                      cellular_model=cellular_model,
                                                      parameter_name_list_in_order=electrophysiology_parameter_name_list_in_order,
                                                      module_name=electrophysiology_module_name,
                                                      propagation_model=propagation_model,
                                                      smoothing_count=smoothing_count,
                                                      smoothing_ghost_distance_to_self=smoothing_ghost_distance_to_self,
                                                      smoothing_past_present_window=smoothing_past_present_window,
                                                      verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    cellular_model = None
    propagation_model = None
    smoothing_count = None
    smoothing_ghost_distance_to_self = None
    smoothing_past_present_window = None
    ####################################################################################################################
    # Step 6: Create ECG calculation method. In this case, the ecg will calculate only the QRS and will use a step
    # function as the AP's upstroke.
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
    freq_cut = 150
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    nb_leads = 8
    # Read clinical data
    untrimmed_clinical_ecg_raw = np.genfromtxt(clinical_data_filename, delimiter=',')
    clinical_ecg_raw = untrimmed_clinical_ecg_raw[:, clinical_qrs_offset:]
    untrimmed_clinical_ecg_raw = None  # Clear Arguments to prevent Argument recycling
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
    frequency = None
    freq_cut = None
    geometry = None  # Clear Geometry
    lead_names = None
    max_len_ecg = None
    max_len_qrs = None
    max_len_st = None
    nb_leads = None
    normalise = None
    zero_align = None
    ####################################################################################################################
    # Step 7: Define instance of the simulation method.
    simulator = SimulateECG(ecg_model=ecg_model, electrophysiology_model=electrophysiology_model, verbose=verbose)
    # Clear Arguments to prevent Argument recycling
    electrophysiology_model = None
    ecg_model = None
    ####################################################################################################################
    # Last Step: Test ECG simulation using new parameters.
    parameter_particle_modules_dict = {propagation_module_name: [], electrophysiology_module_name: [300, 250, 1., 1., 1., 1.]}
    # [apd_max_name, apd_min_name, g_vc_ab_name, g_vc_aprt_name, g_vc_rvlv_name, g_vc_tm_name]
    print('Simulate ECG')
    t = time.time()
    ecg_simulation = simulator.simulate_particle(
        parameter_particle_modules_dict=parameter_particle_modules_dict)
    t = time.time() - t
    print('Simulation time: ', t)
    simulator.ecg_model.visualise_ecg(discrepancy_population=np.array([1., 2.]),
                                                                 ecg_population=np.concatenate(
                                                                     (ecg_simulation[np.newaxis, :, :],
                                                                      ecg_simulation[np.newaxis, :, :]), axis=0))
    print('Visualisation completed')
    quit()

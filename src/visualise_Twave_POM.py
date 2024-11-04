"""This script visualises the POM considered for the inference of repolarisation properties from the T wave"""
import os
from warnings import warn
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from io_functions import export_ensight_timeseries_case, read_dictionary, save_pandas, \
    write_geometry_to_ensight_with_fields, save_csv_file
from geometry_functions import RawEmptyCardiacGeoTet
from postprocess_functions import generate_repolarisation_map, visualise_action_potential_population
from utils import map_indexes, remap_pandas_from_row_index
from adapter_theta_params import AdapterThetaParams, RoundTheta
from cellular_models import CellularModelBiomarkerDictionary
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
        'Caution, all the hyper-parameters are set assuming resolutions of 1000 Hz in all time-series.')
    # TODO: make hyper-parameters relative to the time series resolutions.
    # TODO: enable having different resolutions in different time-series in the code.
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
    source_resolution = 'coarse'
    target_resolution = 'fine'
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
    results_dir = results_dir_root + experiment_type + '_data/' + anatomy_subject_name + '/twave_' \
                  + gradient_ion_channel_str + '_' + ep_model + '/only_endo/'
    # Read hyperparamter dictionary
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_hyperparameter.txt'
    hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
    # Output Paths:
    visualisation_dir = results_dir + 'cellular_figure/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    figure_aligned_action_potential_population_file_name = visualisation_dir + anatomy_subject_name + '_aligned_action_potential_population.png'
    # Clear Arguments to prevent Argument recycling
    clinical_data_filename = None
    data_dir = None
    ecg_subject_name = None
    qrs_lat_prescribed_filename = None
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
    cellular_model = CellularModelBiomarkerDictionary(biomarker_upstroke_name=biomarker_upstroke_name,
                                                      biomarker_apd90_name=biomarker_apd90_name,
                                                      biomarker_celltype_name=biomarker_celltype_name,
                                                      cellular_data_dir=cellular_data_dir_complete,
                                                      cellular_model_name=cellular_model_name,
                                                      list_celltype_name=list_celltype_name, verbose=verbose)
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
    # Step 3: Visualise APs from the ToROrd APD dictionary.
    # Read hyperparameters
    celltype_vc_info = hyperparameter_dict['celltype_vc_info']
    considered_list_celltype_name = list(celltype_vc_info.keys())   # Only visualise celltypes used in the experiments
    # Initialise arguments for plotting
    axes = None
    fig = None
    action_potential_color_list = ['blue', 'red']
    # Plot the aligned population of available action potentials
    for celltype_name_i in range(len(considered_list_celltype_name)):
        celltype_name = considered_list_celltype_name[celltype_name_i]
        action_potential_color = action_potential_color_list[celltype_name_i]
        print('celltype_name ', celltype_name)
        aligned_action_potential_population = cellular_model.get_aligned_action_potential_from_celltype_name(
            celltype_name=celltype_name)
        axes, fig = visualise_action_potential_population(action_potential_list=aligned_action_potential_population,
                                                          axes=axes, action_potential_color=action_potential_color,
                                                          fig=fig, label=celltype_name)
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show(block=False)
    fig.savefig(figure_aligned_action_potential_population_file_name)
    print('Saved aligned action potential population figure: ', figure_aligned_action_potential_population_file_name)
    # Clear Arguments to prevent Argument recycling.
    action_potential_color_list = None
    cellular_model = None
    figure_aligned_action_potential_population_file_name = None
    list_celltype_name = None
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

    # EOF
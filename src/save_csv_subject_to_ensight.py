"""Run personalisation on the full 12-lead ECG beat recording"""
import sys
import multiprocessing
import os
# import time
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from src.cellular_models import MitchellSchaefferAPDdictionary

# import pymp

if __name__ == '__main__':
    anatomy_subject_name = 'DTI004'
    print('anatomy_subject_name: ', anatomy_subject_name)
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
    from geometry_functions import RawEmptyCardiacGeoTet, EikonalGeometry
    from path_config import get_path_mapping
    from io_functions import write_geometry_to_ensight_with_fields

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
    geometric_data_dir = data_dir + 'geometric_data/'
    visualisation_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + resolution + '/'
    ####################################################################################################################
    # Step 3: Generate a cardiac geometry that can run the Eikonal.
    # Argument setup: (in Alphabetical order)
    print('Step 3: Generate a cardiac geometry that cannot run the Eikonal.')
    # Create geometry with a dummy conduction system to allow initialising the geometry.
    geometry = RawEmptyCardiacGeoTet(conduction_system=EmptyConductionSystem(verbose=verbose),
                               geometric_data_dir=geometric_data_dir, resolution=resolution,
                               subject_name=anatomy_subject_name, verbose=verbose)

    # # Create cellular model instance.
    # apd_min_min = 200
    # apd_max_max = 400
    # apd_resolution = 1
    # endo_celltype_name = 'endo'
    # cellular_model = MitchellSchaefferAPDdictionary(apd_max=apd_max_max, apd_min=apd_min_min,
    #                                                 apd_resolution=apd_resolution, cycle_length=800,
    #                                                 list_celltype_name=[endo_celltype_name], verbose=verbose,
    #                                                 vm_max=1., vm_min=0.)
    # vc_ab_name = get_vc_ab_name()
    # vc_ab_cut_name = get_vc_ab_cut_name()
    # vc_aprt_name = get_vc_aprt_name()
    # vc_rt_name = get_vc_rt_name()
    # vc_rvlv_name = get_vc_rvlv_name()
    # vc_tm_name = get_vc_tm_name()
    # vc_name_list = [vc_ab_name, vc_ab_cut_name, vc_aprt_name, vc_rt_name, vc_rvlv_name,
    #                 vc_tm_name]
    #
    # celltype_vc_info = {endo_celltype_name: {vc_tm_name: [0., 1.]}}
    # geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
    #                 conduction_system=EmptyConductionSystem(verbose=verbose),
    #                 geometric_data_dir=geometric_data_dir, resolution=resolution,
    #                 subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)

    write_geometry_to_ensight_with_fields(geometry=geometry, node_field_list=[],
                                          node_field_name_list=[],
                                          subject_name=anatomy_subject_name + '_' + resolution + '_checkpoint',
                                          verbose=verbose,
                                          visualisation_dir=visualisation_dir)
    print('Saved geometry before inference in ', visualisation_dir)
    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

# EOF

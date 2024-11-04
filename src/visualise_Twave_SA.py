import os
from warnings import warn

import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from io_functions import read_dictionary, read_pandas
from postprocess_functions import visualise_tornado_sa, visualise_scatter_sa, visualise_heatmap_sa
from path_config import get_path_mapping
from ecg_functions import correct_qt_interval

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
    source_resolution = 'coarse'
    target_resolution = 'coarse'
    verbose = True
    # Input Paths:
    data_dir = path_dict["data_path"]
    cellular_data_dir = data_dir + 'cellular_data/'
    # clinical_data_filename = data_dir + 'clinical_data/' + ecg_subject_name + '_clinical_full_ecg.csv'
    geometric_data_dir = data_dir + 'geometric_data/'
    # Intermediate Paths: # e.g., results from the QRS inference
    # qrs_lat_prescribed_file_name = path_dict["results_path"] + 'personalisation_data/' + anatomy_subject_name + '/qrs/' \
    #                                + anatomy_subject_name + '_' + source_resolution + '_nodefield_inferred-lat.csv'
    experiment_type = 'sa'
    ep_model = 'GKs5_GKr0.6_tjca60'
    gradient_ion_channel_list = ['sf_IKs']
    gradient_ion_channel_str = '_'.join(gradient_ion_channel_list)
    results_dir = path_dict["results_path"] + experiment_type + '_data/' + anatomy_subject_name + '/twave_' \
                  + gradient_ion_channel_str + '_' + ep_model + '/smoothing_fibre/' #'/only_endo/'
    # Read hyperparamter dictionary
    hyperparameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_hyperparameter.txt'
    hyperparameter_dict = read_dictionary(filename=hyperparameter_result_file_name)
    result_tag = hyperparameter_dict['result_tag']
    theta_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_theta_population.csv'
    parameter_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_parameter_population.csv'
    sobol_indicies_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_sobol_indicies.csv'
    qoi_result_file_name = results_dir + anatomy_subject_name + '_' + source_resolution + '_' + result_tag + '_qoi_population.csv'
    # Output Paths:
    visualisation_dir = results_dir + 'sa_figures/'
    if not os.path.exists(visualisation_dir):
        os.mkdir(visualisation_dir)
    # Tornado
    figure_average_tornado_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_average_tornado.png'
    figure_per_lead_tornado_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_per_lead_tornado.png'
    # Scatter
    figure_average_scatter_theta_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_average_scatter_theta_vs_qoi.png'
    figure_per_lead_scatter_theta_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_per_lead_scatter_theta_vs_qoi.png'
    figure_average_scatter_qoi_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_average_scatter_qoi_vs_qoi.png'
    figure_per_lead_scatter_qoi_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_per_lead_scatter_qoi_vs_qoi.png'
    # Heatmap
    figure_average_heatmap_theta_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_average_heatmap_theta_vs_qoi.png'
    figure_per_lead_heatmap_theta_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_per_lead_heatmap_theta_vs_qoi.png'
    figure_average_heatmap_qoi_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_average_heatmap_qoi_vs_qoi.png'
    figure_per_lead_heatmap_qoi_vs_qoi_file_name = visualisation_dir + anatomy_subject_name + '_' + source_resolution + '_per_lead_heatmap_qoi_vs_qoi.png'
    # Module names:
    propagation_module_name = 'propagation_module'
    electrophysiology_module_name = 'electrophysiology_module'
    # Read hyperparameters
    # clinical_data_filename = hyperparameter_dict['clinical_data_filename']
    # clinical_qrs_offset = hyperparameter_dict['clinical_qrs_offset']
    # Clear Arguments to prevent Argument recycling
    clinical_data_dir_tag = None
    data_dir = None
    ecg_subject_name = None
    lat_dir = None
    results_dir = None
    ####################################################################################################################
    # Step 11: Read the values SA and QOIs.
    print('Step 11: Read the SA and QOI results.')
    # Read hyperparameters
    nb_index_columns = hyperparameter_dict['nb_index_columns']
    # Read data
    theta_population_df = read_pandas(filename=theta_result_file_name)
    qoi_population_df = read_pandas(filename=qoi_result_file_name)
    # Don't use the custom function read_pandas for the next one, because it needs to specify a multiindex
    sobol_indices_df = pd.read_csv(sobol_indicies_result_file_name, delimiter=',', index_col=list(range(nb_index_columns)))
    ####################################################################################################################
    # Step 12: Generate tornado figures from SA results.
    print('Step 12: Generate the figures from SA and QOI results.')
    # Read hyperparameters
    qoi_name_list_for_average = hyperparameter_dict['qoi_name_list_for_average']
    qoi_name_list_per_lead = hyperparameter_dict['qoi_name_list_per_lead']
    sobol_name_list_in_order = hyperparameter_dict['sobol_name_list_in_order']
    theta_name_list = hyperparameter_dict['theta_name_list_in_order']
    value_column_name = hyperparameter_dict['value_column_name']
    qt_dur_name = hyperparameter_dict['qt_dur_name']
    t_pe_name = hyperparameter_dict['t_pe_name']
    t_peak_name = hyperparameter_dict['t_peak_name']
    qtpeak_dur_name = hyperparameter_dict['qtpeak_dur_name']
    t_polarity_name = hyperparameter_dict['t_polarity_name']
    tpeak_dispersion_name = hyperparameter_dict['tpeak_dispersion_name']
    # Define the list of QOIs desired in the figures
    qoi_name_list_for_average_custom = [qt_dur_name, t_pe_name, t_peak_name, t_polarity_name]
    for qoi_name in qoi_name_list_for_average_custom:
        assert qoi_name in qoi_name_list_for_average
    qoi_name_list_for_average = qoi_name_list_for_average_custom
    warn('This is no longer needed!!!')
    # TODO remove the following correction if QTc has already been calculated automatically
    # Correct from QT to QTc
    if anatomy_subject_name == 'DTI004':  # Subject 2
        heart_rate = 48
    # qt_dur_array = qoi_population_df[qt_dur_name].values
    qoi_population_df[qt_dur_name] = correct_qt_interval(heart_rate=heart_rate, qt_dur=qoi_population_df[qt_dur_name].values)
    # print('qtc_dur_array ', qtc_dur_array)
    # raise()
    # correct_qt_interval(heart_rate, qt_dur)
    # qoi_population_df[qt_dur_name] = qoi_population_df[qt_dur_name]

    # print('qoi_name_list_for_average ', qoi_name_list_for_average)
    # raise()

    # HEATMAP S2
    # print('sobol_indices_df ', list(sobol_indices_df.index))
    # sobol_indices_df_S2 = sobol_indices_df.xs('S2', level=1)
    # print('sobol_indices_df_S2 ', sobol_indices_df_S2)
    # aux_sobol = sobol_indices_df_S2.loc['qt_dur']
    # print(aux_sobol)
    # quit()

    # Split the QOIs into subgroups for each lead, so that the "per_lead" figures are legible.
    nb_leads = hyperparameter_dict['nb_leads']
    nb_qoi_per_lead = int(math.ceil(len(qoi_name_list_per_lead) / nb_leads))
    # TODO delete the following code
    # qt_dur_name_list = []
    # t_pe_name_list = []
    # t_peak_name_list = []
    # qtpeak_dur_name_list = []
    # t_polarity_name_list = []
    # # qoi_name_list_per_lead = []
    # for lead_i in range(nb_leads):
    #     # qoi_name_list_per_lead.append(qt_dur_name + '_' + str(lead_i))
    #     # qoi_name_list_per_lead.append(t_pe_name + '_' + str(lead_i))
    #     # qoi_name_list_per_lead.append(t_peak_name + '_' + str(lead_i))
    #     # qoi_name_list_per_lead.append(qtpeak_dur_name + '_' + str(lead_i))
    #     # qoi_name_list_per_lead.append(t_polarity_name + '_' + str(lead_i))
    #     qt_dur_name_list.append(get_biomarker_lead_name(biomarker_lead_name=qt_dur_name, lead_i=lead_i))
    #     t_pe_name_list.append(get_biomarker_lead_name(biomarker_lead_name=t_pe_name, lead_i=lead_i))
    #     t_peak_name_list.append(get_biomarker_lead_name(biomarker_lead_name=t_peak_name, lead_i=lead_i))
    #     qtpeak_dur_name_list.append(get_biomarker_lead_name(biomarker_lead_name=qtpeak_dur_name, lead_i=lead_i))
    #     t_polarity_name_list.append(get_biomarker_lead_name(biomarker_lead_name=t_polarity_name, lead_i=lead_i))
    # qoi_name_list_per_lead = qt_dur_name_list + t_pe_name_list + t_peak_name_list + qtpeak_dur_name_list \
    #                          + t_polarity_name_list
    # qoi_name_list_per_lead = []
    # for lead_i in range(nb_leads):
    #     qoi_name_list_per_lead.append(qt_dur_name + '_' + str(lead_i))
    #     qoi_name_list_per_lead.append(t_pe_name + '_' + str(lead_i))
    #     qoi_name_list_per_lead.append(t_peak_name + '_' + str(lead_i))
    #     qoi_name_list_per_lead.append(qtpeak_dur_name + '_' + str(lead_i))
    #     qoi_name_list_per_lead.append(t_polarity_name + '_' + str(lead_i))
    # TODO delete the above code
    # qoi_name_list_list_per_lead = []
    # for lead_i in range(nb_leads):
    #     qoi_name_list_list_per_lead.append(
    #         qoi_name_list_per_lead[
    #         lead_i * nb_qoi_per_lead:min(len(qoi_name_list_per_lead), (lead_i + 1) * nb_qoi_per_lead)])

    # # HEATMAP
    # # Generate the Heatmap SA figures Theta VS QOI
    # # Average
    # x_axis_df = theta_population_df[
    #     theta_name_list]  # These dataframes have number of rows as number of samples in the SA and columns for each data they store.
    # y_axis_df = qoi_population_df[
    #     qoi_name_list_for_average]  # These dataframes have number of rows as number of samples in the SA and columns for each data they store.
    # fig = visualise_heatmap_sa(x_axis_df=x_axis_df, y_axis_df=y_axis_df)
    # fig.savefig(figure_average_heatmap_theta_vs_qoi_file_name)
    # print('Saved heatmap average figure: ', figure_average_heatmap_theta_vs_qoi_file_name)
    # # Per lead SA results
    # pass  # TODO Would this be interesting to visualise?
    # # TODO To visualise this we will need to split the results for each lead by searching the lead names inside the qoi names
    # # Generate the Heatmap SA figures QOI VS QOI
    # # Average
    # x_axis_df = qoi_population_df[qoi_name_list_for_average]
    # y_axis_df = qoi_population_df[qoi_name_list_for_average]
    # fig = visualise_heatmap_sa(x_axis_df=x_axis_df, y_axis_df=y_axis_df)
    # fig.savefig(figure_average_heatmap_qoi_vs_qoi_file_name)
    # print('Saved heatmap average figure: ', figure_average_heatmap_qoi_vs_qoi_file_name)
    # # Per lead SA results
    # pass  # TODO Would this be interesting to visualise?
    # # TODO To visualise this we will need to split the results for each lead by searching the lead names inside the qoi names

    # SCATTER
    # Generate the Scatter SA figures Theta VS QOI
    # Average
    x_axis_df = theta_population_df[
        theta_name_list]  # These dataframes have number of rows as number of samples in the SA and columns for each data they store.
    y_axis_df = qoi_population_df[
        qoi_name_list_for_average]  # These dataframes have number of rows as number of samples in the SA and columns for each data they store.
    fig = visualise_scatter_sa(x_axis_df=x_axis_df, y_axis_df=y_axis_df)
    fig.savefig(figure_average_scatter_theta_vs_qoi_file_name)
    print('Saved scatter average figure: ', figure_average_scatter_theta_vs_qoi_file_name)
    # Per lead SA results
    pass  # TODO Would this be interesting to visualise?
    # TODO To visualise this we will need to split the results for each lead by searching the lead names inside the qoi names

    # # Generate the Scatter SA figures QOI VS QOI
    # # Average
    # x_axis_df = qoi_population_df[qoi_name_list_for_average]
    # y_axis_df = qoi_population_df[qoi_name_list_for_average]
    # fig = visualise_scatter_sa(x_axis_df=x_axis_df, y_axis_df=y_axis_df)
    # fig.savefig(figure_average_scatter_qoi_vs_qoi_file_name)
    # print('Saved scatter average figure: ', figure_average_scatter_qoi_vs_qoi_file_name)
    # # Per lead SA results
    # pass  # TODO Would this be interesting to visualise?
    # # TODO To visualise this we will need to split the results for each lead by searching the lead names inside the qoi names

    # TORNADO
    # Sobol Total and First indices names
    sobol_indices_name_list_for_tornado = sobol_name_list_in_order[:2]     # This list needs to be in ascending order
    # Generate the Tornado average SA figures
    qoi_name_list_list_for_average = [qoi_name_list_for_average]
    fig = visualise_tornado_sa(qoi_name_list_list=qoi_name_list_list_for_average, sobol_indices_df=sobol_indices_df,
                               sobol_indices_name_list_in_order=sobol_indices_name_list_for_tornado,
                               theta_name_list=theta_name_list, value_column_name=value_column_name)
    fig.savefig(figure_average_tornado_file_name)
    print('Saved tornado average QOI figure: ', figure_average_tornado_file_name)
    # Per lead SA results
    pass  # TODO Would this be interesting to visualise?
    # TODO To visualise this we will need to split the results for each lead by searching the lead names inside the qoi names
    # TODO Dont delete!!
    # TODO think about how to incorporate the fact that the qoi_names may not be in the same order as the code bellow assumes, perhaps we should first sort the qoi_names to be in that order? or use dictionaries?
    # The following function takes a list of list where the each lead has it's own sublist, these sublists can be created with
    # the following piece of code:
    # qoi_name_list_list_per_lead = []
    # for lead_i in range(nb_leads):
    #     qoi_name_list_list_per_lead.append(
    #         qoi_name_list_per_lead[
    #         lead_i * nb_qoi_per_lead:min(len(qoi_name_list_per_lead), (lead_i + 1) * nb_qoi_per_lead)])
    # fig = visualise_tornado_sa(qoi_name_list_list=qoi_name_list_list_per_lead, sobol_indices_df=sobol_indices_df,
    #                            sobol_indices_name_list_in_order=sobol_indices_name_list_for_tornado,
    #                            theta_name_list=theta_name_list, value_column_name=value_column_name)
    # fig.savefig(figure_per_lead_tornado_file_name)
    # print('Saved tornado per lead QOI figure: ', figure_per_lead_tornado_file_name)
    # TODO Dont delete!!

    # # HEATMAP S2
    # sobol_indices_df_S2 = sobol_indices_df.loc['S2']
    # print('sobol_indices_df_S2 ', sobol_indices_df_S2)
    # TODO this is not implemented yet, but the data is being saved

    ####################################################################################################################
    print('END')
    plt.figure()
    plt.show(block=True)

    #EOF



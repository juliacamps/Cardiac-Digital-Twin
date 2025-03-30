import multiprocessing
import os
from warnings import warn

import numba
import numpy as np
import pandas as pd
import pymp

# from postprocess_functions import scatter_visualise_point_cloud

# Environment variables
# TODO change all codes to just use np.nan directly instead of this, becuase future versions of python don't support casting np.nan to integer
nan_value = -2147483648  # min value for int32 #np.array([np.nan]).astype(np.int32)[0]
use_parallel = False

def get_nan_value():
    return nan_value


# MonoAlg3D specific output naming and format
monoalg_vm_file_name_tag = 'Vm.Esca'
monoalg_geo_file_name = 'geometry.geo'
monoalg_drug_folder_name_tag = 'Multiplier_'


def get_monoalg_vm_file_name_tag():
    return monoalg_vm_file_name_tag


def get_monoalg_geo_file_name():
    return monoalg_geo_file_name


def get_monoalg_drug_folder_name_tag():
    return monoalg_drug_folder_name_tag


def get_monoalg_drug_scaling(folder_name):
    monoalg_drug_folder_name_tag = get_monoalg_drug_folder_name_tag()
    drug_scaling_str = folder_name.split(monoalg_drug_folder_name_tag)[-1]
    return float(drug_scaling_str)


def get_monoalg_vm_file_time(filename):
    monoalg_vm_file_name_tag = get_monoalg_vm_file_name_tag()
    time_str = filename.replace(monoalg_vm_file_name_tag, '')
    try:
        time = int(time_str)
    except ValueError:
        raise ValueError("time_str did not contain a number! ", time_str)
    return time


def convert_from_monoalg3D_to_cm_and_translate(monoalg3D_xyz, inference_xyz, scale):
    warn('Hi Julia and Jenny, be super careful with this and check results using Paraview!')
    # scale = np.array([1e+4, 1e+4, 1e+4])
    # scale = np.array([1., 1., 1.])
    monoalg3D_xyz = (monoalg3D_xyz / scale)
    # translate = np.median(inference_xyz, axis=0) - np.median(monoalg3D_xyz, axis=0)
    translate_min = np.amin(inference_xyz, axis=0) - np.amin(monoalg3D_xyz, axis=0)
    translate_max = np.amax(inference_xyz, axis=0) - np.amax(monoalg3D_xyz, axis=0)
    translate = (translate_min + translate_max)/2.
    monoalg3D_xyz = monoalg3D_xyz + translate
    # Note that this is substracting the translations and dividing
    return monoalg3D_xyz


def get_best_str():
    return 'best'


# HPC functions
def run_job(job_file_path):
    os.system('sbatch ' + job_file_path)


def get_parallel_loop(data_size):
    threadsNum = multiprocessing.cpu_count()
    if use_parallel and threadsNum > 1:
        # with pymp.Parallel(min(threadsNum, points_to_map_xyz.shape[0])) as p1:
        p1 = pymp.Parallel(min(threadsNum, data_size))
        iter_gen = p1.range(data_size)
    else:
        print('Running in single thread!')
        iter_gen = range(data_size)
    return iter_gen


# Anatomical landmarks using ventricular coordinates
## Apex to Base (cut, namely, only between apex and base, no valves)
base_ab_cut_value = 1.  # Base value. Note that, ab_cut ventricular coordinate will only cover from apex-to-base and will have invalid values for the valves (usually -1)
apex_ab_cut_value = 0.  # Apex value


def get_base_ab_cut_value():
    return base_ab_cut_value


def get_apex_ab_cut_value():
    return apex_ab_cut_value


# Purkinje and Fast endocaridal ventricular coordinate landmarks
## Apex-to-Base
lv_apical_ab_cut_threshold = 0.4        # This threshold delineates the nodes that are close to the apex in the LV
rv_apical_ab_cut_threshold = 0.2        # This threshold delineates the nodes that are close to the apex in the RV
# Rotational
freewall_center_rt_value = 0.35
freewall_posterior_rt_value = 0.2
freewall_anterior_rt_value = 0.5
assert freewall_posterior_rt_value < freewall_center_rt_value   #
assert freewall_center_rt_value < freewall_anterior_rt_value


def get_lv_apical_ab_cut_threshold():
    return lv_apical_ab_cut_threshold


def get_rv_apical_ab_cut_threshold():
    return rv_apical_ab_cut_threshold


def get_freewall_center_rt_value():
    return freewall_center_rt_value


def get_freewall_posterior_rt_value():
    return freewall_posterior_rt_value


def get_freewall_anterior_rt_value():
    return freewall_anterior_rt_value


# ## Rotational (Cobiveco)
# # Free-wall/Lateral
# freewall_center_rt_value = 0.35
# freewall_posterior_rt_value = 0.2
# freewall_anterior_rt_value = 0.5
# # Septum
# septal_center_rt_value = 0.85
# septal_anterior_rt_value = 0.7
# septal_posterior_rt_value = 1.
# # Paraseptal
# paraseptal_anterior_center_rt_value = 0.6
# paraseptal_posterior_center_rt_value = 0.1
# paraseptal_septal_posterior_rt_value = 0.   # Be careful with this discontinuity in Cobiveco coordinates - Not the same value as septal_posterior_rt_value
# paraseptal_freewall_posterior_rt_value = freewall_posterior_rt_value
# paraseptal_septal_anterior_rt_value = septal_anterior_rt_value
# paraseptal_freewall_anterior_rt_value = freewall_anterior_rt_value


# Purkinje endocardial layer Rules and vc thresholds
'''While the Purkinje can grow higher and is actually delimited by the fast-endocardial layer coverage, this limit is for 
the root nodes!'''
root_node_max_ab_cut_threshold = 0.8     # Purkinje cannot grow above this threshold (80% of apex-to-base)
# purkinje_min_ab_cut_threshold = get_apex_ab_cut_value()      # Purkinje cannot grow bellow this threshold (apex value)


def get_root_node_max_ab_cut_threshold():
    return root_node_max_ab_cut_threshold


# def get_purkinje_min_ab_cut_threshold():
#     return purkinje_min_ab_cut_threshold


# Fast endocardial layer Rules and vc thresholds
'''This threshold also affects how high the his-bundle can grow!'''
endo_fast_and_purkinje_max_ab_cut_threshold = 0.9    # Fast endocardial layer goes as high as threshold (90% of apex-to-base)
assert endo_fast_and_purkinje_max_ab_cut_threshold >= root_node_max_ab_cut_threshold


def get_endo_fast_and_purkinje_max_ab_cut_threshold():
    return endo_fast_and_purkinje_max_ab_cut_threshold


# # Fast/Purkinje region in the ventricles
# ## Apex-to-Base
# ### LV
# lv_endo_fast_ab_min = 0.
# lv_endo_fast_ab_max = 1.
# ### RV
# rv_endo_fast_ab_min = 0.
# rv_endo_fast_ab_max = 1.
# ## Rotational
# ### LV
# lv_endo_fast_rt_min = 0.
# lv_endo_fast_rt_max = 1.
# ### RV
# rv_endo_fast_rt_min = 0.
# rv_endo_fast_rt_max = 1.


# def get_purkinje_max_ab_cut_threshold():
#     return purkinje_max_ab_cut_threshold


# Lead names
lead_I_name = 'I'
lead_II_name = 'II'
unique_limb_lead_name_list = [lead_I_name, lead_II_name]
lead_V1_name = 'V1'
lead_V2_name = 'V2'
lead_V3_name = 'V3'
lead_V4_name = 'V4'
lead_V5_name = 'V5'
lead_V6_name = 'V6'


# def get_lead_I_name():
#     return lead_I_name
#
#
# def get_lead_II_name():
#     return lead_II_name


def get_lead_V1_name():
    return lead_V1_name


precordial_lead_name_list = [lead_V1_name, lead_V2_name, lead_V3_name, lead_V4_name, lead_V5_name, lead_V6_name]
nb_precordial_lead = len(precordial_lead_name_list)


def get_precordial_lead_name_list():
    return precordial_lead_name_list


def get_nb_precordial_lead():
    return nb_precordial_lead


# Lead configurations
unique_lead_name_list = unique_limb_lead_name_list + precordial_lead_name_list  # 8 unique leads
nb_unique_lead = len(unique_lead_name_list)


def get_unique_lead_index(lead_name):
    return unique_lead_name_list.index(lead_name)


def get_lead_V1_index(nb_lead):
    lead_name = get_lead_V1_name()
    if nb_lead == nb_unique_lead:
        lead_index = get_unique_lead_index(lead_name=lead_name)
    else:
        raise NotImplementedError
    return lead_index


def get_unique_lead_name_list():
    return unique_lead_name_list


def get_nb_unique_lead():
    return nb_unique_lead


# Healthy QRS number of positive peaks
qrs_nb_positive_peak_healthy_range = [0, 1]


def get_qrs_nb_positive_peak_healthy_range():
    return qrs_nb_positive_peak_healthy_range


# Healthy QRS number of negative peaks
qrs_nb_negative_peak_healthy_range = [0, 1]


def get_qrs_nb_negative_peak_healthy_range():
    return qrs_nb_negative_peak_healthy_range


# Healthy QRS biomarker values
qrs_nb_peak_healthy_range = [1, 2]


def get_qrs_nb_peak_healthy_range():
    return qrs_nb_peak_healthy_range


## Healthy QRS duration range
qrs_dur_healthy_range = [60., 100.]


def get_qrs_dur_healthy_range():
    return qrs_dur_healthy_range


## Healthy QRS axis range
qrs_axis_healthy_range = [-30., 90.]


def get_qrs_axis_healthy_range():
    return qrs_axis_healthy_range


## Healthy R wave progression
qrs_r_progression_healthy_list = [
    [lead_V1_name, lead_V2_name, lead_V3_name, lead_V4_name, lead_V6_name, lead_V5_name],
    [lead_V2_name, lead_V1_name, lead_V3_name, lead_V4_name, lead_V6_name, lead_V5_name],
    [lead_V1_name, lead_V2_name, lead_V3_name, lead_V4_name, lead_V5_name, lead_V6_name],
    [lead_V2_name, lead_V1_name, lead_V3_name, lead_V4_name, lead_V5_name, lead_V6_name]
]
qrs_r_progression_healthy_index_list = []
for qrs_r_progression_healthy_i in range(len(qrs_r_progression_healthy_list)):
    qrs_r_progression_healthy_index = [precordial_lead_name_list.index(lead_name)
                                       for lead_name in qrs_r_progression_healthy_list[qrs_r_progression_healthy_i]]
    qrs_r_progression_healthy_index_list.append(qrs_r_progression_healthy_index)

# qrs_r_progression_healthy_alternative = [lead_V2_name, lead_V1_name, lead_V3_name, lead_V4_name, lead_V6_name, lead_V5_name]
# qrs_r_progression_healthy_index_alternative = [precordial_lead_name_list.index(lead_name) for lead_name in qrs_r_progression_healthy_alternative]


# def get_qrs_r_progression_healthy():
#     return qrs_r_progression_healthy


def get_qrs_r_progression_healthy_index_list():
    return qrs_r_progression_healthy_index_list


# def get_qrs_r_progression_healthy_index_alternative():
#     return qrs_r_progression_healthy_index_alternative


## Healthy S wave progression
qrs_s_progression_healthy_list = [
    [lead_V6_name, lead_V5_name, lead_V4_name, lead_V3_name, lead_V1_name, lead_V2_name],
    [lead_V5_name, lead_V6_name, lead_V4_name, lead_V3_name, lead_V1_name, lead_V2_name],
    [lead_V6_name, lead_V5_name, lead_V4_name, lead_V3_name, lead_V2_name, lead_V1_name],
    [lead_V5_name, lead_V6_name, lead_V4_name, lead_V3_name, lead_V2_name, lead_V1_name]
]
qrs_s_progression_healthy_index_list = []
for qrs_s_progression_healthy_i in range(len(qrs_s_progression_healthy_list)):
    qrs_s_progression_healthy_index = [precordial_lead_name_list.index(lead_name)
                                       for lead_name in qrs_s_progression_healthy_list[qrs_s_progression_healthy_i]]
    qrs_s_progression_healthy_index_list.append(qrs_s_progression_healthy_index)


# qrs_s_progression_healthy_index = [precordial_lead_name_list.index(lead_name) for lead_name in qrs_s_progression_healthy]
#
# qrs_s_progression_healthy_alternative = [lead_V5_name, lead_V6_name, lead_V4_name, lead_V3_name, lead_V1_name, lead_V2_name]
# qrs_s_progression_healthy_index_alternative = [precordial_lead_name_list.index(lead_name) for lead_name in qrs_s_progression_healthy_alternative]


# def get_qrs_s_progression_healthy():
#     return qrs_s_progression_healthy


def get_qrs_s_progression_healthy_index_list():
    return qrs_s_progression_healthy_index_list


# def get_qrs_s_progression_healthy_index_alternative():
#     return qrs_s_progression_healthy_index_alternative


## Healthy S>R precordial leads
# According to https://ecg.utah.edu/lesson/3#measurements the transition between S>R and R>S can be either in V3 or V4
# this means that at most V3 will still have S>R and at most V4 will still have R>S, because the change cannot occur in
# neither V2 nor V5.
qrs_s_larger_r_progression_healthy = [lead_V1_name, lead_V2_name]  # https://ecg.utah.edu/lesson/3#measurements
qrs_s_larger_r_progression_healthy_index = [precordial_lead_name_list.index(lead_name)
                                            for lead_name in qrs_s_larger_r_progression_healthy]
qrs_s_larger_r_progression_healthy_optional = [lead_V3_name]  # If the change happens at V4, then V3 can still be included here
qrs_s_larger_r_progression_healthy_optional_index = [precordial_lead_name_list.index(lead_name)
                                            for lead_name in qrs_s_larger_r_progression_healthy_optional]


def get_qrs_s_larger_r_progression_healthy():
    return qrs_s_larger_r_progression_healthy


def get_qrs_s_larger_r_progression_healthy_index():
    return qrs_s_larger_r_progression_healthy_index


def get_qrs_s_larger_r_progression_healthy_optional_index():
    return qrs_s_larger_r_progression_healthy_optional_index


## Healthy R>S precordial leads
# According to https://ecg.utah.edu/lesson/3#measurements the transition between S>R and R>S can be either in V3 or V4
# this means that at most V3 will still have S>R and at most V4 will still have R>S, because the change cannot occur in
# neither V2 nor V5.
qrs_r_larger_s_progression_healthy = [lead_V5_name, lead_V6_name]  # https://ecg.utah.edu/lesson/3#measurements
qrs_r_larger_s_progression_healthy_index = [precordial_lead_name_list.index(lead_name)
                                            for lead_name in qrs_r_larger_s_progression_healthy]
qrs_r_larger_s_progression_healthy_optional = [lead_V4_name]  # If the change happens at V3, then V4 can still be included here
qrs_r_larger_s_progression_healthy_optional_index = [precordial_lead_name_list.index(lead_name)
                                            for lead_name in qrs_r_larger_s_progression_healthy_optional]


def get_qrs_r_larger_s_progression_healthy():
    return qrs_r_larger_s_progression_healthy


def get_qrs_r_larger_s_progression_healthy_index():
    return qrs_r_larger_s_progression_healthy_index


def get_qrs_r_larger_s_progression_healthy_optional_index():
    return qrs_r_larger_s_progression_healthy_optional_index


# Clinical fiducial information
qrs_onset_name = 'QRSonset'


def get_qrs_onset_name():
    return qrs_onset_name

# Quantities of interest
## QRS
qrs_dur_name = 'qrs_dur'
qrs_axis_name = 'qrs_axis'
r_wave_progression_name = 'r_wave_progression'
s_wave_progression_name = 's_wave_progression'
rs_wave_progression_name = 'rs_wave_progression'
## Twave
qtc_dur_name = 'qtc_dur'
t_pe_name = 't_pe'
t_peak_name = 't_peak'
qtpeak_dur_name = 'qtpeak_dur'
t_polarity_name = 't_polarity'
tpeak_dispersion_name = 'tpeak_dispersion_v5_v3'


def get_qrs_dur_name():
    return qrs_dur_name


def get_qtc_dur_name():
    return qtc_dur_name


def get_t_pe_name():
    return t_pe_name


def get_t_peak_name():
    return t_peak_name


def get_qtpeak_dur_name():
    return qtpeak_dur_name


def get_t_polarity_name():
    return t_polarity_name


def get_tpeak_dispersion_name():
    return tpeak_dispersion_name


# Simulation Biomarkers
activation_time_map_biomarker_name = 'lat'
repolarisation_time_map_biomarker_name = 'repol'
apd90_biomarker_name = 'apd90'
sf_iks_biomarker_name = 'sf_IKs'


def get_lat_biomarker_name():
    return activation_time_map_biomarker_name


def get_repol_biomarker_name():
    return repolarisation_time_map_biomarker_name


def get_apd90_biomarker_name():
    return apd90_biomarker_name


def get_sf_iks_biomarker_name():
    return sf_iks_biomarker_name


# Ventricular coordinate names and ranges
vc_ranges = {}
vc_range_error_tolerance = 0.001
# apex-to-base/valves (Cobiveco) - apex=0, base/valves=1 - goes up to valves in closed meshes
vc_ab_name = 'ab'
vc_ranges[vc_ab_name] = [0., 1.]
# apex-to-base - cut at base (Cobiveco) - apex=0, base=1., flat-basal-plane<=1, valves<0 - No valves
vc_ab_cut_name = 'ab_cut'
vc_ranges[vc_ab_cut_name] = [0., 1.]
# posterior-to-anterior (Projection) - posterior=0, anterior=1
vc_aprt_name = 'aprt'
vc_ranges[vc_aprt_name] = [0., 1.]
# rotational (Cobiveco) - Septum=1, Posterior=0 - No valves
vc_rt_name = 'rt'
vc_ranges[vc_rt_name] = [-1., 1.]    # When working with closed geometries, they have -1 values in the valves!
valid_vc_rt_range = [0., 1.]
# lv-to-rv (Projection) - LV=0, RV=1
vc_rvlv_name = 'rvlv'
vc_ranges[vc_rvlv_name] = [0., 1.]
# transmural (RV Septum as 0) - Epi=0, Endo=1
vc_tm_name = 'tm'
vc_ranges[vc_tm_name] = [0., 1.]
# lv-and-rv (Binary splitting the septum in half) - LV=0, RV=1
# vc_tv_name = 'tv'
# vc_tv_binary_name = 'rvlv_binary'


def get_vc_ab_name():
    return vc_ab_name


def get_vc_ab_cut_name():
    return vc_ab_cut_name


def get_vc_aprt_name():
    return vc_aprt_name


def get_vc_rt_name():
    return vc_rt_name


def get_valid_vc_rt_range():
    return valid_vc_rt_range


def get_vc_rvlv_name():
    print(vc_rvlv_name, ' should be lvrv instead!')
    return vc_rvlv_name


def get_vc_tm_name():
    return vc_tm_name


# Field names
xyz_name_list = ['x', 'y', 'z']


def get_xyz_name_list():
    return xyz_name_list


# Generic functions
def normalise_field_to_zero_one(field):
    norm_field = (field - np.amin(field)) / (np.amax(field) - np.amin(field))  # Scaled to be between 0 and 1
    return norm_field


def normalise_field_to_range(field, range):
    range_min = range[0]
    range_max = range[1]
    norm_field = normalise_field_to_zero_one(field=field)    # Scaled to be between 0 and 1
    field = norm_field * (range_max - range_min) + range_min    # Scale to match the specified range
    return field


def check_vc_field_ranges(vc_field, vc_name):
    if vc_name in vc_ranges:
        if (np.amin(vc_field) != vc_ranges[vc_name][0]) or (np.amax(vc_field) != vc_ranges[vc_name][1]):
            if ((np.amin(vc_field) - vc_ranges[vc_name][0]) < vc_range_error_tolerance) and \
                    ((np.amax(vc_field) - vc_ranges[vc_name][1]) < vc_range_error_tolerance):
                warn('Careful! Ventricular field ' + vc_name + ' has an error in the ranges, but within the tolerace!'
                     + '\n(Error is small) Ranges of ventricular field ' + vc_name + ' are not as expected!'
                     + '\nExpected range was ' + str(vc_ranges[vc_name]) + ', given values ranged from '
                     + str(np.amin(vc_field)) + ' to ' + str(np.amax(vc_field)))
                vc_field = normalise_field_to_range(field=vc_field, range=vc_ranges[vc_name])
            else:
                # TODO Reactive the exception bellow, it was commented out to test the code on outdated UKB cases
                warn('Please!! Urgently, come to function check_vc_field_ranges in utils.py and uncomment the following call')
                # raise Exception('\n(Error too large) Ranges of ventricular field ' + vc_name + ' are not as expected!\nExpected range was '
                #                 + str(vc_ranges[vc_name]) + ', given values ranged from ' + str(np.amin(vc_field))
                #                 + ' to ' + str(np.amax(vc_field)))
                warn('\n(Error too large) Ranges of ventricular field ' + vc_name + ' are not as expected!\nExpected range was '
                                + str(vc_ranges[vc_name]) + ', given values ranged from ' + str(np.amin(vc_field))
                                + ' to ' + str(np.amax(vc_field)))
    else:
        warn('Careful! Ventricular field ' + vc_name + ' has no controlled ranges!')
    return vc_field


# Eikonal parameter names
fibre_speed_name = 'fibre_speed'
sheet_speed_name = 'sheet_speed'
normal_speed_name = 'normal_speed'
endo_dense_speed_name = 'endo_dense_speed'
endo_sparse_speed_name = 'endo_sparse_speed'
purkinje_speed_name = 'purkinje_speed'
root_node_name_start ='r'


def get_fibre_speed_name():
    return fibre_speed_name


def get_sheet_speed_name():
    return sheet_speed_name


def get_normal_speed_name():
    return normal_speed_name


def get_endo_dense_speed_name():
    return endo_dense_speed_name


def get_endo_sparse_speed_name():
    return endo_sparse_speed_name


def get_purkinje_speed_name():
    return purkinje_speed_name


def get_root_node_name_start():
    return root_node_name_start


def get_root_node_meta_index_population_from_pandas(pandas_parameter_population):
    root_node_name_start = get_root_node_name_start()
    read_all_root_nodes = False
    root_node_meta_index_population = []
    root_i = 0
    while not read_all_root_nodes:
        root_node_name = root_node_name_start + str(root_i)
        if root_node_name in pandas_parameter_population.columns:
            # print('root_node_name ', root_node_name)
            root_node_meta_index_aux = pandas_parameter_population.get(root_node_name)
            root_node_meta_index_aux = np.asarray(root_node_meta_index_aux.tolist())
            # print(root_node_meta_index_aux.shape)
            root_node_meta_index_population.append(root_node_meta_index_aux)
        else:
            read_all_root_nodes = True
        root_i = root_i + 1
    # a = np.stack(root_node_meta_index_population, axis=1)
    # print('np.concatenate(root_node_meta_index_population, axis=1) ', a.shape)
    return np.stack(root_node_meta_index_population, axis=1)


# General file naming
lead_separator = '_'


def get_lead_separator():
    return lead_separator


# Pandas controlled index
row_id_name = 'row_num'


def get_biomarker_lead_name(biomarker_lead_name, lead_i):
    return biomarker_lead_name + get_lead_separator() + str(lead_i)


def translate_from_pandas_to_array(name_list_in_order, pandas_data):
    data_list = []
    for name in name_list_in_order:
        data_list.append(get_values_from_pandas(dictionary=pandas_data, key=name))
    return np.transpose(np.stack(data_list))


def replace_zeros_by_mean(data):
    print('[data!=0] ', np.sum([data != 0]))
    non_zero_mean = np.mean(data[data!=0])
    non_zero_data = data
    non_zero_data[data==0] = non_zero_mean
    return non_zero_data


def change_resolution(data, resolution):
    return resolution * np.round(data/resolution)


def dictionary_to_ndarray(dictionary, key_list_in_order):
    result_list = []
    for key in key_list_in_order:
        result_list.append(dictionary[key][:, np.newaxis])
    return np.concatenate((result_list), axis=1)


# class TestFailed(Exception):
#     def __init__(self, m):
#         self.message = m
#     def __str__(self):
#         return self.message

# try:
#     raise TestFailed('Oops')
# except TestFailed as x:
#     print(x)


def get_sorted_numerical_folder_name_list(folder_name_list, folder_name_string_tag=None, exclude_list=[]):
    folder_name_list = [aux for aux in folder_name_list if aux not in exclude_list] # TODO make this so that it checks if it's a number instead?
    if folder_name_string_tag is None:
        folder_name_number_list = [int(aux) for aux in folder_name_list]
    else:
        folder_name_number_list = [int(aux.replace(folder_name_string_tag, '')) for aux in folder_name_list]
    sorting_index = np.argsort(folder_name_number_list)
    return [folder_name_list[i] for i in sorting_index]


def fold_ecg_matrix(data):
    return np.reshape(data, (data.shape[0], -1), order='C')


def unfold_ecg_matrix(data, nb_leads):
    # Check dimensions of data to decide how to reshape
    if len(data.shape) == 1:
        ecg = np.reshape(data, (1, int(nb_leads), -1), order='C')
    elif len(data.shape) == 2:
        ecg = np.reshape(data, (data.shape[0], int(nb_leads), -1), order='C')
    else:
        raise "How did you save an array with more than 2 dim in a CSV? This was not supported yet in 2023!"
    return ecg


def get_row_id_name():
    return row_id_name


def initialise_pandas_dataframe(df):
    if get_row_id_name() in df.columns:
        df = df.drop(columns=[get_row_id_name()])
    df.insert(loc=0, column=get_row_id_name(), value=np.arange(0, len(df)))
    return df


def get_values_from_pandas(dictionary, key):
    return dictionary[key].values


def get_keys_from_pandas(df):
    if get_row_id_name() in df.columns:
        df = df.drop(columns=[get_row_id_name()])
    return list(df.keys())


def get_pandas_from_value(df, key, value):
    return df.loc[df[key] == value]


def remap_pandas_from_row_index(df, row_index):
    new_df_list = []
    # Populate pandas dataframe
    for new_index in range(0, row_index.shape[0]):
        row_df = initialise_pandas_dataframe(get_pandas_from_value(df=df, key=get_row_id_name(), value=row_index[new_index]))
        row_df[get_row_id_name()] = new_index
        new_df_list.append(row_df)
    return pd.concat(new_df_list, ignore_index=True)


def remove_nan_and_make_list_of_array(mat):
    list_of_array = []
    for i in range(mat.shape[0]):
        root_node_path = mat[i, :][mat[i, :] != get_nan_value()]
        list_of_array.append(root_node_path)
    return list_of_array


def index_list(list, index):
    new_list = []
    for index_i in index:
        new_list.append(list[index_i])
    return new_list


def from_meta_index_list_to_index(meta_index_list, index):
    for meta_index_i in range(len(meta_index_list)):
        meta_index_list[meta_index_i] = index[meta_index_list[meta_index_i]]
    return meta_index_list


def get_edge_from_path_list(path_list):
    edge_list = []
    for path in path_list:
        edge_list_part = get_edge_list_from_path(path)
        if not(edge_list_part.size == 0):     # Check if array is NOT empty
            edge_list.append(edge_list_part)
    edge = np.concatenate(edge_list, axis=0)
    return np.unique(edge, axis=0)


def get_edge_list_from_path(path):
    edge = []
    previous_i = 0
    for current_i in range(1, path.shape[0], 1):
        edge.append(np.array([path[previous_i], path[current_i]]))
        previous_i = current_i
    return np.asarray(edge)


@numba.njit()
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


@numba.njit
def find_first_larger_than(item, vec):
    """return the index of the first occurence of item in vec"""
    for i, v in enumerate(vec):
        if v > item: return i
    return -1


def insert_sorted(aList, newV):
    # Function to insert element
    ini_index = 0
    end_index = len(aList)
    index = int((end_index - ini_index) / 2)
    for i in range(0, len(aList), 1):
        if newV[1] < aList[index][1]:
            if end_index - ini_index <= 1 or index + 1 == end_index:
                index = index + 1
                break
            else:
                ini_index = index + 1
                index = int(index + (end_index - ini_index) / 2 + 1)
        elif newV[1] > aList[index][1]:
            if end_index - ini_index <= 1 or index == ini_index:
                index = index  # Place before the current position
                break
            else:
                end_index = index
                index = int(index - (end_index - ini_index) / 2)
        else:
            index = ini_index
            break
    aList.insert(index, newV)


def map_indexes(points_to_map_xyz, reference_points_xyz, error_tolerance=0.5):
    """The error tolerance is set assuming that all data is in cm units, which should be consistent in all scripts as of March 2024"""
    mapped_indexes = pymp.shared.array((points_to_map_xyz.shape[0]), dtype=int)
    # threadsNum = multiprocessing.cpu_count()
    error_cumm = pymp.shared.array((points_to_map_xyz.shape[0]), dtype=float)
    error_cumm[:] = get_nan_value()  # Make sure that every node is visited or it will give problems later
    # Uncomment the following lines to turn off the parallelisation.
    # if True:
    #     for node_i in range(points_to_map_xyz.shape[0]):
    iter_gen = get_parallel_loop(data_size=points_to_map_xyz.shape[0])
    for node_i in iter_gen:
        if True:
    # with pymp.Parallel(min(threadsNum, points_to_map_xyz.shape[0])) as p1:
    #     for node_i in p1.range(points_to_map_xyz.shape[0]):
            mapped_indexes[node_i] = np.argmin(
                np.linalg.norm(reference_points_xyz - points_to_map_xyz[node_i, :], ord=2, axis=1)).astype(int)
            error_cumm[node_i] = np.amin(
                np.linalg.norm(reference_points_xyz - points_to_map_xyz[node_i, :], ord=2, axis=1)).astype(int)
    # for i in range(points_to_map_xyz.shape[0]):
    #     mapped_indexes[i] = np.argmin(np.linalg.norm(reference_points_xyz - points_to_map_xyz[i, :], ord=2, axis=1)).astype(int)
    # if return_unique_only:  # use the unique function without sorting the contents of the array (meta_indexes)
    #     unique_meta_indexes = np.unique(mapped_indexes, axis=0, return_index=True)[
    #         1]  # indexes to the indexes (meta_indexes) that are unique
    #     mapped_indexes = mapped_indexes[sorted(unique_meta_indexes)]  # TODO this could just be one line of code
    mean_error = np.mean(error_cumm)
    print('mean_error ', mean_error)
    if mean_error > error_tolerance:
        # Cannot plot here because there would be a circular import
        # scatter_visualise_point_cloud(xyz=points_to_map_xyz, title='points_to_map_xyz')
        # scatter_visualise_point_cloud(xyz=reference_points_xyz, title='reference_points_xyz')
        raise Exception("The two sets of poitns don't overlap enough in space, there may be a mistake in the input data.")
    return mapped_indexes

# EOF

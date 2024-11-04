import multiprocessing
import os
from warnings import warn

import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pymp

from ecg_functions import calculate_ecg_augmented_biomarker_from_only_ecg, calculate_r_s_sr_rs_wave_progression, \
    calculate_qrs_width_population, calculate_ecg_qrs_axis_population, calculate_ecg_qrs_nb_peak_pos_neg_population
from postprocess_functions import visualise_ecg
from utils import get_nan_value, get_biomarker_lead_name, dictionary_to_ndarray, \
    get_qrs_s_larger_r_progression_healthy_index, get_qrs_s_larger_r_progression_healthy_optional_index, \
    get_qrs_r_larger_s_progression_healthy_optional_index, get_qrs_r_larger_s_progression_healthy_index, \
    get_qrs_dur_healthy_range, get_qrs_axis_healthy_range, get_nb_precordial_lead, get_qrs_nb_peak_healthy_range, \
    get_nb_unique_lead, get_qrs_nb_positive_peak_healthy_range, get_qrs_nb_negative_peak_healthy_range, \
    get_qrs_r_progression_healthy_index_list, get_qrs_s_progression_healthy_index_list


def calculate_order_rmse_discrepancy_between_index_list(index_list_1, index_list_2):
    # TODO There may be a better way to compute the difference in ordering between two lists, but I can't think of it now
    return math.sqrt(np.mean(np.square(index_list_1-index_list_2)))


def calculate_content_discrepancy_between_index_list(mandatory_list, optional_list, predicted_list):
    # print('isinstance(predicted_list, list) ', isinstance(predicted_list, list))
    if not (isinstance(mandatory_list, list) and isinstance(optional_list, list) and isinstance(predicted_list, list)):
        raise(Exception, 'mandatory_list, optional_list, predicted_list should be type list')

    # print('mandatory_list ', mandatory_list)
    # print('optional_list ', optional_list)
    # print('predicted_list ', predicted_list)

    mandatory_set = set(mandatory_list)
    optional_set = set(optional_list)
    predicted_set = set(predicted_list)
    difference_count = 0

    total_set = set(mandatory_set | predicted_set)
    # table_format = '{:<10} {:<10} {:<10}'
    # print(table_format.format('mandatory_set', 'predicted_set', 'optional_set'))
    # print('-' * 20)
    for elem in sorted(total_set):
        if elem in mandatory_set:
            if elem in predicted_set:
                # print(table_format.format(elem, elem, '-'))
                pass
            else:
                # print(table_format.format(elem, 'Missing', '-'))
                difference_count = difference_count + 1
        elif elem in optional_set:
            # print(table_format.format('-', elem, elem))
            pass
        else:
            # print(table_format.format('Missing', elem, 'Missing'))
            difference_count = difference_count + 1
    return difference_count


def calculate_range_discrepancy(predicted_list, good_range):
    range_discrepancy_list = np.zeros(predicted_list.shape, dtype=float)
    discrepancy_too_small_bool = predicted_list < min(good_range)
    range_discrepancy_list[discrepancy_too_small_bool] = np.abs(predicted_list - min(good_range))[discrepancy_too_small_bool]
    discrepancy_too_large_bool = predicted_list > max(good_range)
    range_discrepancy_list[discrepancy_too_large_bool] = np.abs(predicted_list - max(good_range))[discrepancy_too_large_bool]
    return range_discrepancy_list


def calculate_lead_pcc(lead_1, lead_2):
    # Check which signal is longer to make the function applicable to any case
    if lead_1.shape[0] >= lead_2.shape[0]:
        a = lead_1
        b = lead_2
    else:
        a = lead_2
        b = lead_1
    b_aux = np.zeros(a.shape)
    b_aux[:b.shape[0]] = b
    b_aux[b.shape[0]:] = b[-1]
    b = b_aux
    return np.corrcoef(a, b)[0, 1]


def calculate_ecg_pcc(ecg_1, ecg_2):
    leads_pcc = np.zeros((ecg_1.shape[0]))
    for lead_i in range(ecg_1.shape[0]):
        leads_pcc[lead_i] = calculate_lead_pcc(ecg_1[lead_i, :], ecg_2[lead_i, :])
    return leads_pcc


# def calculate_ecg_pcc_cubic(ecg_1, ecg_2):
#     leads_pcc = np.zeros((ecg_1.shape[0]))
#     for lead_i in range(ecg_1.shape[0]):
#         leads_pcc[lead_i] = calculate_lead_pcc(ecg_1[lead_i, :], ecg_2[lead_i, :])
#     return np.mean(leads_pcc**5) # preserve sign but penalise iverted polarities very strongly


def calculate_ecg_rmse(ecg_list, ecg_reference):
    # print('calculate_ecg_rmse')
    # print('ecg_list ', ecg_list.shape)
    # print('ecg_reference ', ecg_reference.shape)
    # print()
    min_qrs_len = min(ecg_list.shape[2], ecg_reference.shape[1])
    return np.mean(np.sqrt(np.mean((ecg_list[:, :, :min_qrs_len] - ecg_reference[np.newaxis, :, :min_qrs_len]) ** 2, axis=2)), axis=1)


def calculate_ecg_errors(predicted_ecg_list, target_ecg, error_method):
    # print('calculate_ecg_errors')
    # print('predicted_ecg_list ', predicted_ecg_list.shape)
    # print('target_ecg ', target_ecg.shape)
    # print('error_method ', error_method)
    # print()
    # TODO: refactor this function to avoid code repetition
    if error_method == 'rmse':
        result_error = calculate_ecg_rmse(predicted_ecg_list, target_ecg)
    elif error_method == 'rmse_pcc':
        result_rmse = calculate_ecg_rmse(predicted_ecg_list, target_ecg)
        target_ecg_pcc = target_ecg[:, :predicted_ecg_list.shape[2]]
        result_pcc = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
        result_pcc[:] = get_nan_value()
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # if True:
        #     for sample_i in range(result_pcc.shape[0]):
        with pymp.Parallel(min(threads_num, result_pcc.shape[0])) as p1:
            for sample_i in p1.range(result_pcc.shape[0]):
                result_pcc[sample_i] = np.mean(calculate_ecg_pcc(target_ecg_pcc, predicted_ecg_list[sample_i, :, :]))
        # By weighting the rmse and pcc in the following way, it means that the metric is relative to the best and worst cases
        # in the population, which is really bad for inference purposes, but it's acceptable to chose the best from a population
        result_rmse_pcc = ((-result_pcc - np.amin(-result_pcc)) / (
                np.amax(-result_pcc) - np.amin(-result_pcc))) ** 2 + ((result_rmse - np.amin(result_rmse)) / (
                np.amax(result_rmse) - np.amin(result_rmse))) ** 2
        result_error = result_rmse_pcc
    elif error_method == 'rmse_pcc_cubic':  # TODO change name to something more descriptive!! Not cubed anymore
        result_rmse = calculate_ecg_rmse(predicted_ecg_list, target_ecg)
        # TODO remove these lines of code

        # TODO Remove the above lines of code
        target_ecg_pcc = target_ecg[:, :predicted_ecg_list.shape[2]]
        result_pcc = pymp.shared.array((predicted_ecg_list.shape[0], target_ecg.shape[0]), dtype=np.float64)
        result_pcc[:] = get_nan_value()
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # if True:
        #     for sample_i in range(result_pcc.shape[0]):
        with pymp.Parallel(min(threads_num, result_pcc.shape[0])) as p1:
            for sample_i in p1.range(result_pcc.shape[0]):
                result_pcc[sample_i, :] = calculate_ecg_pcc(target_ecg_pcc, predicted_ecg_list[sample_i, :, :])
        # This following strategy weights both the rmse and the pcc together in a way that is suitable for iterative inference methods
        # result_rmse_pcc = ((1.0 - result_pcc)/2.)**2 + (result_rmse ** 2)/(np.abs(np.amax(target_ecg)))#/10.)) # We shift the pcc to positive
        # The weighting factors have been rebalanced to avoid unnecessary computation and to make sure that the PCC has a stronger
        # driving force, compared to the RMSE, at least for the best particles in the first few iterations.

        # aux_a = np.argsort(result_pcc)
        # aux_b = result_pcc[aux_a]
        # aux_c = 1.0-result_pcc
        # aux_c = aux_c[aux_a]
        # plt.plot(aux_b, aux_c)
        # plt.title('x=PCC y=1.PCC')
        # plt.show()
        #TODO Remove these lines of code
        # result_pcc_leads_mean = np.mean(result_pcc, axis=1)
        # print('result_pcc mean: ', np.mean(result_pcc_leads_mean))
        # print('result_pcc std: ', np.std(result_pcc_leads_mean))
        # print('result_pcc ', result_pcc.shape)
        # print('result_pcc_leads_mean ', result_pcc_leads_mean.shape)
        # TODO Remove the above lines of code
        result_pcc_cubic = np.mean((1.0 - result_pcc)**2, axis=1)
        # print(result_pcc_cubic.shape)
        # print(result_pcc.shape)
        # print('max ', np.amax(result_pcc_cubic))
        # print('min ', np.amin(result_pcc_cubic))
        result_pcc_aux = 100*(result_pcc_cubic)
        result_rmse_aux = 2*result_rmse/(np.abs(np.amax(target_ecg))) # TODO, just added x2 2023/07/18
        result_rmse_pcc = result_pcc_aux + result_rmse_aux #/10.)) # We shift the pcc to positive
        # values and normalise the rmse to the maximum amplitude divided by 10 so that it's a small number
        result_error = result_rmse_pcc
        # print('Best ')
        # aux_id = np.argmin(result_error)
        # print('result_rmse ', result_rmse_aux[aux_id])
        # print('result_pcc ', result_pcc_aux[aux_id])     # PCC TO BE 10 TIMES LARGER THAN RMSE
        # print('result_error ', result_error[aux_id])
        # # print('result_error ', result_error.shape)
        # print()
        # print('Worst ')
        # aux_id = np.argmax(result_error)
        # print('result_rmse ', result_rmse_aux[aux_id])
        # print('result_pcc ', result_pcc_aux[aux_id])  # PCC TO BE 10 TIMES LARGER THAN RMSE
        # print('result_error ', result_error[aux_id])
        # # print('result_error ', result_error.shape)
        # print()
    elif error_method == 'rmse_pcc_fudged':  # TODO change name to something more descriptive!! Is this the same as above?
        fudge_factor_for_simulated_ecg = 0.7    # This may ease the later translation to monodomain simulations
        result_rmse = calculate_ecg_rmse(predicted_ecg_list*fudge_factor_for_simulated_ecg, target_ecg)
        target_ecg_pcc = target_ecg[:, :predicted_ecg_list.shape[2]]
        result_pcc = pymp.shared.array((predicted_ecg_list.shape[0], target_ecg.shape[0]), dtype=np.float64)
        result_pcc[:] = get_nan_value()
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # if True:
        #     for sample_i in range(result_pcc.shape[0]):
        with pymp.Parallel(min(threads_num, result_pcc.shape[0])) as p1:
            for sample_i in p1.range(result_pcc.shape[0]):
                result_pcc[sample_i, :] = calculate_ecg_pcc(target_ecg_pcc, predicted_ecg_list[sample_i, :, :])
        # This following strategy weights both the rmse and the pcc together in a way that is suitable for iterative inference methods
        # result_rmse_pcc = ((1.0 - result_pcc)/2.)**2 + (result_rmse ** 2)/(np.abs(np.amax(target_ecg)))#/10.)) # We shift the pcc to positive
        # The weighting factors have been rebalanced to avoid unnecessary computation and to make sure that the PCC has a stronger
        # driving force, compared to the RMSE, at least for the best particles in the first few iterations.
        result_pcc_cubic = np.mean((1.0 - result_pcc)**2, axis=1)
        result_pcc_aux = 100*(result_pcc_cubic)
        result_rmse_aux = 2*result_rmse/(np.abs(np.amax(target_ecg))) # TODO, just added x2 2023/07/18
        result_rmse_pcc = result_pcc_aux + result_rmse_aux #/10.)) # We shift the pcc to positive
        # values and normalise the rmse to the maximum amplitude divided by 10 so that it's a small number
        result_error = result_rmse_pcc
    elif error_method == 'pcc':
        target_ecg_pcc = target_ecg[:, :predicted_ecg_list.shape[2]]
        result_pcc = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
        result_pcc[:] = get_nan_value()
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # if True:
        #     for sample_i in range(result_pcc.shape[0]):
        with pymp.Parallel(min(threads_num, result_pcc.shape[0])) as p1:
            for sample_i in p1.range(result_pcc.shape[0]):
                result_pcc[sample_i] = np.mean(calculate_ecg_pcc(target_ecg_pcc, predicted_ecg_list[sample_i, :, :]))
        result_error = result_pcc
    else:
        raise ('ECG error method not found: ' + error_method)
    return result_error


def dtw_ecg_parallel(predicted_ecg_list, target_ecg, max_slope, mesh_volume, w_max):    # TODO: Refactor this function and simplify behaviour
    """25/02/2021: I have realised that my trianglogram method is equivalent to the vanila paralelogram in practice but less
    computationally efficient. I initially thought that using a parallelogram implied that all warping was to be undone
    towards the end of the signal, like comparing people reading the same text on the same amount of time but with a variability
    on the speed for each part of the sentance. However, that was not the case. The parallelogram serves the same purpose
    as the trianlogram when the constraint of equal ending is put in place, namely, only (N-1, M-1) is evaluated. This
    constraint forces both signals to represent the same information rather than one signal being only half of the other.
    Therefore, by using the trianglogram plus the restriction, the only thing I am achieving is to do more calculations
    strictly necessary. However, the original implementation of the parallelogram allows an amount of warping proportional
    to the length difference between the two signals, which could lead to non-physiological warping. Here instead, the
    maximum amount of warping is defined in w_max and max_slope; this feature is key because the discrepancy calculation
    needs to be equivalent for all signals throughout the method regardless of their length."""

    """Dynamic Time Warping distance specific for comparing electrocardiogram signals.
    It implements a trianglogram constraint (inspired from Itakura parallelogram (Itakura, 1975)).
    It also implements weight penalty with a linear increasing cost away from the true diagonal (i.e. i=j).
    Moreover, it implements a step-pattern with slope-P value = 0.5 from (Sakoe and Chiba, 1978).
    Finally, the algorithm penalises the difference between the lenght of the two signals and adds it to the DTW distance.
    Options
    -------
    max_slope : float Maximum slope of the trianglogram.
    w_max : float weigth coeficient to the distance from the diagonal.
    small_c :  float weight coeficient to the difference in lenght between the signals being compared.
    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).
    .. [2] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).
    """
    # Don't compute repetitions - Numpy Unique has an issue with NaN values: https://github.com/numpy/numpy/issues/2111
    small_c = 0.05 * 171 / mesh_volume  # Scaling factor to account for differences in mesh size # 2022/05/04
    p = np.copy(predicted_ecg_list[:, 0, :])
    p[np.isnan(p)] = np.inf
    p, particle_index, unique_index = np.unique(p, return_index=True, return_inverse=True, axis=0)
    p = None
    predicted_list = predicted_ecg_list[particle_index, :, :]

    # This code has parallel pympy sections as well as numba parallel sections
    nParts = predicted_list.shape[0]
    nLeads = predicted_list.shape[1]
    res = pymp.shared.array(nParts, dtype='float64')
    threads_num = multiprocessing.cpu_count()
    with pymp.Parallel(min(nParts, threads_num)) as p1:
        for conf_i in p1.range(0, nParts):
            # warps_aux = 0.  # TODO DELETE THIS 2022/05/17
            mask = np.logical_not(np.isnan(predicted_list[conf_i, :, :]))
            pred_ecg = np.squeeze(predicted_list[conf_i:conf_i + 1, :,
                                  mask[0, :]])  # using slicing index does not change the shape of the object,
            # however, mixing slicing with broadcasting does change it, which then
            # requires moving the axis with np.moveaxis
            # Lengths of each sequence to be compared
            n_timestamps_1 = pred_ecg.shape[1]
            n_timestamps_2 = target_ecg.shape[1]

            # Computes the region (in-window area using a trianglogram)
            # WARNING: this is a shorter version of the code for generating the region which does not account for special cases, the full version is the fuction "trianglorgram" from myfunctions.py
            max_slope_ = max_slope
            min_slope_ = 1 / max_slope_
            scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2)
            max_slope_ *= scale_max
            scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1)
            min_slope_ *= scale_min
            centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
            lower_bound = min_slope_ * np.arange(n_timestamps_1)
            lower_bound = np.round(lower_bound, 2)
            lower_bound = np.floor(
                lower_bound)  # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path
            upper_bound = max_slope_ * np.arange(n_timestamps_1) + 1
            upper_bound = np.round(upper_bound, 2)
            upper_bound = np.ceil(
                upper_bound)  # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path
            region_original = np.asarray([lower_bound, upper_bound]).astype('int64')
            region_original = np.clip(region_original[:, :n_timestamps_1], 0,
                                      n_timestamps_2)  # Project region on the feasible set

            part_dtw = 0.
            # Compute the DTW for each lead and particle separately so that leads can be wraped differently from each other
            for lead_i in range(nLeads):
                region = np.copy(region_original)
                x = pred_ecg[lead_i, :]
                y = target_ecg[lead_i, :]

                # Computes cost matrix from dtw input
                dist_ = lambda x, y: (x - y) ** 2  # The squared term will penalise differences exponentially, which is desirable especially when relative amplitudes are important

                # Computs the cost matrix considering the window (0 inside, np.inf outside)
                cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
                m = np.amax(cost_mat.shape)
                for i in numba.prange(n_timestamps_1):
                    for j in numba.prange(region[0, i], region[1, i]):
                        # cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) # This new weight considers that wraping in time is cheaper the later it's done #* (w_max/(1+math.exp(-g*(abs(i-j)-m/2)))+1.) # + abs(i-j)*small_c # Weighted version of the DTW algorithm
                        # TODO: BELLOW MODIFIED ON THE 2022/05/16 - to resemble more what it used to be, larger pennalty for warping
                        cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i - j) / max(1., (i + j)) + 1.) + abs(
                            i - j) * small_c * 10  # 07/12/2021 When going from synthetic data into clinical we observe that the
                        # singals cannot be as similar anymore due to the difference in source models between real patients and Eikonal, -
                        # additional penalty for warping:  +  abs( i-j)*small_c*10

                # Computes the accumulated cost matrix
                acc_cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
                acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
                    cost_mat[0, 0: region[1, 0]]
                )
                acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
                    cost_mat[0: region[1, 0], 0]
                )
                region_ = np.copy(region)
                region_[0] = np.maximum(region_[0], 1)
                for i in range(1, n_timestamps_1):
                    for j in range(region_[0, i], region_[1, i]):
                        # Implementation of a Slope-constraint as a step-pattern:
                        # This constraint will enforce that the algorithm can only take up to 2 consecutive steps along the time wraping directions.
                        # I decided to make it symetric because in (Sakoe and Chiba, 1978) they state that symetric means that DTW(A, B) == DTW(B, A), although I am not convinced why it's not the case in the asymetric implementation.
                        # Besides, the asymetric case has a bias towards the diagonal which I thought could be desirable in our case, that said, having DTW(A, B) == DTW(B, A) may prove even more important, especially in further
                        # applications of this algorithm for ECG comparison.
                        # This implementation is further explained in (Sakoe and Chiba, 1978) and correspondes to the one with P = 0.5, (P = n/m, where P is a rule being inforced, n is the number of steps in the diagonal
                        # direction and m is the steps in the time wraping direction).
                        acc_cost_mat[i, j] = min(
                            acc_cost_mat[i - 1, j - 3] + 2 * cost_mat[i, j - 2] + cost_mat[i, j - 1] + cost_mat[i, j],
                            acc_cost_mat[i - 1, j - 2] + 2 * cost_mat[i, j - 1] + cost_mat[i, j],
                            acc_cost_mat[i - 1, j - 1] + 2 * cost_mat[i, j],
                            acc_cost_mat[i - 2, j - 1] + 2 * cost_mat[i - 1, j] + cost_mat[i, j],
                            acc_cost_mat[i - 3, j - 1] + 2 * cost_mat[i - 2, j] + cost_mat[i - 1, j] + cost_mat[i, j]
                        )
                dtw_dist = acc_cost_mat[-1, -1] / (n_timestamps_1 + n_timestamps_2)  # Normalisation M+N according to (Sakoe and Chiba, 1978)
                # signals have the same lenght
                dtw_dist = dtw_dist / np.amax(np.abs(y)) * np.amax(np.abs(target_ecg))  # 2022/05/04 - Normalise by lead amplitude to weight all leads similarly
                part_dtw += math.sqrt(dtw_dist)  # I would rather have leads not compensating for each other # 2022/05/04 Add normalisation by max abs amplitude in each lead
            res[conf_i] = part_dtw / nLeads + small_c * (n_timestamps_1 - n_timestamps_2) ** 2 / min(n_timestamps_1,
                                                                                                     n_timestamps_2)
    return res[unique_index]


# def rmse_ecg(prediction_list, target_ecg):
#     return np.sum(np.sqrt(np.sum((prediction_list - target_ecg[np.newaxis, :]) ** 2, axis=2)), axis=1)


def dtw_trianglorgram(x, y, max_slope=1.5, w_max=10., target_max_amplitude=1.):
    """25/02/2021: I have realised that this method is equivalent to the vanila paralelogram in practice but less
    computationally efficient. I initially thought that using a parallelogram implied that all warping was to be undone
    towards the end of the signal, like comparing people reading the same text on the same amount of time but with a variability
    on the speed for each part of the sentance. However, that was not the case. The parallelogram serves the same purpose
    as the trianlogram when the constraint of equal ending is put in place, namely, only (N-1, M-1) is evaluated. This
    constraint forces both signals to represent the same information rather than one signal being only half of the other.
    Therefore, by using the trianglogram plus the restriction, the only thing I am achieving is to do more calculations
    strictly necessary. However, the original implementation of the parallelogram allows an amount of warping proportional
    to the length difference between the two signals, which could lead to non-physiological warping. Here instead, the
    maximum amount of warping is defined in w_max and max_slope; this feature is key because the discrepancy calculation
    needs to be equivalent for all signals throughout the method regardless of their length. TODO: change method to be the parallelogram to make it more efficient."""
    """Dynamic Time Warping distance specific for comparing electrocardiogram signals.
    It implements a trianglogram constraint (inspired from Itakura parallelogram).
    It also implements weight penalty with a linear increasing cost away from the true diagonal (i.e. i=j).
    Moreover, it implements a step-pattern with slope-P value = 0.5 from (Sakoe and Chiba, 1978).
    Finally, the algorithm penalises the difference between the lenght of the two signals and adds it to the DTW distance.
    Options
    -------
    max_slope : float Maximum slope of the trianglogram.
    w_max : float weigth coeficient to the distance from the diagonal.
    small_c :  float weight coeficient to the difference in lenght between the signals being compared.
    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).
    .. [1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).
    """

    n_timestamps_1 = x.shape[0]
    n_timestamps_2 = y.shape[0]

    small_c = 0.05 * 171 / meshVolume  # Scaling factor to account for differences in mesh size # 2022/05/04

    # print('from code in dtw_trianglogram(...). Sizes:')
    # print(n_timestamps_1)
    # print(n_timestamps_2)

    # Computes the region (in-window area using a trianglogram)
    region = trianglorgram(n_timestamps_1, n_timestamps_2, max_slope)

    # Computes cost matrix from dtw input
    dist_ = lambda x, y: (x - y) ** 2

    region = check_array(region, dtype='int64')
    region_shape = region.shape
    if region_shape != (2, x.size):
        raise ValueError(
            "The shape of 'region' must be equal to (2, n_timestamps_1) "
            "(got {0}).".format(region_shape)
        )

    # Computs the cost matrix considering the window (0 inside, np.inf outside)
    cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
    m = np.amax(cost_mat.shape)
    for i in numba.prange(n_timestamps_1):
        for j in numba.prange(region[0, i], region[1, i]):
            cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i - j) / max(1., (
                    i + j)) + 1.)  # This new weight considers that wraping in time is cheaper the later it's done #* (w_max/(1+math.exp(-g*(abs(i-j)-m/2)))+1.) # + abs(i-j)*small_c # Weighted version of the DTW algorithm

    cost_mat = check_array(cost_mat, ensure_min_samples=2,
                           ensure_min_features=2, ensure_2d=True,
                           force_all_finite=False, dtype='float64')

    # Computes the accumulated cost matrix
    acc_cost_mat = np.ones((n_timestamps_1, n_timestamps_2)) * np.inf
    acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
        cost_mat[0, 0: region[1, 0]]
    )
    acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
        cost_mat[0: region[1, 0], 0]
    )
    region_ = np.copy(region)

    region_[0] = np.maximum(region_[0], 1)
    ant_acc_min_i = -1
    acc_count = 0
    for i in range(1, n_timestamps_1):
        for j in range(region_[0, i], region_[1, i]):
            # Implementation of a Slope-constraint as a step-pattern:
            # This constraint will enforce that the algorithm can only take up to 2 consecutive steps along the time wraping directions.
            # I decided to make it symetric because in (Sakoe and Chiba, 1978) they state that symetric means that DTW(A, B) == DTW(B, A), although I am not convinced why it's not the case in the asymetric implementation.
            # Besides, the asymetric case has a bias towards the diagonal which I thought could be desirable in our case, that said, having DTW(A, B) == DTW(B, A) may prove even more important, especially in further
            # applications of this algorithm for ECG comparison.
            # This implementation is further explained in (Sakoe and Chiba, 1978) and correspondes to the one with P = 0.5, (P = n/m, where P is a rule being inforced, n is the number of steps in the diagonal
            # direction and m is the steps in the time wraping direction).
            acc_cost_mat[i, j] = min(
                acc_cost_mat[i - 1, j - 3] + 2 * cost_mat[i, j - 2] + cost_mat[i, j - 1] + cost_mat[i, j],
                acc_cost_mat[i - 1, j - 2] + 2 * cost_mat[i, j - 1] + cost_mat[i, j],
                acc_cost_mat[i - 1, j - 1] + 2 * cost_mat[i, j],
                acc_cost_mat[i - 2, j - 1] + 2 * cost_mat[i - 1, j] + cost_mat[i, j],
                acc_cost_mat[i - 3, j - 1] + 2 * cost_mat[i - 2, j] + cost_mat[i - 1, j] + cost_mat[i, j]
            )

    dtw_dist = acc_cost_mat[-1, -1] / (
            n_timestamps_1 + n_timestamps_2)  # Normalisation M+N according to (Sakoe and Chiba, 1978)

    dtw_dist = dtw_dist / np.amax(
        np.abs(y)) * target_max_amplitude  # 2022/05/04 - Normalise by lead amplitude to weight all leads similarly
    dtw_dist = math.sqrt(dtw_dist)

    # Penalty for ECG-width differences
    ecg_width_cost = small_c * (n_timestamps_1 - n_timestamps_2) ** 2 / min(n_timestamps_1, n_timestamps_2)

    #     return (dtw_dist+ecg_width_cost, dtw_dist, cost_mat/(n_timestamps_1+n_timestamps_2), acc_cost_mat/(n_timestamps_1+n_timestamps_2), path)
    return dtw_dist, ecg_width_cost


def calculate_ecg_biomarker_errors(nb_leads, predicted_ecg_list, target_ecg, activation_times):
    """This function has hardcoded values for the clinical ECG from DTI004
    This function assumes a 12-lead ECG in the order: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']"""
    # TODO the values for the biomarkers should be automatically calculated from the target_ecg
    warn('This function is obsolete and should not be used if possible!')
    raise(Exception, 'This function is only valid for DTI004')
    clinical_qt_interval =[345, 349, 318, 342, 343, 343, 347, 355]
    clinical_qtpeak_interval = [280, 284, 257, 269, 276, 280, 288, 293]
    clinical_tpeak = [0.138, 0.292, 0.155, 0.471, 0.340, 0.246, 0.187, 0.101]
    clinical_t_polarity = [1., 1., 1., 1., 1., 1., 1., 1.]
    qt_error = np.zeros((nb_leads, predicted_ecg_list.shape[0]))
    qtpeak_error = np.zeros((nb_leads, predicted_ecg_list.shape[0]))
    tpeak_error = np.zeros((nb_leads, predicted_ecg_list.shape[0]))
    tpeak_dispersion_v5_v3_error = np.zeros((predicted_ecg_list.shape[0]))
    t_polarity_error = np.zeros((nb_leads, predicted_ecg_list.shape[0]))
    v3_index = 4
    v5_index = 6
    tpeak_dispersion_v5_v3_ground_truth = clinical_qtpeak_interval[v5_index] - clinical_qtpeak_interval[v3_index]
    for sample_i in range(predicted_ecg_list.shape[0]):
        for lead_i in range(nb_leads):
            qrs_dur, qt_dur, t_pe, t_peak, qtpeak_dur, t_polarity = calculate_ecg_biomarker_old(
                ecg=predicted_ecg_list[sample_i, lead_i, :], lat=activation_times)
            qt_error[lead_i, sample_i] = (qt_dur - clinical_qt_interval[lead_i])/clinical_qt_interval[lead_i]
            qtpeak_error[lead_i, sample_i] = (qtpeak_dur - clinical_qtpeak_interval[lead_i])/clinical_qtpeak_interval[lead_i]
            tpeak_error[lead_i, sample_i] = (t_peak - clinical_tpeak[lead_i]) / clinical_tpeak[lead_i]
            t_polarity_error[lead_i, sample_i] = (t_polarity - clinical_t_polarity[lead_i])/clinical_t_polarity[lead_i]
            if lead_i == v3_index:
                qtpeak_dur_v3 = qtpeak_dur
            if lead_i == v5_index:
                qtpeak_dur_v5 = qtpeak_dur
        tpeak_dispersion_v5_v3 = qtpeak_dur_v5 - qtpeak_dur_v3
        tpeak_dispersion_v5_v3_error[sample_i] = abs((tpeak_dispersion_v5_v3 - tpeak_dispersion_v5_v3_ground_truth) / tpeak_dispersion_v5_v3_ground_truth)
    result_error = np.concatenate((np.linalg.norm(qt_error, axis=0)[:, np.newaxis],
                                   np.linalg.norm(qtpeak_error, axis=0)[:, np.newaxis],
                                   np.linalg.norm(tpeak_error, axis=0)[:, np.newaxis],
                                   np.linalg.norm(t_polarity_error, axis=0)[:, np.newaxis]),
                                   # -result_pcc[:, np.newaxis],
                                   # tpeak_dispersion_v5_v3_error[:, np.newaxis]),
                                  axis=1)
    return result_error


def calculate_ecg_augmented_biomarker_dictionary_from_only_ecg(
        heart_rate,
        lead_v3_i, lead_v5_i, max_lat_list, predicted_ecg_list, qtc_dur_name, qtpeak_dur_name, t_pe_name, t_peak_name,
        t_polarity_name, tpeak_dispersion_name):
    def add_lead_biomarker(biomarker_dictionary, biomarker_lead, biomarker_lead_name, nb_leads):
        for lead_i in range(nb_leads):
            biomarker_dictionary[get_biomarker_lead_name(biomarker_lead_name, lead_i)] = biomarker_lead[:, lead_i]
        return biomarker_dictionary
    qtc_dur, qtc_dur_lead, t_pe, t_pe_lead, t_peak, t_peak_lead, qtpeak_dur, qtpeak_dur_lead, t_polarity, \
        t_polarity_lead, tpeak_dispersion = calculate_ecg_augmented_biomarker_from_only_ecg(
        heart_rate=heart_rate, max_lat_list=max_lat_list, predicted_ecg_list=predicted_ecg_list, lead_v3_i=lead_v3_i,
        lead_v5_i=lead_v5_i)
    nb_leads = predicted_ecg_list.shape[1]
    biomarker_dictionary = {}
    # qt_dur
    biomarker_dictionary[qtc_dur_name] = qtc_dur
    biomarker_dictionary = add_lead_biomarker(biomarker_dictionary=biomarker_dictionary, biomarker_lead=qtc_dur_lead,
                                              biomarker_lead_name=qtc_dur_name, nb_leads=nb_leads)
    # t_pe
    biomarker_dictionary[t_pe_name] = t_pe
    biomarker_dictionary = add_lead_biomarker(biomarker_dictionary=biomarker_dictionary, biomarker_lead=t_pe_lead,
                                              biomarker_lead_name=t_pe_name, nb_leads=nb_leads)
    # qtpeak_dur
    biomarker_dictionary[qtpeak_dur_name] = qtpeak_dur
    biomarker_dictionary = add_lead_biomarker(biomarker_dictionary=biomarker_dictionary, biomarker_lead=qtpeak_dur_lead,
                                              biomarker_lead_name=qtpeak_dur_name, nb_leads=nb_leads)
    # t_peak
    biomarker_dictionary[t_peak_name] = t_peak
    biomarker_dictionary = add_lead_biomarker(biomarker_dictionary=biomarker_dictionary, biomarker_lead=t_peak_lead,
                                              biomarker_lead_name=t_peak_name, nb_leads=nb_leads)
    # t_polarity
    biomarker_dictionary[t_polarity_name] = t_polarity
    biomarker_dictionary = add_lead_biomarker(biomarker_dictionary=biomarker_dictionary, biomarker_lead=t_polarity_lead,
                                              biomarker_lead_name=t_polarity_name, nb_leads=nb_leads)
    # tpeak_dispersion
    biomarker_dictionary[tpeak_dispersion_name] = tpeak_dispersion
    return biomarker_dictionary


def calculate_r_wave_progression_healthy_score(r_wave_progression_index):
    r_wave_progression_healthy_index_list = get_qrs_r_progression_healthy_index_list()
    # print('r_wave_progression_healthy_index_list ', r_wave_progression_healthy_index_list)
    # print('r_wave_progression_index ', r_wave_progression_index)
    r_wave_progression_healthy_score_list = []
    for r_wave_progression_healthy_i in range(len(r_wave_progression_healthy_index_list)):
        r_wave_progression_healthy_index = r_wave_progression_healthy_index_list[r_wave_progression_healthy_i]
        r_wave_progression_healthy_score_list.append(calculate_order_rmse_discrepancy_between_index_list(
            index_list_1=r_wave_progression_index,index_list_2=r_wave_progression_healthy_index))
    return np.amin(r_wave_progression_healthy_score_list)


def calculate_s_wave_progression_healthy_score(s_wave_progression_index):
    s_wave_progression_healthy_index_list = get_qrs_s_progression_healthy_index_list()
    # print('s_wave_progression_healthy_index_list ', s_wave_progression_healthy_index_list)
    # s_wave_progression_healthy_index_alternative = get_qrs_s_progression_healthy_index_alternative()
    # print('s_wave_progression_healthy_index_alternative ', s_wave_progression_healthy_index_alternative)
    # print('s_wave_progression_index ', s_wave_progression_index)
    s_wave_progression_healthy_score_list = []
    for s_wave_progression_healthy_i in range(len(s_wave_progression_healthy_index_list)):
        s_wave_progression_healthy_index = s_wave_progression_healthy_index_list[s_wave_progression_healthy_i]
        s_wave_progression_healthy_score_list.append(calculate_order_rmse_discrepancy_between_index_list(
            index_list_1=s_wave_progression_index, index_list_2=s_wave_progression_healthy_index))
    return np.amin(s_wave_progression_healthy_score_list)


def calculate_s_larger_r_progression_healthy_score(s_larger_r_progression_index):
    s_larger_r_progression_healthy_index = get_qrs_s_larger_r_progression_healthy_index()
    s_larger_r_progression_healthy_optional_index = get_qrs_s_larger_r_progression_healthy_optional_index()
    s_larger_r_progression_index = s_larger_r_progression_index.tolist()
    # print('s_larger_r_progression_index ', type(s_larger_r_progression_index))
    # print('s_larger_r_progression_index- ', s_larger_r_progression_index)
    return calculate_content_discrepancy_between_index_list(
        mandatory_list=s_larger_r_progression_healthy_index,
        optional_list=s_larger_r_progression_healthy_optional_index, predicted_list=s_larger_r_progression_index)


def calculate_r_larger_s_progression_healthy_score(r_larger_s_progression_index):
    r_larger_s_progression_healthy_index = get_qrs_r_larger_s_progression_healthy_index()
    r_larger_s_progression_healthy_optional_index = get_qrs_r_larger_s_progression_healthy_optional_index()
    r_larger_s_progression_index = r_larger_s_progression_index.tolist()
    return calculate_content_discrepancy_between_index_list(
        mandatory_list=r_larger_s_progression_healthy_index,
        optional_list=r_larger_s_progression_healthy_optional_index, predicted_list=r_larger_s_progression_index)


def calculate_r_s_sr_rs_wave_progression_healthy_score_population(max_lat_list, predicted_ecg_list):
    r_wave_progression_healthy_score_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
    s_wave_progression_healthy_score_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
    s_larger_r_progression_healthy_score_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
    r_larger_s_progression_healthy_score_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
    threads_num = multiprocessing.cpu_count()
    # Uncomment the following lines to turn off the parallelisation.
    # if True:
    #     for sample_i in range(predicted_ecg_list.shape[0]):
    with pymp.Parallel(min(threads_num, predicted_ecg_list.shape[0])) as p1:
        for sample_i in p1.range(predicted_ecg_list.shape[0]):
            r_wave_progression_index, s_wave_progression_index, s_larger_r_progression_index, r_larger_s_progression_index = \
                calculate_r_s_sr_rs_wave_progression(max_lat=max_lat_list[sample_i],
                                              predicted_ecg=predicted_ecg_list[sample_i, :, :])
            # print('In the loop')
            # print('r_wave_progression_index ', r_wave_progression_index)
            # print('s_wave_progression_index ', s_wave_progression_index)
            # print('s_larger_r_progression_index ', s_larger_r_progression_index)
            # print('r_larger_s_progression_index ', r_larger_s_progression_index)
            # print('entering functions')
            r_wave_progression_healthy_score = calculate_r_wave_progression_healthy_score(
                r_wave_progression_index=r_wave_progression_index)
            s_wave_progression_healthy_score = calculate_s_wave_progression_healthy_score(
                s_wave_progression_index=s_wave_progression_index)
            s_larger_r_progression_healthy_score = calculate_s_larger_r_progression_healthy_score(
                s_larger_r_progression_index=s_larger_r_progression_index)
            r_larger_s_progression_healthy_score = calculate_r_larger_s_progression_healthy_score(
                r_larger_s_progression_index=r_larger_s_progression_index)

            r_wave_progression_healthy_score_list[sample_i] = r_wave_progression_healthy_score
            s_wave_progression_healthy_score_list[sample_i] = s_wave_progression_healthy_score
            s_larger_r_progression_healthy_score_list[sample_i] = s_larger_r_progression_healthy_score
            r_larger_s_progression_healthy_score_list[sample_i] = r_larger_s_progression_healthy_score

    # print('r_wave_progression_healthy_score_list ', r_wave_progression_healthy_score_list)
    # print('s_wave_progression_healthy_score_list ', s_wave_progression_healthy_score_list)
    # print('s_larger_r_progression_healthy_score_list ', s_larger_r_progression_healthy_score_list)
    # print('r_larger_s_progression_healthy_score_list ', r_larger_s_progression_healthy_score_list)
    return r_wave_progression_healthy_score_list, s_wave_progression_healthy_score_list,\
        s_larger_r_progression_healthy_score_list, r_larger_s_progression_healthy_score_list


def calculate_ecg_qrs_width_healthy_score_population(max_lat_list, predicted_ecg_list):
    qrs_width_list = calculate_qrs_width_population(max_lat_list, predicted_ecg_list)
    # print('qrs_width_list ', qrs_width_list)
    qrs_dur_healthy_range = get_qrs_dur_healthy_range()
    return calculate_range_discrepancy(predicted_list=qrs_width_list, good_range=qrs_dur_healthy_range)
    # qrs_width_healthy_score_population = np.zeros(qrs_width_population.shape, dtype=float)
    # qrs_width_population_too_small = qrs_width_population < min(qrs_dur_healthy_range)
    # qrs_width_healthy_score_population[qrs_width_population_too_small] = np.abs(qrs_width_population - min(qrs_dur_healthy_range))
    # qrs_width_population_too_large = qrs_width_population > max(qrs_dur_healthy_range)
    # qrs_width_healthy_score_population[qrs_width_population_too_large] = np.abs(
    #     qrs_width_population - max(qrs_dur_healthy_range))
    # return qrs_width_healthy_score_population


def calculate_ecg_qrs_axis_healthy_score_population(max_lat_list, predicted_ecg_list):
    qrs_axis_list = calculate_ecg_qrs_axis_population(max_lat_list=max_lat_list, predicted_ecg_list=predicted_ecg_list)
    qrs_axis_healthy_range = get_qrs_axis_healthy_range()
    # print('qrs_axis_list ', qrs_axis_list)
    return calculate_range_discrepancy(predicted_list=qrs_axis_list, good_range=qrs_axis_healthy_range)


def calculate_ecg_qrs_nb_peak_healthy_score_population(max_lat_list, predicted_ecg_list):
    qrs_nb_peak_pos_neg_per_lead_list = calculate_ecg_qrs_nb_peak_pos_neg_population(max_lat_list=max_lat_list, predicted_ecg_list=predicted_ecg_list)
    # print('qrs_nb_peak_per_lead_list ', qrs_nb_peak_pos_neg_per_lead_list)
    qrs_nb_positive_peak_healthy_range = get_qrs_nb_positive_peak_healthy_range()
    qrs_nb_negative_peak_healthy_range = get_qrs_nb_negative_peak_healthy_range()
    qrs_nb_peak_healthy_range = get_qrs_nb_peak_healthy_range()
    # nb_peak_healthy_score_population = np.zeros((predicted_ecg_list.shape[0]))
    nb_peak_healthy_score_population = pymp.shared.array((predicted_ecg_list.shape[0], predicted_ecg_list.shape[1]), dtype=np.float64)
    threads_num = multiprocessing.cpu_count()
    # Uncomment the following lines to turn off the parallelisation.
    # if True:
    #     for lead_i in range(predicted_ecg_list.shape[1]):
    with pymp.Parallel(min(threads_num, predicted_ecg_list.shape[1])) as p1:
        for lead_i in p1.range(predicted_ecg_list.shape[1]):
            nb_peak_healthy_score_population[:, lead_i] = \
                + calculate_range_discrepancy(
                    predicted_list=qrs_nb_peak_pos_neg_per_lead_list[:, lead_i, 0],
                    good_range=qrs_nb_positive_peak_healthy_range) \
                + calculate_range_discrepancy(
                    predicted_list=qrs_nb_peak_pos_neg_per_lead_list[:, lead_i, 1],
                    good_range=qrs_nb_negative_peak_healthy_range) \
                + calculate_range_discrepancy(
                    predicted_list=qrs_nb_peak_pos_neg_per_lead_list[:, lead_i, 0]
                                   + qrs_nb_peak_pos_neg_per_lead_list[:, lead_i, 1],
                    good_range=qrs_nb_peak_healthy_range)
    # print('np.sum(nb_peak_healthy_score_population, axis=1) ', np.sum(nb_peak_healthy_score_population, axis=1))
    return np.sum(nb_peak_healthy_score_population, axis=1)


# def calculate_ecg_qrs_nb_peak_healthy_score_population(max_lat_list, predicted_ecg_list):
#     nb_peak_healthy_score_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
#     threads_num = multiprocessing.cpu_count()
#     # Uncomment the following lines to turn off the parallelisation.
#     if True:
#         for sample_i in range(predicted_ecg_list.shape[0]):
#     # with pymp.Parallel(min(threads_num, predicted_ecg_list.shape[0])) as p1:
#     #     for sample_i in p1.range(predicted_ecg_list.shape[0]):
#             r_wave_progression_index, s_wave_progression_index = \
#                 calculate_ecg_qrs_nb_peak_pos_neg(max_lat=max_lat_list[sample_i],
#                                                   predicted_ecg=predicted_ecg_list[sample_i, :, :])
#             print('In the loop')
#             print('r_wave_progression_index ', r_wave_progression_index)
#             print('s_wave_progression_index ', s_wave_progression_index)
#             print('entering functions')
#             r_wave_progression_healthy_score = calculate_r_wave_progression_healthy_score(
#                 r_wave_progression_index=r_wave_progression_index)
#
#             r_wave_progression_healthy_score_list[sample_i] = r_wave_progression_healthy_score
#
#     print('r_wave_progression_healthy_score_list ', r_wave_progression_healthy_score_list)
#     return r_wave_progression_healthy_score_list
#
#     # qrs_nb_peak_per_lead_list = pymp.shared.array((predicted_ecg_list.shape[0], predicted_ecg_list.shape[1]), dtype=np.float64)
#     # threads_num = multiprocessing.cpu_count()
#     # # Uncomment the following lines to turn off the parallelisation.
#     # # if True:
#     # #     for sample_i in range(qrs_nb_peak_list.shape[0]):
#     # with pymp.Parallel(min(threads_num, qrs_nb_peak_per_lead_list.shape[0])) as p1:
#     #     for sample_i in p1.range(qrs_nb_peak_per_lead_list.shape[0]):
#     #         qrs_nb_peak_per_lead_list[sample_i, :] = calculate_ecg_qrs_nb_peak(max_lat=max_lat_list[sample_i],
#     #                                                                            predicted_ecg=predicted_ecg_list[sample_i, :, :])
#     # return qrs_nb_peak_per_lead_list


def calculate_ecg_qrs_healthy_score(max_lat_list, predicted_ecg_list):
    '''Calculates, aggregates and normalises:
        1) QRS duration error between simulated and healthy - normalised by healthy range width
        2) QRS axis error between simulated and healthy - normalised by healthy range width
        3) R wave progression RMSE between simulated and healthy - normalised by number of (precordial) leads used to evaluate minus 1 - 6-1
        4) S wave progression RMSE between simulated and healthy - normalised by number of (precordial) leads used to evaluate minus 1 - 6-1
        5) S>R -> R>R transition error between simulated and healthy - normalised by number of (precordial) leads used to evaluate - 6
            This last one is computed in two separate steps and then added together.
        6) QRS number of peaks minimized to discourage notched QRS complexes - normalised by number of leads used - 8
            This metric gives results about 3 times larger than the rest because it's internally composed by three metrics:
            6.1) QRS number of positive peaks (positive value)
            6.2) QRS number of negative peaks (negative value)
            6.3) QRS total number of peaks
    '''
    # print('predicted_ecg_list ', predicted_ecg_list.shape)
    # TODO REMOVE FOLLOWING CODE
    # if True:
    #     # TODO REMOVE FOLLOWING CODE
    #     print('TODO REMOVE FOLLOWING CODE')
    #     anatomy_subject_name = 'DTI032'  #'UKB_1000532' #'UKB_1000268' #   'UKB_1000532' 'DTI004'  #'rodero_13' # 'rodero_13'  # 'DTI004'  # 'UKB_1000532' #'UKB_1000268'
    #     ecg_subject_name = 'DTI032'  #'UKB_1000532' #'UKB_1000268' #'UKB_1000532' # 'UKB_1000268' #  'UKB_1000532' #'DTI004'  # 'DTI004'  # 'UKB_1000532' # 'UKB_1000268'  # Allows using a different ECG for the personalisation than for the anatomy
    #     print('anatomy_subject_name: ', anatomy_subject_name)
    #     print('ecg_subject_name: ', ecg_subject_name)
    #     # ####################################################################################################################
    #     # # TODO THIs kills all the processes every time you run the inference because it tries to exceed the allowed memory
    #     # # Set the memory limit to 100GB (in bytes) - Heartsrv has 126GB
    #     # memory_limit = 60 * 1024 * 1024 * 1024
    #     # # Set the memory limit for the current process
    #     # resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    #     # ####################################################################################################################
    #     # print(
    #     #     'Caution, all the hyper-parameters are set assuming resolutions of 1000 Hz in all time-series.')  # TODO: make hyper-parameters relative to the time series resolutions.
    #     # # TODO: enable having different resolutions in different time-series in the code.
    #     # # Get the directory of the script
    #     # script_directory = os.path.dirname(os.path.realpath(__file__))
    #     # print('Script directory:', script_directory)
    #     # # Change the current working directory to the script dierctory
    #     # os.chdir(script_directory)
    #     # working_directory = os.getcwd()
    #     # print('Working directory:', working_directory)
    #     # # Clear Arguments to prevent Argument recycling
    #     # script_directory = None
    #     # working_directory = None
    #     ####################################################################################################################
    #     # LOAD FUNCTIONS AFTER DEFINING THE WORKING DIRECTORY
    #     from conduction_system import EmptyConductionSystem
    #     from ecg_functions import PseudoQRSTetFromStepFunction
    #     from geometry_functions import SimpleCardiacGeoTet
    #     from cellular_models import StepFunctionUpstrokeEP
    #     from path_config import get_path_mapping
    #     from utils import get_vc_rt_name, get_vc_ab_cut_name
    #
    #     print('All imports done!')
    #     ####################################################################################################################
    #     # Load the path configuration in the current server
    #     if os.path.isfile('../.custom_config/.your_path_mapping.txt'):
    #         path_dict = get_path_mapping()
    #     else:
    #         raise 'Missing data and results configuration file at: ../.custom_config/.your_path_mapping.txt'
    #     ####################################################################################################################
    #     # Step 1: Define paths and other environment variables.
    #     # General settings:
    #     source_resolution = 'coarse'
    #     verbose = True
    #     # Input Paths:
    #     data_dir = path_dict["data_path"]
    #     # TODO revert back to using the qrs only for this script. This is a temporary hack to test UKB subjects.
    #     clinical_data_filename = 'clinical_data/' + ecg_subject_name + '_clinical_qrs_ecg.csv'
    #     clinical_data_filename_path = data_dir + clinical_data_filename
    #     geometric_data_dir = data_dir + 'geometric_data/'
    #     # Output Paths:
    #     ep_model_qrs = 'stepFunction'
    #     # Module names:
    #     propagation_module_name = 'propagation_module'
    #     electrophysiology_module_name = 'electrophysiology_module'
    #
    #     ####################################################################################################################
    #     # Step 2: Create Cellular Electrophysiology model. In this case, it will use a step function as the AP's upstroke.
    #     print('Step 2: Create Cellular Electrophysiology model, using a ToROrd APD dictionary.')
    #     # Arguments for cellular model:
    #     resting_vm_value = 0.
    #     upstroke_vm_value = 1.
    #     if ep_model_qrs == 'stepFunction':
    #         cellular_model = StepFunctionUpstrokeEP(resting_vm_value=resting_vm_value, upstroke_vm_value=upstroke_vm_value,
    #                                                 verbose=verbose)
    #     else:
    #         raise Exception('Uncontrolled cellular model for the inference of the activation properties from QRS signals!')
    #     ####################################################################################################################
    #     # Step 3: Generate an Eikonal-friendly geometry.
    #     print('Step 3: Generate a cardiac geometry that can run the Eikonal.')
    #     # Argument setup: (in Alphabetical order)
    #     # vc_ab_name = get_vc_ab_name()
    #     vc_ab_cut_name = get_vc_ab_cut_name()
    #     vc_rt_name = get_vc_rt_name()
    #     # vc_tm_name = get_vc_tm_name()
    #     # vc_name_list = [vc_ab_cut_name, vc_rt_name, get_vc_tm_name(), get_vc_aprt_name(), get_vc_rvlv_name()]
    #     vc_name_list = [vc_ab_cut_name, vc_rt_name]  # , vc_tm_name]#, vc_tv_name]
    #     # Only one celltype/no-celltype, because its using a step function as an action potential.
    #     celltype_vc_info = {}
    #     # Create geometry with a dummy conduction system to allow initialising the geometry.
    #     geometry = SimpleCardiacGeoTet(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
    #                                conduction_system=EmptyConductionSystem(verbose=verbose),
    #                                geometric_data_dir=geometric_data_dir, resolution=source_resolution,
    #                                subject_name=anatomy_subject_name, vc_name_list=vc_name_list, verbose=verbose)
    #     ####################################################################################################################
    #     # Step 7: Create ECG calculation method. In this case, the ecg will calculate only the QRS and will use a step
    #     # function as the AP's upstroke.
    #     print('Step 7: Create ECG calculation method. Using step function.')
    #     # Arguments for ECG calculation:
    #     filtering = True
    #     max_len_qrs = 256  # This hyper-paramter is used when paralelising the ecg computation, because it needs a structure to synchronise the results from the multiple threads.
    #     max_len_ecg = max_len_qrs
    #     normalise = True
    #     zero_align = True
    #     frequency = 1000  # Hz
    #     if frequency != 1000:
    #         warn(
    #             'The hyper-parameter frequency is only used for filtering! If you dont use 1000 Hz in any time-series in the code, the other hyper-parameters will not give the expected outcome!')
    #     low_freq_cut = 0.5
    #     high_freq_cut = 150
    #     I_name = 'I'
    #     II_name = 'II'
    #     v3_name = 'V3'
    #     v5_name = 'V5'
    #     lead_names = [I_name, II_name, 'V1', 'V2', v3_name, 'V4', v5_name, 'V6']
    #     nb_leads = len(lead_names)
    #     # Read clinical data
    #     clinical_ecg_raw = np.genfromtxt(clinical_data_filename_path,
    #                                      delimiter=',')  # No offset allowed in clinical QRS recordings
    #     # TODO revert this change, this is only needed when using an ECG that has the full beat instead of a trimmed QRS, and the
    #     # TODO cutting point will change from subject to subject, either have an automatic delineator, or preprocess the QRS beforehand
    #     clinical_ecg_raw = clinical_ecg_raw[:, :max_len_qrs]  # TODO remove this line
    #     # Create ECG model
    #     ecg_model = PseudoQRSTetFromStepFunction(electrode_positions=geometry.electrode_xyz, filtering=filtering,
    #                                              frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
    #                                              low_freq_cut=low_freq_cut,
    #                                              max_len_qrs=max_len_qrs, nb_leads=nb_leads, nodes_xyz=geometry.node_xyz,
    #                                              normalise=normalise, reference_ecg=clinical_ecg_raw, tetra=geometry.tetra,
    #                                              tetra_centre=geometry.get_tetra_centre(), verbose=verbose,
    #                                              zero_align=zero_align)
    #     clinical_ecg = ecg_model.preprocess_ecg(clinical_ecg_raw)
    #     # clinical_ecg_new = np.zeros((predicted_ecg_list.shape[1], predicted_ecg_list.shape[2]))
    #     # if clinical_ecg.shape[1] >= clinical_ecg_new.shape[1]:
    #     #     for lead_i in range(clinical_ecg_new.shape[0]):
    #     #         clinical_ecg_new[lead_i, :] = clinical_ecg[lead_i, :clinical_ecg_new.shape[1]]
    #     # else:
    #     #     for lead_i in range(clinical_ecg_new.shape[0]):
    #     #         clinical_ecg_new[lead_i, :] = clinical_ecg[lead_i, -1]
    #     #         clinical_ecg_new[lead_i, :clinical_ecg.shape[1]] = clinical_ecg[lead_i, :]
    #     # predicted_ecg_list[0, :, :] = clinical_ecg_new
    #     # max_lat_list[0] = clinical_ecg.shape[1]
    #
    #     max_lat_list = np.asarray([clinical_ecg.shape[1]])
    #     predicted_ecg_list = clinical_ecg[np.newaxis, :, :]
    #     # max_lat_list = max_lat_list[0][np.newaxis]
    #     # predicted_ecg_list = predicted_ecg_list[0, :, :][np.newaxis, :, :]
    # TODO REMOVE ABOVE CODE

    # QRS duration
    qrs_dur_healthy_range = get_qrs_dur_healthy_range()
    qrs_duration_normalise_factor = max(qrs_dur_healthy_range) - min(qrs_dur_healthy_range)
    qrs_duration_error_list = calculate_ecg_qrs_width_healthy_score_population(max_lat_list,
                                                                               predicted_ecg_list) / qrs_duration_normalise_factor
    # QRS axis
    qrs_axis_healthy_range = get_qrs_axis_healthy_range()
    qrs_axis_normalise_factor = max(qrs_axis_healthy_range) - min(qrs_axis_healthy_range)
    qrs_axis_error_list = calculate_ecg_qrs_axis_healthy_score_population(max_lat_list,
                                                                           predicted_ecg_list) / qrs_axis_normalise_factor
    # QRS wave progressions
    nb_precordial_lead = get_nb_precordial_lead()
    r_wave_progression_healthy_score_list, s_wave_progression_healthy_score_list, \
        s_larger_r_progression_healthy_score_list, r_larger_s_progression_healthy_score_list = \
        calculate_r_s_sr_rs_wave_progression_healthy_score_population(max_lat_list, predicted_ecg_list)
    r_wave_progression_error_list = r_wave_progression_healthy_score_list/(nb_precordial_lead-1)
    s_wave_progression_error_list = s_wave_progression_healthy_score_list/(nb_precordial_lead-1)
    s_larger_r_progression_error_list = s_larger_r_progression_healthy_score_list/nb_precordial_lead
    r_larger_s_progression_error_list = r_larger_s_progression_healthy_score_list/nb_precordial_lead

    # QRS number of peaks
    nb_unique_lead = get_nb_unique_lead()
    nb_peak_error_list = calculate_ecg_qrs_nb_peak_healthy_score_population(max_lat_list, predicted_ecg_list)/nb_unique_lead

    # Error aggregation
    error_aggregation = qrs_duration_error_list + qrs_axis_error_list + r_wave_progression_error_list \
        + s_wave_progression_error_list + s_larger_r_progression_error_list + r_larger_s_progression_error_list \
                        + nb_peak_error_list

    # best_i = np.argmin(error_aggregation)
    # print('best_i ', best_i)
    # best_ecg = predicted_ecg_list[best_i, :, :]
    #
    # print('best error ', error_aggregation[best_i])
    # print('qrs_duration_error ', qrs_duration_error_list[best_i])
    # print('qrs_axis_error ', qrs_axis_error_list[best_i])
    # print('r_wave_progression_error ', r_wave_progression_error_list[best_i])
    # print('s_wave_progression_error ', s_wave_progression_error_list[best_i])
    # print('s_larger_r_progression_error ', s_larger_r_progression_error_list[best_i])
    # print('r_larger_s_progression_error ', r_larger_s_progression_error_list[best_i])
    # print('nb_peak_error ', nb_peak_error_list[best_i])
    #
    # axes, fig = visualise_ecg(ecg_list=best_ecg[np.newaxis, :, :], ecg_color='g', linewidth=3.)
    #
    # best_i = np.argmax(error_aggregation)
    # print('worst_i ', best_i)
    # best_ecg = predicted_ecg_list[best_i, :, :]
    #
    # print('worst error ', error_aggregation[best_i])
    # print('qrs_duration_error ', qrs_duration_error_list[best_i])
    # print('qrs_axis_error ', qrs_axis_error_list[best_i])
    # print('r_wave_progression_error ', r_wave_progression_error_list[best_i])
    # print('s_wave_progression_error ', s_wave_progression_error_list[best_i])
    # print('s_larger_r_progression_error ', s_larger_r_progression_error_list[best_i])
    # print('r_larger_s_progression_error ', r_larger_s_progression_error_list[best_i])
    # print('nb_peak_error ', nb_peak_error_list[best_i])
    #
    # axes, fig = visualise_ecg(ecg_list=best_ecg[np.newaxis, :, :], ecg_color='r', linewidth=3., axes=axes, fig=fig)
    #
    # plt.show(block=False)
    # raise()

    return error_aggregation


# def calculate_qrs_biomarker_errors(predicted_ecg_list, predicted_max_lat_list, target_biomarker_dict):
#     result_error_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
#     threads_num = multiprocessing.cpu_count()
#     # Uncomment the following lines to turn off the parallelisation.
#     # if True:
#     #     for sample_i in range(result_pcc.shape[0]):
#     with pymp.Parallel(min(threads_num, predicted_ecg_list.shape[0])) as p1:
#         for sample_i in p1.range(predicted_ecg_list.shape[0]):
#             qrs_axis_error = abs(target_qrs_axis - predicted_qrs_axis)
#             result_error =
#             result_error_list[sample_i] = result_error
#
#     return result_error_list


# def calculate_qt_dur_biomarker_from_only_ecg(max_lat, predicted_ecg_list): # TODO reformat this and do properly
#     qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat, predicted_ecg_list)
#     return np.mean(qt_dur_lead, axis=1)
#
#
# def calculate_qt_dur_biomarker_from_only_ecg(max_lat, predicted_ecg_list):
#     """This is not to be used with clinical data because it assumes that the signal returns to baseline after the end
#     of the T wave.
#     """
#     # TODO: Assumes ECGs are at 1000 Hz.
#     nb_leads = predicted_ecg_list.shape[1]
#     qt_dur_lead = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads), dtype=np.float64)
#     qt_dur_lead[:, :] = get_nan_value()
#     threads_num = multiprocessing.cpu_count()
#     # Uncomment the following lines to turn off the parallelisation.
#     # if True:
#     #     for sample_i in range(result_pcc.shape[0]):
#     with pymp.Parallel(min(threads_num, qt_dur_lead.shape[0])) as p1:
#         for sample_i in p1.range(qt_dur_lead.shape[0]):
#             for lead_i in range(nb_leads):
#                 qt_dur, t_pe, t_peak, qtpeak_dur, t_polarity = calculate_ecg_lead_biomarker_from_only_ecg(
#                     ecg_lead=predicted_ecg_list[sample_i, lead_i, :], max_lat=max_lat)
#                 qt_dur_lead[sample_i, lead_i] = qt_dur
#     return qt_dur_lead


# def calculate_t_pe_biomarker_from_only_ecg(max_lat, predicted_ecg_list): # TODO reformat this and do properly
#     qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat, predicted_ecg_list)
#     return np.mean(t_pe_lead, axis=1)
#
#
# def calculate_t_peak_biomarker_from_only_ecg(max_lat, predicted_ecg_list): # TODO reformat this and do properly
#     qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat, predicted_ecg_list)
#     return np.mean(t_peak_lead, axis=1)
#
#
# def calculate_qtpeak_dur_biomarker_from_only_ecg(max_lat, predicted_ecg_list): # TODO reformat this and do properly
#     qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat, predicted_ecg_list)
#     return np.mean(qtpeak_dur_lead, axis=1)
#
#
# def calculate_t_polarity_biomarker_from_only_ecg(max_lat, predicted_ecg_list): # TODO reformat this and do properly
#     qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat, predicted_ecg_list)
#     return np.mean(t_polarity_lead, axis=1)
#
#

#
#
# def calculate_tpeak_dispersion_biomarker_from_only_ecg(max_lat, predicted_ecg_list, lead_v3_i, lead_v5_i): # TODO reformat this and do properly
#     qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat, predicted_ecg_list)
#     return calculate_tpeak_dispersion(qtpeak_dur_lead, lead_v3_i, lead_v5_i)


def calculate_ecg_biomarker_old(ecg, lat):
    """This is not to be used with clinical data because it assumes that the signal returns to baseline after the end
    of the T wave."""
    # TODO: Assumes ECGs are at 1000 Hz.
    dV = abs(np.gradient(ecg))
    ddV = abs(np.gradient(dV))
    dV[0:2] = 0.0 # remove gradient artefacts
    ddV[0:2] = 0.0
    # Find Q start
    dVTOL_end_of_Twave = 0.0002 # mV/ms
    # dVTOL_start_of_QRS = 0.0002
    ddVTOL_start_of_Twave = 0.0002 # mV/ms^2

    # for i in range(len(V)):
    #     if (dV[i] > dVTOL_start_of_QRS) & (i > 10):
    #         break
    # q_start_idx = i
    q_start_idx = np.nanmin(lat)

    # Set QRS end
    qrs_end_idx = np.nanmax(lat)
    qrs_dur = qrs_end_idx - q_start_idx # TODO code how to find end of QRS

    # Find T peak and amplitude
    segment = ecg[qrs_end_idx:]
    t_amplitude = abs(segment).max()
    t_peak_idx = np.where(abs(segment) == t_amplitude)[0][0] + qrs_end_idx
    t_sign = np.sign(segment[t_peak_idx - qrs_end_idx])
    t_peak = t_sign * t_amplitude
    t_min = np.amin(segment)
    t_max = abs(np.amax(segment))
    #t_polarity = t_max/t_min * 1/(max(abs(t_max),abs(t_min))) # Value close to 1 is positive monophasic, close to 0 is negative monophasic, around 0.5 is biphasic.
    t_polarity = (t_max + t_min)/(max(abs(t_max), abs(t_min)))
    # Find T-wave end
    for i in range(len(ecg) - 1, t_peak_idx, -1):
        if (dV[i] > dVTOL_end_of_Twave):
            break
    t_end_idx = i

    # Find T start
    # segment = ddV[qrs_end_idx:t_peak_idx]
    # min_dd_idx= np.where(segment == segment.min())[0][0] + qrs_end_idx
    # for i in range(min_dd_idx, t_peak_idx):
    #     if (abs(ddV[i] > ddVTOL_start_of_Twave)):
    #         break
    # t_start_idx = i

    # t_dur = t_end_idx - t_start_idx
    qt_dur = t_end_idx - q_start_idx
    t_pe = t_end_idx - t_peak_idx
    qtpeak_dur = t_peak_idx - q_start_idx
    # t_op = t_start_idx - t_peak_idx

    # landmarks = np.array([[q_start_idx, ecg[q_start_idx]], [qrs_end_idx, ecg[qrs_end_idx]], [t_peak_idx, ecg[t_peak_idx]], [t_end_idx, ecg[t_end_idx]]])
    return qrs_dur, qt_dur, t_pe, t_peak, qtpeak_dur, t_polarity #, landmarks


class Metric:
    def __init__(self):
        pass

    def evaluate_metric(self, predicted_data):
        raise NotImplementedError

    def evaluate_metric_population(self, predicted_data_population):
        raise NotImplementedError


class BiomarkerFromOnlyECG(Metric):
    def __init__(self, biomarker_name_list, heart_rate, lead_v3_i, lead_v5_i, qtc_dur_name, qtpeak_dur_name, t_pe_name,
                 t_peak_name, t_polarity_name, tpeak_dispersion_name):
        super().__init__()
        print('biomarker_name_list ', biomarker_name_list)
        self.biomarker_name_list = biomarker_name_list
        self.heart_rate = heart_rate
        # self.max_lat = max_lat
        self.lead_v3_i = lead_v3_i
        self.lead_v5_i = lead_v5_i
        self.qtc_dur_name = qtc_dur_name
        self.qtpeak_dur_name = qtpeak_dur_name
        self.t_pe_name = t_pe_name
        self.t_peak_name = t_peak_name
        self.t_polarity_name = t_polarity_name
        self.tpeak_dispersion_name = tpeak_dispersion_name

    def evaluate_metric(self, max_lat, predicted_data):
        warn('This function has NOT been tested on just one particle!! It is possible that the average of biomarkers'
             ' is averaging accross the wrong axis!!')
        # qt_dur, qt_dur_lead, t_pe, t_pe_lead, t_peak, t_peak_lead, qtpeak_dur, qtpeak_dur_lead, t_polarity, \
        #     t_polarity_lead, tpeak_dispersion = calculate_ecg_augmented_biomarker_from_only_ecg(
        #     max_lat=self.max_lat, predicted_ecg_list=predicted_data[np.newaxis, :, :], lead_v3_i=self.lead_v3_i,
        #     lead_v5_i=self.lead_v5_i)
        #
        #
        # qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = \
        #     calculate_ecg_biomarker_from_only_ecg(max_lat=self.max_lat, predicted_ecg_list=predicted_data[np.newaxis, :, :])
        # qt_dur_mean = np.mean(qt_dur_lead, axis=1)
        # t_pe_mean = np.mean(t_pe_lead, axis=1)
        # t_peak_mean = np.mean(t_peak_lead, axis=1)
        # qtpeak_dur_mean = np.mean(qtpeak_dur_lead, axis=1)
        # t_polarity_mean = np.mean(t_polarity_lead, axis=1)
        # tpeak_dispersion_v5_v3 = calculate_tpeak_dispersion(qtpeak_dur_lead=qtpeak_dur_lead, lead_v3_i=self.lead_v3_i,
        #                                                     lead_v5_i=self.lead_v5_i)
        # biomarker_mean = np.concatenate(([qt_dur_mean, t_pe_mean, t_peak_mean, qtpeak_dur_mean, t_polarity_mean,
        #                                   tpeak_dispersion_v5_v3]), axis=1)
        # return np.concatenate(([biomarker_mean, qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead]), axis=1)
        # calculate_ecg_augmented_biomarker_from_only_ecg(
        #     max_lat=self.max_lat, predicted_ecg_list=predicted_data[np.newaxis, :, :], lead_v3_i=self.lead_v3_i,
        #     lead_v5_i=self.lead_v5_i)
        biomarker_dictionary = calculate_ecg_augmented_biomarker_dictionary_from_only_ecg(
            heart_rate=self.heart_rate,
            lead_v3_i=self.lead_v3_i, lead_v5_i=self.lead_v5_i, max_lat_list=np.array([max_lat]),
            predicted_ecg_list=predicted_data[np.newaxis, :, :], qtc_dur_name=self.qtc_dur_name,
            qtpeak_dur_name=self.qtpeak_dur_name, t_pe_name=self.t_pe_name, t_peak_name=self.t_peak_name,
            t_polarity_name=self.t_polarity_name, tpeak_dispersion_name=self.tpeak_dispersion_name)
        return dictionary_to_ndarray(dictionary=biomarker_dictionary, key_list_in_order=self.biomarker_name_list)

    def evaluate_metric_population(self, max_lat_population, predicted_data_population):
        # The heart rate is in beats/minute
        # qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(max_lat=self.max_lat, predicted_ecg_list=predicted_data_population)
        # qt_dur_mean = np.mean(qt_dur_lead, axis=1)
        # t_pe_mean = np.mean(t_pe_lead, axis=1)
        # t_peak_mean = np.mean(t_peak_lead, axis=1)
        # qtpeak_dur_mean = np.mean(qtpeak_dur_lead, axis=1)
        # t_polarity_mean = np.mean(t_polarity_lead, axis=1)
        # tpeak_dispersion_v5_v3 = calculate_tpeak_dispersion(qtpeak_dur_lead=qtpeak_dur_lead, lead_v3_i=self.lead_v3_i,
        #                                                     lead_v5_i=self.lead_v5_i)
        # biomarker_mean = np.concatenate(([qt_dur_mean, t_pe_mean, t_peak_mean, qtpeak_dur_mean, t_polarity_mean,
        #                                   tpeak_dispersion_v5_v3]), axis=1)
        # return np.concatenate(([biomarker_mean, qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead]),
        #                       axis=1)
        biomarker_dictionary = calculate_ecg_augmented_biomarker_dictionary_from_only_ecg(
            heart_rate=self.heart_rate,
            lead_v3_i=self.lead_v3_i, lead_v5_i=self.lead_v5_i, max_lat_list=max_lat_population,
            predicted_ecg_list=predicted_data_population, qtc_dur_name=self.qtc_dur_name,
            qtpeak_dur_name=self.qtpeak_dur_name, t_pe_name=self.t_pe_name, t_peak_name=self.t_peak_name,
            t_polarity_name=self.t_polarity_name, tpeak_dispersion_name=self.tpeak_dispersion_name)
        return dictionary_to_ndarray(dictionary=biomarker_dictionary, key_list_in_order=self.biomarker_name_list)
        # return calculate_ecg_augmented_biomarker_from_only_ecg(
        #     max_lat=self.max_lat, predicted_ecg_list=predicted_data_population, lead_v3_i=self.lead_v3_i,
        #     lead_v5_i=self.lead_v5_i)


class DiscrepancyToTargetData(Metric):
    def __init__(self):
        super().__init__()

    def evaluate_metric(self, predicted_data, target_data):
        raise NotImplementedError

    def evaluate_metric_population(self, predicted_data_population, target_data):
        raise NotImplementedError


class DiscrepancyECGDtw(DiscrepancyToTargetData):  # TODO Rename to a name that includes the fact that its using a target ECG
    def __init__(self, max_slope, mesh_volume, w_max):
        super().__init__()
        self.max_slope = max_slope
        self.mesh_volume = mesh_volume
        self.w_max = w_max

    def evaluate_metric(self, predicted_data, target_data):
        return dtw_ecg_parallel(predicted_ecg_list=predicted_data[np.newaxis, :, :], target_ecg=target_data, max_slope=self.max_slope,
                                mesh_volume=self.mesh_volume, w_max=self.w_max)

    def evaluate_metric_population(self, predicted_data_population, target_data):
        return dtw_ecg_parallel(predicted_ecg_list=predicted_data_population, target_ecg=target_data, max_slope=self.max_slope,
                                mesh_volume=self.mesh_volume, w_max=self.w_max)


class DiscrepancyECG(DiscrepancyToTargetData):  # TODO Rename to a name that includes the fact that its using a target ECG
    def __init__(self, error_method_name):
        super().__init__()
        self.error_method_name = error_method_name

    def evaluate_metric(self, predicted_data, target_data):
        return calculate_ecg_errors(predicted_ecg_list=predicted_data[np.newaxis, :, :], target_ecg=target_data, error_method=self.error_method_name)

    def evaluate_metric_population(self, predicted_data_population, target_data):
        return calculate_ecg_errors(predicted_ecg_list=predicted_data_population, target_ecg=target_data, error_method=self.error_method_name)


class DiscrepancyHealthyQRS(Metric):  # TODO Rename to a name that includes the fact that its using a target ECG
    def __init__(self):
        super().__init__()

    def evaluate_metric(self, predicted_data):
        predicted_ecg = predicted_data[0]
        max_lat = predicted_data[1]
        return calculate_ecg_qrs_healthy_score(max_lat_list=[max_lat], predicted_ecg_list=predicted_ecg[np.newaxis, :, :])

    def evaluate_metric_population(self, predicted_data_population):
        predicted_ecg_list = predicted_data_population[0]
        max_lat_list = predicted_data_population[1]
        return calculate_ecg_qrs_healthy_score(max_lat_list=max_lat_list, predicted_ecg_list=predicted_ecg_list)


# END

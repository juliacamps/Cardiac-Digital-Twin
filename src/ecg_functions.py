# This file contains the ECG computation class
# And all the methods for ECG computation that can be considered
# import multiprocessing
# from abc import ABC

import numpy as np
import pymp
from scipy import signal
import matplotlib.pyplot as plt
from warnings import warn

from postprocess_functions import plot_histogram, visualise_ecg
from utils import get_nan_value, get_nb_precordial_lead, get_lead_V1_index, \
    normalise_field_to_zero_one, get_nb_unique_lead, get_parallel_loop


# Generic calculations
def get_cycle_length(heart_rate):
    '''Input: heart_rate in beats/minute
    Output: cycle_length in ms'''
    # Convert from beats/minute to beats/ms
    # heart_rate_ms = heart_rate / 60000.  # beats / ms
    # cycle_length = 1. / heart_rate_ms  # ms
    return 60000./heart_rate


def get_rr_interval(heart_rate):
    '''Input: heart_rate in beats/minute
    Output: rr_interval in seconds'''
    # Convert from beats/minute to beats/second
    # heart_rate_seconds = heart_rate / 60.  # beats / second
    # rr_interval = 1. / heart_rate_seconds  # seconds
    return 60./heart_rate


def correct_qt_interval(heart_rate, qt_dur):
    rr_interval = get_rr_interval(heart_rate=heart_rate)  # seconds
    # Calculate (corrected) QT (or QTc) interval
    return qt_dur / (rr_interval ** (1. / 3.))


def zero_align_ecg(original_ecg):
    return original_ecg - (original_ecg[:, 0:1] + original_ecg[:, -2:-1]) / 2  # align at zero


def normalise_ecg_between_0_and_1(original_ecg):
    normalised_ecg = normalise_field_to_zero_one(field=original_ecg)  # (original_ecg - np.amin(original_ecg)) / (np.amax(original_ecg) - np.amin(original_ecg))
    normalised_ecg = zero_align_ecg(normalised_ecg) # It gets missaligned after normalising
    return normalised_ecg


def filter_butterworth_ecg(b, a, ecg):
    return signal.filtfilt(b, a, ecg)


def delineate_lead_q_wave_onset(ecg_lead):
    warn('Currently using Zaragozas Matlab code to delineate the clincial ECG signals')
    # TODO Implement a strategy for ECG delineation that we can rely on
    raise NotImplementedError


def delineate_ecg_q_wave_onset(ecg):
    warn('Not implemented yet.')
    nb_leads = ecg.shape[0]
    q_wave_onset_list = np.zeros((nb_leads)) + get_nan_value() # make sure that no value is without defining
    for lead_i in range(nb_leads):
        q_wave_onset_list[lead_i] = delineate_lead_q_wave_onset(ecg[lead_i, :])
    return q_wave_onset_list


# def delineate_ecg_lead_qrs(ecg_lead, max_lat):
#     return ecg_lead[:int(max_lat)]


def delineate_ecg_qrs_end(max_lat, predicted_ecg):
    return predicted_ecg[:, :int(max_lat)]


# def calculate_qrs_width(max_lat, predicted_ecg):
#     # This function assumes that the LATs have been aligned to start at either 0 or 1 ms -> LATs = LATs - min(LATs)
#     # TODO Ideally this fucntion would call the delineation function and actually compute the width of the QRS
#     return max_lat


def calculate_qrs_width_population(max_lat_list, predicted_ecg_list):
    # This function assumes that the LATs have been aligned to start at either 0 or 1 ms -> LATs = LATs - min(LATs)
    # TODO Ideally this fucntion would call the delineation function and actually compute the width of the QRS
    return max_lat_list


# TODO review this code and make sure it's correct! It has a lot of hacked in values without proper comments
def calculate_ecg_lead_biomarker_from_only_ecg(ecg_lead, max_lat, heart_rate): # TODO review this code and make sure it's correct!
    """This is not to be used with clinical data because it assumes that the signal returns to baseline after the end
        of the T wave.
        Returns the following biomarkers: corrected qt_dur (QTc), t_pe, t_peak, qtpeak_dur, t_polarity
        Input: max_lat is in ms, heart_rate is in beats/minute
        QTc=QT/(RR^(1⁄3)) # RR is in seconds
        Thus, when HR is 60 beats/minute, the QT == QTc
    """
    # TODO: Assumes ECGs are at 1000 Hz.
    dV = abs(np.gradient(ecg_lead))
    ddV = abs(np.gradient(dV))
    dV[0:2] = 0.0 # remove gradient artefacts
    ddV[0:2] = 0.0
    # Find Q start
    dVTOL_end_of_Twave = 0.002 # 0.0002 # mV/ms # TODO This trick will only work with simulated signals

    # Find T peak and amplitude
    qrs_end_idx = int(max_lat)   # The maximum LAT is used as the end of the QRS assuming 1000Hz
    segment = ecg_lead[qrs_end_idx:]
    t_amplitude = abs(segment).max()
    t_peak_idx = np.where(abs(segment) == t_amplitude)[0][0] + qrs_end_idx
    t_sign = np.sign(segment[t_peak_idx - qrs_end_idx])
    t_peak = t_sign * t_amplitude
    t_min = np.amin(segment)
    t_max = abs(np.amax(segment))
    t_polarity = (t_max + t_min)/(max(abs(t_max), abs(t_min)))
    # Find T-wave end
    i = len(ecg_lead) - 1
    for i in range(len(ecg_lead) - 1, t_peak_idx, -1):
        if (dV[i] > dVTOL_end_of_Twave):
            break
    t_end_idx = i

    qt_dur = t_end_idx  # Non-corrected QT interval in ms
    # Correct QT interval using the heart rate
    qtc_dur = correct_qt_interval(heart_rate=heart_rate, qt_dur=qt_dur)
    t_pe = t_end_idx - t_peak_idx
    qtpeak_dur = t_peak_idx
    return qtc_dur, t_pe, t_peak, qtpeak_dur, t_polarity


def calculate_ecg_biomarker_from_only_ecg(heart_rate, max_lat_list, predicted_ecg_list): # TODO use dictionaries??? or separate this into multiple functions one per biomarker!
    """This is not to be used with clinical data because it assumes that the signal returns to baseline after the end
    of the T wave.
    Returns the following biomarkers: "corrected qt_dur (QTc), t_pe, t_peak, qtpeak_dur, t_polarity, tpeak_dispersion_v5_v3"
    """
    # TODO: Give as input a list of the desired biomarkers or compute them in separate functions!!!
    # TODO: Assumes ECGs are at 1000 Hz.

    # v3_index = 4
    # v5_index = 6
    nb_leads = predicted_ecg_list.shape[1]
    # nb_biomarker = 5
    qtc_dur_lead = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads), dtype=np.float64)
    t_pe_lead = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads), dtype=np.float64)
    t_peak_lead = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads), dtype=np.float64)
    qtpeak_dur_lead = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads), dtype=np.float64)
    t_polarity_lead = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads), dtype=np.float64)
    # tpeak_dispersion_v5_v3 = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
    # ecg_lead_biomarker = pymp.shared.array((predicted_ecg_list.shape[0], nb_leads * nb_biomarker), dtype=np.float64)
    # ecg_mean_biomarker = pymp.shared.array((predicted_ecg_list.shape[0], nb_biomarker + 1), dtype=np.float64)
    qtc_dur_lead[:, :] = get_nan_value()
    t_pe_lead[:, :] = get_nan_value()
    t_peak_lead[:, :] = get_nan_value()
    qtpeak_dur_lead[:, :] = get_nan_value()
    t_polarity_lead[:, :] = get_nan_value()
    # tpeak_dispersion_v5_v3[:] = get_nan_value()
    # ecg_lead_biomarker[:, :] = get_nan_value()
    # ecg_mean_biomarker[:, :] = get_nan_value()
    # threads_num = multiprocessing.cpu_count()
    # Uncomment the following lines to turn off the parallelisation.
    # if True:
    #     for sample_i in range(qt_dur_lead.shape[0]):
    iter_gen = get_parallel_loop(data_size=qtc_dur_lead.shape[0])
    for sample_i in iter_gen:
        if True:
    # with pymp.Parallel(min(threads_num, qtc_dur_lead.shape[0])) as p1:
    #     for sample_i in p1.range(qtc_dur_lead.shape[0]):
            # qt_dur_list = np.zeros((nb_leads))
            # t_pe_list = np.zeros((nb_leads))
            # t_peak_list = np.zeros((nb_leads))
            # qtpeak_dur_list = np.zeros((nb_leads))
            # t_polarity_list = np.zeros((nb_leads))
            for lead_i in range(nb_leads):
                qtc_dur, t_pe, t_peak, qtpeak_dur, t_polarity = calculate_ecg_lead_biomarker_from_only_ecg(
                    ecg_lead=predicted_ecg_list[sample_i, lead_i, :], max_lat=max_lat_list[sample_i], heart_rate=heart_rate)
                qtc_dur_lead[sample_i, lead_i] = qtc_dur
                t_pe_lead[sample_i, lead_i] = t_pe
                t_peak_lead[sample_i, lead_i] = t_peak
                qtpeak_dur_lead[sample_i, lead_i] = qtpeak_dur
                t_polarity_lead[sample_i, lead_i] = t_polarity
                # if lead_i == v3_index:
                #     qtpeak_dur_v3 = qtpeak_dur
                # if lead_i == v5_index:
                #     qtpeak_dur_v5 = qtpeak_dur
            # ECG average biomarkers
            # tpeak_dispersion_v5_v3[sample_i] = qtpeak_dur_v5 - qtpeak_dur_v3
            # ecg_mean_biomarker[sample_i, :] = [np.mean(qt_dur_list), np.mean(t_pe_list), np.mean(t_peak_list),
            #                                    np.mean(qtpeak_dur_list), np.mean(t_polarity_list),
            #                                    tpeak_dispersion_v5_v3]
            # # Per Lead biomarkers
            # if return_per_lead_biomarkers:
            #     aux_i_s = 0
            #     ecg_lead_biomarker[sample_i, aux_i_s:nb_leads + aux_i_s] = qt_dur_list
            #     aux_i_s = aux_i_s + nb_leads
            #     ecg_lead_biomarker[sample_i, aux_i_s:nb_leads + aux_i_s] = t_pe_list
            #     aux_i_s = aux_i_s + nb_leads
            #     ecg_lead_biomarker[sample_i, aux_i_s:nb_leads + aux_i_s] = t_peak_list
            #     aux_i_s = aux_i_s + nb_leads
            #     ecg_lead_biomarker[sample_i, aux_i_s:nb_leads + aux_i_s] = qtpeak_dur_list
            #     aux_i_s = aux_i_s + nb_leads
            #     ecg_lead_biomarker[sample_i, aux_i_s:nb_leads + aux_i_s] = t_polarity_list
    warn("The order of the biomarkers is static: qt_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead")
    # if return_per_lead_biomarkers:
    #     result_ecg_biomarker = np.concatenate((ecg_mean_biomarker, ecg_lead_biomarker), axis=1)
    # else:
    #     result_ecg_biomarker = ecg_mean_biomarker
    return qtc_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead


def calculate_tpeak_dispersion(qtpeak_dur_lead, lead_v3_i, lead_v5_i):
    return qtpeak_dur_lead[:, lead_v5_i] - qtpeak_dur_lead[:, lead_v3_i]


def calculate_ecg_augmented_biomarker_from_only_ecg(heart_rate, max_lat_list, predicted_ecg_list, lead_v3_i, lead_v5_i):
    qtc_dur_lead, t_pe_lead, t_peak_lead, qtpeak_dur_lead, t_polarity_lead = calculate_ecg_biomarker_from_only_ecg(
        heart_rate, max_lat_list, predicted_ecg_list)
    qtc_dur = np.mean(qtc_dur_lead, axis=1)
    t_pe = np.mean(t_pe_lead, axis=1)
    t_peak = np.mean(t_peak_lead, axis=1)
    qtpeak_dur = np.mean(qtpeak_dur_lead, axis=1)
    t_polarity = np.mean(t_polarity_lead, axis=1)
    tpeak_dispersion = calculate_tpeak_dispersion(qtpeak_dur_lead, lead_v3_i, lead_v5_i)
    return qtc_dur, qtc_dur_lead, t_pe, t_pe_lead, t_peak, t_peak_lead, qtpeak_dur, qtpeak_dur_lead, t_polarity, \
        t_polarity_lead, tpeak_dispersion


def calculate_ecg_qrs_axis(max_lat, predicted_ecg):
    # ignore_percentage_threshold = 0.1  # Ignore differences between leads less than 10% in the net amplitude of the QRS
    nb_leads = predicted_ecg.shape[0]
    if nb_leads != get_nb_unique_lead():
        raise(Exception, 'This function assumes only 8 leads in the default arrangement: I, II, V1, ..., V6')
    predicted_qrs = delineate_ecg_qrs_end(max_lat, predicted_ecg)
    I_qrs = predicted_qrs[0, :]
    II_qrs = predicted_qrs[1, :]
    # Calculate remining limb and augmented leads using only leads I and II
    III_qrs = II_qrs - I_qrs  # lead III = II - I
    aVR_qrs = (-1. / 2.) * (I_qrs + II_qrs)  # aVR = - 1/2 * (I + II)
    aVL_qrs = (-1. / 2.) * (III_qrs - I_qrs)  # aVL = - 1/2 * (III - I)
    aVF_qrs = (1. / 2.) * (II_qrs + III_qrs)  # aVF = 1/2 * (II + III)
    # limb_augmented_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    limb_augmented_angles = [0., 60., 120., -150., -30., 90.] # https://ecg.utah.edu/lesson/2-1 and https://litfl.com/ecg-axis-interpretation/
    limb_augmented_qrs = np.zeros((len(limb_augmented_angles), I_qrs.shape[0]))
    limb_augmented_qrs[0, :] = I_qrs
    limb_augmented_qrs[1, :] = II_qrs
    limb_augmented_qrs[2, :] = III_qrs
    limb_augmented_qrs[3, :] = aVR_qrs
    limb_augmented_qrs[4, :] = aVL_qrs
    limb_augmented_qrs[5, :] = aVF_qrs
    limb_augmented_qrs_net_amplitude = np.sum(limb_augmented_qrs, axis=1)
    # print('limb_augmented_qrs_net_amplitude ', limb_augmented_qrs_net_amplitude.shape)
    # absolute_limb_augmented_qrs_net_amplitude = np.abs(limb_augmented_qrs_net_amplitude)
    # limb_augmented_index_list = np.argsort(absolute_limb_augmented_qrs_net_amplitude)
    # print('limb_augmented_index ', limb_augmented_qrs_net_amplitude[limb_augmented_index_list])
    # isoelectric_lead_index_list = []
    # positive_lead_index_list = []
    # negative_lead_index_list = []
    # lowest_absolute_limb_augmented_qrs_net_amplitude = abs(limb_augmented_qrs_net_amplitude[limb_augmented_index_list[0]])
    # for limb_augmented_i in range(limb_augmented_index_list.shape[0]):
    #     limb_augmented_index = limb_augmented_index_list[limb_augmented_i]
    #     # limb_augmented_name = limb_augmented_names[limb_augmented_index]
    #     limb_augmented_net_amplitude = limb_augmented_qrs_net_amplitude[limb_augmented_index_list[limb_augmented_i]]
    #     percentage_difference = abs(
    #         lowest_absolute_limb_augmented_qrs_net_amplitude - abs(limb_augmented_net_amplitude)
    #     ) / lowest_absolute_limb_augmented_qrs_net_amplitude
    #     if percentage_difference < ignore_percentage_threshold:
    #         isoelectric_lead_index_list.append(limb_augmented_index)
    #     if limb_augmented_net_amplitude > 0.:
    #         positive_lead_index_list.append(limb_augmented_index)
    #     elif limb_augmented_net_amplitude < 0.:
    #         negative_lead_index_list.append(limb_augmented_index)
    #
    # # isoelectric_lead_names = limb_augmented_names[np.asarray(isoelectric_lead_index_list)]
    # print('isoelectric_lead_index_list ', isoelectric_lead_index_list)
    # # positive_lead_names = limb_augmented_names[np.asarray(positive_lead_index_list)]
    # print('positive_lead_index_list ', positive_lead_index_list)
    # # negative_lead_names = limb_augmented_names[np.asarray(negative_lead_index_list)]
    # print('negative_lead_index_list ', negative_lead_index_list)

    positive_limb_augmented_lead_index = np.argmax(limb_augmented_qrs_net_amplitude)
    # print('limb_augmented_qrs_net_amplitude ', limb_augmented_qrs_net_amplitude)
    # positive_name = limb_augmented_names[positive_limb_augmented_lead_index]
    # print('positive_limb_augmented_lead_index ', positive_limb_augmented_lead_index)
    # print('positive_name ', positive_name)
    # print('value ', limb_augmented_qrs_net_amplitude[positive_limb_augmented_lead_index])
    positive_angle = limb_augmented_angles[positive_limb_augmented_lead_index]
    # print('positive_angle ', positive_angle)
    isoelectric_limb_augmented_lead_index = np.argmin(np.abs(limb_augmented_qrs_net_amplitude))
    # print('isoelectric_limb_augmented_lead_index ', isoelectric_limb_augmented_lead_index)
    # isoelectric_name = limb_augmented_names[isoelectric_limb_augmented_lead_index]
    # print('isoelectric_name ', isoelectric_name)
    isoelectric_angle = limb_augmented_angles[isoelectric_limb_augmented_lead_index]
    # print('isoelectric_angle ', isoelectric_angle)
    d_iso_pos_angle = isoelectric_angle - positive_angle
    # Some corrections on the angle values may be necessary due the discontinuity at 180 degrees
    # print('d_iso_pos_angle ', d_iso_pos_angle)
    # d_iso_pos_angle_2 = d_iso_pos_angle
    if d_iso_pos_angle > 180:
        # If the difference is larger than 180, it means that it needs to cross through the discontinuity
        # d_iso_pos_angle = -(d_iso_pos_angle % 180)  # take the module of 180 and change the direction of the difference
        d_iso_pos_angle = d_iso_pos_angle % (-180)
    elif d_iso_pos_angle < -180:
        # d_iso_pos_angle = -(d_iso_pos_angle % (-180))
        d_iso_pos_angle = d_iso_pos_angle % 180
    # print('d_iso_pos_angle mod ', d_iso_pos_angle)
    # print('d_iso_pos_angle_2 mod ', d_iso_pos_angle_2)
    # if d_iso_pos_angle != d_iso_pos_angle_2:
    #     warn('d_iso_pos_angle_2 is different')
    if d_iso_pos_angle < 0.:  # isoelectric_angle < positive_angle:  # Careful, when aVR is involved, there is a discontunity
        qrs_axis = isoelectric_angle + 90
    elif d_iso_pos_angle > 0.:  # isoelectric_angle > positive_angle:
        qrs_axis = isoelectric_angle - 90
    else:
        # If both agles are the same, this is a strange case, so set the angle to something strange to make sure it gets penalised
        qrs_axis = 179  # https://litfl.com/ecg-axis-interpretation/
    # print('qrs_axis pre ', qrs_axis)
    # if abs(qrs_axis) > 180:
    # qrs_axis_2 = qrs_axis
    if qrs_axis > 180:
        # qrs_axis = qrs_axis - 360.  # convert it to a negative value in the correct part of the diagram
        qrs_axis = qrs_axis % (-180)
    elif qrs_axis < -180:
        # qrs_axis = qrs_axis + 360.  # convert it to a positive value in the correct part of the diagram
        qrs_axis = qrs_axis % 180
    # print('qrs_axis mod ', qrs_axis)
    # print('qrs_axis_2 mod ', qrs_axis_2)
    # if qrs_axis != qrs_axis_2:
    #     warn('qrs_axis is different')
    #
    # nb_cols = 3
    # nb_rows = 4
    #
    # fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(10, 15))
    #
    # ecg_12_leads = np.zeros((12, predicted_qrs.shape[1]))
    # print('ecg_12_leads ', ecg_12_leads.shape)
    # print('simulated_ecg ', predicted_ecg.shape)
    # ecg_12_leads[0, :] = I_qrs
    # ecg_12_leads[1, :] = II_qrs
    # ecg_12_leads[2, :] = III_qrs
    # ecg_12_leads[3, :] = aVR_qrs
    # ecg_12_leads[4, :] = aVL_qrs
    # ecg_12_leads[5, :] = aVF_qrs
    # ecg_12_leads[6:, :] = predicted_qrs[2:, :]
    #
    # axes = np.reshape(axes, ecg_12_leads.shape[0])
    #
    # lead_names_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # for lead_i in range(ecg_12_leads.shape[0]):
    #     axes[lead_i].set_title(lead_names_12[lead_i], fontsize=30)
    #     time_steps = np.arange(ecg_12_leads.shape[1])
    #     axes[lead_i].plot(time_steps, ecg_12_leads[lead_i, :], label='Sim', color='k', linewidth=2.)
    #     axes[lead_i].set_ylim([-1.5, 1.5])
    #     for tick in axes[lead_i].xaxis.get_major_ticks():
    #         tick.label1.set_fontsize(14)
    #     for tick in axes[lead_i].yaxis.get_major_ticks():
    #         tick.label1.set_fontsize(14)
    # axes[lead_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    #
    # plt.show(block=False)

    return qrs_axis


def calculate_ecg_qrs_axis_population(max_lat_list, predicted_ecg_list):
    qrs_axis_list = pymp.shared.array((predicted_ecg_list.shape[0]), dtype=np.float64)
    # threads_num = multiprocessing.cpu_count()
    # Uncomment the following lines to turn off the parallelisation.
    # if True:
    #     for sample_i in range(qrs_axis_list.shape[0]):
    iter_gen = get_parallel_loop(data_size=qrs_axis_list.shape[0])
    for sample_i in iter_gen:
        if True:
    # with pymp.Parallel(min(threads_num, qrs_axis_list.shape[0])) as p1:
    #     for sample_i in p1.range(qrs_axis_list.shape[0]):
            qrs_axis_list[sample_i] = calculate_ecg_qrs_axis(max_lat=max_lat_list[sample_i],
                                              predicted_ecg=predicted_ecg_list[sample_i, :, :])
    return qrs_axis_list


def calculate_r_s_sr_rs_wave_progression(max_lat, predicted_ecg):
    lead_V1_index = get_lead_V1_index(nb_lead=predicted_ecg.shape[0])
    precordial_qrs = delineate_ecg_qrs_end(max_lat, predicted_ecg[lead_V1_index:, :])
    if precordial_qrs.shape[0] != get_nb_precordial_lead():
        raise (Exception, 'This function requires 6 leads after lead_V1_index in the default arrangement: V1, ..., V6')
    # R wave progression
    precordial_r_wave_amplitude_index = np.argmax(precordial_qrs, axis=1)
    # print('precordial_r_wave_amplitude_index ', precordial_r_wave_amplitude_index)
    precordial_r_wave_amplitude = np.zeros(precordial_r_wave_amplitude_index.shape, dtype=float)
    for lead_i in range(precordial_r_wave_amplitude_index.shape[0]):
        precordial_r_wave_amplitude[lead_i] = precordial_qrs[lead_i, precordial_r_wave_amplitude_index[lead_i]]
    precordial_r_wave_progression_index = np.argsort(precordial_r_wave_amplitude)
    # S wave progression
    precordial_s_wave_absolute_amplitude = np.zeros(precordial_r_wave_amplitude_index.shape, dtype=float)
    for lead_i in range(precordial_r_wave_amplitude_index.shape[0]):
        # Make sure that the S wave is found after the R wave
        precordial_s_wave_absolute_amplitude[lead_i] = np.abs(np.amin(precordial_qrs[lead_i, precordial_r_wave_amplitude_index[lead_i]:]))
    precordial_s_wave_progression_index = np.argsort(precordial_s_wave_absolute_amplitude)
    # S>R wave progression
    # print('precordial_r_wave_amplitude ', precordial_r_wave_amplitude)
    # print('precordial_s_wave_absolute_amplitude ', precordial_s_wave_absolute_amplitude)
    precordial_sr_wave_progression_index = np.nonzero(precordial_s_wave_absolute_amplitude > precordial_r_wave_amplitude)[0]  # no need to have an np.abs(precordial_r_wave_amplitude), because it will still come out smaller if its negative
    # R>S wave progression
    precordial_rs_wave_progression_index = np.nonzero(
        precordial_r_wave_amplitude > precordial_s_wave_absolute_amplitude)[0]  # no need to have an np.abs(precordial_r_wave_amplitude), because it will still come out smaller if its negative

    return precordial_r_wave_progression_index, precordial_s_wave_progression_index, \
        precordial_sr_wave_progression_index, precordial_rs_wave_progression_index


def calculate_ecg_qrs_nb_peak_pos_neg_lead(ecg_lead):
    '''This function counts the number of positive and negative peaks in the ECG that have an amplitude difference of
    at least 10% of the signal's total amplitude range.
    This large threshold ensures that the metric will only focus on large deflections and won't count small notches that
    could derive from the coarseness of the mesh and of the Eikonal (pathchy LAT maps)
    It considers a peak to be positive if the value at that point is >= 0 regardless of the gradients.
    It considers a peak to be negative if the value at that point is < 0 regardless of the gradients.
    Thus, this function requires the ECG to be aligned at zero.
    '''
    significance_threshold = ((np.amax(ecg_lead)-np.amin(ecg_lead))/100.)*10  # 10% of the amplitude range of the ECG lead
    gradient = np.gradient(ecg_lead)
    # We replace the gradient values of zero by the previous gradient value so that we actually evaluate crossings through zero instead of plateaus
    gradient_zero_i = np.nonzero(gradient==0)[0]  # get indexes of zero values
    gradient[gradient_zero_i] = gradient[gradient_zero_i-1]  # replace them by previous value
    peak_index = np.nonzero(np.diff(np.sign(gradient)))[0]
    # plt.figure()
    # plt.plot(ecg_lead)
    # plt.vlines(peak_index, -1.5, 1.5, 'r')
    # plt.show(block=False)
    # print('peak_index before threshold filtering ', peak_index)
    peak_index = peak_index[peak_index > 0]   # The first value cannot be a peak
    peak_index = peak_index[peak_index < ecg_lead.shape[0]-1]  # The last value cannot be a peak
    peak_index = np.unique(peak_index)  # Each index should only appear once
    # print('peak_index before threshold filtering 2 ', peak_index)
    # Revise peak_index to make sure that only "significant" peaks are kept
    ## Forward pass
    preak_index_keep = np.ones(peak_index.shape, dtype=bool)
    prev_index = 0
    for peak_index_i in range(peak_index.shape[0]):
        # Before signal part
        part_signal_diff_max = np.amax(np.abs(ecg_lead[prev_index:peak_index[peak_index_i]]-ecg_lead[prev_index]))
        if part_signal_diff_max < significance_threshold:
            preak_index_keep[peak_index_i] = False
        else:
            prev_index = peak_index[peak_index_i]
    peak_index = peak_index[preak_index_keep]
    # print('peak_index mod 1 ', peak_index)
    ## Reverse pass
    preak_index_keep = np.ones(peak_index.shape, dtype=bool)
    prev_index = -1
    for peak_index_i in range(peak_index.shape[0]-1, 0, -1):
        # print('peak_index_i reverse ', peak_index_i)
        # After signal part
        part_signal_diff_max = np.amax(np.abs(ecg_lead[peak_index[peak_index_i]:prev_index] - ecg_lead[prev_index]))
        if part_signal_diff_max < significance_threshold:
            preak_index_keep[peak_index_i] = False
        else:
            prev_index = peak_index[peak_index_i]
    peak_index = peak_index[preak_index_keep]
    # print('peak_index mod 2 ', peak_index)

    peak_pos = np.sum(ecg_lead[peak_index]>=0)
    peak_neg = np.sum(ecg_lead[peak_index]<0)

    # plt.figure()
    # plt.plot(ecg_lead)
    # pos_index = peak_index[ecg_lead[peak_index]>=0]
    # plt.vlines(pos_index, -1.5, 1.5, 'c')
    # neg_index = peak_index[ecg_lead[peak_index]<0]
    # plt.vlines(neg_index, -1.5, 1.5, 'm')
    # # plt.show(block=False)
    # # gradient = np.gradient(gradient)
    # # plt.figure()
    # # plt.plot(np.sign(gradient))
    # # print('gradient ', gradient)
    # plt.show(block=False)

    return peak_pos, peak_neg


def calculate_ecg_qrs_nb_peak_pos_neg(max_lat, predicted_ecg):
    predicted_qrs = delineate_ecg_qrs_end(max_lat, predicted_ecg)
    qrs_nb_peak_pos_neg_lead = np.zeros((predicted_qrs.shape[0], 2), dtype=float)
    for lead_i in range(predicted_qrs.shape[0]):
        # print('lead_i ', lead_i)
        qrs_nb_peak_pos_neg_lead[lead_i, :] = calculate_ecg_qrs_nb_peak_pos_neg_lead(predicted_qrs[lead_i, :])
        # print('qrs_nb_peak_pos_neg_lead[lead_i, :] ', qrs_nb_peak_pos_neg_lead[lead_i, :])
    # raise()
    return qrs_nb_peak_pos_neg_lead


def calculate_ecg_qrs_nb_peak_pos_neg_population(max_lat_list, predicted_ecg_list):
    qrs_nb_peak_per_lead_list = pymp.shared.array((predicted_ecg_list.shape[0], predicted_ecg_list.shape[1], 2), dtype=np.float64)
    # threads_num = multiprocessing.cpu_count()
    # Uncomment the following lines to turn off the parallelisation.
    # if True:
    #     for sample_i in range(qrs_nb_peak_per_lead_list.shape[0]):
    iter_gen = get_parallel_loop(data_size=qrs_nb_peak_per_lead_list.shape[0])
    for sample_i in iter_gen:
        if True:
    # with pymp.Parallel(min(threads_num, qrs_nb_peak_per_lead_list.shape[0])) as p1:
    #     for sample_i in p1.range(qrs_nb_peak_per_lead_list.shape[0]):
            qrs_nb_peak_per_lead_list[sample_i, :, :] = calculate_ecg_qrs_nb_peak_pos_neg(
                max_lat=max_lat_list[sample_i], predicted_ecg=predicted_ecg_list[sample_i, :, :])
    return qrs_nb_peak_per_lead_list


def align_ecg_to_qrs_onset(qrs_onset_per_lead, qrs_offset_per_lead, untrimmed_original_ecg):  # Pre-trim all leads to align their QRS onsets
    warn('This function is vulnerable to outlier mistakes in the delineation process! Thus, requires visual inspection!')
    # visualise QRS onset distribution
    # print('qrs_onset_per_lead ', qrs_onset_per_lead)
    # plot_histogram(qrs_onset_per_lead)
    # Initialise variables
    min_qrs_onset = np.amin(np.asarray(qrs_onset_per_lead))
    trimmed_original_ecg_list = []
    nb_leads = untrimmed_original_ecg.shape[0]
    min_ecg_len = untrimmed_original_ecg.shape[1]
    qrs_offset_per_lead_aligned = np.zeros(qrs_offset_per_lead.shape, dtype=int)
    # print('min_qrs_onset ', min_qrs_onset)
    # print('min_ecg_len ', min_ecg_len)
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        alignment = qrs_onset_per_lead[lead_i]-min_qrs_onset
        trimmed_original_lead = untrimmed_original_ecg[lead_i, alignment:]
        qrs_offset_per_lead_aligned[lead_i] = int(qrs_offset_per_lead[lead_i] - alignment)
        min_ecg_len = min(min_ecg_len, len(trimmed_original_lead))
        trimmed_original_ecg_list.append(trimmed_original_lead)
    # print('min_ecg_len ', min_ecg_len)
    trimmed_original_ecg_array = np.zeros((nb_leads, min_ecg_len)) + get_nan_value() # make sure that no value is without defining
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        trimmed_original_lead = trimmed_original_ecg_list[lead_i]
        trimmed_original_ecg_array[lead_i, :] = trimmed_original_lead[:min_ecg_len]
    # axes, fig = visualise_ecg(ecg_list=[trimmed_original_ecg_array])    # visualise resulting ecg for visual inspection
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # fig.suptitle('align', fontsize=16)
    # plt.show(block=False)
    # print('trimmed_original_ecg_array ', trimmed_original_ecg_array.shape)
    return trimmed_original_ecg_array, min_qrs_onset, qrs_offset_per_lead_aligned


def trim_ecg_to_qrs_onset(qrs_onset_per_lead, untrimmed_original_ecg):
    warn('This function is vulnerable to outlier mistakes in the delineation process! Thus, requires visual inspection!')
    # visualise QRS onset distribution
    # plot_histogram(qrs_onset_per_lead)
    # Initialise variables
    trimmed_original_ecg_list = []
    nb_leads = untrimmed_original_ecg.shape[0]
    min_ecg_len = untrimmed_original_ecg.shape[1]
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        trimmed_original_lead = untrimmed_original_ecg[lead_i, qrs_onset_per_lead[lead_i]:]
        min_ecg_len = min(min_ecg_len, len(trimmed_original_lead))
        trimmed_original_ecg_list.append(trimmed_original_lead)
    trimmed_original_ecg_array = np.zeros((nb_leads, min_ecg_len)) + get_nan_value() # make sure that no value is without defining
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        trimmed_original_lead = trimmed_original_ecg_list[lead_i]
        trimmed_original_ecg_array[lead_i, :] = trimmed_original_lead[:min_ecg_len]
    # axes, fig = visualise_ecg(ecg_list=[trimmed_original_ecg_array])    # visualise resulting ecg for visual inspection
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # fig.suptitle('trim', fontsize=16)
    # plt.show(block=False)
    return trimmed_original_ecg_array


def trim_ecg_to_qrs_offset(qrs_offset_per_lead, untrimmed_original_ecg):
    warn('This function is vulnerable to outlier mistakes in the delineation process! Thus, requires visual inspection!')
    # visualise QRS onset distribution
    # plot_histogram(qrs_onset_per_lead)
    # Initialise variables
    trimmed_original_ecg_list = []
    nb_leads = untrimmed_original_ecg.shape[0]
    min_ecg_len = untrimmed_original_ecg.shape[1]
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        trimmed_original_lead = untrimmed_original_ecg[lead_i, :qrs_offset_per_lead[lead_i]]
        min_ecg_len = min(min_ecg_len, len(trimmed_original_lead))
        trimmed_original_ecg_list.append(trimmed_original_lead)
    trimmed_original_ecg_array = np.zeros((nb_leads, min_ecg_len)) + get_nan_value() # make sure that no value is without defining
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        trimmed_original_lead = trimmed_original_ecg_list[lead_i]
        trimmed_original_ecg_array[lead_i, :] = trimmed_original_lead[:min_ecg_len]
    # axes, fig = visualise_ecg(ecg_list=[trimmed_original_ecg_array])    # visualise resulting ecg for visual inspection
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # fig.suptitle('trim', fontsize=16)
    # plt.show(block=False)
    return trimmed_original_ecg_array


def resample_ecg(desired_freq, original_ecg, original_freq, original_x=None):
    nb_leads = original_ecg.shape[0]
    if original_freq is not None:
        current_len = original_ecg.shape[1]
        ini_time = 0.
        end_time = current_len/original_freq
        current_x = np.linspace(ini_time, end_time, current_len)
    elif original_x is not None:
        current_x = original_x/1000     # times are expected in ms
        ini_time = current_x[0]
        end_time = current_x[-1]
    else:
        raise('Either original_freq or original_x should be defined')
    desired_len = int(round(end_time*desired_freq))
    desired_x = np.linspace(ini_time, end_time, desired_len)
    resampled_ecg = np.zeros((nb_leads, desired_len)) + get_nan_value() # make sure that no value is without defining
    for lead_i in range(nb_leads):  # Iterate over ecg leads
        original_lead = original_ecg[lead_i, :]
        resampled_ecg[lead_i, :] = np.interp(desired_x, current_x, original_lead, left=None, right=None, period=None)
    # axes, fig = visualise_ecg(ecg_list=[resampled_ecg])  # visualise resulting ecg for visual inspection
    # axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    # fig.suptitle('resample', fontsize=16)
    # plt.show(block=False)
    return resampled_ecg


class CalculateEcg: # TODO refactor this sequence of classes so that they dont need a reference ECG to be initialised
    def __init__(self, filtering, frequency, high_freq_cut, lead_names, low_freq_cut, max_len_ecg, max_len_qrs,
                 nb_leads, normalise, reference_ecg, verbose, zero_align, qrs_onset=0):
        if verbose:
            print('Initialising ECG calculation')
        self.filtering = filtering
        self.normalise = normalise
        self.zero_align = zero_align
        self.nb_leads = nb_leads  # Number of ECG leads to simulate.
        self.frequency = frequency  # Hz
        # self.high_freq_cut = high_freq_cut  # High cut-off frequency of the filter Hz
        high_w = high_freq_cut / (self.frequency / 2)  # Normalize the frequency
        self.high_b_filtfilt, self.high_a_filtfilt = signal.butter(4, high_w, 'low')  # Butterworth filter of fourth order.
        low_w = low_freq_cut / (self.frequency / 2)  # Normalize the frequency
        self.low_b_filtfilt, self.low_a_filtfilt = signal.butter(4, low_w, 'high')  # Butterworth filter of fourth order.
        # Note: Eight order slowed down the inference in 2019 without significant benefit on the signal quality.
        self.lead_names = lead_names
        self.max_len_ecg = max_len_ecg  # This parameter controls the maximum possible length of the recordings.
        # It's important to modulate this parameter based on weather you need only QRS or also T wave
        self.max_len_qrs = max_len_qrs  # This parameter may have the same value as the one above, when the one above
        self.verbose = verbose
        # only considers the QRS complex. This parameter can be used for normalisation and locating parts of the QRS.
        self.reference_ecg = self.__preprocess_ecg_without_normalise(reference_ecg) # Temporal preprocessing until self.reference_lead_is_positive is computed
        self.reference_lead_is_positive = None  # This indicates which leads have a positive R peak (maximum value at
        # the start of the signal)
        # self.max_qrs_end = qrs_onset + max_len_qrs
        self.__set_reference_lead_is_positive(max_qrs_end=qrs_onset + max_len_qrs)
        self.reference_ecg = self.preprocess_ecg(reference_ecg, qrs_onset=qrs_onset)

    def calculate_ecg(self, lat, vm):
        raise NotImplementedError

    def calculate_ecg_population(self, lat_population, vm_population):
        raise NotImplementedError

    def __filter_ecg(self, original_ecg):
        processed_ecg = original_ecg
        # First we filter out the low frequencies using a high-pass filter and the lower thresholds
        processed_ecg = filter_butterworth_ecg(b=self.low_b_filtfilt, a=self.low_a_filtfilt, ecg=processed_ecg)
        # Secondly we filter out the high frequencies using a low-pass filter and the higher thresholds
        processed_ecg = filter_butterworth_ecg(b=self.high_b_filtfilt, a=self.high_a_filtfilt, ecg=processed_ecg)
        return processed_ecg

    def __normalise_ecg(self, original_ecg, qrs_onset=0):
        if original_ecg.shape[0] == 8:
            normalised_ecg = self.__normalise_ecg_based_on_rwave_8_leads(original_ecg=original_ecg, qrs_onset=qrs_onset)
        elif self.nb_leads == original_ecg.shape[0]:
            warn('Your ECG is not standard with 8 leads: I, II, V1, ..., V6')
            normalised_ecg = self.__normalise_ecg_based_on_rwave_any_leads(original_ecg=original_ecg, qrs_onset=qrs_onset)
        else:
            warn('Your ECG has different number of leads than your defined nb_leads. Thus, it cannot be safely normalised using the reference/clinical recording.')
            normalised_ecg = normalise_ecg_between_0_and_1(original_ecg=original_ecg)
        return normalised_ecg

    def preprocess_ecg(self, original_ecg, qrs_onset=0):
        processed_ecg = original_ecg
        if self.filtering:
            processed_ecg = self.__filter_ecg(processed_ecg)
        if self.zero_align:
            processed_ecg = zero_align_ecg(processed_ecg)
        if self.normalise:
            processed_ecg = self.__normalise_ecg(original_ecg=processed_ecg, qrs_onset=qrs_onset)
        return processed_ecg

    def __preprocess_ecg_without_normalise(self, original_ecg):
        processed_ecg = original_ecg
        if self.filtering:
            processed_ecg = self.__filter_ecg(processed_ecg)
        if self.zero_align:
            processed_ecg = zero_align_ecg(processed_ecg)
        return processed_ecg

    def plot_ecgs(self, ecg_list):
        raise NotImplementedError

    # def preprocess_raw_ecg(self, raw_ecg): # TODO
    #     raise NotImplementedError

    def __set_reference_lead_is_positive(self, max_qrs_end):
        approx_qrs_end = min(self.reference_ecg.shape[1], max_qrs_end)  # Approximate end of QRS.
        reference_lead_max = np.absolute(np.amax(self.reference_ecg[:, :approx_qrs_end], axis=1))
        reference_lead_min = np.absolute(np.amin(self.reference_ecg[:, :approx_qrs_end], axis=1))
        reference_lead_is_positive = reference_lead_max >= reference_lead_min
        reference_amplitudes = np.zeros(shape=self.nb_leads, dtype=np.float64)
        reference_amplitudes[reference_lead_is_positive] = reference_lead_max[reference_lead_is_positive]
        reference_amplitudes[np.logical_not(reference_lead_is_positive)] = reference_lead_min[
            np.logical_not(reference_lead_is_positive)]
        # if self.verbose:
        #     print('reference_lead_is_positive')
        #     print(reference_lead_is_positive)
        #     print('reference_amplitudes')
        #     print(reference_amplitudes)
        self.reference_lead_is_positive = reference_lead_is_positive  # Have some R progression by normalising by the
        # largest absolute amplitude lead

    def __normalise_ecg_based_on_rwave_8_leads(self, original_ecg, qrs_onset):
        if self.nb_leads != 8 or original_ecg.shape[0] != 8:
            raise(Exception, 'This function is hardcoded for the specific ECG configuration: I, II, V1, ..., V6')
        # print('Normalising ECG ', original_ecg.shape)
        approx_qrs_end = min(self.reference_ecg.shape[1], self.max_len_qrs+qrs_onset)  # Approximate end of QRS.
        # approx_qrs_width = min(original_ecg.shape[1], self.max_len_qrs)  # This approximation is more robust.
        # print('approx_qrs_end ', approx_qrs_end)
        # print(np.amax(original_ecg[:, :approx_qrs_end]))
        reference_amplitudes = np.empty(shape=self.nb_leads, dtype=np.float64)
        reference_amplitudes[self.reference_lead_is_positive] = np.absolute(
            np.amax(original_ecg[:, :approx_qrs_end], axis=1)[
                self.reference_lead_is_positive])
        reference_amplitudes[np.logical_not(self.reference_lead_is_positive)] = \
            np.absolute(np.amin(original_ecg[:, :approx_qrs_end], axis=1))[np.logical_not(
                self.reference_lead_is_positive)]
        normalised_ecg = np.zeros(original_ecg.shape)
        normalised_ecg[:2, :] = original_ecg[:2, :] / np.mean(
            reference_amplitudes[:2])  # Normalise limb leads separatedly
        normalised_ecg[2:self.nb_leads, :] = original_ecg[2:self.nb_leads, :] / np.mean(
            reference_amplitudes[2:self.nb_leads])
        return normalised_ecg


    def __normalise_ecg_based_on_rwave_any_leads(self, original_ecg, qrs_onset):
        warn('This function will normalise all leads together as if they were precordial and needed to preserve R progression.')
        approx_qrs_end = min(self.reference_ecg.shape[1], self.max_len_qrs+qrs_onset)  # Approximate end of QRS.
        reference_amplitudes = np.empty(shape=self.nb_leads, dtype=np.float64)
        reference_amplitudes[self.reference_lead_is_positive] = np.absolute(
            np.amax(original_ecg[:, :approx_qrs_end], axis=1)[
                self.reference_lead_is_positive])
        reference_amplitudes[np.logical_not(self.reference_lead_is_positive)] = \
            np.absolute(np.amin(original_ecg[:, :approx_qrs_end], axis=1))[np.logical_not(
                self.reference_lead_is_positive)]
        normalised_ecg = original_ecg / np.mean(reference_amplitudes)
        return normalised_ecg


# Implementation of the pseudo-ECG method for tetrahedral meshes
class PseudoEcgTetFromVM(CalculateEcg):
    """This subclass can compute the ECG (QRS + Twave) and expects a VM time series per node"""
    def __init__(self, electrode_positions, filtering, frequency, high_freq_cut, lead_names, low_freq_cut, max_len_ecg, max_len_qrs,
                 nb_leads, nodes_xyz, normalise, reference_ecg, tetra, tetra_centre, verbose, zero_align):
        super().__init__(filtering=filtering, frequency=frequency, high_freq_cut=high_freq_cut, lead_names=lead_names,
                         low_freq_cut=low_freq_cut, max_len_ecg=max_len_ecg, max_len_qrs=max_len_qrs,
                         nb_leads=nb_leads, normalise=normalise, reference_ecg=reference_ecg, verbose=verbose,
                         zero_align=zero_align)
        if frequency != 1000:
            print('All functions assume frequency of 1000 Hz in the AP and the ECG signals!\nThis does not only '
                  'affect the filtering functions, but also the assumptions when calculating the ECG.\nIncreasing the '
                  'frequency should not invalidate the latter, but otherwise, the code will need some rethinking to '
                  'ensure that the sampling rate of the AP is not too sparse in time.')
            raise NotImplementedError
        self.nb_bsp = electrode_positions.shape[0]
        self.tetra = tetra  # The Pseudo-ECG method needs the tetrahedrons for the final computation. If we pass the tetrahedrons as an input, it may overload the memory when parallelising the code.
        self.__calculate_pseudo_ecg_dxyz_dr(nodes_xyz, tetra_centre, electrode_positions)

    '''functionality'''
    def calculate_ecg(self, lat, vm):  # This function gets overwritten by child classes
        return self.pseudo_ecg_from_vm(lat=lat, vm=vm)#, used_step_function_vm=False)

    def calculate_ecg_population(self, lat_population, vm_population):  # Overwrites the parent function with only one possible computation method
        # print('vm_population ', vm_population.shape)
        # print('self.max_len_ecg ', self.max_len_ecg)
        vm_population = vm_population[:, :, :self.max_len_ecg]
        # vm_population_start = vm_population[:, :, 1]
        # vm_population_start = vm_population_start.flatten()
        # vm_population_start = np.sort(vm_population_start)
        # vm_population_end = vm_population[:, :, 500]
        # vm_population_end = vm_population_end.flatten()
        # vm_population_end = np.sort(vm_population_end)
        # plt.figure()
        # plt.plot(vm_population_start)
        # plt.ylim(-.01, .01)
        # plt.title('start2')
        # plt.show(block=False)
        # plt.figure()
        # plt.plot(vm_population_end)
        # plt.ylim(-.01, .01)
        # plt.title('end2')
        # plt.show(block=False)
        # # for i in range(vm_population.shape[0]):
        # #     for j in range(vm_population.shape[1]):
        # #         plt.plot(vm_population[i, j, :])
        # # plt.title('2')
        # # plt.show(block=False)
        # TODO make a unique functino that is more efficient than the basic unic function from numpy in large arrays with multiple dimensions
        # vm_population_unique, unique_indexes, inverse_unique_indexes = np.unique(vm_population, return_index=True, return_inverse=True, axis=0)
        vm_population_unique = vm_population
        # unique_indexes = np.ones((vm_population.shape[0]), dtype=bool)
        # inverse_unique_indexes = np.ones((vm_population.shape[0]), dtype=bool)
        lat_population_unique = lat_population#[unique_indexes, :]
        ecg_population_unique = pymp.shared.array((vm_population_unique.shape[0], self.nb_leads,
                                                   vm_population_unique.shape[2]), dtype=np.float64)
        ecg_population_unique[:, :, :] = get_nan_value()
        # threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation of the ecg computation.
        # if True:
        #     print('Parallel loop turned off in ecg module')
        #     for conf_i in range(ecg_population_unique.shape[0]):
        iter_gen = get_parallel_loop(data_size=ecg_population_unique.shape[0])
        for conf_i in iter_gen:
            if True:
        # with pymp.Parallel(min(threads_num, ecg_population_unique.shape[0])) as p1:
        #     for conf_i in p1.range(ecg_population_unique.shape[0]):
                aux_ecg = self.calculate_ecg(lat=lat_population_unique[conf_i, :], vm=vm_population_unique[conf_i, :, :])
                # ecg_len = int(aux_ecg.shape[1])
                ecg_population_unique[conf_i, :, :aux_ecg.shape[1]] = aux_ecg
        return ecg_population_unique#[inverse_unique_indexes, :, :]

    def pseudo_ecg_from_vm(self, lat, vm):#, used_step_function_vm):
        # vm = np.transpose(vm) # TODO edit the following code so that this transpose is no longer needed
        # TODO vm is (node_i, time) indexed everywhere, except in this function, where it is (time, node_i) indexed.

        # TODO ECGs should be aligned using the LATs upon creation!
        # print('lat ', np.amax(lat), np.amin(lat))

        nb_time_steps = vm.shape[1]  # 1000 Hz is one evaluation every 1 ms
        # print('Is nan in VM?')
        # aux = vm == get_nan_value()
        # print('aux.shape')
        # aux = np.sum(aux, axis=0)
        # print(aux.shape)
        # print(aux)
        # print(np.sum(vm == get_nan_value()))
        # print(np.any(np.isnan(vm)))
        # print(np.sum(vm))
        # print(np.amax(vm))
        # print(np.amin(vm))
        simulated_ecg = np.zeros((self.nb_leads, nb_time_steps), dtype=np.float64) + get_nan_value() #np.full((self.nb_leads, nb_time_steps), np.nan, dtype=np.float64)

        # bsp is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        bsp = np.zeros((self.nb_bsp, nb_time_steps), dtype=np.float64)
        ele_contrib = np.zeros((self.nb_bsp, self.tetra.shape[0]), dtype=np.float64)

        # if used_step_function_vm and (lat is not None):
        # for time_step_i in range(1, nb_time_steps, 1):
        #     # Shortcuts when using a step function to define the vm values for the QRS calculation.
        #     active_nodes = np.nonzero(lat == time_step_i)[0].astype(np.int32)
        #     if not len(active_nodes) == 0:
        #         active_tetra = np.unique(
        #             np.concatenate([self.tetra_per_node[node_index] for node_index in active_nodes]))
        #         # print('active_tetrahedra ', np.amax(active_tetrahedra))
        #         # print('self.tetrahedrons ', self.tetrahedrons.shape)
        #         b_m_vm = (vm[self.tetra[active_tetra, 0:3], time_step_i]
        #                   - vm[self.tetra[active_tetra, 3], time_step_i][:, np.newaxis])
        #         bd_vm = np.squeeze(
        #             np.matmul(self.dxyz[active_tetra, :, :], b_m_vm[:, :, np.newaxis]), axis=2)
        #         ele_contrib[:, active_tetra] = np.sum(self.dr[:, active_tetra, :] * bd_vm, axis=2)
        #         bsp[:, time_step_i] = np.sum(ele_contrib, axis=1)
        #     else:
        #         bsp[:, time_step_i] = bsp[:, time_step_i - 1]
        # else:
        for time_step_i in range(1, nb_time_steps, 1):
            b_m_vm = (vm[self.tetra[:, 0:3], time_step_i] - vm[self.tetra[:, 3], time_step_i][:, np.newaxis])  # change to this: tetrahedrons[:, 3], essentially it's changing from
            # "active_ele_i:active_ele_i+1" to ":"
            bd_vm = np.squeeze(
                np.matmul(self.dxyz[:, :, :], b_m_vm[:, :, np.newaxis]), axis=2)
            ele_contrib[:, :] = np.sum(self.dr[:, :, :] * bd_vm, axis=2)
            bsp[:, time_step_i] = np.sum(ele_contrib, axis=1)
        ele_contrib = None  # Clear memory
        # Applying Einthoven's triangle, bsp is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        bsp_LA = bsp[0, :]
        bsp_RA = bsp[1, :]
        bsp_LL = bsp[2, :]
        # bsp_RL = bsp[3, :]
        # bsp_V1 = bsp[4, :]
        # bsp_V2 = bsp[5, :]
        # bsp_V3 = bsp[6, :]
        # bsp_V4 = bsp[7, :]
        # bsp_V5 = bsp[8, :]
        # bsp_V6 = bsp[9, :]

        # Calculate limb leads: I, II - Other leads can be recovered later from just these two, uncomment the following code to test this.
        simulated_ecg[0, :] = (bsp_LA - bsp_RA) # lead I = LA - RA
        simulated_ecg[1, :] = (bsp_LL - bsp_RA) # lead II = LL - RA
        # aux_III = (bsp_LL - bsp_LA) # lead III = LL - LA
        # re_aux_III = simulated_ecg[1, :] - simulated_ecg[0, :] # lead III = II - I
        # print()
        # print('III ', np.sum(np.abs(aux_III-re_aux_III)))
        # aux_aVR = bsp_RA - (1./2.)*(bsp_LA + bsp_LL) # aVR = RA - 1/2 * (LA + LL)
        # re_aux_aVR =  (-1./2.)*(simulated_ecg[0, :]+simulated_ecg[1, :]) # aVR = - 1/2 * (I + II)
        # print('aVR ', np.sum(np.abs(aux_aVR-re_aux_aVR)))
        # aux_aVL = bsp_LA - (1. / 2.) * (bsp_RA + bsp_LL)  # aVL = LA - 1/2 * (RA + LL)
        # re_aux_aVL = (-1. / 2.) * (aux_III - simulated_ecg[0, :]) # aVL = - 1/2 * (III - I)
        # print('aVL ', np.sum(np.abs(aux_aVL - re_aux_aVL)))
        # aux_aVF = bsp_LL - (1./2.)*(bsp_RA + bsp_LA) # aVF = LL - 1/2 * (RA + LA)
        # re_aux_aVF = (1. / 2.) * (simulated_ecg[1, :] + aux_III) # aVF = 1/2 * (II + III)
        # print('aVF ', np.sum(np.abs(aux_aVF - re_aux_aVF)))
        # print('example I-II ', np.sum(np.abs(simulated_ecg[0, :] - simulated_ecg[1, :])))
        # print()

        # Calculate precordial leads: 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        BSPecg = bsp - np.mean(bsp[0:3, :], axis=0)
        bsp = None  # Clear Memory
        simulated_ecg[2:self.nb_leads, :] = BSPecg[4:self.nb_bsp, :]
        # print('example V5-V6 ', np.sum(np.abs(simulated_ecg[6, :] - simulated_ecg[7, :])))
        # print()

        # nb_cols = 3
        # nb_rows = 4
        #
        # fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(15, 20))
        #
        # ecg_12_leads = np.zeros((12, simulated_ecg.shape[1]))
        # print('best_ecg ', ecg_12_leads.shape)
        # print('simulated_ecg ', simulated_ecg.shape)
        # ecg_12_leads[0:2, :] = simulated_ecg[0:2, :]
        # ecg_12_leads[2, :] = re_aux_III
        # ecg_12_leads[3, :] = re_aux_aVR
        # ecg_12_leads[4, :] = re_aux_aVL
        # ecg_12_leads[5, :] = re_aux_aVF
        # ecg_12_leads[6:, :] = simulated_ecg[2:, :]
        #
        # ecg_12_leads = self.preprocess_ecg(ecg_12_leads)
        #
        # axes = np.reshape(axes, ecg_12_leads.shape[0])
        #
        # lead_names_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # for lead_i in range(ecg_12_leads.shape[0]):
        #     time_steps = np.arange(ecg_12_leads.shape[1])
        #     # axes[lead_i].plot(time_steps, self.reference_ecg[lead_i, :], label='Clinical', color='lime', linewidth=3.)
        #     axes[lead_i].set_title(lead_names_12[lead_i])
        #     time_steps = np.arange(ecg_12_leads.shape[1])
        #     axes[lead_i].plot(time_steps, ecg_12_leads[lead_i, :], label='Sim', color='k', linewidth=2.)
        #     axes[lead_i].set_ylim([-1., 1.])
        #     for tick in axes[lead_i].xaxis.get_major_ticks():
        #         tick.label1.set_fontsize(14)
        #     for tick in axes[lead_i].yaxis.get_major_ticks():
        #         tick.label1.set_fontsize(14)
        # axes[lead_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        #
        # if self.verbose:
        #     plt.show(block=False)
        simulated_ecg = self.preprocess_ecg(simulated_ecg)
        # calculate_ecg_qrs_axis(max_lat=np.amax(lat), predicted_ecg=simulated_ecg)
        return simulated_ecg

    # TODO remove this function and call the visualise_ecg function in the postprocess_functions.py instead.
    def visualise_ecg(self, discrepancy_population, ecg_population):#, save_dir=None):
        warn('This function is outdated! Use the one in preprocessing_functinos.py instead')
        best_index = np.argmin(discrepancy_population)
        if self.verbose:
            print('Best discrepancy value: ', discrepancy_population[best_index])
            plt.hist(discrepancy_population)
            plt.show(block=False)
        # Try to make half as many rows as columns
        # rows*cols=nb_leads; rows=cols/2; cols/2*cols=nb_leads; cols=(nb_leads*2)**0.5
        nb_cols = (self.nb_leads * 2) ** 0.5
        if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
            nb_rows = nb_cols / 2
        else:
            # Try to make 2 rows and the necessary columns
            nb_cols = self.nb_leads / 2
            if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
                nb_rows = nb_cols / 2
            else:
                warn('This number of leads cannot be plotted in the default configurations!')
                nb_cols = self.nb_leads
                nb_rows = 1

        # # Plot clinical TODO remove this
        # fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10))
        # axes = np.reshape(axes, self.nb_leads)
        # for lead_i in range(self.nb_leads):
        #     time_steps = np.arange(self.reference_ecg.shape[1])
        #     axes[lead_i].plot(time_steps, self.reference_ecg[lead_i, :], label='Clinical', color='k', linewidth=3.)
        #     axes[lead_i].set_title(self.lead_names[lead_i])
        #     axes[lead_i].set_ylim([-1.5, 1.5])
        #     for tick in axes[lead_i].xaxis.get_major_ticks():
        #         tick.label1.set_fontsize(14)
        #     for tick in axes[lead_i].yaxis.get_major_ticks():
        #         tick.label1.set_fontsize(14)
        # axes[lead_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        # plt.show()
        #
        # # Plot cloud without best TODO remove this
        # fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10))
        # axes = np.reshape(axes, self.nb_leads)
        # for lead_i in range(self.nb_leads):
        #     for ecg_i in range(len(ecg_population)):
        #         time_steps = np.arange(ecg_population.shape[2])
        #         axes[lead_i].plot(time_steps, ecg_population[ecg_i, lead_i, :], color='gray', linewidth=1.)
        #     time_steps = np.arange(self.reference_ecg.shape[1])
        #     axes[lead_i].plot(time_steps, self.reference_ecg[lead_i, :], label='Clinical', color='k', linewidth=3.)
        #     axes[lead_i].set_title(self.lead_names[lead_i])
        #     axes[lead_i].set_ylim([-1.5, 1.5])
        #     for tick in axes[lead_i].xaxis.get_major_ticks():
        #         tick.label1.set_fontsize(14)
        #     for tick in axes[lead_i].yaxis.get_major_ticks():
        #         tick.label1.set_fontsize(14)
        # axes[lead_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        # plt.show()
        # quit()


        fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10))
        axes = np.reshape(axes, self.nb_leads)
        # colours = ['k', 'b'] # Needs to be as long as ecgs.shape[0] TODO this needs to be fixed!
        # clin_ecg = self.clinical_ecg
        # clin_time_steps = np.arange(len(clin_ecg[0, :]))
        best_ecg = ecg_population[best_index, :, :]
        for lead_i in range(self.nb_leads):
            for ecg_i in range(len(ecg_population)):
                time_steps = np.arange(ecg_population.shape[2])
                # axes[lead_i].plot(time_steps, ecg_list[ecg_i][lead_i, :], colours[ecg_i], label=labels[ecg_i], linewidth=1.)
                axes[lead_i].plot(time_steps, ecg_population[ecg_i, lead_i, :], color='gray', linewidth=1.)
            time_steps = np.arange(self.reference_ecg.shape[1])
            #TODO I have commented the following line for Debbie's simulations
            axes[lead_i].plot(time_steps, self.reference_ecg[lead_i, :], label='Clinical', color='lime', linewidth=3.)
            axes[lead_i].set_title(self.lead_names[lead_i])
            time_steps = np.arange(best_ecg.shape[1])
            axes[lead_i].plot(time_steps, best_ecg[lead_i, :], label='Best', color='k', linewidth=2.)
            axes[lead_i].set_ylim([-1.5, 1.5])
            for tick in axes[lead_i].xaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
            for tick in axes[lead_i].yaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
        axes[lead_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

        # if save_dir is not None:
        #     plt.savefig(save_dir + 'ecg_inference_result.png')
        if self.verbose:
            plt.show(block=False)
        return fig

    # Private function to precompute parts of the pseudo-ECG. This function should only be called on initialisation.
    def __calculate_pseudo_ecg_dxyz_dr(self, nodes_xyz, tetrahedron_centre, electrode_positions):
        tetra_per_node = [[] for i in range(0, nodes_xyz.shape[0], 1)]
        for tetra_i in range(0, self.tetra.shape[0], 1):
            tetra_per_node[self.tetra[tetra_i, 0]].append(tetra_i)
            tetra_per_node[self.tetra[tetra_i, 1]].append(tetra_i)
            tetra_per_node[self.tetra[tetra_i, 2]].append(tetra_i)
            tetra_per_node[self.tetra[tetra_i, 3]].append(tetra_i)
        self.tetra_per_node = [np.array(n) for n in tetra_per_node]  # list of arrays instead of list of lists

        # Evaluate first spatial derivative of the mesh
        D = nodes_xyz[self.tetra[:, 3], :]  # RECYCLED
        A = nodes_xyz[self.tetra[:, 0], :] - D  # RECYCLED
        B = nodes_xyz[self.tetra[:, 1], :] - D  # RECYCLED
        C = nodes_xyz[self.tetra[:, 2], :] - D  # RECYCLED
        D = None  # Clear Memory

        tVolumes = np.reshape(
            np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1), (np.cross(B, C)[:, :, np.newaxis]))),
            self.tetra.shape[0])  # Tetrahedrons volume, no need to divide by 6
        # since it's being normalised by the sum which includes this 6 scaling factor
        meshVolume = np.sum(
            tVolumes) / 6.  # used to scale the relevance of signal-length discrepancies in small vs large geometries
        # print(meshName+'volume: '+ str(meshVolume))
        tVolumes = tVolumes / np.sum(tVolumes)

        # Calculate the tetrahedron (temporal) voltage gradients
        Mg = np.stack((A, B, C), axis=-1)
        A = None  # Clear Memory
        B = None  # Clear Memory
        C = None  # Clear Memory

        # Calculate the gradients
        G_pseudo = np.zeros(Mg.shape)
        for i in range(Mg.shape[0]):
            G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
            # If you obtain a Singular Matrix error type, this may be because one of the elements in the mesh is
            # really tinny if you are using a truncated mesh generated with Paraview, the solution is to do a
            # crinkle clip, instead of a regular smooth clip, making sure that the elements are of similar size
            # to each other. The strategy to identify the problem is to search for what element in Mg is giving
            # a singular matrix and see what makes it "special".
        G_pseudo = np.moveaxis(G_pseudo, 1, 2)
        Mg = None  # clear memory

        r = np.moveaxis(np.reshape(np.repeat(tetrahedron_centre, electrode_positions.shape[0], axis=1),
                                   (tetrahedron_centre.shape[0],
                                    tetrahedron_centre.shape[1], electrode_positions.shape[0])), 1,
                        -1) - electrode_positions
        d_r = np.moveaxis(np.multiply(
            np.moveaxis(r, [0, 1], [-1, -2]),
            np.multiply(np.moveaxis(np.sqrt(np.sum(r ** 2, axis=2)) ** (-3), 0, 1), tVolumes)), 0, -1)
        tVolumes = None
        self.dxyz = G_pseudo
        self.dr = d_r




class PseudoQRSTetFromStepFunction(PseudoEcgTetFromVM):
    """This subclass can only compute the ECG's QRS complex, and uses a step function"""

    def __init__(self, electrode_positions, filtering, frequency, high_freq_cut, lead_names, low_freq_cut, max_len_qrs,
                 nb_leads, nodes_xyz, normalise, reference_ecg, tetra, tetra_centre, verbose, zero_align):
        super().__init__(electrode_positions=electrode_positions, filtering=filtering, frequency=frequency,
                         high_freq_cut=high_freq_cut, lead_names=lead_names, low_freq_cut=low_freq_cut,
                         max_len_ecg=max_len_qrs, max_len_qrs=max_len_qrs, nb_leads=nb_leads, nodes_xyz=nodes_xyz,
                         normalise=normalise, reference_ecg=reference_ecg, tetra=tetra, tetra_centre=tetra_centre,
                         verbose=verbose, zero_align=zero_align)

    '''functionality'''
    def calculate_ecg(self, lat, vm):  # Overwrites the parent function with only one possible computation method
        return self.pseudo_ecg_from_vm_step_function(lat=lat, vm=vm)

    # def calculate_ecg_population(self, lat_population, vm_population):  # Overwrites the parent function with only one possible computation method
    #     vm_population = vm_population[:, :, :self.max_len_ecg]
    #     vm_population_unique, unique_indexes, reverse_unique_indexes = np.unique(vm_population, return_index=True, return_inverse=True, axis=0)
    #     lat_population_unique = lat_population[unique_indexes, :]
    #     ecg_population_unique = pymp.shared.array((lat_population_unique.shape[0], self.nb_leads, self.max_len_qrs), dtype=np.float64)
    #     ecg_population_unique[:, :, :] = get_nan_value()
    #     threads_num = multiprocessing.cpu_count()
    #     # Uncomment the following lines to turn off the parallelisation of the ecg computation.
    #     # if True:
    #     #     for conf_i in range(ecg_population_unique.shape[0]):
    #     with pymp.Parallel(min(threads_num, ecg_population_unique.shape[0])) as p1:
    #         for conf_i in p1.range(ecg_population_unique.shape[0]):
    #             aux_ecg = self.calculate_ecg(lat=lat_population_unique[conf_i, :], vm=vm_population_unique[conf_i, :, :])
    #             ecg_len = int(aux_ecg.shape[1])
    #             ecg_population_unique[conf_i, :, :ecg_len] = aux_ecg
    #     return ecg_population_unique[reverse_unique_indexes, :, :]

    def pseudo_ecg_from_vm_step_function(self, lat, vm):
        nb_time_steps = min(self.max_len_qrs, vm.shape[1])  # 1000 Hz is one evaluation every 1 ms
        simulated_ecg = np.full((self.nb_leads, nb_time_steps), np.nan, dtype=np.float64)

        # bsp is  ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        bsp = np.zeros((self.nb_bsp, nb_time_steps), dtype=np.float64)
        ele_contrib = np.zeros((self.nb_bsp, self.tetra.shape[0]), dtype=np.float64)

        for time_step_i in range(1, nb_time_steps, 1):
            # Shortcuts when using a step function to define the vm values for the QRS calculation.
            active_nodes = np.nonzero(lat == time_step_i)[0].astype(np.int32)
            if not len(active_nodes) == 0:
                active_tetra = np.unique(
                    np.concatenate([self.tetra_per_node[node_index] for node_index in active_nodes]))
                # print('active_tetrahedra ', np.amax(active_tetrahedra))
                # print('self.tetrahedrons ', self.tetrahedrons.shape)
                b_m_vm = (vm[self.tetra[active_tetra, 0:3], time_step_i]
                          - vm[self.tetra[active_tetra, 3], time_step_i][:, np.newaxis])
                bd_vm = np.squeeze(
                    np.matmul(self.dxyz[active_tetra, :, :], b_m_vm[:, :, np.newaxis]), axis=2)
                ele_contrib[:, active_tetra] = np.sum(self.dr[:, active_tetra, :] * bd_vm, axis=2)
                bsp[:, time_step_i] = np.sum(ele_contrib, axis=1)
            else:
                bsp[:, time_step_i] = bsp[:, time_step_i - 1]
        ele_contrib = None  # Clear memory
        # Applying Einthoven's triangle
        simulated_ecg[0, :] = (bsp[0, :] - bsp[1, :])
        simulated_ecg[1, :] = (bsp[2, :] - bsp[1, :])
        BSPecg = bsp - np.mean(bsp[0:3, :], axis=0)
        bsp = None  # Clear Memory
        simulated_ecg[2:self.nb_leads, :] = BSPecg[4:self.nb_bsp, :]
        return self.preprocess_ecg(simulated_ecg)

# EOF






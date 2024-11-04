# This is a stand alone script that plots ECGs from body surface potentials
# simulated from monodomain simulations.
# The normalisation strategy is the same as in ecg_functions.py.
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
# Helper functions from ecg_functions.py:

def filter_butterworth_ecg(b, a, ecg):
    return signal.filtfilt(b, a, ecg)

def __filter_ecg(original_ecg, low_a_filtfilt, low_b_filtfilt, high_a_filtfilt, high_b_filtfilt):
    # First we filter out the low frequencies using a high-pass filter and the lower thresholds
    processed_ecg = filter_butterworth_ecg(b=low_b_filtfilt, a=low_a_filtfilt, ecg=original_ecg)
    # Secondly we filter out the high frequencies using a low-pass filter and the higher thresholds
    return filter_butterworth_ecg(b=high_b_filtfilt, a=high_a_filtfilt, ecg=processed_ecg)

def zero_align_ecg(original_ecg):
    return original_ecg - (original_ecg[:, 0:1] + original_ecg[:, -2:-1]) / 2  # align at zero


def __preprocess_ecg_without_normalise(original_ecg, filtering, zero_align, frequency, high_freq_cut, low_freq_cut):
    processed_ecg = original_ecg
    if filtering:
        high_w = high_freq_cut / (frequency / 2)  # Normalize the frequency
        high_b_filtfilt, high_a_filtfilt = signal.butter(4, high_w,'low')  # Butterworth filter of fourth order.
        low_w = low_freq_cut / (frequency / 2)  # Normalize the frequency
        low_b_filtfilt, low_a_filtfilt = signal.butter(4, low_w, 'high')
        processed_ecg = __filter_ecg(processed_ecg, low_a_filtfilt=low_a_filtfilt, low_b_filtfilt=low_b_filtfilt,
                                     high_a_filtfilt=high_a_filtfilt, high_b_filtfilt=high_b_filtfilt)
    if zero_align:
        processed_ecg = zero_align_ecg(processed_ecg)
    return processed_ecg

def __set_reference_lead_is_positive(max_qrs_end, reference_ecg):
    approx_qrs_end = min(reference_ecg.shape[1], max_qrs_end)  # Approximate end of QRS.
    reference_lead_max = np.absolute(np.amax(reference_ecg[:, :approx_qrs_end], axis=1))
    reference_lead_min = np.absolute(np.amin(reference_ecg[:, :approx_qrs_end], axis=1))
    reference_lead_is_positive = reference_lead_max >= reference_lead_min
    reference_amplitudes = np.zeros(shape=nb_leads, dtype=np.float64)
    reference_amplitudes[reference_lead_is_positive] = reference_lead_max[reference_lead_is_positive]
    reference_amplitudes[np.logical_not(reference_lead_is_positive)] = reference_lead_min[
        np.logical_not(reference_lead_is_positive)]
    # if verbose:
    #     print('reference_lead_is_positive')
    #     print(reference_lead_is_positive)
    #     print('reference_amplitudes')
    #     print(reference_amplitudes)
    return reference_lead_is_positive  # Have some R progression by normalising by the
    # largest absolute amplitude lead

def __normalise_ecg_based_on_rwave_8_leads(original_ecg, qrs_onset, reference_ecg, max_len_qrs, reference_lead_is_positive):
    if nb_leads != 8 or original_ecg.shape[0] != 8:
        raise(Exception, 'This function is hardcoded for the specific ECG configuration: I, II, V1, ..., V6')
    # print('Normalising ECG ', original_ecg.shape)
    approx_qrs_end = min(reference_ecg.shape[1], max_len_qrs+qrs_onset)  # Approximate end of QRS.
    # approx_qrs_width = min(original_ecg.shape[1], max_len_qrs)  # This approximation is more robust.
    # print('approx_qrs_end ', approx_qrs_end)
    # print(np.amax(original_ecg[:, :approx_qrs_end]))
    reference_amplitudes = np.empty(shape=nb_leads, dtype=np.float64)
    reference_amplitudes[reference_lead_is_positive] = np.absolute(
        np.amax(original_ecg[:, :approx_qrs_end], axis=1)[
            reference_lead_is_positive])
    reference_amplitudes[np.logical_not(reference_lead_is_positive)] = \
        np.absolute(np.amin(original_ecg[:, :approx_qrs_end], axis=1))[np.logical_not(
            reference_lead_is_positive)]
    normalised_ecg = np.zeros(original_ecg.shape)
    normalised_ecg[:2, :] = original_ecg[:2, :] / np.mean(
        reference_amplitudes[:2])  # Normalise limb leads separatedly
    normalised_ecg[2:nb_leads, :] = original_ecg[2:nb_leads, :] / np.mean(
        reference_amplitudes[2:nb_leads])
    return normalised_ecg

def preprocess_ecg(original_ecg, reference_ecg, filtering, zero_align, frequency, high_freq_cut, low_freq_cut,
                   max_len_qrs, qrs_onset, reference_lead_is_positive):
    processed_ecg = original_ecg
    if filtering:
        high_w = high_freq_cut / (frequency / 2)  # Normalize the frequency
        high_b_filtfilt, high_a_filtfilt = signal.butter(4, high_w,'low')  # Butterworth filter of fourth order.
        low_w = low_freq_cut / (frequency / 2)  # Normalize the frequency
        low_b_filtfilt, low_a_filtfilt = signal.butter(4, low_w, 'high')
        processed_ecg = __filter_ecg(processed_ecg, low_a_filtfilt=low_a_filtfilt, low_b_filtfilt=low_b_filtfilt,
                                     high_a_filtfilt=high_a_filtfilt, high_b_filtfilt=high_b_filtfilt)
    if zero_align:
        processed_ecg = zero_align_ecg(processed_ecg)
    if normalise:
        processed_ecg = __normalise_ecg_based_on_rwave_8_leads(original_ecg=processed_ecg, qrs_onset=qrs_onset,
                                                               reference_ecg=reference_ecg, max_len_qrs=max_len_qrs,
                                                               reference_lead_is_positive=reference_lead_is_positive)
    return processed_ecg

def import_simulated_ecg_8leads_raw(filename, monoalg_activation_offset):
    data = np.genfromtxt(filename, skip_footer=1)
    t = data[:, 0] - monoalg_activation_offset
    LA = data[:, 1]
    RA = data[:, 2]
    LL = data[:, 3]
    RL = data[:, 4]
    V1 = data[:, 5]
    V2 = data[:, 6]
    V3 = data[:, 7]
    V4 = data[:, 8]
    V5 = data[:, 9]
    V6 = data[:, 10]

    # Ealuate Wilson's central terminal
    VW = 1.0 / 3.0 * (RA + LA + LL)

    # Evaluate simulated ECG lead traces
    V1 = V1 - VW
    V2 = V2 - VW
    V3 = V3 - VW
    V4 = V4 - VW
    V5 = V5 - VW
    V6 = V6 - VW
    I = LA - RA
    II = LL - RA
    III = LL - LA
    aVL = LA - (RA + LL) / 2.0
    aVF = LL - (LA + RA) / 2.0
    aVR = RA - (LA + LL) / 2.0
    ecgs = np.vstack((I, II, V1, V2, V3, V4, V5, V6))
    return t, ecgs

def visualise_ecgs(nb_leads, reference_ecg, simulated_ecgs, simulated_t,  lead_names, casename):
    nb_cols = (nb_leads * 2) ** 0.5
    if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
        nb_rows = nb_cols / 2
    else:
        # Try to make 2 rows and the necessary columns
        nb_cols = nb_leads / 2
        if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
            nb_rows = nb_cols / 2
        else:
            nb_cols = nb_leads
            nb_rows = 1
    fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10))
    axes = np.reshape(axes, nb_leads)
    for lead_i in range(nb_leads):
        time_steps = np.arange(reference_ecg.shape[1])
        axes[lead_i].plot(time_steps, reference_ecg[lead_i, :], label='Clinical', color='lime', linewidth=3.)
        axes[lead_i].plot(simulated_t, simulated_ecgs[lead_i, :], color='k', label='Simulation', linewidth=1.)
        axes[lead_i].set_title(lead_names[lead_i])
        axes[lead_i].set_ylim([-1.5, 1.5])
        for tick in axes[lead_i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in axes[lead_i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
    axes[nb_leads-1].legend(loc='center left', bbox_to_anchor=(0.1, 0.2), fontsize=14)
    fig.suptitle(casename)
    plt.show()


def read_csv_file(filename, skiprows=0, usecols=None):
    return np.loadtxt(filename, delimiter=',', dtype=float, skiprows=skiprows, usecols=usecols)

def read_ecg_from_csv(filename, nb_leads):
    folded_data = read_csv_file(filename=filename)
    return unfold_ecg_matrix(data=folded_data, nb_leads=nb_leads)

def unfold_ecg_matrix(data, nb_leads):
    # Check dimensions of data to decide how to reshape
    if len(data.shape) == 1:
        ecg = np.reshape(data, (1, int(nb_leads), -1), order='C')
    elif len(data.shape) == 2:
        ecg = np.reshape(data, (data.shape[0], int(nb_leads), -1), order='C')
    else:
        raise "How did you save an array with more than 2 dim in a CSV? This was not supported yet in 2023!"
    return ecg


###################################################################################################################
## Main script
filtering = True
normalise = True
zero_align = True
frequency = 1000
high_freq_cut = 150
low_freq_cut = 0.5
max_len_ecg = 450
max_len_qrs = 200
nb_leads = 8
qrs_onset = 0
casename = 'DTI032'

# Preprocess the clinical data
print('Preprocessing clinical ECGs')
clinical_data_filename_path = '/mnt/scratch/jenny/'+casename+'_clinical_full_ecg.csv'
reference_ecg = np.genfromtxt(clinical_data_filename_path, delimiter=',')
reference_ecg = __preprocess_ecg_without_normalise(original_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
                                                   frequency=frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut)
max_qrs_end = qrs_onset + max_len_qrs
reference_lead_is_positive = __set_reference_lead_is_positive(max_qrs_end=max_qrs_end, reference_ecg=reference_ecg)
reference_ecg = preprocess_ecg(original_ecg=reference_ecg, reference_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
                                frequency=frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut, max_len_qrs=max_len_qrs,
                               qrs_onset=qrs_onset, reference_lead_is_positive=reference_lead_is_positive)

# Read in simulated ECGs
print('Importing and preprocessing simulated monoAlg3D ECGs')
simulated_t, simulated_ecgs_8leads = import_simulated_ecg_8leads_raw(filename=casename+'_ecg.txt', monoalg_activation_offset=37)
simulation_frequency = simulated_ecgs_8leads.shape[1]
max_len_qrs = 200 * 4
processed_simulated_ecgs = preprocess_ecg(original_ecg=simulated_ecgs_8leads, reference_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
                                frequency=simulation_frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut, max_len_qrs=max_len_qrs,
                               qrs_onset=qrs_onset, reference_lead_is_positive=reference_lead_is_positive)

# Plot together
print('Visualising ECGs together')
visualise_ecgs(nb_leads=nb_leads, reference_ecg=reference_ecg, simulated_ecgs=processed_simulated_ecgs, simulated_t=simulated_t,
               lead_names=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], casename=casename)

# # Test if code is working properly by reading in an Eikonal ECG:
# # Read in simulated ECGs
# print('Importing and preprocessing simulated monoAlg3D ECGs')
# # simulated_t, simulated_ecgs_8leads = import_simulated_ecg_8leads_raw(filename=casename+'_eikonal.txt', monoalg_activation_offset=37)
# simulated_ecgs_8leads = read_ecg_from_csv(filename='./DTI032_eikonal.csv', nb_leads=nb_leads)[0]
# simulated_t = np.arange(simulated_ecgs_8leads.shape[1])
# simulation_frequency = simulated_ecgs_8leads.shape[1]
# # max_len_qrs = 200 * 4
# processed_simulated_ecgs = preprocess_ecg(original_ecg=simulated_ecgs_8leads, reference_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
#                                 frequency=simulation_frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut, max_len_qrs=max_len_qrs,
#                                qrs_onset=qrs_onset, reference_lead_is_positive=reference_lead_is_positive)
#
# # Plot together
# print('Visualising ECGs together')
# visualise_ecgs(nb_leads=nb_leads, reference_ecg=reference_ecg, simulated_ecgs=simulated_ecgs_8leads, simulated_t=simulated_t,
#                lead_names=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], casename=casename)
#



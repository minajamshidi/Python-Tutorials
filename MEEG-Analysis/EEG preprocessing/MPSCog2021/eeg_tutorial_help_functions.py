"""
This script includes some help functions for the EEG tutorial of
the course Methods in Cognitive Neuroscience @ Max Planck School of Cognition


Prepared by Mina Jamshidi Idaji @ MPI CBS, Leipzig, Germany, jamshidi@cbs.mpg.de

June 2020
"""


import numpy as np
import scipy.signal as sp
from matplotlib import pyplot as plt
import mne


def plot_psd(data, fs, f_max=None, overlap_perc=0.5, freq_res=0.5, newfig=True):
    """
    This function plots the spectrum of the input signal
    :param data: ndarray [n_chan x n_samples]
                 data array . can be multi-channel
    :param fs: sampling frequency
    :param f_max: maximum frequency in the plotted spectrum
    :param overlap_perc: overlap percentage of the sliding windows in welch method
    :param freq_res: frequency resolution, in Hz
    :return: no output, plots the spectrum
    """
    if np.iscomplexobj(data):
        data = np.real(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    nfft = 2 ** np.ceil(np.log2(fs / freq_res))
    noverlap = np.floor(overlap_perc * nfft)
    f, pxx = sp.welch(data, fs=fs, nfft=nfft, nperseg=nfft, noverlap=noverlap)
    if f_max is not None:
        pxx = pxx[:, f <= f_max]
        f = f[f <= f_max]
    if newfig:
        plt.figure()
    ax = plt.subplot(111)
    plt.plot(f, 10*np.log10(pxx.T))
    plt.ylabel('PSD (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    return ax


def bandpass_filter_raw_plot(data, fs, f1, f2):
    """
    This function builds a butterworth bandpass filter of order 4 with the pass-band [f1, f2]
    and plots its frequency response. It, then, filters the input data and returns the
    filtered signal.

    :param data: ndarray [n_chan x n_samples]
                 data array . can be multi-channel
    :param fs: int
               sampling frequency
    :param f1: float
               low cut-off frequency of the filter
    :param f2: float
               high cut-off frequency of the filter
    :return: filtered data
    """
    b, a = sp.butter(N=2, Wn=np.array([f1, f2]) / fs * 2, btype='bandpass')  # build a bandpass butterworth filter of order 4, with cut-off frequencies 1 and 45
    w, h = sp.freqz(b, a)  # compute the frequency response of the filter
    f = w / np.pi * fs / 2
    plt.figure()
    plt.plot(f, 10 * np.log10(abs(h)))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('frequency response of butterworth bandpass [1, 45]Hz')
    plt.grid()

    data1 = sp.filtfilt(b, a, data)
    return data1


def notch_filter_raw_plot(data, fs, fc):
    """
        This function builds a iir notch filter with the stop-band centered at fc
        and plots its frequency response. It, then, filters the input data and returns the
        filtered signal.

        :param data: ndarray [n_chan x n_samples]
                     data array . can be multi-channel
        :param fs: int
                   sampling frequency
        :param fc: float
                   central frequency of the stop-band
        :return: filtered data
        """
    b, a = sp.iirnotch(w0=fc / fs * 2, Q=100)
    w, h = sp.freqz(b, a)
    f = w / np.pi * fs / 2
    plt.figure()
    plt.plot(f, 10 * np.log10(abs(h)))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('frequency response of notch filter at 50Hz')
    plt.grid()

    data1 = sp.filtfilt(b, a, data)
    return data1


def extract_ec_condition(raw):
    """
    This function is written to extract the EC condition of LEMON resting-state EEG data.
    :param raw: MNE raw class
    :return: MNE raw object with oly EC data.
    """
    events_ = mne.events_from_annotations(raw)  # extract the events
    events_mat = events_[0]
    annot_onset = events_mat[:, 0]
    annot_description = events_mat[:, -1]

    # mark the 210 events
    ec_array = np.zeros(annot_description.shape)
    ec_array[annot_description == 210] = 1

    # find the onset and offset of the condition
    ec_array_diff = np.diff(ec_array)
    ind_start = np.where(ec_array_diff == 1)[0] + 1  # find the onset of EC -> where ec_array_diff is 1
    ind_end = np.where(ec_array_diff == -1)[0] + 1  # find the end of EC -> where ec_array_diff is -1
    if ind_end.shape[0] != ind_start.shape[0]:
        ind_end = np.append(ind_end, ec_array_diff.shape[0])

    onset_ec = annot_onset[ind_start]  # the sample corresponding to onset of EC
    duration_ec = annot_onset[ind_end] - onset_ec  # the duration of corresponding EC

    # cut the data --> select the sc condition
    data = raw.get_data()
    ind_start_data = onset_ec
    ind_end_data = annot_onset[ind_end]
    n_ec_segments = ind_start.shape[0]
    data_new = np.empty((data.shape[0], 0))
    for i_seg in range(n_ec_segments):
        data_new = np.append(data_new, data[:, ind_start_data[i_seg]:ind_end_data[i_seg]], axis=1)
    print('duration of data=' + str(data_new.shape[1] / raw.info['sfreq'] / 60) + ' (s)')
    raw_ec = mne.io.RawArray(data_new, raw.info)
    raw_ec._annotations = mne.Annotations(onset_ec, duration_ec, ['new ec segment'] * n_ec_segments,
                                          orig_time=raw.info['meas_date'])
    return raw_ec

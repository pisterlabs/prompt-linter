## IMPORTS
import multiprocessing
import numpy as np
import datetime
from os.path import isdir, getsize
from glob import glob
from sys import exit
import matplotlib.pylab as plt
import math
import EMtools as DCIP
from scipy import fftpack
import spectrum
from scipy.signal import detrend
import datfiles_lib_parallel as datlib
from classes import Node
import sys
from statsmodels.tsa.stattools import adfuller
from scipy.signal import coherence
from classes import DataCouple

def synthetic_current(sample_freq, time_base, length):
    """

    # Creation of synthetic current signal of amplitude 1.
    # Input:
    # sample_freq: desired sample frequency of output current file
    # time_base: desired time base of synthetic current signal
    # Output:
    # time: time array
    # syn_current: time serie of synthetic signal
    """
    N = length
    time = np.linspace(1, N, N) / sample_freq
    N_cycle = time_base * 4
    syn_current = np.asarray([0. for i in range(N)])
    for i in range(N):
        if np.mod(time[i], N_cycle) < time_base:
            syn_current[i] = 1.
        elif time_base <= np.mod(time[i], N_cycle) < time_base * 2:
            syn_current[i] = 0.
        elif time_base * 2 <= np.mod(time[i], N_cycle) < time_base * 3:
            syn_current[i] = -1.
        elif time_base * 3 <= np.mod(time[i], N_cycle) < time_base * 4:
            syn_current[i] = -0.

    return time, syn_current


def cycle_current(sample_freq, time_base, length):
    # Creation of synthetic current signal of amplitude 1.
    # Input:
    # sample_freq: desired sample frequency of output current file
    # time_base: desired time base of synthetic current signal
    # Output:
    # time: time array
    # syn_current: time serie of synthetic signal
    N = length
    time = np.linspace(1, N, N) / sample_freq
    N_cycle = time_base * 4
    syn_current = np.asarray([0. for i in range(N)])
    for i in range(N_cycle):
        if np.mod(time[i], N_cycle) < time_base:
            syn_current[i] = 1.
        elif time_base <= np.mod(time[i], N_cycle) < time_base * 2:
            syn_current[i] = 0.
        elif time_base * 2 <= np.mod(time[i], N_cycle) < time_base * 3:
            syn_current[i] = -1.
        elif time_base * 3 <= np.mod(time[i], N_cycle) < time_base * 4:
            syn_current[i] = -0.

    return time, syn_current


def get_maxima(period, nfft, freq, signal_fft):
    # Get local maxima and minima of current waveform using the derivative
    # Creation of synthetic current signal of amplitude 1.
    # Input:
    # period: not used anymore, to be removed
    # nfft: not used anymore, to be removed
    # freq: frequency array corresponding to the input FFT
    # signal_fft: FFT of signal to get maxima
    # Output:
    # periods: frequency array corresponding to maxima locations
    derivative = [signal_fft[i + 1] - signal_fft[i] for i in range(int(len(freq) / 2))]

    periods = []
    index = []
    for i in range(int(len(freq) / 2) - 1):
        if (derivative[i] > 0 and derivative[i + 1]) < 0:
            periods.append(freq[i])
            index.append(i)
    return np.asarray(periods)


def get_frequency_index(freq, periods):
    # Return frequency index corresponding to the input periods
    # Input:
    # freq: Frequency array where index have to be extracted from
    # periods: Periods of analysis
    # Output:
    # index: List of index of freq array corresponding to periods
    index = [0 for i in range(len(periods))]

    count_periods = 0
    for i in range(int(np.floor(len(freq) / 2))):
    #for i in range(len(freq)):
        if freq[i + 1] > periods[count_periods]:
            index[count_periods] = i
            count_periods += 1
            if count_periods == len(periods):
                break
    if count_periods < len(periods):
        index = index[:count_periods]

    return index

def nextpow2(i):
    # Getting next power of 2, greater than i
    # Input:
    # i: number you wish to get the next power of 2
    # outputs:
    # n: power of 2 above i
    n = 1
    while n < i:
        n *= 2
    return n

def make_pow2(time, data):
    # Make signal length a power of 2 for FFT purposes
    # Input:
    # time: time array
    # data: signal array
    # Outputs:
    # time_2: time array padded with zeros so its length its a power of 2
    # data_2: data array padded with zeros so its length its a power of 2
    nfft = nextpow2(len(data))
    N = nfft - len(data)
    if not N == 0:
        data_2 = data.tolist()
        time_2 = time.tolist()
        data_2.extend([0 for i in range(N)])
        time_2.extend([time[-1] + (time[1] - time[0]) * i for i in range(N)])

    return np.asarray(time_2), np.asarray(data_2)

def get_power_spectra(sample_freq, data):
    # Get robust power spectra based on multitaper method
    # Input:
    # sample_freq: sample frequency of input data
    # data: Input data from which the power spectrum has to be derived
    # Output:
    # freq: frequency array of power spectrum
    # spec: robust power spectra
    nfft = 1024
    tbw = 3
    [tapers, eigen] = spectrum.dpss(nfft, tbw, 1)
    #res = spectrum.pmtm(data, e=tapers, v=eigen, show=False)
    amp, weights, _ = spectrum.pmtm(data, NW=3, show=False)
    freq = np.linspace(0, 1, len(amp[0])) * sample_freq

    spec = [0. for i in range(len(amp[0]))]

    for i in range(len(amp[0])):
        for j in range(len(amp)):
            spec[i] += amp[j][i] * weights[i][j]

    return freq, np.abs(spec)

def get_harmonics_amplitude(periods, freq, spectrum):
    # Get amplitude of harmonics
    # Input:
    # sample_freq: sample frequency of input data
    # data: Input data from which the power spectrum has to be derived
    # Output:
    # freq: frequency array of power spectrum
    # spec: robust power spectra

    df_log = 0.1 # width around harmonic for amplitude estimation
    df_log = 5 # for now, use indexes
    ps_values = [0. for i in range(len(periods))]

    for i in range(len(periods)):
        try:
            ps_values[i] = np.max([spectrum[int(periods[i]) - df_log:int(periods[i]) + df_log]])
        except ValueError:
            ps_values[i] = np.nan
    return ps_values


def harmonic_analysis_robust(time, data, periods):

    freq, amp = get_power_spectra(150, data)
    #freq, amp, _ = get_fft(time, data)                          # FFT calculation
    index = get_frequency_index(freq, periods)
    ps_values = get_harmonics_amplitude(index, freq, amp)

    return ps_values, periods


def harmonic_analysis(time, data, periods):

    #freq, amp = get_power_spectra(150, data)
    freq, amp, _ = get_fft(time, data)                          # FFT calculation
    index = get_frequency_index(freq, periods)
    ps_values = get_harmonics_amplitude(index, freq, amp)

    return ps_values, periods


def coherency_analysis(f, C, periods):

    index = get_frequency_index(f, periods)
    coherency = get_harmonics_amplitude(index, f, C)

    return coherency


def get_vp(data, sample_half_t):
    start_vp = 50
    end_vp = 90

    num_half_T = np.round(np.asarray(data).size / sample_half_t) - 2
    trim_length = num_half_T * sample_half_t
    data = np.asarray(data[0:int(trim_length)])

    try:
        window = DCIP.createHanningWindow(num_half_T)   # creates filter window
        tHK = DCIP.filterKernal(filtershape=window)     # creates filter kernal
        tHK.kernel = tHK.kernel[1:-1]
        stack = tHK * data                               # stack data
        Vp = DCIP.getPrimaryVoltage(start_vp, end_vp, stack)         # Use of DCIP Tools for VP calculation
    except:
        Vp = np.nan

    return Vp


def get_harmonics_and_vp(node, on_time, periods, injection):


    messages = []
    test = 1
    sample_freq = 150
    sample_half_t = 600.0
    print("[" + multiprocessing.current_process().name + "] recovering Vp and harmonics of file " + node.file_name)
    messages.append("[" + multiprocessing.current_process().name + "] taking care of node " + node.id + '\n')
    messages.append("[" + multiprocessing.current_process().name + "] \tAnalyzing file: " + node.file_name + '\n')
    fIn = open(node.file_name, 'r', encoding="utf8", errors='ignore')
    linesFIn = fIn.readlines()
    fIn.close()
    try:
        time, data = datlib.read_data(linesFIn)
        time, data = datlib.trim_injection_time(time, data, injection.start_date, injection.end_date)
    except:
        messages.append("[" + multiprocessing.current_process().name + "] \t -> Cannot read file." + '\n')
        test = 0
    if len(time) < 600:
        test = 0
    if test:
        time, data = datlib.trim_on_time(time, data, on_time, 2, 150)

        #result = adfuller(data)
        #node.adf = result[1] # getting p value
        variance = moving_variance_signal(data, 2, 150)
        if any(v > 50 for v in variance):
            messages.append("\t\tAbrupt change in signal variance. Check signal." + '\n')
            _, _ = trim_variance(time, data, 150, 2, variance)
        try:
            node.harmonics, node.freq_harmonics = harmonic_analysis_robust(time, data, periods)
            #node.harmonics, node.freq_harmonics = harmonic_analysis(time, data, periods)
        except:
            messages.append("[" + multiprocessing.current_process().name + "] \t -> Issue with file." + '\n')
            test = 0
        try:
            node.Vp = get_vp(data, sample_half_t)
        except:
            pass

    return node, messages


def get_coherency(node, nodes, periods, injection):

    messages = []
    print("[" + multiprocessing.current_process().name + "] doing coherency analysis of node " + node.id)
    messages.append("[" + multiprocessing.current_process().name + "] taking care of node " + node.id + '\n')
    messages.append("[" + multiprocessing.current_process().name + "] \tCoherency analysis for file: " + node.file_name[:] + '\n')
    test = 1
    try:
        fIn = open(node.file_name, 'r', encoding="utf8", errors='ignore')
        linesFIn = fIn.readlines()
        fIn.close()
        t, data = datlib.read_data(linesFIn)
        t, data = datlib.trim_injection_time(t, data, injection.start_date, injection.end_date)
        if len(t) < 600:
            test = 0
    except:
        messages.append("[" + multiprocessing.current_process().name + '] \t -> Cannot read file.' + '\n')
        test = 0
    dataCouples = []
    if test:
        nodesSelected = get_N_closest_nodes(node, nodes)
        for j in range(len(nodesSelected)):
            if not nodes[j].id == node.id:
                test = 1
                try:
                    fIn = open(nodesSelected[j].file_name, 'r', encoding="utf8", errors='ignore')
                    linesFIn = fIn.readlines()
                    fIn.close()
                    time_2, data_2 = datlib.read_data(linesFIn)
                except:
                    messages.append("[" + multiprocessing.current_process().name + '] \t -> Cannot read file.' + '\n')
                    test = 0
                if test and len(time_2) > 600:
                    time_1, time_2, data_1, data_2, test = datlib.get_common_ts(t, time_2, data, data_2)
                    if test and len(time_1) > 600:
                        if len(time_1) > 2400:
                            #f, C = coherence(data_1, data_1, fs=150.)
                            f, C = coherence(data_1 - np.mean(data_1), data_2 - np.mean(data_2), fs=150., window=('dpss', 3), nperseg=2400, noverlap=1800, nfft=4096)
                        else:
                            f, C = coherence(data_1 - np.mean(data_1), data_2 - np.mean(data_2), fs=150., window=('dpss', 3), nperseg=len(time_1), noverlap=0, nfft=nextpow2(len(time_1)))
                        tmp = coherency_analysis(f, C, periods)
                        dataCouples.append(DataCouple(distance=get_distance_two_nodes(node, nodesSelected[j]),
                                           couple=[node.file_name, nodesSelected[j].file_name],
                                           coherency=tmp,
                                           cluster=-1,
                                           mem=None,
                                           node_index=[j]))
            if len(dataCouples) == 10:
                break
    return dataCouples, messages


def check_vp_outliers(node, nodes):

    messages = []
    print("[" + multiprocessing.current_process().name + "] doing Vp and harmonics analysis of node " + node.id)
    messages.append("[" + multiprocessing.current_process().name + "] taking care of node " + node.id + '\n')
    messages.append("[" + multiprocessing.current_process().name + "] \tVp and harmonics analysis for file: " + node.file_name[:] + '\n')

    nodesSelected = get_N_closest_nodes(node, nodes)
    count = 0
    index = 0
    while count < 4 and index < len(nodesSelected):
        if get_distance_two_nodes(node, nodesSelected[index]) < 500 and not nodesSelected[index].is_quarantine:
            harmonics_ratio = np.median([np.abs(np.log10(node.harmonics[j]) - np.log10(nodesSelected[index].harmonics[j]) /np.log10(node.harmonics[j])) * 100 for j in range(10)])
            Vp_ratio = np.abs((np.log10(np.abs(node.Vp)) - np.log10(np.abs(nodesSelected[index].Vp))) / np.log10(np.abs(node.Vp))) * 100
            messages.append("[" + multiprocessing.current_process().name + "] log10(Vp) ratio:" + str(Vp_ratio) + "; Harmonics ratio: " + str(harmonics_ratio) + "\n")
            #print(np.log10(np.abs(node.Vp)), np.log10(np.abs(nodesSelected[index].Vp)), Vp_ratio, nodesSelected[index].file_name)
            #print(np.log10(node.harmonics[6]), np.log10(nodesSelected[index].harmonics[6]), harmonics_ratio, nodesSelected[index].file_name)
            if Vp_ratio < 50 and harmonics_ratio < 50:
                messages.append("[" + multiprocessing.current_process().name + "] Node" + node.id + " is not considered outlier.\n")
                node.is_quarantine = 0
                #print(Vp_ratio, 'maybe not outlier')
            count += 1
        index += 1

    return node, messages

def get_distances(nodes):
    # Calculates distances between two nodes
    # Input:
    # utm: list of 2 lists. utm[0]: list of easting; utm[1]: list of northing
    # Output:
    # distances: array of size(len(utm[0]), len(utm[0])) containing distances between nodes
    distances = np.asarray([[0. for i in range(len(nodes))] for j in range(len(nodes))])
    for i in range(len(nodes)):
        distances[i, :] = np.asarray([np.sqrt((nodes[i].location[0] - nodes[j].location[0]) ** 2 + (nodes[i].location[1] - nodes[j].location[1]) ** 2) for j in range(len(nodes))])

    return distances

def get_distance_two_nodes(node_a, node_b):

    return np.sqrt((node_a.location[0] - node_b.location[0]) ** 2 + (node_a.location[1] - node_b.location[1]) ** 2)


def get_N_closest_nodes(node, nodes):

    distance = [np.nan for i in range(len(nodes))]
    fIn = open(node.file_name, 'r', encoding="utf8", errors='ignore')
    linesFIn = fIn.readlines()
    fIn.close()
    time_1 = datlib.get_start_end_time(linesFIn)
    #time, data = datlib.read_data(linesFIn)
    for i in range(len(nodes)):
        if len(nodes[i].file_name):
            try:
                fIn = open(nodes[i].file_name, 'r', encoding="utf8", errors='ignore')
                linesFIn = fIn.readlines()
                fIn.close()
                time_2 = datlib.get_start_end_time(linesFIn)
                test, time_span = datlib.is_common_ts(time_1, time_2, 150)
                if test and time_span > 2400:
                    distance[i] = np.sqrt((node.location[0] - nodes[i].location[0]) ** 2 + (node.location[1] - nodes[i].location[1]) ** 2)
                else:
                    distance[i] = np.nan
            except:
                distance[i] = np.nan

    nodes_select = []
    tmp_nodes_names = []
    ind = np.argsort(distance) # np.nan are at the end
    for i in range(len(ind)):
        if nodes[ind[i]].id not in tmp_nodes_names and (not nodes[ind[i]].id == node.id):
            nodes_select.append(nodes[ind[i]])
            tmp_nodes_names.append(nodes[ind[i]].id)

    return nodes_select



def get_distance_from_point(nodes, point):
    distances = [[], []]
    distances[0] = [nodes[i].location[0] - point[0] for i in range(len(nodes))]
    distances[1] = [nodes[i].location[1] - point[1] for i in range(len(nodes))]

    return distances


def get_power_ratio(nodes):
    # Calculates power ratio between two nodes
    # Input:
    # values: list of power for harmonics
    # Output:
    # ratios: array of size(len(values), len(values)) containing power ratio between nodes
    ratios = [np.asarray([[np.nan for i in range(len(nodes))] for j in range(len(nodes))]) for k in range(len(nodes[0].harmonics))]
    for k in range(len(nodes[0].harmonics)):
        for i in range(len(nodes)):
            ratios[k][i, :] = np.log10( np.asarray([nodes[i].harmonics[k] / nodes[j].harmonics[k] for j in range(len(nodes))]))
    return ratios


def get_Vp_differences(values):
    # Calculates power ratio between two nodes
    # Input:
    # values: list of Vp differences
    # Output:
    # ratios: array of size(len(values), len(values)) containing power ratio between nodes
    differences = np.asarray([[np.nan for i in range(len(values))] for j in range(len(values))])
    for i in range(len(values)):
        differences[i, :] = np.asarray([(values[i] - values[j]) / values[i] for j in range(len(values))])

    return differences


def get_Vp_ratios(values):
    # Calculates power ratio between two nodes
    # Input:
    # values: list of Vp ratios
    # Output:
    # ratios: array of size(len(values), len(values)) containing power ratio between nodes
    differences = np.asarray([[np.nan for i in range(len(values))] for j in range(len(values))])
    for i in range(len(values)):
        differences[i, :] = np.asarray([np.abs(values[i] / values[j]) for j in range(len(values))])

    return differences


def remove_outliers(ps_values, periods):
    # Removes obvious outlier from harmonic selection based on variation in power and period
    # Input:
    # ps_values: values for power spectra
    # periods: frequencies corresponding to ps_values
    # Output:
    # ps_values_filt, periods_filt: list of filtered ps_values and periods

    error = [(ps_values[i + 1] - ps_values[i]) / ps_values[i] \
             for i in range(len(periods) - 1)]
    error_period = [(np.log10(periods[i + 2]) - np.log10(periods[i + 1])) / (np.log10(periods[i + 1]) - np.log10(periods[i])) \
             for i in range(len(periods) - 2)]

    ps_values_filt = []
    periods_filt = []
    for i in range(len(error)):
        if i < len(error) - 2:
            if error[i] < 5 and error_period[i] < 3:
                ps_values_filt.append(ps_values[i])
                periods_filt.append(periods[i + 1])
        else:
            if error[i] < 5:
                ps_values_filt.append(ps_values[i])
                periods_filt.append(periods[i])
    return ps_values_filt, periods_filt


def get_fft(time, data):
    # Get fft of 1D time series
    # Input:
    # time: array of time values
    # data: array of values for the 1D time series
    # Output:
    # freq: array of frequencies for spectral analysis
    # amp: Amplitude of power spectra
    # phase: phase of power spectra

    t_inc = time[1] - time[0]
    t_inc = 1. / 150.0
    if not np.mod(len(data), 2) == 0:
        nfft = nextpow2(len(data))
    else:
        nfft = len(data)
    spectrum = fftpack.fft(data, nfft) / nfft
    amp = np.abs(spectrum)
    phase = np.arctan2(spectrum.imag, spectrum.real)
    freq = np.linspace(0, 1, nfft) / t_inc

    return freq, amp, phase


def moving_variance_signal(data, time_base, sample_freq):

    N = time_base * 4 * sample_freq
    stability = [0. for i in range(int(len(data) / N))]
    t_stability = [N * i for i in range(int(len(data) / N))]
    variation_variance = [0. for i in range(len(stability) - 1)]
    for i in range(int(len(data) / N)):
        stability[i] = np.std(detrend(data[i * N:(i + 1) * N] - np.median(data[i * N:(i + 1) * N])))
        if i > 0:
            variation_variance[i - 1] = np.abs(stability[i] - stability[i - 1]) / stability[i - 1]

    return variation_variance


def get_current_stability(current, sample_freq, time_base):

    current = current - np.median(current)
    N = time_base * 4 * sample_freq
    t_stability = [N * i for i in range(int(len(current) / N))]

    current_value = [[0., 0.] for i in range(int(len(current) / N))]
    for i in range(int(len(current) / N)):
        current_value[i][0] = np.mean((current[i * N:i * N + time_base * sample_freq]))
        current_value[i][1] = np.mean((current[i * N + int(N / 2):i * N + int(N / 2) + time_base * sample_freq]))

    variability = [0., 0.]
    for i in range(len(current_value)):
        for j in range(len(current_value)):
            tmp = np.abs(current_value[i][0] - current_value[j][0]) / current_value[i][0] * 100
            if tmp > variability[0]:
                variability[0] = tmp
            tmp = np.abs(current_value[i][1] - current_value[j][1]) / np.abs(current_value[i][1]) * 100
            if tmp > variability[1]:
                variability[1] = tmp

    return variability

def trim_variance(time, data, sample_freq, time_base, variation_variance):

    N = time_base * 4 * sample_freq

    test = []
    i = 0
    data_out = [[]]
    time_out = [[]]
    for i in range(len(variation_variance)):
        if variation_variance[i] < 5:
            #print("Ok, carry on")
            data_out[-1].extend(data[i * N: (i + 1) * N])
            time_out[-1].extend(time[i * N: (i + 1) * N])
        else:
            data_out.append([])
            time_out.append([])

    #fig, (ax1) = plt.subplots(1,1)
    #ax1.plot(time, data)
    #for i in range(len(data_out)):
    #        ax1.plot(time_out[i], data_out[i], '*')
    #plt.show()

    return time_out, data_out

def sort_in_bins_2d(data, L):

    x_min = data[0].distance
    x_max = data[0].distance
    for i in range(len(data)):
        #print(data[i].distance)
        if data[i].distance > x_max:
            x_max = data[i].distance
        if data[i].distance < x_min:
            x_min = data[i].distance

    x_min = int(np.floor(x_min / L) * L)
    x_max = int(np.ceil(x_max / L) * L)
    print(x_min, x_max)
    N_bins = int((x_max - x_min) / L)
    bins = np.linspace(0, N_bins - 1, N_bins) * 100 + x_min
    binCenter = np.asarray([bins[i] + L / 2 for i in range(len(bins) - 1)])
    for i in range(len(data)):
        distance = np.asarray([np.abs(data[i].distance - binCenter[j]) for j in range(len(binCenter))])
        min_distance = np.min(distance)
        cluster = [j for j in range(len(binCenter)) if np.abs(data[i].distance - binCenter[j]) == min_distance]
        cluster = cluster[0]
        data[i].distance = binCenter[cluster]
        data[i].cluster = cluster

    return data, N_bins

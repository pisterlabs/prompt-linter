from scipy.signal import coherence, welch, csd, decimate, lfilter
import numpy as np
import multiprocessing as mp
import time

def imag_coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1):
    r"""
    Copied from signal.coherence to calculate the imaginary part
    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    Generate two test signals with some common features.
    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> b, a = signal.butter(2, 0.25, 'low')
    >>> x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*np.sin(2*np.pi*freq*time)
    >>> y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
    Compute and plot the coherence.
    >>> f, Cxy = signal.img_coherence(x, y, fs, nperseg=1024)
    >>> plt.semilogy(f, Cxy)
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Imaginary Coherence')
    >>> plt.show()
    """

    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft, detrend=detrend,
                       axis=axis)
    _, Pyy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)

    ImgCxy = np.imag(Pxy)**2 / Pxx / Pyy
    #Cxy = np.imag(Pxy**2 / Pxx / Pyy)

    return freqs, ImgCxy #Cxy


class parallel_coh():

  def __init__(self, voltage_state, sampling_r, frequency_r, b, a, coh_type):
    self.volt_state = voltage_state
    self.sampling_r = sampling_r
    self.frequency_r = frequency_r
    self.fnotch = 50 # notching filter at 50 Hz
    self.b = b
    self.a = a
    self.coh_type = coh_type # absolute or imaginary part of the coherence

  def calculate(self, first_elect, sec_elect):
    first_voltages = self.volt_state[:, first_elect]
    second_voltages =  self.volt_state[:, sec_elect]

    filtered_first = lfilter(self.b, self.a, first_voltages) # filtering power notch at 50 Hz
    filtered_second = lfilter(self.b, self.a, second_voltages)
    if self.coh_type == 'abs':
      f_loop, Cxy_loop = coherence(filtered_first, filtered_second, self.sampling_r, nperseg=self.frequency_r)
    else:
      f_loop, Cxy_loop = imag_coherence(filtered_first, filtered_second, self.sampling_r, nperseg=self.frequency_r)
    return f_loop, Cxy_loop



class session_coherence():

  def __init__(self, raw_times, raw_voltages, downsampling,
                montage_name, n_electrodes, sampling_rate, brain_state):
    self.raw_times = raw_times
    self.montage_name = montage_name
    self.n_electrodes = n_electrodes
    self.final_srate = sampling_rate/downsampling
    self.downfreq_ratio = sampling_rate*2/downsampling
    self.f_ratio = 2 # 2 samples per Hz
    self.down_voltages = []
    self.brain_state = brain_state # 0, 1, 2, 4. 3 for 1 and 2 together.
    self.coherence_short = []
    self.coherence_long = []
    self.f_long = []
    self.f_short =[]
    self.z_long = []
    self.z_short = []
    self.volt_state = np.transpose(raw_voltages)
    self.time_state = np.size(raw_voltages[0,:])
    self.time_wake = 0
    self.time_REM = 0
    self.time_NoREM = 0

    
  # It will be different depending on the brain state
  def calc_cohe_short(self, comb_short_distance, s_processes=34, s_chunk=1, b = np.array([0,0,0]), a=np.array([0,0,0]), ch_type='abs'):
    
    ### PARALLEL ###
    start_time = time.time()
    coh_short = parallel_coh(self.volt_state, self.final_srate, self.downfreq_ratio, b, a, ch_type)
    pool = mp.Pool(s_processes)
    # starmap only returns one value, even if the function returns more than one
    coherence_short_parallel = pool.starmap(coh_short.calculate, comb_short_distance, chunksize=s_chunk)
    pool.close()
    self.f_short = np.asarray(coherence_short_parallel[0][0]) # just need a frequency array. They are all the same
    coherences = []
    for ind_coh in list(coherence_short_parallel):
      coherences.append(ind_coh[1]) # first element is the list of frequencies, the second are the coherences
    self.coherence_short = np.asarray(coherences)
    print(f'--- The SHORT distance coherence took {(time.time() - start_time)} seconds ---')


  def calc_cohe_long(self, comb_long_distance, l_processes=48, l_chunk=1, b = np.array([0,0,0]), a=np.array([0,0,0]), ch_type='abs'):

    ### PARALLEL ###
    start_time = time.time()
    coh_long = parallel_coh(self.volt_state, self.final_srate, self.downfreq_ratio, b, a, ch_type)
    pool = mp.Pool(l_processes)
    # starmap only returns one value, even if the function returns more than one
    coherence_long_parallel = pool.starmap(coh_long.calculate, comb_long_distance, chunksize=l_chunk)
    pool.close()
    self.f_long = np.asarray(coherence_long_parallel[0][0]) # just need a frequency array. They are all the same
    coherences = []
    for ind_coh in list(coherence_long_parallel):
      coherences.append(ind_coh[1])
    print(f'--- The LONG distance coherence took {(time.time() - start_time)} seconds ---')
    self.coherence_long = np.asarray(coherences)


  def calc_areas_coh(self, f_list = []):
    k_top_freq = self.set_top_freq()
    self.f_w=self.f_short[1*self.f_ratio: k_top_freq*self.f_ratio + 1] # 1.5-100 Hz
    self.coh_areas_animal  = self.coherence_short[:, 1*self.f_ratio: k_top_freq*self.f_ratio + 1]
    
  
  def calc_zcoh_short(self, f_list = []):
    k_top_freq = self.set_top_freq()
    self.f_w=self.f_short[1*self.f_ratio: k_top_freq*self.f_ratio + 1] # 1.5-100 Hz
    Cxy_w_short  = self.coherence_short[:, 0*self.f_ratio: k_top_freq*self.f_ratio + 1]
    # First pass everything to z
    Cxy_w_short_z = np.arctanh(Cxy_w_short)
    # Then average in z
    # For line plotting (array of numbers, from 1.5 Hz to k_top_freq Hz bins)
    short_line_plot_m_z=np.mean(Cxy_w_short_z[:, 1*self.f_ratio : k_top_freq*self.f_ratio + 1], axis=0)
    print("******short mean z coherences*****")
    print(Cxy_w_short_z[:, 1*self.f_ratio : k_top_freq*self.f_ratio + 1])

    # Z inverse transform
    self.short_line_plot_1rec_m = np.tanh(short_line_plot_m_z)


    # Same for every freq band. Both for bar plots and significance statistics
    self.short_1rec_m = []
    for freq_band in f_list:
      short_m_z = np.mean(Cxy_w_short_z[:, freq_band[1]*self.f_ratio : freq_band[2]*self.f_ratio + 1], axis=(0,1))
      self.short_1rec_m.append(np.tanh(short_m_z))


  def calc_zcoh_long(self, f_list):
    k_top_freq = self.set_top_freq()
    self.f_w=self.f_long[1*self.f_ratio: k_top_freq*self.f_ratio + 1] # 1.5-k_top_freq Hz
    Cxy_w_long  = self.coherence_long[:, 0*self.f_ratio: k_top_freq*self.f_ratio + 1]
    # First pass everything to z
    Cxy_w_long_z = np.arctanh(Cxy_w_long)
    # Then average in z
    long_line_plot_m_z=np.mean(Cxy_w_long_z[:, 1*self.f_ratio : k_top_freq*self.f_ratio + 1], axis=0)
    self.long_line_plot_1rec_m = np.tanh(long_line_plot_m_z)

    # Same for every freq band. Both for bar plots and significance statistics
    self.long_1rec_m = []
    for freq_band in f_list:
      long_m_z = np.mean(Cxy_w_long_z[:, freq_band[1]*self.f_ratio : freq_band[2]*self.f_ratio + 1], axis=(0,1))
      self.long_1rec_m.append(np.tanh(long_m_z))

  def set_top_freq(self):
    if self.final_srate == 1000:
      top_freq = 400
    elif self.final_srate == 500:
      top_freq = 200
    elif self.final_srate == 250:
      top_freq = 100
    else:
      top_freq = 50
    return top_freq



  def parallel_coh(self, first_elect, sec_elect):
    f_loop, Cxy_loop = coherence(self.volt_state[:, first_elect + 2], self.volt_state[:, sec_elect + 2], self.final_srate, nperseg=self.downfreq_ratio)
    return f_loop, Cxy_loop
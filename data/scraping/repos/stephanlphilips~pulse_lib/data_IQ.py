"""
data class to store IQ based signals.
"""
from scipy import signal
import numpy as np

from dataclasses import dataclass

class envelope_generator():
    """
    Object that handles envelope functions that can be used in spin qubit experiments.
    Key properties
        * Makes sure average amplitude of the evelope is the one expressed
        * Executes some subsampling functions to give greater time resolution than the sample rate of the AWG.
        * Allows for plotting the FT of the envelope function.
    """
    def __init__(self, AM_envelope_function = None, PM_envelope_function = None):
        """
        define envelope funnctions.
        Args
            AM_envelope_function (str/tuple) : spec of a windowing as defined for the scipy.signal.windows.get_window function
            AM_envelope_function (lamba M) : (function overload) function where M is the number of points where the evelopen should be rendered for.
            AM_envelope_function (lamba M) : (function overload) function where M is the number of points where the evelopen should be rendered for.
        """
        self.AM_envelope_function = AM_envelope_function
        self.PM_envelope_function = PM_envelope_function

    def get_AM_envelope(self, delta_t, sample_rate=1):
        """
        Render the envelope for the given waveshape (in init).

        Args:
            delta_t (float) : time of the pulse (5.6 ns)
            sample_rate (float) : number of samples per second (e.g. 1GS/s)

        Returns:
            envelope (np.ndarray[ndim=1, dtype=double]) : envelope function in DC
        """

        n_points = delta_t*sample_rate
        if n_points < 1: #skip
            return 0.0

        if self.AM_envelope_function is None:
            envelope = 1.0  #assume constant envelope
        elif isinstance(self.AM_envelope_function, tuple) or isinstance(self.AM_envelope_function, str):
            envelope = signal.get_window(self.AM_envelope_function, int(n_points*10))[::10][:int(n_points)] #ugly fix
        else:
            envelope = self.AM_envelope_function(delta_t, sample_rate) # user reponsible to do good subsampling him/herself.

        return envelope

    def get_PM_envelope(self, delta_t, sample_rate=1):
        """
        Render the envelope for the given waveshape (in init).

        Args:
            delta_t (float) : time of the pulse (5.6 ns)
            sample_rate (float) : number of samples per second (e.g. 1GS/s)

        Returns:
            envelope (np.ndarray[ndim=1, dtype=double]) : envelope function in DC
        """

        n_points = delta_t*sample_rate
        if n_points < 1: #skip
            return 0

        if self.PM_envelope_function is None:
            envelope = 0
        elif isinstance(self.PM_envelope_function, tuple) or isinstance(self.PM_envelope_function, str):
            envelope = signal.get_window(self.PM_envelope_function, int(n_points*10))[::10]
        else:
            envelope = self.PM_envelope_function(delta_t, sample_rate) # user reponsible to do good subsampling him/herself.

        return envelope

def make_chirp(f_start, f_stop, time0, time1):
    '''
    Make a chirp.

    Args:
        f_start (float) : start frequency (Hz)
        f_stop (stop frequency) : stop frequency (Hz)
    '''

    chirp_constant = (f_stop - f_start)/(time1*1e-9-time0*1e-9)/2

    def my_chirp(delta_t, sample_rate = 1):
        """
        Function that makes a phase envelope to make a chirped pulse

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """

        n_points = int(delta_t*sample_rate + 0.9)
        t = np.linspace(0, n_points/sample_rate*1e-9, n_points)
        out = 2*np.pi*chirp_constant*t*t
        return out

    return my_chirp

@dataclass
class IQ_data_single:
    start : float = 0
    stop : float = 0
    amplitude : float = 1
    frequency : float = 0
    phase_offset : float = 0
    ''' offset from coherent pulse  '''
    envelope : envelope_generator = None
    ref_channel : str = None
    coherent_pulsing : bool = True

    @property
    def start_phase(self):
        ''' Old name for phase_offset '''
        return self.phase_offset

@dataclass
class Chirp:
    start : float
    stop : float
    amplitude : float
    start_frequency : float
    stop_frequency : float
    ref_channel : str = None
    phase : float = 0.0
    ''' Phase of the chirp. Only used for I/Q rendering. '''

    def phase_mod_generator(self):
        return make_chirp(self.start_frequency,
                          self.stop_frequency,
                          self.start, self.stop)

if __name__ == '__main__':

    """
    Example on how the envelope generator works.
    """
    import matplotlib.pyplot as plt

    # empty envelope
    from pulse_lib.segments.data_classes.data_IQ import envelope_generator
    env = envelope_generator('blackman')
    env.get_AM_envelope(50)

    # example generation of envelopes with different timings.
    env = envelope_generator('blackman')

    data_1 = env.get_AM_envelope(50)
    data_2 = env.get_AM_envelope(50.2)
    data_3 = env.get_AM_envelope(50.4)
    data_4 = env.get_AM_envelope(50.6)

    # plt.figure("general windowing function")
    # plt.plot(data_1, label = "50.0 ns pulse")
    # plt.plot(data_2, label = "50.2 ns pulse")
    # plt.plot(data_3, label = "50.4 ns pulse")
    # plt.plot(data_4, label = "50.6 ns pulse")
    # plt.legend()
    # plt.xlabel("time (ns)")
    # plt.ylabel("amp (a.u.)")

    # make your own envelope
    def gaussian_sloped_envelope(delta_t, sample_rate = 1):
        """
        function that has blackman slopes at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """

        n_points = int(delta_t*sample_rate + 0.9)
        envelope = np.empty([n_points], np.double)
        if delta_t < 16:
            envelope = signal.get_window('blackman', n_points*10)[::10]
        else:
            time_slope = (16 + delta_t)*sample_rate - int(delta_t*sample_rate)
            envelope_left_right = signal.get_window('blackman', int(time_slope*10))[::10]

            half_pt_gauss = int(time_slope/2)

            envelope[:half_pt_gauss] = envelope_left_right[:half_pt_gauss]
            envelope[half_pt_gauss:half_pt_gauss+n_points-int(time_slope)] = 1
            envelope[n_points-len(envelope_left_right[half_pt_gauss:]):] = envelope_left_right[half_pt_gauss:]

        return envelope

    # example generation of envelopes with different timings.
    env = envelope_generator(gaussian_sloped_envelope)

    data_1 = env.get_AM_envelope(20)
    data_2 = env.get_AM_envelope(20.5)
    data_3 = env.get_AM_envelope(21.)
    data_4 = env.get_AM_envelope(21.5)

    plt.figure("custom windowing function")
    plt.plot(data_1, label = "20.0 ns pulse")
    plt.plot(data_2, label = "20.2 ns pulse")
    plt.plot(data_3, label = "20.4 ns pulse")
    plt.plot(data_4, label = "20.6 ns pulse")
    plt.legend()
    plt.xlabel("time (ns)")
    plt.ylabel("amp (a.u.)")
    plt.show()

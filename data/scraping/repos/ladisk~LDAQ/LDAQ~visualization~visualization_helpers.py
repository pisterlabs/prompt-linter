import numpy as np
from scipy.signal import coherence, csd
import types


def compute_nth(max_points_to_refresh, t_span, n_lines, sample_rate):
    points_per_line = max_points_to_refresh/n_lines
    nth = int(np.ceil(sample_rate * (t_span) / points_per_line))
    return nth


def check_subplot_options_validity(subplot_options, layout):
    """
    Check if the plot layout is valid with the given rowspan and colspan.

    Args:
        subplot_options (dict): Options for each subplot (xlim, ylim, axis_style, etc.).
        layout (list): The layout of the QT application. It specifies which channels are plotted on each subplot.

    Returns:
        bool: True if the layout is valid, False otherwise.
    """
    max_row = max([channels['pos'][0] for acq_name, acq_layout in layout.items() for channels in acq_layout if isinstance(channels['pos'], tuple)])
    max_col = max([channels['pos'][1] for acq_name, acq_layout in layout.items() for channels in acq_layout if isinstance(channels['pos'], tuple)])

    # Create a matrix to keep track of the occupied cells in the subplot grid
    occupied_cells = [[False] * (max_col + 1) for _ in range(max_row + 1)]

    # Loop through each subplot and check if it is valid
    for pos, options in subplot_options.items():
        row, col = pos
        rowspan = options.get('rowspan', 1)
        colspan = options.get('colspan', 1)

        # Check if the subplot is out of bounds
        if row + rowspan - 1 > max_row or col + colspan - 1 > max_col:
            return False

        # Check if the subplot overlaps with another subplot
        for i in range(row, row + rowspan):
            for j in range(col, col + colspan):
                if occupied_cells[i][j]:
                    return False
                occupied_cells[i][j] = True

    # If all subplots are valid, return True
    return True

# ------------------------------------------------------------------------------
#  Prepared plot Functions
# ------------------------------------------------------------------------------

def _fun_fft(self, data):
   amp = np.fft.rfft(data) * 2 / len(data)
   freq = np.fft.rfftfreq(len(data), d=1/self.acquisition.sample_rate)

   return np.array([freq, np.abs(amp)]).T

class _FRF_calculation():
    """
    This class can be used for averaging the FRF over multiple calls of the same function.
    """
    def __init__(self) -> None:
        self.first_call = True
        self.N_samples = 0
        self.freq = None
        self.H1 = None
        self.n_avg = 10
        self.n_calls = 1
        
        self.last_fun_call = None
    
    def _calc_frf(self, self_vis, channel_data):   
        if self_vis.acquisition.is_ready == False: # this means that was run for the first time
            self.first_call = True
        
        # estimate FRF:
        x, y = channel_data.T
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        Sxy = np.conj(X) * Y
        Sxx = np.conj(X) * X
        H1 = Sxy / Sxx
        
        if self.first_call: # create the frequency vector and H1
            self.N_samples = len(channel_data)
            self.freq = np.fft.rfftfreq(self.N_samples, d=1/self_vis.acquisition.sample_rate)
            self.first_call = False
            self.H1 = H1
        else: # only update H1
            n = self.n_calls if self.n_calls < self.n_avg else self.n_avg
            
            self.H1 = self.H1 * (n-1)/n + H1 * 1/n
            
        self.n_calls += 1
    
    def get_frf_abs(self, self_vis, channel_data):
        self._calc_frf(self_vis, channel_data) # always update the FRF
        self.last_fun_call = 'abs'
        return np.array([self.freq, np.abs(self.H1)]).T
    
    def get_frf_phase(self, self_vis, channel_data):
        if self.last_fun_call == 'phase' or self.last_fun_call is None: # only update if the last call was phase
            self._calc_frf(self_vis, channel_data)
        self.last_fun_call = 'phase'
        return np.array([self.freq, np.angle(self.H1)*180/np.pi]).T
    
def _fun_frf_amp(self_vis, channel_data):   
    """Default function for calculating the FRF amplitude.

    Args:
        self_vis (class): visualization class object
        channel_data (array): 2D numpy array with (time, channel) shape.

    Returns:
        2D numpy array: np.array([freq, np.abs(H1)]).T
    """
    # estimate FRF:
    # x, y = channel_data.T
    # X = np.fft.rfft(x)
    # Y = np.fft.rfft(y)
    # Sxy = np.conj(X) * Y
    # Sxx = np.conj(X) * X
    # H1 = Sxy / Sxx
    
    # freq = np.fft.rfftfreq(len(channel_data), d=1/self_vis.acquisition.sample_rate)
    
    x, y = channel_data.T
    fs = self_vis.acquisition.sample_rate
    freq, Sxy = csd(x, y, fs, nperseg=int(fs))
    freq, Sxx = csd(x, x, fs, nperseg=int(fs))
    H1 = Sxy / Sxx
    
    return np.array([freq, np.abs(H1)]).T
    
def _fun_frf_phase(self_vis, channel_data):   
    """Default function for calculating the FRF phase.
    
    Args:
        self_vis (class): visualization class object
        channel_data (array): 2D numpy array with (time, channel) shape.
        
    Returns:
        2D numpy array: np.array([freq, np.angle(H1)*180/np.pi]).T
    """

    x, y = channel_data.T
    fs = self_vis.acquisition.sample_rate
    freq, Sxy = csd(x, y, fs, nperseg=int(fs))
    freq, Sxx = csd(x, x, fs, nperseg=int(fs))
    H1 = Sxy / Sxx
    
    return np.array([freq, np.angle(H1)*180/np.pi]).T

def _fun_coh(self_vis, channel_data):   
    """Default function for calculating the coherence.
    
    Args:
        self_vis (class): visualization class object
        channel_data (array): 2D numpy array with (time, channel) shape.
        
    Returns:
        2D numpy array: np.array([freq, coherence]).T
    """
    # estimate FRF:
    x, y = channel_data.T
    fs   = self_vis.acquisition.sample_rate    
    freq, coh = coherence(x, y, fs, nperseg=fs)
    
    return np.array([freq, coh]).T
        
        
        
        
        
    


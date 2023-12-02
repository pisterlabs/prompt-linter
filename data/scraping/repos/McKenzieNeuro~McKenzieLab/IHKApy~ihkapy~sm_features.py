import numpy as np
from scipy.signal import coherence
from itertools import combinations as combin
import warnings # temporary


"""
Each feature is a function that takes an array-like 3d segment 
The segemnt has shape (n_raw_chan,n_samples,n_freq_chan)
The feature functions return a dictionary of features
    key   : str   = name of sub-feature
    value : float = value of the feature evaluated
The keys of this returned dictionary are to be used a column names for
the data-frame in which we save our features.

Example 1. Mean Power raw. 
Takes a segment of data which has (for example) 4 raw channels and 41 
freq channels with 10_000 samples.
For each raw channel
    Compute the mean power 
    Save this to dictionary under key=f"mean_power_ch{n_raw_chan}"
Return the dictionary of mean powers

Example 2. Coherence feature.
Takes a segment of data with four raw channels and 41 freq channels
Discard all the freq channels that are convolutions, only keep raw recording
For each pair of raw channels
    Compute the coherence
    Save selected frequencies to dictionary 
Return the dictionary of raw-chan pairwise coherence



Naming Convention:
    ch_raw refers to the raw electrode channel index
    ch_freq refers to the binary (wavelet freq) file channel index


To add a new feature:
    - Write a function that takes a segment and returns a dictionary of features
    - Add a conditional statement in get_feats() that calls that function
        and adds it to the returned dictionary.
    - Append the name of that feature to the FEATURES list in Options.toml
"""

### Formatting helpers

def _select_freq_chans(segment,chan_freq_idxs) -> (np.ndarray,np.ndarray):
    """Formatting helper for all features. 

    Assumes segment array-like w/ shape (n_chan_raw,n_samples,n_chan_freq)

    Selects frequency (wavelet binary) channels of interest and formats 
    chan_freq_idxs as a 1d numpy ndarray.
    """
    # If None, select all frequency channels
    segment = np.asarray(segment)
    if chan_freq_idxs is None or chan_freq_idxs == "all": 
        chan_freq_idxs = np.arange(segment.shape[2])
    elif type(chan_freq_idxs) == type(1): # int
        chan_freq_idxs = np.asarray([chan_freq_idxs])
        segment = segment[:,:,chan_freq_idxs] 
    else:
        chan_freq_idxs = np.asarray(chan_freq_idxs)
        segment = segment[:,:,chan_freq_idxs] 
    return segment,chan_freq_idxs

def _feats_as_dict_from_2d_array(
        feat_name : str,
        feats_all : np.ndarray,
        chan_freq_idxs : np.ndarray
        ):
    """Helper for single-channel features.

    Turns a 2d feats array* with dims = (n_raw_chans , n_freq_chans)
    into our standard dictionary format that can easily be added to a
    pandas dataframe.

    *Each element is assumed to be a feature (float) corresponding it's 
    index's raw channel and frequency index.
    """
    feats_dict = {}
    for ch_raw,feats_ch_raw in enumerate(feats_all):
        for chb,feat in zip(chan_freq_idxs,feats_ch_raw):
            feats_dict[f"{feat_name}_chraw_{str(ch_raw).zfill(3)}_chfreq_{str(chb).zfill(3)}"] = feat
    return feats_dict


### Begin: Features on individual channels

def mean_power(
        segment : list or np.ndarray,
        chan_freq_idxs : np.ndarray or int = None
        ) -> dict:
    """Get mean power across each channel.

    Parameters
    ----------
    `segment : array-like`
        A list of segment samples. Dims = (n_raw , n_samples , n_freq_chan)

    `chan_freq_idxs : ndarray or int = None`
        The 1d list of binary file indices we are interested in, e.g.
        if we're interested in all wavelet channels (aka binary file 
        channels) pass None. If we're interested in the first two we 
        can pass [0,1] or np.array([0,1]). If we're only interested in
        the raw data we can simply pass an int, 0 in this case. 

    Returns
    -------
    `dict`
        A dictionary where the key,value pairs are `name of feature`
        and `computed feature float`. All keys are strings, all values
        are floats.
    """
    # Select binary (aka wavelet/freq) file channels
    segment,chan_freq_idxs = _select_freq_chans(segment,chan_freq_idxs)
    # Compute the features along samples axis
    mp_all = np.mean(segment,axis=1)
    # Save and return, formatted as dict
    mp_feats = _feats_as_dict_from_2d_array("mean_power",mp_all,chan_freq_idxs)
    return mp_feats


def var(
        segment         : list or np.ndarray,
        chan_freq_idxs  : np.ndarray or int = None
        ) -> dict:
    """Get the variance of the signal"""
    segment,chan_freq_idxs = _select_freq_chans(segment,chan_freq_idxs)
    var_all = np.var(segment,axis=1) # all channels
    var_feats = _feats_as_dict_from_2d_array("var",var_all,chan_freq_idxs)
    return var_feats

### End: Features on individual channels


### Begin: Features comparing two channels

# coherence is always only computed for a single index
def coherence_all_pairs(
        segment         : list or np.ndarray,
        fs              : float or int # = 2000
        ) -> dict:
    """Cohere each pair of signals"""
    # NB: The chan_freq_idxs parameter selects all freqs of relevance
    # TODO: chan_freq_idxs is hard coded... change this to whatever is 
    # specified in toml gets fed to this function 
    chan_freq_idxs = 0 # only select raw channel
    segment,chan_freq_idxs = _select_freq_chans(segment,chan_freq_idxs)
    segment = np.squeeze(segment) # 3d array -> 2d array
    raw_chans = list(range(len(segment)))    
    coherence_dict = {}
    for (chx,chy),(x,y) in zip(combin(raw_chans,2),combin(segment,2)):
        # There parameters defined here are similar to the default 
        # MatLab options
        sample_freqs,cxy = coherence(x,y,fs=fs,window='hann',
                nperseg=256,noverlap=None,nfft=None,
                detrend='constant',axis=0)
        # Select a subset of the 129 freqs: 129 = 256 // 2 + 1
        idxs = np.array([0,2,4,8,16,32,64,-1]) # A logarithmic subset of freqs
        sample_freqs,cxy = sample_freqs[idxs] , cxy[idxs]
        
        # By default noverlap=None defaults it to half the segment length
        # NB: FFT and therefore scipy.signal.coherence samples 
        # frequencies linearly, not logarithmically
        
        # TODO: Do we want to implement something closer to the MatLab 
        # or is this good enough?
        # Scipy does not provide an easy way of computing the coherence
        # at specific frequencies as mscohere does in MatLab. As a 
        # temporary measure I'm returning a logarithmic subset of the 
        # freqs computed by scipy.signal.coherence().
        # We could potentially reimplement mscohere by hand (in Python) 
        # so that it can take a list of frequencies as input.

        # Serialize the freqs
        for sf,cohxy in zip(sample_freqs,cxy):
            feat_name = f"coherence_freq_{round(sf,2)}_chx_{str(chx).zfill(3)}_chy_{str(chy).zfill(3)}"
            coherence_dict[feat_name] = cohxy 

    return coherence_dict


### End: Features comparing two channels


# Compute all the features on a segments
def get_feats(
        segment : np.ndarray or list,
        features_list : list,
        data_ops : dict
        ) -> dict:
    """Computes all features specified in featurelist.

    Computes each feature specified in featurelist on all of the
    segment, then assembles output into a features_dict—a dictionary
    of all features computed, this corresponds to a single row to be 
    appended to our features dataframe.

    To write a new feature, write the method in this module 
    (`sm_features.py`) then add it to execut conditionally in this 
    function (`get_feats`), finally add a string representation to the 
    FEATURES in your `Options.toml` config file.

    Parameters
    ----------
    `segment : np.ndarray or list`
        Is a list of segment samples ordered by the raw_chan index.
        Shape = (n_raw_chan , n_samples , n_wavelet_chan)

    `features_list : list`
        A list of strings: the names of the features you'd like to compute.

    `data_ops : dict`
        Dictionary containing metadata, as defined in Options.toml.

    Returns
    -------
    `dict`
        A dictionary of all the features we computed.
    """
    IMPLEMENTED_FEATS = ["mean_power","var","coherence"]
    for feat in features_list:
        if feat not in IMPLEMENTED_FEATS:
            warnings.warn(f"{feat} has not yet been implemented.")
    all_feats = {}
    if "mean_power" in features_list:
        AMP_FREQ_IDX_ALL = data_ops["AMP_FREQ_IDX_ALL"]
        mp_feats = mean_power(segment,AMP_FREQ_IDX_ALL) 
        all_feats.update(mp_feats) # add it to greater dictionary
    if "var" in features_list:
        var_feats = var(segment,"all")
        all_feats.update(var_feats)
    if "coherence" in features_list:
        FS = data_ops["FS"]
        coherence_feats = coherence_all_pairs(
                segment,
                fs = FS
                )
        all_feats.update(coherence_feats)
    # print()
    # for i,j in all_feats.items(): print(f"{i}:{j}") # debug
    return all_feats

# Entropy, read more: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7515030/ 

def _a_mjr(sig,m,j,r):
    N = len(sig)
    assert N >= j+m
    a1 = sig[j:j+m]
    acc = 0
    for i in range(j+1):
        d = dist(a1 , sig[i:i+m])
        if d < r: acc+=1
    return acc

def _a_mr(sig,m,r):
    N = len(sig)
    acc = sum([a_mjr(sig,m,j,r) for j in range(1,N-m+1)])
    return acc

def samp_en(sig,m,r):
    """Not yet implemented as features..."""
    return a_mr(sig,m+1,r) / a_mr(sig,m,r)



if __name__ == "__main__":
    # Unit tests
    print("Runnin unit tests")
    ### HELPERS

    # TEST _select_freq_chans() 
    segment = [np.random.normal(0,1,(10000,41)) for i in range(4)]
    chan_freq_idxs = [0,1,2,3,4,10,12]
    segment,chan_freq_idxs = _select_freq_chans(segment,chan_freq_idxs)
    try:
        assert segment.shape == (4,10000,7)
        print("Passed Test #1 _select_freq_chans()")
    except:
        print("Failed Test #1 _select_freq_chans()")
    segment = [np.random.normal(0,1,(10000,41)) for i in range(4)]
    chan_freq_idxs = "all"
    segment,chan_freq_idxs = _select_freq_chans(segment,chan_freq_idxs)
    try:
        assert segment.shape == (4,10000,41)
        assert (chan_freq_idxs == np.arange(41)).all()
        print("Passed Test #2 _select_freq_chans()")
    except:
        print("Failed Test #2 _select_freq_chans()")

    # TEST _feats_as_dict_from_2d_array()
    n_chan_raw, n_chan_freq = 4,3
    feats_array = np.random.normal(0,1,(4,3))
    feats_dict = _feats_as_dict_from_2d_array(
        "dummy",
        feats_all = feats_array,
        chan_freq_idxs = np.array([1,5,9])
        )
    print("\nTest _feats_as_dict_from_2d_array() by human inspection.")
    print("Does this output look good?:")
    for i,j in feats_dict.items():
        print(f"{i} : {j}")
    print()

    ### FEATURES
    n_ch_raw = 4
    n_ch_freq = 3
    n_samples = 10000
    segment_1 = [np.ones((n_samples,n_ch_freq)) for i in range(n_ch_raw)]
    segment_0 = [np.zeros((n_samples,n_ch_freq)) for i in range(n_ch_raw)]
    segment_r = [np.random.normal(0,1,(n_samples,n_ch_freq)) for i in range(n_ch_raw)]
    # TEST mean_power()
    print("\nTest mean_power() by human inspection.")
    print("Does this output look good?")
    mp_dict = mean_power(segment_1,"all") 
    for i,j in mp_dict.items():
        print(f"{i} : {j}")

    # TEST var()
    

    # TEST coherence_all_pairs() 






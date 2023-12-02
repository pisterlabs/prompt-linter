import numpy as np
import os
from ihkapy.fileio import utils
from ihkapy.fileio.utils import get_all_valid_session_basenames
from ihkapy.fileio.binary_io import get_n_samples_from_dur_fs,load_binary_multiple_segments
from ihkapy.fileio.metadata_io import get_seizure_start_end_times
from ihkapy.fileio.options_io import load_ops_as_dict # For loading Options.toml
from scipy.signal import coherence
from ihkapy import sm_features # for featurizing
import warnings
import pandas as pd
from tqdm import tqdm


# pure function (not in the strict sense, but doesn't r/w)
def _get_x_pct_time_of_interval(
        start_time : float,     # in seconds
        end_time : float,       # in seconds
        segment_length : float,# in seconds
        pct : float             # proportion of times to sample
        ) -> np.ndarray:
    """Get a randomly sampled 1d array of start times

    Parameters
    ----------
    `start_time : float`
        The beginning timestamp, in seconds, of the interval we sample from.

    `end_time : float`
        The end timestamp, in seconds, of the interval we sample from.

    `segment_length : float`
        The time, in seconds of our samples.

    `pct : float`
        The proportion of segments to select, must be between 0 and 1. 

    Returns
    -------
    `np.ndarray`
        A 1d numpy array of start times for the segments (in seconds). 
        Together with the segment length, these fully define the segments
        of interest to us that we would like to sample from. 
    """
    assert end_time > start_time
    assert pct >= 0.0 and pct <= 1.0
    # The number of segments that fit in the interval
    n_segments = int((end_time - start_time) // segment_length) 
    # List of all start times of max number of non-overlapping segments
    # that fit in the interval.
    segment_start_times = np.linspace(start_time,end_time-segment_length,n_segments)
    if pct == 1.0: return segment_start_times
    # Choose and sort a random sample according to pct
    n_select = int(np.ceil(n_segments * pct)) # num win to select
    segment_start_times = np.random.choice(segment_start_times,n_select,replace=False)
    segment_start_times.sort()
    return segment_start_times


# Helper for calc_features()
# pure
def _get_bin_absolute_intervals(
        start_seiz              : float,
        end_seiz                : float,
        preictal_bins           : list,
        postictal_delay         : int,
        bin_names               : list,
        all_start_times         : list,
        all_end_times           : list,
        total_session_time      : float
        ):
    """Returns dictionary of all valid bin intervals for single seizure.

    The function is named 'absolute' because it measures the absolute 
    number of seconds from the begining of the session (aka recording/file)


    Parameters
    ----------
    `start_seiz : float or int`
        The start time of the seizure in question, in seconds. (Relative
        to the start of the session.)

    `end_seiz : float or int`
        The end time of the seizure in questionm, in seconds. (Relative
        to the start of the session.)

    `preictal_bins : list`
        The list found in Options.toml config file. It's a list of floats
        that specify, in seconds, the pre-ictal bin timestamps.

    `postictal_delay : int or float`
        Specifies post-ictal period after the end of a seizure in seconds
        (e.g. 600)

    `bin_names : list`
        A list of bin names (strings). Must have length of preictal_bins + 2. 
        This is because we have one single intra-ictal and one single post-
        ictal bin. 

    `all_start_times : list`
        A list of the start times of every seizure (in seconds from start 
        of session). Needed to check validity of intervals.

    `all_end_times : list `
        A list of all end times of every seizure (in seconds from start of
        session). Needed to check validity of intervals. 

    `total_session_time : float`
        Total number of seconds from start to end of the sessions.
    
    Returns
    -------
    `dict`
        The keys are names of time bins (e.g. pre_ictal_bin1, post_ictal, etc.)
        If the corresponding bin for the seizure is valid, the value will be a 
        tuple (bin_start_time,bin_end_time). If the bin is not valid, the value
        will be None. A siezure's time bin is considered valid if all of these
        conditions are satisfied:
            1. The bin does not over-lap with any other seizure.
            2. The bin starts after the beginning of the session. 
            3. The bin ends before the end of the session. 
    """
    # Some tests and checks
    assert len(preictal_bins) + 2 == len(bin_names), "Args have incompatible length."
    assert len(all_start_times) == len(all_end_times), "Logic err, start and end times lists must match in length."

    bin_name_intraictal = bin_names[-2]     # string
    bin_name_postictal = bin_names[-1]      # string
    bin_names_preictal = bin_names[:-2]     # list of strings
    
    # Init bin_intervals (the dict we will return)
    # NB: The seizure interval is always valid
    bin_intervals = {bin_name_intraictal : (start_seiz,end_seiz)} 

    ### PRE-ICTAL
    # The absolute time of pre-ictal intervals, (we will iterate through them)
    pre_bins_abs = [start_seiz - i for i in preictal_bins] 
    # Important, not to forget this final implied pre-ictal slice 
    # (the start of seizure)
    pre_bins_abs.append(start_seiz) 
    for bin_name_preictal,start_bin,end_bin in zip(bin_names_preictal,pre_bins_abs[:-1],pre_bins_abs[1:]):
        # TODO: compare with MatLab
        #       Notably verify if there are missing checks

        bin_intervals[bin_name_preictal] = (start_bin,end_bin)
        # Check whether another (different) seizure overlaps with our bin
        # if so, define it as none
        for start_seiz_2,end_seiz_2 in zip(all_start_times,all_end_times):
            # TODO: check MatLab confirm that postictal delay term necessary
            condition0 = start_bin < end_seiz_2 + postictal_delay 
            condition1 = start_seiz > start_seiz_2
            condition2 = start_bin < 0
            if (condition0 and condition1) or condition2:
                # The pre-ictal bin interval is not valid
                # Re-define it as None and break out of for-loop
                bin_intervals[bin_name_preictal] = None
                break 

    ### POST-ICTAL
    # If the post-ictal interval is valid, define it.
    # Post-ictal is valid iff that there is no second seizure right 
    # after the one in question. 
    end_postictal = end_seiz + postictal_delay
    bin_intervals[bin_name_postictal] = (end_seiz , end_postictal)
    # Check if invalid: redefine to None
    if end_postictal > total_session_time: 
        bin_intervals[bin_name_postictal] = None
    for start_seiz_2 in all_start_times:
        if end_postictal > start_seiz_2 and end_seiz < start_seiz_2:
            bin_intervals[bin_name_postictal] = None
    return bin_intervals


def _get_total_session_time_from_binary(
        binary_filepaths:list,
        fs=2000,
        n_chan_binary=41,
        precision="int16"
        ) -> float:
    """Determines the total session time from binary files."""
    fsize_bytes = os.path.getsize(binary_filepaths[0])
    # Check they are all exactly the same size, they must be
    for i in binary_filepaths[1:]: 
        assert fsize_bytes == os.path.getsize(i), "Corrupt data, Raw channel binaries mismatched filesize"
    bytes_per_sample = np.dtype(precision).itemsize
    n_samples_per_chan = fsize_bytes / bytes_per_sample / n_chan_binary
    assert n_samples_per_chan == int(n_samples_per_chan), "Logic error, possibly n_chan_binary is incorrect: this the number of channels saved in the binary file."
    total_duration_in_seconds = n_samples_per_chan / fs # total duration in seconds
    return total_duration_in_seconds


def _get_feats_df_column_names(
        features_list   : list,
        data_ops        : dict,
        ):
    """Calls the featurizing method on dummy segments, forwards columnames returned."""
    N_CHAN_RAW      = data_ops["N_CHAN_RAW"]
    N_CHAN_BINARY   = data_ops["N_CHAN_BINARY"]
    # Rem: session_basename identifies the recording, time_bin is the class label
    # Dummy segments random noise not cnst or div by zero error in coherence (psd=0)
    dummy_segments = np.random.normal(0,1,(N_CHAN_RAW,10000,N_CHAN_BINARY)) # 10000 samples is a 5s segment at 2000Hz
    dummy_features = sm_features.get_feats(dummy_segments,features_list,data_ops)
    colnames = ["session_basename","time_bin"] # first two colnames by default
    colnames += [k for k in dummy_features.keys()] # concatenate
    return colnames

def _get_match_basename_in_dir_paths(directory,basename):
    """Returns list of all full paths of files starting with b in basenames, in directory."""
    return [os.path.join(directory,i) for i in os.listdir(directory) if i[:len(basename)]==basename]


# Calculates all the features for one session and returns them in a 
# pandas dataframe
def calc_features(
        fio_ops : dict,
        data_ops : dict,
        feature_ops : dict,
        session_basename : str,
        ) -> pd.DataFrame:
    """Compute all features in all channels for multiple sessions.

    Several noteworthy assumptions are made: It is assumed that the
    binary data file and the metadata text file share the same basename.
    (Asside: metadata txt file is parsed by parse_metadata in metadata_io.py)
    
    Parameters
    ----------
    `session_basenames_list : list or str`
        If the list is empty, it will default to all valid edf files in the 
        directory. 

    Returns
    -------
    `dict`
        The keys are the bin names to which the features belong, and the
        values are pandas dataframes with columns { sessions_basename , 
        start_time , feat_01_name , feat_02_name , ... }
        
    `dict`
        Options that where used to generate the data. This could be a 
        serialized copy of the Options.toml file. 
    """
    # feature_functions = # TODO: this is a list of functions
    # features = # TODO: 2d np array shape = (n_segments,n_feats_per_segment)

    # Unpack fio_ops params
    RAW_DATA_PATH         = fio_ops["RAW_DATA_PATH"]
    WAVELET_BINARIES_PATH = fio_ops["WAVELET_BINARIES_PATH"]

    # Unpack data params
    SCALE_PHASE     = data_ops["SCALE_PHASE"]
    SCALE_POWER     = data_ops["SCALE_POWER"]
    FS              = data_ops["FS"]
    N_CHAN_RAW      = data_ops["N_CHAN_RAW"]
    N_CHAN_BINARY   = data_ops["N_CHAN_BINARY"]
    PRECISION       = data_ops["PRECISION"]
    # TODO: clean up amp and phase indexing conventions
    amp_idx         = data_ops["AMP_IDX"]
    AMP_IDX = np.array(amp_idx) * 2 + 1 
    ph_idx          = data_ops["PH_IDX"]
    PH_IDX  = np.array(ph_idx)  * 2 + 2 
 
    # Unpack relevant parameters, params.feature in Options.toml
    BIN_NAMES       = feature_ops["BIN_NAMES"]
    PREICTAL_BINS   = feature_ops["PREICTAL"]["BINS"] # in (negative) seconds
    POSTICTAL_DELAY = feature_ops["POSTICTAL"]["DELAY"] # in seconds
    PREICTAL_PCT    = feature_ops["PREICTAL"]["PCT"]
    INTRAICTAL_PCT  = feature_ops["INTRAICTAL"]["PCT"]
    POSTICTAL_PCT   = feature_ops["POSTICTAL"]["PCT"]
    PCT     = PREICTAL_PCT + [INTRAICTAL_PCT] + [POSTICTAL_PCT] # Concatenation
    PCT_DIC = {b:pct for b,pct in zip(BIN_NAMES,PCT)} # a handy pct dictionary 
    DUR_FEAT        = feature_ops["DUR_FEAT"]
    N_SAMPLES       = get_n_samples_from_dur_fs(DUR_FEAT,FS) # n samples per feature
    FEATURES        = feature_ops["FEATURES"]

    # Init Pandas DataFrame with right colnames
    colnames = _get_feats_df_column_names(FEATURES,data_ops)
    feats_df = pd.DataFrame({name:[] for name in colnames})

    # Retrieve seizure times from metadata
    session_metadata_path = os.path.join(RAW_DATA_PATH, session_basename+".txt")
    start_times,end_times = get_seizure_start_end_times(session_metadata_path)
    # All of the session binaries
    session_binary_paths = _get_match_basename_in_dir_paths(WAVELET_BINARIES_PATH,session_basename)
    # Get total session time, assumes all binary's exact same length
    total_session_time = _get_total_session_time_from_binary(
            session_binary_paths, 
            fs=FS,
            n_chan_binary=N_CHAN_BINARY,
            precision=PRECISION) # in secs

    # TODO: The following three nested for loops are potentially hard to
    # swallow, consider refactoring them. Q: is this necessary?

    # For each seizure in the session
    print(f"Computing features for session {session_basename}")
    for start_time,end_time in tqdm(list(zip(start_times,end_times))):
        # _get_bin_absolute_intervals() returns a dic with 
        # keys=BIN_NAMES,values=(strtbin,endbin) (in s). 
        # If the intervals of a bin are not valid, either because 
        # they start before or end after a file or because they 
        # overlap with another seizure: the value at bin_name 
        # is None.
        bins_absolute_intervals = _get_bin_absolute_intervals(
                start_seiz          = start_time,
                end_seiz            = end_time,
                preictal_bins       = PREICTAL_BINS, 
                postictal_delay     = POSTICTAL_DELAY,
                bin_names           = BIN_NAMES,
                all_start_times     = start_times,
                all_end_times       = end_times,
                total_session_time  = total_session_time
                )

        # For each time bin (=interval label) corresponding to this session
        for bin_name in BIN_NAMES:
            interval = bins_absolute_intervals[bin_name] # interval in seconds
            pct      = PCT_DIC[bin_name] # % of intervals to grab, float in (0,1]
            # If the interval is valid, get the segment start-times
            if interval:
                start_time,end_time = interval # unpack interval tuple
                # Get set of timestamps corresponding to start of segments
                segment_starts = _get_x_pct_time_of_interval(
                        start_time      = start_time,
                        end_time        = end_time,
                        segment_length   = DUR_FEAT,
                        pct             = pct
                        )
                # This holds the segments in this time-bin
                bin_segments = np.zeros((
                    N_CHAN_RAW,
                    len(segment_starts),
                    N_SAMPLES, 
                    N_CHAN_BINARY
                    ))
                for raw_ch_idx in range(N_CHAN_RAW):
                    # Is it dangerous to use such a strict file-naming convention?
                    # Yes, TODO: refactor all nameings of things that derive 
                    # from basenames to utility method
                    session_binary_chan_raw_path = utils.fmt_binary_chan_raw_path(
                            WAVELET_BINARIES_PATH,
                            session_basename,
                            raw_ch_idx
                            )
                    # Load all segments from specific time bin and raw chan
                    ws = load_binary_multiple_segments(
                            file_path       = session_binary_chan_raw_path,
                            n_chan          = N_CHAN_BINARY,
                            sample_rate     = FS,
                            offset_times    = segment_starts,
                            duration_time   = DUR_FEAT,
                            precision       = PRECISION
                            )
                    assert ws.shape == (len(segment_starts),N_SAMPLES,N_CHAN_BINARY)
                    bin_segments[raw_ch_idx,:,:,:] = ws

                # Get features for segment 
                for segment in bin_segments.transpose((1,0,2,3)):
                    # This a single segment, all channels, raw and wavelet/binary
                    assert segment.shape == (N_CHAN_RAW,N_SAMPLES,N_CHAN_BINARY) 
                    feats = sm_features.get_feats(segment,FEATURES,data_ops)
                    # Add the session name and time bin to the row dictionary
                    feats.update({
                        "session_basename"  : session_basename,
                        "time_bin"          : bin_name
                        })
                    # Add the row to our pandas dataframe
                    feats_df.loc[len(feats_df.index)] = feats
    return feats_df

def calc_features_all(options_path="Options.toml"):
    """Computes features csv for each binary file in the raw binary data directory."""
    # Unpack parameters and user-defined constants
    ops = load_ops_as_dict(options_path=options_path)
    data_ops = ops["params"]["data"]
    fio_ops = ops["fio"]
    feature_ops = ops["params"]["feature"]
    RAW_DATA_PATH = fio_ops["RAW_DATA_PATH"]
    FEATURES_PATH = fio_ops["FEATURES_PATH"]
    valid_basenames = get_all_valid_session_basenames(RAW_DATA_PATH)
    print(f"Computing features for {len(valid_basenames)} sessions...")
    for basename in valid_basenames:
        # Get the features of one session
        df = calc_features(fio_ops,data_ops,feature_ops,session_basename=basename)
        # Save the them
        csv_out_path = utils.fmt_features_df_csv_path(FEATURES_PATH,basename)
        df.to_csv(csv_out_path)
        # TODO: delete following line, above two lines replaced it
        # df.to_csv(os.path.join(FEATURES_PATH,f"{basename}.csv"))
    return         


if __name__=="__main__":
    import array
    ### UNIT TESTS ###

    ### TEST _get_x_pct_time_of_interval()
    arr = _get_x_pct_time_of_interval(
            start_time    = 5.0,
            end_time      = 152.6,
            segment_length = 1.0,
            pct           = 0.05)
    print("Test _get_x_pct_time_of_interval(), array returned:")
    print(arr)
    print()


    ### TEST _get_bin_absolute_intervals()  
    # Test _get_bin_absolute_intervals() 1 
    bin_abs_intervals = _get_bin_absolute_intervals(
            start_seiz           = 100,
            end_seiz             = 130,
            preictal_bins        = [50 , 25 , 10],
            postictal_delay      = 60,
            bin_names            = ["pre1","pre2","pre3","intra","post"],
            all_start_times      = [100,500,1000],
            all_end_times        = [130,550,1100],
            total_session_time = 2000
            )
    assert bin_abs_intervals["pre1"] == (50,75)
    assert bin_abs_intervals["pre2"] == (75,90)
    assert bin_abs_intervals["pre3"] == (90,100)
    assert bin_abs_intervals["intra"] == (100,130)
    assert bin_abs_intervals["post"] == (130,190)

    # Test _get_bin_absolute_intervals() 2 
    bin_abs_intervals = _get_bin_absolute_intervals(
            start_seiz           = 100,
            end_seiz             = 130,
            preictal_bins        = [50 , 25 , 10],
            postictal_delay      = 10,
            bin_names            = ["pre1","pre2","pre3","intra","post"],
            all_start_times      = [50,100,135],
            all_end_times        = [60,130,170],
            total_session_time = 2000 
            )
    assert bin_abs_intervals["pre1"] == None    # Overlaps with previous post-ictal
    assert bin_abs_intervals["pre2"] == (75,90)
    assert bin_abs_intervals["pre3"] == (90,100)
    assert bin_abs_intervals["intra"] == (100,130)
    assert bin_abs_intervals["post"] == None        # Overlaps with next seizure

    # Test _get_bin_absolute_intervals() 3
    bin_abs_intervals = _get_bin_absolute_intervals(
            start_seiz           = 15,
            end_seiz             = 100,
            preictal_bins        = [50 , 25 , 10],
            postictal_delay      = 60,
            bin_names            = ["pre1","pre2","pre3","intra","post"],
            all_start_times      = [15],
            all_end_times        = [100],
            total_session_time = 150 
            )
    assert bin_abs_intervals["pre1"] == None        # Before file start
    assert bin_abs_intervals["pre2"] == None        # Before file start
    assert bin_abs_intervals["pre3"] == (5,15)      # Valid
    assert bin_abs_intervals["intra"] == (15,100)   # Valid
    assert bin_abs_intervals["post"] == None        # Ends after end of file
    # Not every single edge-case is tested... (low priority TODO)

    ### TEST _get_total_session_time()
    # Generate array, serialize it, then measure it's length
    arr = array.array("h", np.arange(256)) # highly divisible by two test array
    with open("temp_test.dat","wb") as file:
        arr.tofile(file)
    fs,n_chan_binary = 2,4
    total_duration_in_seconds = _get_total_session_time_from_binary(
        binary_filepaths = ["./temp_test.dat"],
        fs=fs,
        n_chan_binary=n_chan_binary,
        precision="int16"
        )
    os.remove("temp_test.dat") # delete the test file
    try: 
        assert total_duration_in_seconds == len(arr) // fs // n_chan_binary
        print("Passed Test _get_total_session_time()")
    except: 
        print("Failed Test _get_total_session_time()")

    ### TEST _get_feats_df_column_names(), and sm_features.get_feats() implicitly
    features = ["mean_power","coherence"]
    data_ops = {"FS":2000,"NUM_FREQ":20,"LOW_FREQ":0.5,"HIGH_FREQ":200,
            "SPACING":"LOGARITHMIC","ZSCORE_POWER":True,"SCALE_PHASE":1000,
            "SCALE_POWER":1000,"N_CHAN_RAW":4,"CH_PHASE_AMP":2,
            "TS_FEATURES":["WAVELET_POWER","WAVELET_PHASE"],
            "N_CHAN_BINARY":3,"PRECISION":"int16",
            "AMP_IDX":[],#[14,15,16,17,18,19],
            "PH_IDX":[0],#[0,1,2,3,4,5,6,7,8,9],
            "AMP_FREQ_IDX_ALL":[1],#,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],
            "PH_FREQ_IDX_ALL":[2]}#,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]}
    colnames = _get_feats_df_column_names(features,data_ops)
    print("\nTest, first eight column names are:")
    for i in colnames[:8]: print(i)

    ### TEST calc_features
    fio_ops = {"RAW_DATA_PATH":"/Users/steve/Documents/code/unm/data/h24_data/raw",
            "WAVELET_BINARIES_PATH":"/Users/steve/Documents/code/unm/data/h24_data/binaries"}
    # data_ops, use same dict as in above sm_features.get_feats() test, 
    # but update it to include all amplitude and phase indices
    data_ops["N_CHAN_BINARY"] = 41
    data_ops["AMP_IDX"] = [14,15,16,17,18,19]
    data_ops["PH_IDX"] = [0,1,2,3,4,5,6,7,8,9]
    data_ops["AMP_FREQ_IDX_ALL"] = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]
    data_ops["PH_FREQ_IDX_ALL"] = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
    feature_ops = {"DUR_FEAT":5,"N_PREICTAL_BINS":4,
            "PREICTAL":{"BINS":[10800,3600,600,10],"PCT":[0.05,0.05,0.2,1.0]},
            "INTRAICTAL":{"PCT":1.0},
            "POSTICTAL":{"DELAY":600,"PCT":0.2},
            "BIN_NAMES":["pre1","pre2","pre3",
                "pre4","intra","post"],
            "FEATURES":["mean_power","var","coherence"]}
    session_basename = "AC75a-5 DOB 072519_TS_2020-03-23_17_30_04"
    feats_df = calc_features(fio_ops,data_ops,feature_ops,session_basename)
    # serialize feat dataframe to look at it in jupyter notebook
    print("Writing to CSV 'feats_df.csv'")
    feats_df.to_csv("feats_df.csv")
    # saveas = "feats_df.pkl"
    # print(f"Pickling features {saveas}")
    # feats_df.to_pickle(saveas)
    print("Passed test calc_features()")

    print("\nTests All Passed: _get_bin_absolute_intervals()")






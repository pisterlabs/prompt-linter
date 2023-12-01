"""
In : This module expects a json file exported from Trinity's mongodb edl collection.

Methods:
- Extract data from mongodb export.
- Transform extracted data into desired format
- Load/export transformed data into format of preference.

Out : A dataframe indexed by (experimental_group, subject_id, session_no, trial_no)
with flattened columns for each relevant metric.

Use :
- From an analysis file in the same directory use 'from clean import etl'
  then call 'df = etl(path_to_json)'.
- From the command line, start virtual envirnoment and run 'python clean.py'

Comments :
- For exploring and understanding the structure of the json file, use 'explore.py'.
- Use 'tinker.ipynb' to test small modifications and manipulations.
- Create a new branch on git before modifying this file.
"""

################################################################################
################ Imports #######################################################
################################################################################
import pandas as pd # For manipulating tabular structures in data
import itertools as it # For manipulating lists when building mutli_index
import toolz.functoolz as ftz
from toolz.curried import map, curry
import numpy as np
from scipy.special import erf





################################################################################
################ Extraction Methods ############################################
################################################################################

def target_records(frame):
    """
    In: raw dataframe read in by pd.read_json()
    Methods:
    - iterate over session dataframe
    - Restrict to records with 41 trials
    - Identify treatment_group for each id
    - Restrict to trials listing a player_tag (excludes empty trial_no == 0)
    - Set outputs
    Out:
    - records is a list of dictionaries with the same fields the db has
    - groups_by_id is a dictionary in the form subject_id: experimental_group
    - sessions is a set of all session_no encountered
    - trials is a set of all trial_no encountered
    Use: Call `records, groups_by_id, sessions, trials = target_records(frame)'
    """
    # Initialize outputs
    records = []
    groups_by_id = { 3:'na_2D', 
                     4:'na_vr', 
                     5:'2U1D_locked',
                     7:'na_vr',
                     8:'na_2D',
                     9:'na_2D',
                     10:'na_vr', 
                     11:'2U1D_locked', 
                     12:'2U1D_locked', 
                     13:'na_2D', 
                     14:'na_vr', 
                     15:'2U1D_locked', 
                     16:'na_vr', 
                     17:'na_2D', 
                     18:'2U1D_locked', 
                     19:'na_vr', 
                     20:'na_vr', 
                     21:'na_vr',
                     23:'2U1D_locked', 
                     24:'na_2D',
                     26:'na_2D', 
                     27:'2U1D_locked', 
                     28:'2U1D_unlocked', 
                     29:'2U1D_unlocked', 
                     30:'1U1D', 
                     31:'2U1D_unlocked', 
                     32:'1U1D', 
                     33:'2U1D_locked', 
                     34:'2U1D_unlocked',
                     38:'2U1D_unlocked', 
                     39:'1U1D', 
                     40:'1U1D', 
                     41:'2U1D_unlocked', 
                     42:'1U1D',
                     44:'2U1D_unlocked', 
                     45:'1U1D',
                     47:'2U1D_unlocked', 
                     48:'1U1D',
                     50:'1U1D',
                     54:'MFP',
                     55:'MFP',
                     56:'MFP',
                     57:'MFP',
                     58:'MFP',
                     59:'na_2D',
                     60:'MFP',
                     61:'MFP',
                     62:'MFP'                                         
                    }
    session_nos = [1,2,3,4]
    trial_nos =  [1,2,3,4,5,6,7,8,9,10]

    # Iterate over frame
    for _id,sessions in zip(frame['_id'],frame['sessions']): # For each subject retrieve list of sessions
        # Sessions is a list of len 1  containing a dict with key 'trials'
        
        # Check that id is supposed to be included
        if int(_id) not in groups_by_id:
            print(f"Detected unknown ID in data: {_id}.  Excluding subject...")
            continue
        # Check consistency for subject's treatment group
        elif treatment_group(sessions[0]['trials']) != groups_by_id[int(_id)]:
            print(f'Detected unusual treatment condition for id: {trial["player_tag"]}. Excluding subject...')
            continue

        for trial in sessions[0]['trials']: # For each trial
            if 'player_tag' in trial: # Exclude trial 0, which has no data and no reference to player_tag
                
                # Triple check player_tag is expected
                if int(trial['player_tag']) in groups_by_id:
                    # Set output variables

                    # Gets reset many times for each id.  Will set group
                    # To all ids in this list of trials.  Should be safe
                    # Because function treatment_group verifies that
                    # All trials in the list have the same player_tag
                        
                    # Ensure sessionNumber expected
                    if int(trial['sessionNumber']) not in session_nos:
                        print(f"Detected unknown session number: {trial['sessionNumber']}. Excluding trial...")
                        continue
                    # Ensure trial number expected
                    if trial['trial_no'] not in trial_nos:
                        print(f"Detected unknown trial number: {trial['trial_no']}, id: {trial['player_tag']}.  Will adjust at the end...")
                    records.append(trial) # Add trial dictionary to the list of records.
                else :
                    print(f"Detected unexpected player tag: {trial['player_tag']}. Excluding trial...")
                        
                        

    # Return output
    return groups_by_id, session_nos, trial_nos, records


# Switch for determining which treatment group each subject belongs to
def treatment_group(trials):
    """
    In : trials is raw list of dictionaries containing trial data from the db
    Methods :
    - iterate over trials and detect treatment group by examining 'environment'
      and 'paradigm' fields
    - check that the apparent treatment group is consistent for all trials
    Out : String listing treatment group
    - '2D_control' means environment 2 (2D screen) appears
    - 'non_adaptive_vr' means environment 1 (vr) and paradigm 0 (non-adaptive)
    - 'adaptive_vr' means environment 1 (vr) and paradigm 1 (adaptive)
    Use : Call 'group = treatment_group(trials)'
    """

    # Checks before assigning new group that old and new are identical
    def assign_group(current_group, proposed_group):
        if current_group == None: # Group hasn't been assigned yet.
            return proposed_group # Assign proposed_group
        else: # Group has been assigned
            if current_group == proposed_group: # Check new assignment is eq
                return current_group # Return same
            else: # Proposed new assignment not the same
                raise ValueError # Raise an error


    group = None # Initialize output

    for trial in trials: # Iterate over trials
        if 'environment' in trial: # Excludes empty trials
            if trial['environment']==0: # If this trial is in the physical mockup
                continue # We can't tell which group this trial came from
            elif trial['environment']==2 : # If subject in control group
                group = assign_group(group, 'na_2D') # Consistency  Formerly 2D_control
            elif trial['environment']==1: # If subject in experimental group
                if trial['paradigm']==0 : # If subject in non-adaptive VR group
                    group = assign_group(group, "na_vr") # consistency Formerly non_adaptive_vr
                elif trial['paradigm']==1: # If subject in adaptive VR group
                    group = assign_group(group, "2U1D_locked") # consistency Formerly adaptive_vr
                elif trial['paradigm']==2: # If subject in 2 up 1 down unlocked
                    group = assign_group(group, "2U1D_unlocked")
                elif trial['paradigm']==3: # If subject in 1 up 1 down
                    group = assign_group(group, "1U1D")
                elif trial['paradigm']==5: # If subject in median fixed progression
                    group = assign_group(group, "MFP")                
                else : # Encountered unknown paradigm
                    print(f"Found unknown paradigm: {trial['paradigm']}")
                    raise ValueError # Raise an error to stop the program
            else: # Encountered an unknown environment
                print(f"Found unknown environment: {trial['environment']}")
                raise ValueError # Raise an error to stop the program

    # Return the group
    return group


def get_multi_index():
    """
    Methods:
    - Produces an empty index with levels for group, id, session, trial
    Out : A pandas MultiIndex ready to contain data.
    """
    return pd.MultiIndex.from_arrays([[],[],[],[]], names=("group", "id","session","trial"))



def build_dataframe(idx, cols, ids, records):
    """
    In :
    - idx is the target MultiIndex in the form ('group', 'id', 'session', 'trial')
    - cols is the list of column names to use for the df
    - ids is a dictionary in the form subject_id : experimental_group
    - records is a list of dictionaries corresponding to data for each trial
    Methods :
    - Iterate of records
    - Insert data from each record at the appropriate location in the index.
    - Assure that assignment of columns occurs in the correct order.
    Out : Dataframe indexed ('group', 'id', 'session', 'trial') with cols
          corresponding to each data point in db.
    """

    def idx_iter(start, stop):
        """
        Helper function for creating an iterator that returns all the intended multiindex indices
        between start (inclusive) stop (exclusive)
        In :
        - start is a tuple in the form (group, id, session, trial).  
        - stop is a tuple in the form (group, id, session, trial). 
        Out :
        - iterator of every index between start and stop (not included).
        Comments:
        - Example: ("1U1D",23,2,1), ("1U1D", 23, 2, 2), ... ("1U1D", 23, 2, 10), ("1U1D", 23, 3, 1)
        - Handles sessions and trials.  Never changes group or id.
        """
        groups = it.repeat(start[0]) # Same group over and over endlessly
        ids = it.repeat(start[1]) # Same id over and over endlessly
        repeat10 = curry(it.repeat)(times=10) # Build a function that repeats the argument 10 times
        sessions = ftz.pipe(it.count(1), map(repeat10), it.chain.from_iterable) # Repeats every natural number 10 times infinitely
        trials = it.cycle( np.arange(1,11) ) # Repeats the first ten numbers infinitely
        drop_until_start = curry(it.dropwhile)(lambda idx : not idx == start) # Function for starting idx start
        take_until_stop = curry(it.takewhile)(lambda idx : not idx == stop) # Function for ending at idx stop
        return ftz.pipe(zip(groups,ids,sessions,trials), drop_until_start, take_until_stop) # Isolate terms between start / stop

    df = pd.DataFrame(index=idx, columns=cols, dtype=object) # Initialze df with target idx; index=idx, 
    
    id_with_error = None
    for record in records: # Iterate over records
        # Assign record to corresponding location in the index.
        # Note the following line will generate a visible deprecation warning
        # Because fix_ordering gives a list containing lists of varying length.
        id = int(record['player_tag'])
        group = ids[id]
        session = record['sessionNumber']
        trial = record['trial_no']
        try :
            if any(pd.notnull(df.loc[(group,id,session,trial)])): # The simulator reset and started counting trial nos over
                if id_with_error != id:
                    print(f"Detected simulator reset.  id:{id}.  Will remove extra trials if necessary...")
                    id_with_error = id
                
                # Detect the max trial_no so far
                last_trial = df.dropna().xs(level=('id','session'), key=(id,session)).reset_index(level='trial')['trial'].max()
                trial = last_trial + 1
                
        except KeyError:
            pass
        df.loc[( group, id, session, trial)]  = record
    df = df.sort_index()

    trials_above_11 = df.loc[(slice(None),slice(None),slice(None),slice(11,None))].index
    ids_above_11 = trials_above_11.unique(level="id")
    for id in ids_above_11:
        df = df.sort_index()
        trials = df.loc[(slice(None),id,slice(1,3),slice(None))]
        old_idx = trials.index
        start = old_idx[0]
        stop = (start[0],start[1],4,1) # not included
        new_idx = idx_iter(start,stop)
        for old,new in zip(old_idx, new_idx):
            df.loc[new] = trials.loc[old]
    df.drop(trials_above_11, inplace=True)

    if df.isna().any(axis=None): # If any value in df has not yet been set
        print('\n\n\nDataFrame contains NA\n--------')
        print('\n\nColumns containing NAs')
        print(df.isna().any(axis=0)) #Print which cols contain nas
        print('\n\nIDs containing NAs')
        print(df.isna().any(axis=1).groupby(level="id").describe()) #Which id has the NAs
    return  df # Return the new df with sorted index and no NaNs.

def extract_data(json_filepath):
    """
    Pull together all the functions in this module to produce a df
    In : Path to a json file exported by mongodb from the EDL collection
    Methds :
    - Read in the file
    - Extract records of interests
    - Build a multi_index from the records
    - Build the dataframe.
    Out : Dataframe indexed ('group', 'id', 'session', 'trial') with cols
          corresponding to each data point in db.
    """
    raw = pd.read_json(json_filepath) # Read in data

    groups_by_id, sessions, trials, records = target_records(raw) # Extract records.
    multi_index = get_multi_index() # Make multi_index

    # Specify columns for the df with keys from the first record.
    # All records should have the same columns. Explicitly checked by
    # fix_ordering().
    cols = list(records[0].keys()) # Initialize with keys from first record.

    # Initialize dataframe
    return build_dataframe(multi_index, cols, groups_by_id, records)








################################################################################
################ Transformation Methods ########################################
################################################################################




def unpack_col(df, col, names):
    """
    This function modifies df inplace.
    In :
    - df is a DataFrame to modify.
    - col is the name of column containing a list of items to be unpacked.
    - names is an ordered tuple of new column names to use for each val unpacked.
    Methods :
    - Iterate over names
    - Extract one value for each name
    - Calculate new column of df to contain value for each df row.
    Out : Null
    """

    def extract_val(col, idx):
        """
        Helper that makes a function for use in df.apply().  Useful for
        factoring out like code and iterating over columns.
        In :
        - col is the name of a column containing a list to extract from.
        - idx is the index of the value to be extracted.
        Out :
        - function of form (dataframe_row) => (val at col[idx])
        """
        def func(row):
            try:
                return row[col][idx]
            except TypeError: # Occurs if this row[col] is np.NaN instead of a list
                return np.NaN
        return func

    # Iterate over names and the corresponding index of each name.
    for name, idx in zip(names,range(len(names))):
        # Assign new column called name to extracted val from col.
        # Axis=1 specifies that the operation is performed on every row.
        df[name] = df.apply(extract_val(col,idx), axis=1)

def drop_cols(df, cols_to_drop):
    """
    This function modifies df inplace.
    In :
    - df is a DataFrame to modify
    - cols_to_drop is a list of column names to drop.
    Methods :
    - Iterate over cols_to_drop
    - Drop each corresponding column from df.
    Out : Null
    """
    for col in cols_to_drop: # For each requested drop.
        # axis="columns" means drop a column
        # inplace means modify df instead of copying.
        # note drop will raise error if col not in df.
        df.drop(col, axis="columns", inplace=True )

def rel_error(actual, approx):
    """
    Calculates the relative error between actual and approx.
    In :
    - actual is a pandas Series of numericals. Assumed no NaN or zeros
    - approx is a pandas Series of numericals.  Assumed no NaN
    Methods :
    - subtract actual from approx and divide by actual
    Out :
    - pandas Series of relative errors
    """
    return (approx - actual) / actual

def linear_adjust(skill_col, excellent, avg_adequate):
    """
    Performs linear transformation of skill_col such that erf(excellent) = 0.95 and erf(avg_adequate) = 0.0
    Note erf(0.8135) = 0.75 and erf(0) = 0.
    In :
    - skill_col is pandas Series of skill values to scale
    - excellent is the skill value that should be mapped to 0.75
    - avg_adequate is the skill value that should be mapped to 0.
    Methods :
    - Construct a line passing through (excellent, 0.8135) and (avg_adequate, 0).
    - Apply the transformation
    Out :
    - Pandas Series transformed by the line
    """
    slope = (0.8135 - 0) / (excellent - avg_adequate)
    return slope * (skill_col - avg_adequate)

def difficulty_control(skill_col, difficulty_col):
    """
    Adjusts the skill column (after erf) by the difficulty of each trial
    In :
    - skill_col is pandas Series of skill values post erf apply
    - difficulty col is a pandas Series of natural numbers 1 : 24 representing diff of each trial
    Methods :
    - Construct a line passing through (24, 1) and (0, 0)
    - Apply this transformation to difficulty_col and multiply elementwise by skill_col
    - Do not scale negative values, only positive ones.
    """
    return np.where(skill_col >= 0, skill_col * (1 / 24) * difficulty_col, skill_col)

def compute_skill(metric_col, crash_col, excellent, avg_adequate):
    """
    Factors out general procedures for computing the skill column
    In :
    - Metric_col is pandas Series of target raw metric to adjust
    - difficulty_col is pandas Series of difficulty levels at each trial 1:25
    - crash_col is 0's and 1's indicating whether a crash occurred this trial
    - excellent is the threshold to use for when val from metric_col is excellent performance
    - avg_adequate is threshold to use for when valu from metric_col is average adequate performance
    Methods:
    - Linear adjustment to match excellent threshold -> 0.75 and average adequate -> 0.0.
    - Take the error function to map to [-1,1].  Linear in center, compresses edges
    - Send crashes to a skill of -1
    Out:
    - pandas Series of normalized skill metric.
    """

    skill = linear_adjust(metric_col, excellent, avg_adequate) # Set val of excellent to 0.8135, avg_adequate to 0
    skill = (1/2)*(1 + erf(skill)) # Compress outliers to lie on -1,1 then linear shifts to [0,1].  Excellent threshold is 0.75, and avg_adequate is 0
    skill = np.where(crash_col, 0, skill) # Set crashes to a value of -1
    return skill # Return the new column

def ls_skill(ls_subject, ls_cpu, ls_crash):
    """
    Interprets raw performance data for ls task as a continuous performance
    value on [-1,1]
    In :
    - ls_subject is pandas series of sum distances to science sites from subject's selected landing site
    - ls_cpu is pandas series of sum distances to science sites from cpu's selected best landing site
    - ls_crash is pandas series of 0's and 1's indicating whether the subject's selected site is too steep.
    - ls_difficulty is a pandas series of natural numbers from 1 : 25 indicating difficulty level of trial
    Methods :
    - Filter out ls_cpu values of zero
    - Calculate relative error between subject and cpu distances
    - Compute cases where a crash or other catastrophic failure occurred
    - Compute normalized skill column
    Out :
    - pandas Series of normalized skill values
    Comments :
    - The ls_cpu values are sometimes zero. We use NaN at these points.
    """

    cpu = ls_cpu.where(ls_cpu!=0,np.nan) # Restrict cases where cpu value is nonzero
    subject = ls_subject.where(ls_subject<9999, np.nan) # val 10000 means failure to select
    rel = rel_error(cpu, subject) # Relative error of remaining values
    excellent = 0.1 # Within 10% of optimal site
    avg_adequate = 0.3 # Within 30% of optimal site
    # Mark a crash if terrain steep or if subject didn't select a site in time.
    crash = np.logical_or(ls_crash, subject.isna()) 

    # Be sure to remove cases where cpu is na.  This comes up if subject > 9999 and cpu = 0.
    return np.where(cpu.isna(), np.nan, compute_skill(rel, crash, excellent, avg_adequate))

def mc_skill(mc_rms,  mc_crash, mc_difficulty):
    """
    Interprets raw performance data for mc task as a continuous performance
    value on [-1,1]
    In :
    - mc_rms is pandas series of root mean squared error deviations from guidance cue
    - mc_difficulty is pandas series of difficulty level 1 : 25 for mc task
    - mc_crash is pandas series of 0's and 1's indicating whether subject ran out of fuel
    Methods :
    - Calculate thresholds for excellent and avg adequate performance based on difficulty
    - Compute normalized skill metric
    Out :
    - pandas series of skill levels
    """
    def mc_thresholds(mc_difficulty):
        """
        Calculates thresholds for excellent and avg_adequate.  Uses formula from Unity code.
        In :
        - mc_difficulty is pandas series of difficulty levels for mc task
        Methods :
        - mod sends 9 or less to 0, 12 or less to 1, 15 or less to 2, ...
        - Increment "center" for rms calculation by 1/2 per mod.  
        Out :
        - pandas Series of thresholds to use for excellent
        - pandas Series of thresholds to use for avg_adequate
        Comments :
        - rms higher with higher wind
        - This adjustment accounts for difficulty in rms metric
        """
        # np.where is a vectorized if statement
        mod = np.where( mc_difficulty <= 9, 0, np.ceil( (mc_difficulty - 9) / 3) )
        # linear function of mod, copied from Unity code
        excellents = 0.5 * mod + 3
        avg_adequates = 0.5 * mod + 3.5
        return excellents, avg_adequates

    excellent, avg_adequate = mc_thresholds(mc_difficulty)
    return compute_skill(mc_rms, mc_crash, excellent, avg_adequate)

def de_skill(de_speed, de_crash):
    """
    Calculates skill for descent engine task on continuous metric [-1, 1]
    In :
    - de_speed is pandas Series of vertical speeds at landing
    - de_crash is 0's / 1's indicating whether subject ran out of fuel
    Methods :
    - Compute normalized skill metric for the de task
    Out :
    - pandas series of skill levels
    """
    excellent = 121 # Speed below which subject was excellent
    avg_adequate =  190.5
    return compute_skill(de_speed, de_crash, excellent, avg_adequate)

def transform_data(df, unpack=True, drop=True):
    """
    For documentation on data annotation implemented here, see google doc
    titled: 'Data Annotation'.  Request access if needed.

    In : Dataframe indexed ('group', 'id', 'session', 'trial') with cols
         corresponding to each data point in db.

    Methods : Modify df inplace instead of copying.
    - Unpack lists in aggregated columns.  Create a column for each list elem.
    - Drop unneeded columns.

    Out : Dataframe indexed ('group', 'id', 'session', 'trial') with cols
         corresponding to each data point of interest for analysis.
    """


    if (unpack):
        # Unpack difficulty_level column.
        # Check ordering of levels.  Unclear on what MC stands for.
        unpack_col(df,
                'difficulty_level',
                ('mc_difficulty', 'ls_difficulty', 'de_difficulty'))

        # Unpack performance column
        # Check ordering of performance.  Unclear on what MC stands for.
        unpack_col(df,
                'performance',
                ('mc_performance', 'ls_performance', 'de_performance'))

        # Unpack rawPerformance column
        # Check ordering of rawPerformance.  Unclear on what MC, LG stands for.
        unpack_col(df,
                'rawPerformance',
                ('mc_rms','ls_subject',
                    'ls_cpu', 'de_speed',
                    'de_fuel', 'mc_fuel', "mc_crash",
                    'ls_crash','de_crash'))

        # Unpack next_level column
        # Check ordering of next_level.  Unclear on what MC stands for.
        unpack_col(df,
                'next_level',
                ('mc_next_level', 'ls_next_level', 'de_next_level'))

    # Calculate skill without difficulty control
    df['ls_raw_skill'] = ls_skill(df['ls_subject'], df['ls_cpu'], df['ls_crash'])
    df['mc_raw_skill'] = mc_skill(df['mc_rms'],  df['mc_crash'], df['mc_difficulty'])
    df['de_raw_skill'] = de_skill(df['de_speed'], df['de_crash'])

    # Detect cases where subject failed mc and thus could not be scored on ls or de
    df['ls_raw_skill'] = df['ls_raw_skill'].where(df['ls_cpu']!= 0, np.NaN )
    df['de_raw_skill'] = df['de_raw_skill'].where(df['ls_cpu']!= 0, np.NaN)

    # Adding difficulty control to a new column
    df['ls_skill'] = difficulty_control(df['ls_raw_skill'], df['ls_difficulty'])
    df['mc_skill'] = difficulty_control(df['mc_raw_skill'], df['mc_difficulty'])
    df['de_skill'] = difficulty_control(df['de_raw_skill'], df['de_difficulty'])

    if (drop):
        # Drop unneeded columns
        drop_cols(df,
                ('environment','paradigm', 'difficulty_level', 'performance',
                'player_tag', 'sessionNumber', 'rawPerformance', 'bedford',
                'trial_no', 'next_level', 'SART', 'lockstep'))

    

    return df







################################################################################
################ Transformation Methods ########################################
################################################################################


def load_data(df, export=False):
    """
    Exports the data frame to a csv suitable for importing into R
    """
    if export:
        df.reset_index().to_csv(path_or_buf="cleaned.csv", index=False)
    return df







################################################################################
################ Main method ###################################################
################################################################################

def etl(json_filepath):
    """
    Wrapper for clean imports.
    """
    return load_data(transform_data(extract_data(json_filepath)))

if __name__ == '__main__': # When called as a script from the command line.
    load_data(transform_data(extract_data('MATRIKS-EDL.json')),export=True)

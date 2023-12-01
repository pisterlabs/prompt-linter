"""

Notes: context = 1 for motion task, -1 for color task
"""

from glob import glob
import scipy.ndimage
from scipy.io import loadmat
import numpy as np
import itertools
import pandas as pd
import multiprocessing as mp
from itertools import repeat

dir_monkeyA = '../../../data/mante/PFC data 1/'
dir_monkeyF = '../../../data/mante/PFC data 2/'
dir_monkeyG = '../../../data/mante/PFC data 3/'


class RawManteUnit:
    """
    Each unit is characterized by:
    - response : tensor (shape ntrials x ntimesteps),
    - trials : pandas DataFrame where each row corresponds to a trial, containing detailed information about each trial.
    - time : a numpy array of shape #timesteps containing the equivalent in ms of each timestep
    - bin_width : the time step in ms
    """

    def __init__(self, response, trials, time):
        self.response = response
        self.trials = trials
        self.time = time
        self.bin_width = self.time[1] - self.time[0]

    @classmethod
    def from_mat(cls, filename):
        data = loadmat(filename)
        response = data['unit'][0][0][0]
        trials = pd.DataFrame({'stim_dir': data['unit'][0][0][1][0][0][0].ravel(),
                  'stim_col': data['unit'][0][0][1][0][0][1].ravel(),
                  'targ_dir': data['unit'][0][0][1][0][0][2].ravel(),
                  'targ_col': data['unit'][0][0][1][0][0][3].ravel(),
                  'context': data['unit'][0][0][1][0][0][4].ravel(),
                  'correct': data['unit'][0][0][1][0][0][5].ravel(),
                  'congruent': data['unit'][0][0][1][0][0][6].ravel(),
                  'stim_dir2col': data['unit'][0][0][1][0][0][7].ravel(),
                  'stim_col2dir': data['unit'][0][0][1][0][0][8].ravel(),
                  'stim_trial': data['unit'][0][0][1][0][0][9].ravel()})
        time = data['unit'][0][0][2][0] * 1000
        return RawManteUnit(response, trials, time)

    def resample(self, target_bin, ignore_extra=True):
        assert target_bin % self.bin_width == 0, \
            'target_bin must be an integer multiple of bin_width.'
        resample_factor = int(round(target_bin / self.bin_width))
        arr = self.response[:, :-(self.response.shape[1] % resample_factor)]
        if self.response.shape[1] % resample_factor != 0 and not ignore_extra:
            extra_sum = self.response[:, -(self.response.shape[1] % resample_factor):].sum(axis=-1)
        else:
            extra_sum = None
        arr = arr.reshape((arr.shape[0], arr.shape[1] // resample_factor, resample_factor)).sum(axis=-1).astype('float32')
        if extra_sum is not None:
            arr = np.hstack([arr, extra_sum.reshape((-1, 1))])
        arr = arr / 50 * 1000
        self.response = arr
        self.time = self.time[::resample_factor]
        self.bin_width = self.time[1] - self.time[0]

    def smooth(self, width):
        bin_std = width / self.bin_width
        # win_len = int(6 * bin_std)
        # window = signal.gaussian(win_len, bin_std, sym=True)
        # window /= np.sum(window)
        new_response = np.zeros_like(self.response, dtype='float')
        for i in range(self.response.shape[0]):
        #     new_response[i] = np.convolve(self.response[i], window, mode='same')
            new_response[i] = scipy.ndimage.gaussian_filter1d(self.response[i], bin_std)
        self.response = new_response

    def check_consistency(self):
        correct_trials = self.trials['correct'].astype('bool')
        choices_inferred = np.zeros_like(self.trials['stim_dir'])
        mask = correct_trials & (self.trials['context'] == 1)
        choices_inferred[mask] = np.sign(self.trials.loc[mask, 'stim_dir'])
        mask = correct_trials & (self.trials['context'] == -1)
        choices_inferred[mask] = np.sign(self.trials.loc[mask, 'stim_col2dir'])
        mask = ~correct_trials & (self.trials['context'] == 1)
        choices_inferred[mask] = -np.sign(self.trials.loc[mask, 'stim_dir'])
        mask = ~correct_trials & (self.trials['context'] == -1)
        choices_inferred[mask] = -np.sign(self.trials.loc[mask, 'stim_col2dir'])
        assert np.all(choices_inferred == self.trials['targ_dir'])
        self.trials['choices_inferred'] = choices_inferred


def _task_coh_levels(data):
    """
    for parallelization. Transform the columns containing non-uniform coherence values to a set of 6 uniformized
    coherence levels
    """
    stim_dirs = np.sort(np.unique(data.trials['stim_dir'])).tolist()
    dir_lvls = {stim_dirs[i]: i for i in range(len(stim_dirs))}
    data.trials['stim_dir_lvl'] = np.vectorize(dir_lvls.get)(data.trials['stim_dir'])

    stim_cols = np.sort(np.unique(data.trials['stim_col2dir'])).tolist()
    col_lvls = {stim_cols[i]: i for i in range(len(stim_cols))}
    data.trials['stim_col_lvl'] = np.vectorize(col_lvls.get)(data.trials['stim_col2dir'])
    return data


def _task_avg_cond(data, conditions, variables, correct_only):
    """
    for parallelization. Generate condition averaged-trajectories for one unit for an ndarray of conditions
    corresponding to a list of variables.
    :param data: ManteDataset
    :param conditions: numpy array, shape #conditions x #variables, containing the specification of each condition
    :param variables: list of strings (names of the columns corresponding to "conditions"
    :param correct_only: bool
    :return: - data_avg : numpy array of shape #conditions x #timesteps
             - ntrials : numpy array of shape #conditions
    """
    data_avg = np.zeros((len(conditions), data.response.shape[1]))
    ntrials = np.zeros(len(conditions))
    for j, cond in enumerate(conditions):
        mask = np.ones_like(data.trials['stim_dir']).astype('bool')
        for k, var in enumerate(variables):
            mask = mask & (data.trials[var] == cond[k])
        if correct_only:
            mask = mask & (data.trials['correct'] == 1)
        mask = mask.ravel()
        if np.sum(mask) > 0:
            data_avg[j] = np.mean(data.response[mask, :], axis=0)
        else:
            data_avg[j] = np.nan
        ntrials[j] = np.sum(mask)
    return data_avg, ntrials


class ManteDataset:
    """
    A dataset is characterized by:
    - units : a list of RawManteUnits
    - conditions: None or a pandas DataFrame containing the variables values for each condition of the condition-avg
    tensor
    - data_avg : None or a tensor of shape #conditions x #timesteps x #neurons
    - ntime : the #timesteps
    - bin_width
    - smoothing_width
    """

    def __init__(self, monkey='A', bin_width=None, smoothing_width=None, cavg=False, correct_only=False):
        self.bin_width = bin_width
        self.smoothing_width = smoothing_width
        self.units = []

        print(f'Loading data for monkey {monkey}')
        if monkey == 'F':
            datadir = dir_monkeyF
        elif monkey == 'A':
            datadir = dir_monkeyA
        else:
            datadir = dir_monkeyG

        files = glob(datadir + '*.mat')
        for file in files:
            data = RawManteUnit.from_mat(file)
            if bin_width is not None:
                data.resample(bin_width)
            if smoothing_width is not None:
                data.smooth(smoothing_width)
            self.units.append(data)
        print(f"loaded {len(self.units)} units, binned at {(self.units[0].time[1] - self.units[0].time[0])}ms")
        Ts = [unit.response.shape[1] for unit in self.units]
        assert all(t == Ts[0] for t in Ts)
        self.ntime = Ts[0]

        if cavg:
            self.condition_average_wrapper(correct_only)

    def condition_average_wrapper(self, correct_only=False):
        # Prepare condition averaging: replace exact coherence values by coherence levels (with parallelization)
        pool = mp.Pool(mp.cpu_count())
        self.units = pool.map(_task_coh_levels, self.units)

        if correct_only:
            self.condition_average(['stim_dir_lvl', 'stim_col_lvl', 'context'], correct_only=True)
        else:
            self.condition_average(['correct', 'stim_dir_lvl', 'stim_col_lvl', 'context'])

        # Reconvert from coherence levels to mean coherences for each level
        self.coherence_values = np.array([np.sort(np.unique(data.trials['stim_dir'])) for data in self.units]).mean(axis=0).tolist()
        self.conditions['stim_dir'] = np.vectorize(self.coherence_values.__getitem__)(self.conditions.stim_dir_lvl)
        self.conditions['stim_col'] = np.vectorize(self.coherence_values.__getitem__)(self.conditions.stim_col_lvl)
        self.conditions['choice'] = np.sign(self.conditions.stim_dir) * ((self.conditions.context == 1) & (self.conditions.correct == 1)) + \
                                    np.sign(self.conditions.stim_col) * ((self.conditions.context == -1) & (self.conditions.correct == 1)) - \
                                    np.sign(self.conditions.stim_dir) * ((self.conditions.context == 1) & (self.conditions.correct == 0)) - \
                                    np.sign(self.conditions.stim_col) * ((self.conditions.context == -1) & (self.conditions.correct == 0))

    def condition_average(self, variables, correct_only=False):
        data0 = self.units[0]
        conditions = list(itertools.product(*[np.unique(data0.trials[var]).tolist() for var in variables]))
        print(f'Averaging over {len(conditions)} conditions')
        pool = mp.Pool(mp.cpu_count())

        args = zip(self.units, repeat(conditions, len(self.units)), repeat(variables, len(self.units)),
                   repeat(correct_only, len(self.units)))
        results = pool.starmap(_task_avg_cond, args)

        data_avgs, ntrials = zip(*results)
        self.data_avg = np.array(data_avgs).transpose((1, 2, 0))
        self.ntrials = np.array(ntrials).T
        self.conditions = pd.DataFrame(data=conditions, columns=variables)
        if correct_only:
            self.conditions['correct'] = 1

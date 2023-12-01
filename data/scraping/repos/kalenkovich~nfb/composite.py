import numpy as np
import sympy

from ..signal_processing.filters import Coherence


class CompositeSignal:
    """
    Class for composite signal
    """
    def __init__(self, signals, expression, name, ind, fs):
        """
        Constructor
        :param signals: list of all signals
        :param expression: str expression
        """
        self.ind = ind
        self.name = name
        self.signals = signals
        self.coh_filter = None
        if 'coh' in expression.lower():
            names = ''.join([ch if ch.isalnum() else ' ' for ch in expression]).split()[1:]
            self.signals_idx = [j for j, signal in enumerate(self.signals) if signal.name in names]
            self.signals = [self.signals[j] for j in self.signals_idx]
            self.expression_lambda = self.coherence
            self.coh_filter = Coherence(500, fs, (8, 12))
        elif expression == '':
            self.expression_lambda = self.push_zeros
        else:
            self._signals_names = [signal.name for signal in self.signals]
            self.expression = sympy.sympify(expression)
            self.expression_lambda = sympy.lambdify(self._signals_names, self.expression, modules="numpy")
            self.signals_idx = list(range(len(signals)))
        self.current_sample = 0
        self.current_chunk = None
        # signal statistics
        self.scaling_flag = False
        self.mean = np.nan
        self.std = np.nan
        # signal statistics accumulators
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

    def update(self, chunk):
        chunk_size = chunk.shape[0]
        self.current_sample = self.expression_lambda(*[signal.current_chunk for signal in self.signals])
        self.mean_acc = (self.n_acc * self.mean_acc + chunk_size * self.current_sample.sum()) / (self.n_acc + chunk_size)
        self.var_acc = (self.n_acc * self.var_acc + chunk_size * (self.current_sample - self.mean_acc).sum() ** 2) / (
            self.n_acc + chunk_size)
        self.std_acc = self.var_acc ** 0.5
        self.n_acc += chunk_size
        if self.scaling_flag and self.std>0:
            self.current_sample = (self.current_sample - self.mean) / self.std
        self.current_chunk = self.current_sample*np.ones(len(chunk))
        pass

    def coherence(self, x1, x2):
        X = np.vstack([x1, x2]).T
        return self.coh_filter.apply(X)[-1]

    def push_zeros(self, *args):
        return np.zeros(len(args[0]))

    def update_statistics(self, raw=None, emulate=False, from_acc=False,
                          signals_recorder=None, stats_previous=None, updated_derived_signals_recorder=None,
                          drop_outliers=0):
        if from_acc:
            self.mean = self.mean_acc
            self.std = self.std_acc
            self.reset_statistic_acc()
            return None

        from time import time
        timer = time()
        if updated_derived_signals_recorder is None:
            signals_data = signals_recorder[:, self.signals_idx].copy()
            for j, signal_data in enumerate(signals_data.T):
                if np.isfinite(stats_previous[j][0]) and np.isfinite(stats_previous[j][1]):
                    signal_data = signal_data * stats_previous[j][1] + stats_previous[j][0]
                if self.signals[j].std > 0:
                    signals_data[:, j] = (signal_data - self.signals[j].mean) / self.signals[j].std
        else:
            signals_data = updated_derived_signals_recorder.copy()

        if self.coh_filter is None:
            if signals_data.shape[1] > 1:
                signal_recordings = self.expression_lambda(*signals_data.T)
            else:
                signal_recordings = np.apply_along_axis(self.expression_lambda, 0, signals_data)
            # drop outliers
            if drop_outliers and signal_recordings.std() > 0:
                signal_recordings_clear = signal_recordings[
                    np.abs(signal_recordings - signal_recordings.mean()) < drop_outliers * signal_recordings.std()]
            else:
                signal_recordings_clear = signal_recordings
            self.mean = np.mean(signal_recordings_clear)
            self.std = np.std(signal_recordings_clear)
        else:
            self.mean = self.mean_acc
            self.std = self.std_acc
        self.reset_statistic_acc()
        print('*** COMPOSITE TIME ELAPSED', time() - timer)

    def reset_statistic_acc(self):
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0
        if self.coh_filter is not None:
            self.coh_filter.buffer *= 0

    def enable_scaling(self):
        self.scaling_flag = True
        pass
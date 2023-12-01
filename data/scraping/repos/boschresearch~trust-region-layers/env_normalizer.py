#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Sequence, Union

import numpy as np


class RunningMeanStd(object):
    """
    The following class is derived from OpenAI baselines
    https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/running_mean_std.py
    Copyright (c) 2017 OpenAI (http://openai.com), licensed under the MIT license,
    cf. 3rd-party-licenses.txt file in the root directory of this source tree.

    Running mean and std metrics for normalization.
    """

    def __init__(self, epsilon=0., shape=(), dtype=np.float64):
        super().__init__()
        self._mean = np.zeros(shape, dtype=dtype)
        self._std = np.zeros(shape, dtype=dtype)
        self._var = np.zeros(shape, dtype=dtype)
        self._count = np.array(epsilon, dtype=dtype)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def count(self):
        return self._count

    @property
    def shape(self):
        return self.mean.shape

    @property
    def std(self):
        return self._std

    def __call__(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        tot_count = self.count + batch_count

        if tot_count == batch_count:
            self._mean[...] = batch_mean
            self._var[...] = batch_var
            self._std[...] = np.sqrt(batch_var)
            self._count[...] = tot_count
        else:
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / tot_count

            # Welford's online algorithm
            M2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / tot_count
            self._var = M2 / (tot_count - 1)
            self._std = np.sqrt(self._var)

            self._mean = new_mean
            self._count = tot_count


class BaseNormalizer(object):
    """
    An identity mapping
    """

    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self, dones:np.ndarray=None):
        """
        Reset Normalizer at the end of the episode.
        Args:
            dones: Flags which episodes ended.
        Return:
        """
        pass


class MovingAvgNormalizer(BaseNormalizer):

    def __init__(self, prev_filter: BaseNormalizer, shape: Sequence[int], center: bool = True, scale: bool = True,
                 gamma: float = 0.99, clip: Union[None, float] = None):
        """
        Normalize as f(x) = (x-mean)/std with running estimates of mean and std.
        Args:
            prev_filter: previous filter instance when stacking filters.
            shape: Shape to compute the metrics for
            center: Substract mean
            scale: Divide by scale
            gamma: Discounting for moving avgerage. Mainly used for special reward normalization.
            clip: Optional clipping value, which is applied after transforming the values.
        """
        self.prev_filter = prev_filter

        self.rs = RunningMeanStd(shape=shape)

        self.center = center
        self.scale = scale
        self.clip = clip

        self.gamma = gamma

        # TODO this should have the size of n_envs
        self.ret = np.zeros((1,))

    def __call__(self, x, update=True, **kwargs):
        """
        Normalize values x
        Args:
            x: Values to normalize
            update: Update running metrics, normally disabled during testing.
            kwargs: arguments for previous filters

        Returns:
            Normalized values
        """
        x = self.prev_filter(x, **kwargs)

        # compute discount of previous step. Only required for reward normalization.
        # When gamma == 0, no discounting is applied.
        if update:
            self.ret = self.ret * self.gamma + x
            self.rs(self.ret)

        if self.center:
            x = x - self.rs.mean

        if self.scale:
            x = x / (self.rs.std + 1e-16)

        if self.clip:
            x = np.clip(x, -self.clip, self.clip)

        return x

    def reset(self, dones=None):
        self.prev_filter.reset(dones)
        self.ret[dones] = 0.

#!/usr/bin/env python3

"""
Adapted from cherry:
https://github.com/learnables/cherry/blob/master/cherry/plot.py

"""

import numpy as np


def _one_sided_smoothing(x_before, y_before, smoothing_temperature=1.0):
    """
    ## Decription

    One side (regular) exponential moving average for smoothing a curve
    It evenly resamples points baesd on x-axis and then averages y values with
    weighting factor decreasing exponentially.

    ## Arguments

    * **x_before** (ndarray) - x values. Required to be in accending order.
    * **y_before** (ndarray) - y values. Required to have same size as x_before.
    * **smoothing_temperature** (float, *optional*, default=1.0) - the number of previous
      steps trusted. Used to calculate the decay factor.

    ## Return

    * **x_after** (ndarray) - x values after resampling.
    * **y_after** (ndarray) - y values after smoothing.
    * **y_count** (ndarray) - decay values at each steps.

    ## Credit

    Adapted from OpenAI's baselines implementation.

    ## Example

    ~~~python
    from plotify.smoothing import _one_sided_smoothing as osmooth
    x_smoothed, y_smoothed, y_counts = osmooth(
        x_original,
        y_original,
        smoothing_temperature=1.0,
    )
    ~~~
    """

    if x_before is None:
        x_before = np.arange(len(y_before))

    assert len(x_before) == len(y_before), \
        'x_before and y_before must have equal length.'
    assert all(x_before[i] <= x_before[i+1] for i in range(len(x_before)-1)), \
        'x_before needs to be sorted in ascending order.'

    # Resampling
    size = len(x_before)
    x_after = np.linspace(x_before[0], x_before[-1], size)
    y_after = np.zeros(size, dtype=float)
    y_count = np.zeros(size, dtype=float)

    # Weighting factor for data of previous steps
    alpha = np.exp(-1./smoothing_temperature)
    x_before_length = x_before[-1] - x_before[0]
    x_before_index = 0
    decay_period = x_before_length/(size-1)*smoothing_temperature

    for i in range(len(x_after)):
        # Compute current EMA value based on the value of previous time step
        if(i != 0):
            y_after[i] = alpha * y_after[i-1]
            y_count[i] = alpha * y_count[i-1]

        # Compute current EMA value by adding weighted average of old points
        # covered by the new point
        while x_before_index < size:
            if x_after[i] >= x_before[x_before_index]:
                difference = x_after[i] - x_before[x_before_index]
                # Weighting factor for y value of each old points
                beta = np.exp(-(difference/decay_period))
                y_after[i] += y_before[x_before_index] * beta
                y_count[i] += beta
                x_before_index += 1
            else:
                break

    y_after = y_after/y_count
    return x_after, y_after, y_count


def exponential_smoothing(x, y=None, temperature=1.0):
    """
    ## Decription

    Two-sided exponential moving average for smoothing a curve.
    It performs regular exponential moving average twice from two different
    sides and then combines the results together.

    ## Arguments

    * **x** (ndarray/tensor/list) - x values, in accending order.
    * **y** (ndarray/tensor/list) - y values.
    * **temperature** (float, *optional*, default=1.0) - The higher,
      the smoother.

    ## Return

    * ndarray - x values after resampling.
    * ndarray - y values after smoothing.

    ## Credit

    Adapted from OpenAI's baselines implementation.

    ## Example

    ~~~python
    from plotify.smoothing import exponential_smoothing
    x_smoothed, y_smoothed, _ = exponential_smoothing(
        x_original,
        y_original,
        temperature=3.,
    )
    ~~~
    """

    if y is None:
        y = x
        x = np.arange(0, len(y))

    if isinstance(y, (list, tuple)):
        y = np.array(y)

    if isinstance(x, (list, tuple)):
        x = np.array(x)

    assert x.shape == y.shape
    assert len(x.shape) == 1
    x_after1, y_after1, y_count1 = _one_sided_smoothing(x,
                                                        y,
                                                        temperature)
    x_after2, y_after2, y_count2 = _one_sided_smoothing(-x[::-1],
                                                        y[::-1],
                                                        temperature)

    y_after2 = y_after2[::-1]
    y_count2 = y_count2[::-1]

    y_after = y_after1 * y_count1 + y_after2 * y_count2
    y_after /= (y_count1 + y_count2)
    return x_after1.tolist(), y_after.tolist()


def smooth(x, y=None, temperature=1.0):
    """
    ## Decription

    Simple wrapper around exponential smoothing.

    Returns 1 value (smoothed `x`) if `y` is None.

    Returns 2 values (smoothed `x` and `y`) if `y` is not None.

    ## Example

    ~~~python
    x = np.arange(100)
    y = np.logspace(-50, 50)

    # single value
    smooth_x = plotify.smoothing.smooth(x, temperature=20.0)

    # double value
    smooth_x, smooth_y = plotify.smoothing.smooth(x, y)
    ~~~
    """
    # Not officially supported.
    result = exponential_smoothing(x=x, y=y, temperature=temperature)
    if y is None:
        return result[1]
    return result

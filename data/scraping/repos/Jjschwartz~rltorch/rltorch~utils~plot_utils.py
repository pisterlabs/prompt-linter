import numpy as np


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.,
                  low_counts_threshold=1e-8):
    """From openai.baselines.common.plot_util.py

    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments
    ---------
    xolds : array or list
        x values of data. Needs to be sorted in ascending order
    yolds : array of list
        y values of data. Has to have the same length as xolds
    low : float
        min value of the new x grid. By default equals to xolds[0]
    high : float
        max value of the new x grid. By default equals to xolds[-1]
    n : int
        number of points in new x grid
    decay_steps : float
        EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
        y values with counts less than this value will be set to NaN

    Returns
    -------
    xs : array
        with new x grid
    ys : array
        of EMA of y at each point of the new x grid
    count_ys : array
        of EMA of y counts at each point of the new x grid
    """

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, \
        f'low={low} < xolds[0]={xolds[0]} - extrapolation not permitted!'
    assert xolds[-1] >= high, \
        f'high={high} > xolds[-1]={xolds[-1]} - extrapolation not permitted!'
    assert len(xolds) == len(yolds), \
        f'len of xolds ({len(xolds)}) and yolds ({len(yolds)}) do not match!'

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0    # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.,
                  low_counts_threshold=1e-8):
    """From openai.baselines.common.plot_util.py

    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments
    ---------
    xolds : array or list
        x values of data. Needs to be sorted in ascending order
    yolds : array of list
        y values of data. Has to have the same length as xolds
    low : float
        min value of the new x grid. By default equals to xolds[0]
    high : float
        max value of the new x grid. By default equals to xolds[-1]
    n : int
        number of points in new x grid
    decay_steps : float
        EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
        y values with counts less than this value will be set to NaN

    Returns
    -------
    xs : array
        with new x grid
    ys : array
        of EMA of y at each point of the new x grid
    count_ys : array
        of EMA of y counts at each point of the new x grid
    """
    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    xs, ys1, count_ys1 = one_sided_ema(
        xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0
    )
    _,  ys2, count_ys2 = one_sided_ema(
        -xolds[::-1], yolds[::-1], -high, -low, n, decay_steps,
        low_counts_threshold=0
    )
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


def plot_xy(ax, xs, ys, label=None):
    """Create a line plot on given axis, with stderr included

    Will plot an errorbar plot if len(xs) == 1

    Parameters
    ----------
    ax : Matplotlib.pyplot.axis
       axis to plot on
    xs : array
       of x-axis values
    ys : array
       of y-axis values
    label : str, optional
       a label for the line (default=None)
    """
    print(f"Plotting {label}")
    try:
        if len(ys[0]):
            # list of lists
            y_mean = np.mean(ys, axis=0)
            y_std = np.std(ys, axis=0)
    except Exception:
        y_mean = ys
        y_std = 0

    if len(xs) > 1:
        ax.plot(xs, y_mean, label=label)
        ax.fill_between(xs, y_mean-y_std, y_mean+y_std, alpha=0.25)
    else:
        ax.errorbar(
            xs, y_mean, yerr=(y_mean-y_std, y_mean+y_std),
            fmt='o', label=label, capsize=10
        )

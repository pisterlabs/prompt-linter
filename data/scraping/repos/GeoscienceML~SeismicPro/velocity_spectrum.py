"""Implements VerticalVelocitySpectrum and ResidualVelocitySpectrum classes."""

# pylint: disable=not-an-iterable, too-many-arguments
import math

import numpy as np
from numba import njit, prange
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt


from .utils import coherency_funcs
from .interactive_plot import VelocitySpectrumPlot
from ..containers import SamplesContainer
from ..decorators import batch_method, plotter
from ..stacking_velocity import StackingVelocity, StackingVelocityField
from ..utils import add_colorbar, set_ticks, set_text_formatting, get_first_defined
from ..gather.utils.correction import apply_constant_velocity_nmo
from ..const import DEFAULT_STACKING_VELOCITY


COHERENCY_FUNCS = {
    "stacked_amplitude": coherency_funcs.stacked_amplitude,
    "S": coherency_funcs.stacked_amplitude,
    "normalized_stacked_amplitude": coherency_funcs.normalized_stacked_amplitude,
    "NS": coherency_funcs.normalized_stacked_amplitude,
    "semblance": coherency_funcs.semblance,
    "NE": coherency_funcs.semblance,
    'crosscorrelation': coherency_funcs.crosscorrelation,
    'CC': coherency_funcs.crosscorrelation,
    'ENCC': coherency_funcs.energy_normalized_crosscorrelation,
    'energy_normalized_crosscorrelation': coherency_funcs.energy_normalized_crosscorrelation
}


class BaseVelocitySpectrum(SamplesContainer):
    """Base class for vertical velocity spectrum calculation.

    Implements general computation logic and a method for spectrum visualization.

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate velocity spectrum for.
    window_size : float
        Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother the
        resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
    mode: str, defaults to `semblance`
        A measure for estimating hodograph coherency. See `COHERENCY_FUNCS` for available options.
    max_stretch_factor : float, defaults to np.inf
        Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO correction.
        This mute is applied after NMO correction for each provided velocity and before coherency calculation. The
        lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
        Reasonably good value is 0.65.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which velocity spectrum calculation was called.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the velocity spectrum. Measured in samples.
    coherency_func : callable
        A function that estimates the chosen coherency measure for a hodograph.
    max_stretch_factor : float
        Maximum allowable factor for stretch muter.
    """

    def __init__(self, gather, window_size, mode='semblance', max_stretch_factor=np.inf):
        self.gather = gather.copy()
        self.half_win_size_samples = math.ceil((window_size / gather.sample_interval / 2))
        self.max_stretch_factor = max_stretch_factor

        self.coherency_func = COHERENCY_FUNCS.get(mode)
        if self.coherency_func is None:
            raise ValueError(f"Unknown mode {mode}, available modes are {COHERENCY_FUNCS.keys()}")

    @property
    def samples(self):
        """np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.gather.samples

    @property
    def sample_interval(self):
        """float: Sample interval of seismic traces. Measured in milliseconds."""
        return self.gather.sample_interval

    @property
    def delay(self):
        """float: Delay recording time of seismic traces. Measured in milliseconds."""
        return self.gather.delay

    @property
    def coords(self):
        """Coordinates or None: Spatial coordinates of the velocity spectrum. Determined by the underlying gather.
        `None` if the gather is indexed by unsupported headers or required coords headers were not loaded or
        coordinates are non-unique for traces of the gather."""
        return self.gather.coords

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in meters/milliseconds) by their indices (possibly non-integer) in
        velocity spectrum."""
        _ = time_ix, velocity_ix
        raise NotImplementedError

    def get_time_knots(self, stacking_velocity):
        """Return a sorted array of `stacking_velocity` times, that lie within the time range used for spectrum
        calculation. The first and the last spectrum times are always included."""
        valid_times_mask = (stacking_velocity.times > self.times[0]) & (stacking_velocity.times < self.times[-1])
        valid_times = np.sort(stacking_velocity.times[valid_times_mask])
        return np.concatenate([[self.times[0]], valid_times, [self.times[-1]]])

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_single_velocity_spectrum(coherency_func, gather_data, times, offsets, velocity, sample_interval, delay,
                                      half_win_size_samples, t_min_ix, t_max_ix, max_stretch_factor=np.inf,
                                      interpolate=True, out=None):
        """Calculate velocity spectrum for a given range of zero-offset traveltimes and constant velocity.

        Parameters
        ----------
        coherency_func: njitted callable
            A function that estimates hodograph coherency.
        gather_data : 2d np.ndarray
            Gather data for velocity spectrum calculation.
        times : 1d np.ndarray
            Recording time for each trace value. Measured in milliseconds.
        offsets : array-like
            The distance between source and receiver for each trace. Measured in meters.
        velocity : array-like
            Seismic wave velocity for velocity spectrum computation. Measured in meters/milliseconds.
        sample_interval : float
            Sample interval of seismic traces. Measured in milliseconds.
        delay : float
            Delay recording time of seismic traces. Measured in milliseconds.
        half_win_size_samples : int
            Half of the temporal size for smoothing the velocity spectrum. Measured in samples.
        t_min_ix : int
            Time index in `times` array to start calculating velocity spectrum from. Measured in samples.
        t_max_ix : int
            Time index in `times` array to stop calculating velocity spectrum at. Measured in samples.
        max_stretch_factor : float, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO
            correction. The lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
            Reasonably good value is 0.65.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude
            at the nearest time sample is used.
        out : np.array, optional
            The buffer to store result in. If not provided, a new array is allocated.

        Returns
        -------
        out : 1d np.ndarray
            Calculated velocity spectrum values for a specified `velocity` in time range from `t_min_ix` to `t_max_ix`.
        """
        t_win_size_min_ix = max(0, t_min_ix - half_win_size_samples)
        t_win_size_max_ix = min(len(times) - 1, t_max_ix + half_win_size_samples)

        corrected_gather_data = apply_constant_velocity_nmo(gather_data, offsets, sample_interval, delay,
                                                            times[t_win_size_min_ix: t_win_size_max_ix + 1], velocity,
                                                            max_stretch_factor, interpolate)
        numerator, denominator = coherency_func(corrected_gather_data)

        if out is None:
            out = np.empty(t_max_ix - t_min_ix, dtype=np.float32)

        for t in prange(t_min_ix, t_max_ix):
            t_rel = t - t_win_size_min_ix
            ix_from = max(0, t_rel - half_win_size_samples)
            ix_to = min(corrected_gather_data.shape[1] - 1, t_rel + half_win_size_samples)
            out[t - t_min_ix] = np.sum(numerator[ix_from : ix_to]) / (np.sum(denominator[ix_from : ix_to]) + 1e-8)
        return out

    def _plot(self, title=None, x_label=None, x_ticklabels=None, x_ticker=None, y_ticklabels=None, y_ticker=None,
              grid=False, stacking_velocity_ix=None, velocity_bounds_ix=None, colorbar=True,
              clip_threshold_quantile=0.99, n_levels=10, ax=None, **kwargs):
        """Plot vertical velocity spectrum and, optionally, stacking velocity.

        Parameters
        ----------
        title : str, optional, defaults to None
            Plot title.
        x_label : str, optional, defaults to None
            The title of the x-axis.
        x_ticklabels : list of str, optional, defaults to None
            An array of labels for the x-axis.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticklabels : list of str, optional, defaults to None
            An array of labels for the y-axis.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        stacking_velocity_ix : tuple of two 1d np.ndarray, optional
            Indices of times and velocities of a stacking velocity to show on the plot.
        velocity_bounds_ix : tuple of three 1d np.ndarray, optional
            Indices of times and velocities of left and right bounds. An area between these bounds will be highlighted
            on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the velocity spectrum plot.
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the velocity spectrum values by given quantile.
        n_levels : int, optional, defaults to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        """
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        cmap = plt.get_cmap('seismic')
        level_values = np.linspace(0, np.quantile(self.velocity_spectrum, clip_threshold_quantile), n_levels)
        norm = mcolors.BoundaryNorm(level_values, cmap.N, clip=True)
        img = ax.imshow(self.velocity_spectrum, norm=norm, aspect='auto', cmap=cmap)
        add_colorbar(ax, img, colorbar, y_ticker=y_ticker)
        ax.set_title(**{"label": None, **title})

        if stacking_velocity_ix is not None:
            stacking_times_ix, stacking_velocities_ix = stacking_velocity_ix
            ax.plot(stacking_velocities_ix, stacking_times_ix, c='#fafcc2', linewidth=2.5,
                    marker="o", markevery=slice(1, -1))
        if velocity_bounds_ix is not None:
            ax.fill_betweenx(*velocity_bounds_ix, color="white", alpha=0.2)
        if grid:
            ax.grid(c='k')

        set_ticks(ax, "x", x_label, x_ticklabels, **x_ticker)
        set_ticks(ax, "y", "Time", y_ticklabels, **y_ticker)

    def plot(self, *args, interactive=False, **kwargs):
        """Plot velocity spectrum in interactive or non-interactive mode."""
        if not interactive:
            return self._plot(*args, **kwargs)
        return VelocitySpectrumPlot(self, *args, **kwargs).plot()


class VerticalVelocitySpectrum(BaseVelocitySpectrum):
    r"""A class for vertical velocity spectrum calculation and processing.

    Velocity spectrum is a measure of hodograph coherency. The higher the values of velocity spectrum are, the more
    coherent the signal is along a hyperbolic trajectory over the entire spread length of the gather.

    Velocity spectrum instance can be created either directly by passing source gather (and optional parameters such as
    velocity range, window size, coherency measure and a factor for stretch mute) to its `__init__` or by calling
    :func:`~Gather.calculate_vertical_velocity_spectrum` method (recommended way).

    The velocity spectrum is computed by:
    :math:`VS(k, v) = \frac{\sum^{k+N/2}_{i=k-N/2} numerator(i, v)}
                           {\sum^{k+N/2}_{i=k-N/2} denominator(i, v)},
    where:
     - VS - velocity spectrum value for starting time index `k` and velocity `v`,
     - N - temporal window size,
     - numerator(i, v) - numerator of the coherency measure,
     - denominator(i, v) - denominator of the coherency measure.

    For different coherency measures the numerator and denominator are calculated as follows:

    - Stacked Amplitude, "S":
        numerator(i, v) = abs(sum^{M-1}_{j=0} f_{j}(i, v))
        denominator(i, v) = 1

    - Normalized Stacked Amplitude, "NS":
        numerator(i, v) = abs(sum^{M-1}_{j=0} f_{j}(i, v))
        denominator(i, v) = sum^{M-1}_{j=0} abs(f_{j}(i, v))

    - Semblance, "NE":
        numerator(i, v) = (sum^{M-1}_{j=0} f_{j}(i, v))^2 / M
        denominator(i, v) = sum^{M-1}_{j=0} f_{j}(i, v)^2

    - Crosscorrelation, "CC":
        numerator(i, v) = ((sum^{M-1}_{j=0} f_{j}(i, v))^2 - sum^{M-1}_{j=0} f_{j}(i, v)^2) / 2
        denominator(i, v) = 1

    - Energy Normalized Crosscorrelation, "ENCC":
        numerator(i, v) = ((sum^{M-1}_{j=0} f_{j}(i, v))^2 - sum^{M-1}_{j=0} f_{j}(i, v)^2) / (M - 1)
        denominator(i, v) = sum^{M-1}_{j=0} f_{j}(i, v)^2

    where f_{j}(i, v) is the amplitude value on the `j`-th trace being NMO-corrected for time index `i` and velocity
    `v`. Thus the amplitude is taken for the time defined by :math:`t(i, v) = \sqrt{t_0^2 + \frac{l_j^2}{v^2}}`, where:
    :math:`t_0` - start time of the hyperbola associated with time index `i`,
    :math:`l_j` - offset of the `j`-th trace,
    :math:`v` - velocity value.

    See the COHERENCY_FUNCS for the full list of available coherency measures.

    The resulting matrix :math:`VS(k, v)` has shape (n_times, n_velocities) and contains vertical velocity spectrum
    values based on hyperbolas with each combination of the starting point :math:`k` and velocity :math:`v`.

    The algorithm for velocity spectrum calculation looks as follows:
    For each velocity from the given velocity range:
        1. Calculate NMO-corrected gather.
        2. Estimate numerator and denominator for given coherency measure for each timestamp.
        3. Get the values of velocity spectrum as a ratio of rolling sums of numerator and denominator in temporal
           windows of a given size.

    Examples
    --------
    Calculate velocity spectrum for 200 velocities from 2000 to 6000 m/s and a temporal window size of 16 ms:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather()
    >>> spectrum = gather.calculate_vertical_velocity_spectrum(velocities=np.linspace(2000, 6000, 200), window_size=16)

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate velocity spectrum for.
    velocities : 1d np.ndarray, optional, defaults to None
        An array of stacking velocities to calculate the velocity spectrum for. Measured in meters/seconds. If not
        provided, `stacking_velocity` is evaluated for gather times to estimate the velocity range being examined.
        The resulting velocities are then evenly sampled from this range being additionally extended by
        `relative_margin` * 100% in both directions with a step of `velocity_step`.
    stacking_velocity : StackingVelocity or StackingVelocityField, optional, defaults to DEFAULT_STACKING_VELOCITY
        Stacking velocity around which vertical velocity spectrum is calculated if `velocities` are not given.
        `StackingVelocity` instance is used directly. If `StackingVelocityField` instance is passed, a
        `StackingVelocity` corresponding to gather coordinates is fetched from it.
    relative_margin : float, optional, defaults to 0.2
        Relative velocity margin to additionally extend the velocity range obtained from `stacking_velocity`: an
        interval [`min_velocity`, `max_velocity`] is mapped to [(1 - `relative_margin`) * `min_velocity`,
        (1 + `relative_margin`) * `max_velocity`].
    velocity_step : float, optional, defaults to 50
        A step between two adjacent velocities for which vertical velocity spectrum is calculated if `velocities` are
        not passed. Measured in meters/seconds.
    window_size : int, optional, defaults to 50
        Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother the
        resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
    mode: str, optional, defaults to 'semblance'
        The measure for estimating hodograph coherency.
        The available options are:
            `semblance` or `NE`,
            `stacked_amplitude` or `S`,
            `normalized_stacked_amplitude` or `NS`,
            `crosscorrelation` or `CC`,
            `energy_normalized_crosscorrelation` or `ENCC`.
    max_stretch_factor : float, defaults to np.inf
        Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO correction.
        This mute is applied after NMO correction for each provided velocity and before coherency calculation. The
        lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
        Reasonably good value is 0.65.
    interpolate: bool, optional, defaults to True
        Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude at
        the nearest time sample is used.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which velocity spectrum calculation was called.
    velocity_spectrum : 2d np.ndarray
        An array with calculated vertical velocity spectrum values.
    velocities : 1d np.ndarray
        Range of velocity values for which vertical velocity spectrum was calculated. Measured in meters/seconds.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the vertical velocity spectrum. Measured in samples.
    max_stretch_factor : float
        Maximum allowable factor for stretch muter.
    """
    def __init__(self, gather, velocities=None, stacking_velocity=None, relative_margin=0.2, velocity_step=50,
                 window_size=50, mode='semblance', max_stretch_factor=np.inf, interpolate=True):
        super().__init__(gather, window_size, mode, max_stretch_factor)
        if stacking_velocity is None:
            stacking_velocity = DEFAULT_STACKING_VELOCITY
        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(self.coords)
        if velocities is None:
            velocities = self.get_velocity_range(stacking_velocity, relative_margin, velocity_step)
        else:
            velocities = np.sort(velocities)

        self.velocities = np.asarray(velocities, dtype=np.float32)  # m/s
        self.stacking_velocity = stacking_velocity
        self.relative_margin = relative_margin

        velocities_ms = self.velocities / 1000  # from m/s to m/ms
        kwargs = {"spectrum_func": self.calc_single_velocity_spectrum, "coherency_func": self.coherency_func,
                  "gather_data": self.gather.data, "times": self.times, "offsets": self.gather.offsets,
                  "velocities": velocities_ms, "sample_interval": self.sample_interval, "delay": self.delay,
                  "half_win_size_samples": self.half_win_size_samples, "max_stretch_factor": max_stretch_factor,
                  "interpolate": interpolate}
        self.velocity_spectrum = self._calc_spectrum_numba(**kwargs)

    @property
    def n_velocities(self):
        """int: The number of velocities the spectrum was calculated for."""
        return len(self.velocities)

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_spectrum_numba(spectrum_func, coherency_func, gather_data, times, offsets, velocities, sample_interval,
                             delay, half_win_size_samples, max_stretch_factor, interpolate):
        """Parallelized and njitted method for vertical velocity spectrum calculation.

        Parameters
        ----------
        spectrum_func : njitted callable
            Base function for velocity spectrum calculation for a single velocity and a time range.
        coherency_func : njitted callable
            A function for hodograph coherency estimation.
        other parameters : misc
            Passed directly from class attributes or `__init__` arguments (except for `velocities` which are converted
            from m/s to m/ms).

        Returns
        -------
        velocity_spectrum : 2d np.ndarray
            Array with vertical velocity spectrum values.
        """
        velocity_spectrum = np.empty((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for j in prange(len(velocities)):  # pylint: disable=consider-using-enumerate
            spectrum_func(coherency_func=coherency_func, gather_data=gather_data, times=times, offsets=offsets,
                          velocity=velocities[j], sample_interval=sample_interval, delay=delay,
                          half_win_size_samples=half_win_size_samples, t_min_ix=0, t_max_ix=gather_data.shape[1],
                          max_stretch_factor=max_stretch_factor, interpolate=interpolate, out=velocity_spectrum[:, j])
        return velocity_spectrum

    def get_velocity_range(self, stacking_velocity, relative_margin, velocity_step):
        """Return an array of stacking velocities for spectrum calculation:
        1. First `stacking_velocity` is evaluated for gather times to estimate the velocity range being examined.
        2. Then the range is additionally extended by `relative_margin` * 100% in both directions.
        3. The resulting velocities are then evenly sampled from this range with a step of `velocity_step`.
        """
        interpolated_velocities = stacking_velocity(self.times)
        min_velocity = np.min(interpolated_velocities) * (1 - relative_margin)
        max_velocity = np.max(interpolated_velocities) * (1 + relative_margin)
        n_velocities = math.ceil((max_velocity - min_velocity) / velocity_step) + 1
        return min_velocity + velocity_step * np.arange(n_velocities)

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in meters/milliseconds) by their indices (possibly non-integer) in
        velocity spectrum."""
        time = None
        if 0 <= time_ix <= self.n_times - 1:
            time = self.delay + self.sample_interval * time_ix

        velocity = None
        if 0 <= velocity_ix <= self.n_velocities - 1:
            velocity = np.interp(velocity_ix, np.arange(self.n_velocities), self.velocities) / 1000  # from m/s to m/ms

        return time, velocity

    def _plot(self, stacking_velocity=None, *, plot_bounds=True, title=None, x_ticker=None, y_ticker=None, grid=False,
              colorbar=True, ax=None, **kwargs):
        """Plot vertical velocity spectrum."""
        # Add a stacking velocity line on the plot
        stacking_velocity_ix = None
        velocity_bounds_ix = None
        if stacking_velocity is not None:
            stacking_times = self.get_time_knots(stacking_velocity)
            stacking_velocities = stacking_velocity(stacking_times)
            stacking_times_ix = self.times_to_indices(stacking_times)
            stacking_velocities_ix = np.interp(stacking_velocities, self.velocities, np.arange(self.n_velocities))
            stacking_velocity_ix = (stacking_times_ix, stacking_velocities_ix)

            if plot_bounds and stacking_velocity.bounds is not None:
                left_bound, right_bound = stacking_velocity.bounds
                left_times = self.get_time_knots(left_bound)
                right_times = self.get_time_knots(right_bound)
                bounds_times = np.unique(np.concatenate([left_times, right_times]))
                left_velocities = left_bound(bounds_times)
                right_velocities = right_bound(bounds_times)
                bounds_times_ix = self.times_to_indices(bounds_times)
                left_velocities_ix = np.interp(left_velocities, self.velocities, np.arange(self.n_velocities))
                right_velocities_ix = np.interp(right_velocities, self.velocities, np.arange(self.n_velocities))
                velocity_bounds_ix = (bounds_times_ix, left_velocities_ix, right_velocities_ix)

        super()._plot(title=title, x_label="Velocity, m/s", x_ticklabels=self.velocities,
                      x_ticker=x_ticker, y_ticklabels=self.times, y_ticker=y_ticker, ax=ax, grid=grid,
                      stacking_velocity_ix=stacking_velocity_ix, velocity_bounds_ix=velocity_bounds_ix,
                      colorbar=colorbar, **kwargs)
        return self

    @plotter(figsize=(10, 9), args_to_unpack="stacking_velocity")
    def plot(self, stacking_velocity=None, *, plot_bounds=True, title=None, interactive=False, **kwargs):
        """Plot vertical velocity spectrum.

        Parameters
        ----------
        stacking_velocity : StackingVelocity or str, optional
            Stacking velocity to plot if given. If its times are sampled less than once every 50 ms, each point will be
            highlighted with a circle.
            May be `str` if plotted in a pipeline: in this case it defines a component with stacking velocities to use.
        plot_bounds : bool, optional, defaults to True
            Whether to display bound used for stacking velocity calculation if they exist.
        title : str, optional
            Plot title. If not provided, equals to stacked lines "Vertical Velocity Spectrum" and coherency func name.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the velocity spectrum plot.
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the velocity spectrum values by given quantile.
        n_levels : int, optional, defaults to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        interactive : bool, optional, defaults to `False`
            Whether to plot velocity spectrum in interactive mode. This mode also plots the gather used to calculate
            the velocity spectrum. Clicking on velocity spectrum highlights the corresponding hodograph on the gather
            plot and allows performing NMO correction of the gather with the selected velocity.
            Interactive plotting must be performed in a JupyterLab environment with the `%matplotlib widget`
            magic executed and `ipympl` and `ipywidgets` libraries installed.
        sharey : bool, optional, defaults to True, only for interactive mode
            Whether to share y axis of velocity spectrum and gather plots.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.

        Returns
        -------
        velocity_spectrum : VerticalVelocitySpectrum
            Self unchanged.
        """
        if title is None:
            title = f"Vertical Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(self.coords)
        return super().plot(stacking_velocity=stacking_velocity, plot_bounds=plot_bounds, interactive=interactive,
                            title=title, **kwargs)

    @batch_method(target="for", args_to_unpack="init", copy_src=False)
    def calculate_stacking_velocity(self, init=None, bounds=None, relative_margin=None, acceleration_bounds="auto",
                                    times_step=100, max_offset=5000, hodograph_correction_step=25, max_n_skips=2):
        """Calculate stacking velocity by vertical velocity spectrum.

        Notes
        -----
        A detailed description of the proposed algorithm and its implementation can be found in
        :func:`~velocity_model.calculate_stacking_velocity` docs.

        Parameters
        ----------
        init : StackingVelocity or str, optional
            A rough estimate of the stacking velocity being picked. Used to calculate `bounds` as
            [`init` * (1 - `relative_margin`), `init` * (1 + `relative_margin`)] if they are not given.
            May be `str` if called in a pipeline: in this case it defines a component with stacking velocities to use.
            If not given, `self.stacking_velocity` is used.
        bounds : array-like of two StackingVelocity, optional
            Left and right bounds of an area for stacking velocity picking. If not given, `init` must be passed.
        relative_margin : positive float, optional
            A fraction of stacking velocities defined by `init` used to estimate `bounds` if they are not given.
            If not given, `self.relative_margin` is used.
        acceleration_bounds : tuple of two positive floats or "auto" or None, optional
            Minimal and maximal acceleration allowed for the stacking velocity function. If "auto", equals to the range
            of accelerations of stacking velocities in `bounds` extended by 50% in both directions. If `None`, only
            ensures that picked stacking velocity is monotonically increasing. Measured in meters/seconds^2.
        times_step : float, optional, defaults to 100
            A difference between two adjacent times defining graph nodes.
        max_offset : float, optional, defaults to 5000
            An offset for hodograph time estimation. Used to create graph nodes and calculate their velocities for each
            time.
        hodograph_correction_step : float, optional, defaults to 25
            The maximum difference in arrival time of two hodographs starting at the same zero-offset time and two
            adjacent velocities at `max_offset`. Used to create graph nodes and calculate their velocities for each
            time.
        max_n_skips : int, optional, defaults to 2
            Defines the maximum number of intermediate times between two nodes of the graph. Greater values increase
            computational costs, but tend to produce smoother stacking velocity.

        Returns
        -------
        stacking_velocity : StackingVelocity
            Calculated stacking velocity.
        """
        kwargs = {"init": get_first_defined(init, self.stacking_velocity), "bounds": bounds,
                  "relative_margin": get_first_defined(relative_margin, self.relative_margin),
                  "acceleration_bounds": acceleration_bounds, "times_step": times_step, "max_offset": max_offset,
                  "hodograph_correction_step": hodograph_correction_step, "max_n_skips": max_n_skips}
        return StackingVelocity.from_vertical_velocity_spectrum(self, **kwargs)


class ResidualVelocitySpectrum(BaseVelocitySpectrum):
    """A class for residual vertical velocity spectrum calculation and processing.

    Residual velocity spectrum is a hodograph coherency measure for a CDP gather along picked stacking velocity. The
    method of its computation for a given time and velocity completely coincides with the calculation of
    :class:`~VerticalVelocitySpectrum`, however, residual velocity spectrum is computed in a small area around given
    stacking velocity, thus allowing for additional optimizations.

    The boundaries in which calculation is performed depend on time `t` and are given by:
    `stacking_velocity(t)` * (1 +- `relative_margin`).

    Since the length of this velocity range varies for different timestamps, the residual velocity spectrum values are
    interpolated to obtain a rectangular matrix of size (`n_times`, max(right_boundary - left_boundary)), where
    `left_boundary` and `right_boundary` are arrays of left and right boundaries for all timestamps respectively.

    Thus the residual velocity spectrum is a function of time and relative velocity margin. Zero margin line
    corresponds to the given stacking velocity and generally should pass through local velocity spectrum maxima.

    Residual velocity spectrum instance can be created either directly by passing gather, stacking velocity and other
    arguments to its `__init__` or by calling :func:`~Gather.calculate_residual_velocity_spectrum` method (recommended
    way).

    Examples
    --------
    First let's sample a CDP gather from a survey:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather()

    Now let's calculate stacking velocity by velocity spectrum of the gather:
    >>> velocity_spectrum = gather.calculate_vertical_velocity_spectrum()
    >>> velocity = velocity_spectrum.calculate_stacking_velocity()

    Residual velocity spectrum for the gather and calculated stacking velocity can be obtained as follows:
    >>> residual_spectrum = gather.calculate_residual_velocity_spectrum(velocity)

    Parameters
    ----------
    gather : Gather
        Seismic gather to calculate residual velocity spectrum for.
    stacking_velocity : StackingVelocity or StackingVelocityField
        Stacking velocity around which residual velocity spectrum is calculated. `StackingVelocity` instance is used
        directly. If `StackingVelocityField` instance is passed, a `StackingVelocity` corresponding to gather
        coordinates is fetched from it.
    relative_margin : float, optional, defaults to 0.2
        Relative velocity margin, that determines the velocity range for velocity spectrum calculation for each time
        `t` as `stacking_velocity(t)` * (1 +- `relative_margin`).
    velocity_step : float, optional, defaults to 25
        A step between two adjacent velocities for which residual velocity spectrum is calculated. Measured in
        meters/seconds.
    window_size : int, optional, defaults to 50
        Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother the
        resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
    mode: str, optional, defaults to 'semblance'
        The measure for estimating hodograph coherency.
        The available options are:
            `semblance` or `NE`,
            `stacked_amplitude` or `S`,
            `normalized_stacked_amplitude` or `NS`,
            `crosscorrelation` or `CC`,
            `energy_normalized_crosscorrelation` or `ENCC`.
    max_stretch_factor : float, defaults to np.inf
        Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO correction.
        This mute is applied after NMO correction for each provided velocity and before coherency calculation. The
        lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
        Reasonably good value is 0.65.
    interpolate: bool, optional, defaults to True
        Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude at
        the nearest time sample is used.

    Attributes
    ----------
    gather : Gather
        Seismic gather for which residual velocity spectrum calculation was called.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual velocity spectrum was calculated.
    velocity_spectrum : 2d np.ndarray
        An array with calculated residual vertical velocity spectrum values.
    margins : 1d np.ndarray
        An array of velocity margins the spectrum was calculated for.
    margin_step : float
        A step between each two adjacent margins.
    relative_margin : float
        Relative velocity margin, that determines the velocity range for velocity spectrum calculation for each time.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the velocity spectrum. Measured in samples.
    max_stretch_factor: float
        Maximum allowable factor for stretch muter.
    """
    def __init__(self, gather, stacking_velocity, relative_margin=0.2, velocity_step=25, window_size=50,
                 mode='semblance', max_stretch_factor=np.inf, interpolate=True):
        super().__init__(gather, window_size, mode, max_stretch_factor)
        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(self.coords)
        self.stacking_velocity = stacking_velocity
        self.relative_margin = relative_margin

        stacking_velocities = self.stacking_velocity(self.times)
        kwargs = {"spectrum_func": self.calc_single_velocity_spectrum, "coherency_func": self.coherency_func,
                  "gather_data": self.gather.data, "times": self.times, "offsets": self.gather.offsets,
                  "stacking_velocities": stacking_velocities, "relative_margin": relative_margin,
                  "velocity_step": velocity_step, "sample_interval": self.sample_interval, "delay": self.delay,
                  "half_win_size_samples": self.half_win_size_samples, "max_stretch_factor": max_stretch_factor,
                  "interpolate": interpolate}
        self.velocity_spectrum = self._calc_spectrum_numba(**kwargs)
        self.margins, self.margin_step = np.linspace(-self.relative_margin, self.relative_margin, self.n_margins,
                                                     retstep=True)

    @property
    def n_margins(self):
        """int: The number of velocity margins the spectrum was calculated for."""
        return self.velocity_spectrum.shape[1]

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_spectrum_numba(spectrum_func, coherency_func, gather_data, times, offsets, stacking_velocities,
                             relative_margin, velocity_step, sample_interval, delay, half_win_size_samples,
                             max_stretch_factor, interpolate):
        """Parallelized and njitted method for residual vertical velocity spectrum calculation.

        Parameters
        ----------
        spectrum_func : njitted callable
            Base function for velocity spectrum calculation for a single velocity and a time range.
        coherency_func : njitted callable
            A function for hodograph coherency estimation.
        other parameters : misc
            Passed directly from class attributes or `__init__` arguments (except for `stacking_velocities` which are
            the values of `stacking_velocity` evaluated at gather times).

        Returns
        -------
        residual_velocity_spectrum : 2d np.ndarray
            Array with residual vertical velocity spectrum values.
        """
        # Calculate velocity bounds and a range of velocities for residual spectrum calculation
        left_bound = stacking_velocities * (1 - relative_margin)
        right_bound = stacking_velocities * (1 + relative_margin)
        min_velocity = left_bound.min()
        max_velocity = right_bound.max()
        n_velocities = math.ceil((max_velocity - min_velocity) / velocity_step) + 1
        velocities = (min_velocity + velocity_step * np.arange(n_velocities)).astype(np.float32)

        # Convert bounds to their indices in the array of velocities and construct a binary mask that stores True
        # values for (time, velocity) pairs for which spectrum should be calculated
        left_bound_ix = np.empty(len(left_bound), dtype=np.int32)
        right_bound_ix = np.empty(len(right_bound), dtype=np.int32)
        spectrum_mask = np.zeros((gather_data.shape[1], len(velocities)), dtype=np.bool_)
        for i in prange(len(left_bound_ix)):
            left_bound_ix[i] = np.argmin(np.abs(left_bound[i] - velocities))
            right_bound_ix[i] = np.argmin(np.abs(right_bound[i] - velocities))
            spectrum_mask[i, left_bound_ix[i] : right_bound_ix[i] + 1] = True

        # Calculate only necessary part of the vertical velocity spectrum
        velocity_spectrum = np.zeros((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for i in prange(len(velocities)):
            t_ix = np.where(spectrum_mask[:, i])[0]
            if len(t_ix) == 0:
                continue
            t_min_ix = t_ix[0]
            t_max_ix = t_ix[-1]

            spectrum_func(coherency_func=coherency_func, gather_data=gather_data, times=times, offsets=offsets,
                          velocity=velocities[i] / 1000, sample_interval=sample_interval, delay=delay,
                          half_win_size_samples=half_win_size_samples, t_min_ix=t_min_ix, t_max_ix=t_max_ix+1,
                          max_stretch_factor=max_stretch_factor, interpolate=interpolate,
                          out=velocity_spectrum[t_min_ix : t_max_ix + 1, i])

        # Interpolate velocity spectrum to get a rectangular image
        residual_velocity_spectrum_len = (right_bound_ix - left_bound_ix).max()
        residual_velocity_spectrum = np.empty((len(times), residual_velocity_spectrum_len), dtype=np.float32)
        for i in prange(len(residual_velocity_spectrum)):
            cropped_spectrum = velocity_spectrum[i, left_bound_ix[i] : right_bound_ix[i] + 1]
            cropped_velocities = velocities[left_bound_ix[i] : right_bound_ix[i] + 1]
            target_velocities = np.linspace(left_bound[i], right_bound[i], residual_velocity_spectrum_len)
            residual_velocity_spectrum[i] = np.interp(target_velocities, cropped_velocities, cropped_spectrum)
        return residual_velocity_spectrum

    def get_time_velocity_by_indices(self, time_ix, velocity_ix):
        """Get time (in milliseconds) and velocity (in meters/milliseconds) by their indices (possibly non-integer) in
        residual velocity spectrum."""
        if (time_ix < 0) or (time_ix > self.n_times - 1):
            return None, None
        time = self.delay + self.sample_interval * time_ix

        if (velocity_ix < 0) or (velocity_ix > self.n_margins - 1):
            return time, None
        margin = -self.relative_margin + velocity_ix * self.margin_step
        velocity = self.stacking_velocity(time) * (1 + margin) / 1000  # from m/s to m/ms
        return time, velocity

    def _plot(self, *, acceptable_margin=None, title=None, x_ticker=None, y_ticker=None, grid=False, colorbar=True,
              ax=None, **kwargs):
        """Plot residual vertical velocity spectrum."""
        stacking_times_ix = self.times_to_indices(self.get_time_knots(self.stacking_velocity))
        stacking_velocities_ix = np.full_like(stacking_times_ix, (self.n_margins - 1) / 2)
        stacking_velocity_ix = (stacking_times_ix, stacking_velocities_ix)

        velocity_bounds_ix = None
        if acceptable_margin is not None:
            bounds = [-acceptable_margin, acceptable_margin]
            left_ix, right_ix = np.interp(bounds, self.margins, np.arange(self.n_margins))
            velocity_bounds_ix = ([0, self.n_times - 1], left_ix, right_ix)

        super()._plot(title=title, x_label="Relative velocity margin, %", x_ticklabels=self.margins * 100,
                      x_ticker=x_ticker, y_ticklabels=self.times, y_ticker=y_ticker, ax=ax, grid=grid,
                      stacking_velocity_ix=stacking_velocity_ix, velocity_bounds_ix=velocity_bounds_ix,
                      colorbar=colorbar, **kwargs)
        return self

    @plotter(figsize=(10, 9))
    def plot(self, *, acceptable_margin=None, title=None, interactive=False, **kwargs):
        """Plot residual vertical velocity spectrum. The plot always has a vertical line in the middle, representing
        the stacking velocity it was calculated for.

        Parameters
        ----------
        acceptable_margin : float, optional
            Defines an area around central stacking velocity that will be highlighted on the spectrum plot as
            `stacking_velocity(t)` * (1 +- `acceptable_margin`) for each time `t`. May be used for visual quality
            control of stacking velocity picking by setting this value low enough and checking that local maximas of
            velocity spectrum corresponding to primaries lie inside the highlighted area.
        title : str, optional
            Plot title. If not provided, equals to stacked lines "Residual Velocity Spectrum" and coherency func name.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the y-axis; see `.utils.set_ticks` for more details.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the residual velocity spectrum plot.
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the residual velocity spectrum values by given quantile.
        n_levels : int, optional, defaults to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        interactive : bool, optional, defaults to `False`
            Whether to plot residual velocity spectrum in interactive mode. This mode also plots the gather used to
            calculate the residual velocity spectrum. Clicking on residual velocity spectrum highlights the
            corresponding hodograph on the gather plot and allows performing NMO correction of the gather with the
            selected velocity. Interactive plotting must be performed in a JupyterLab environment with the
            `%matplotlib widget` magic executed and `ipympl` and `ipywidgets` libraries installed.
        sharey : bool, optional, defaults to True, only for interactive mode
            Whether to share y axis of residual velocity spectrum and gather plots.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.

        Returns
        -------
        velocity_spectrum : ResidualVelocitySpectrum
            Self unchanged.
        """
        if title is None:
            title = f"Residual Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        return super().plot(interactive=interactive, acceptable_margin=acceptable_margin, title=title, **kwargs)

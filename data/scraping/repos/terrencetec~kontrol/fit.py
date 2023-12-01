"""Fitting tools for system models and noise.
"""
from scipy.optimize import minimize, differential_evolution
from inspect import signature
import numpy as np
from ..utils import zpk

def noise_fit(noise_model, f, noise_data, weight=None, x0=None, **kwargs):
    """Noise model fit, follow the argument format of scipy.optimize.curve_fit

    The fitting is done by minimizing the mean-squared whitened error. The
    errors are whitened by the noise data.

    Parameters
    ----------
    noise_model: function
        The noise function in the form of noise_model(f, a, b, c, \.\.\.),
        where a, b, c, \.\.\. are the parameters defining the noise models.
        Example models are in kontrol.model.noise.lvdt_noise() and
        kontrol.model.noise.geophone_noise(). The number of parameters will
        be estimated by introspection (by the number of commas).
    f: array_like
        The frequency axis of the noise
    noise_data: array_like
        The noise data to be fitted. Must have same length as f.
    weight: array_like, optional
        Weightings in frequency domain that will be multiplied to the
        residues before summing. This can be used to filter unwanted data
        or to emphasize particular frequency regions.
    x0: array_like, optional
        Initial guess for of the noise model parameters. If not specified,
        it will be default to ones.
    \*\*kwargs:
        keyword arguments that will be passed to scipy.optimize.minimize.

    Returns
    -------
    args: numpy.ndarray
        The arguments of the noise_model. The fitted noise model can be
        called by noise_model(f, \*args).

    Notes
    -----
    The cost function is defined by the summation of ((noise_model(n) -
    noise data(n))/noise_data(n))^2)^1/2*weight(n)
    """
    if weight is None:
        weight = np.ones_like(noise_data)
    if x0 is None:
        no_of_params = str(signature(noise_model)).count(',')
        x0 = np.ones(no_of_params)
    def cost(args):
        return(sum(np.sqrt((noise_model(f, *args)/noise_data
            - np.ones_like(noise_data))**2)*weight))
    res = minimize(cost, x0, options={'disp':True},
        method='Nelder-Mead', **kwargs)
    args = res.x
    return args

def make_weight(x, *segments, default_weight=1.):
    """Make weighting functions for data fitting

    Parameters
    ----------
    x: list or np.ndarray
        The data points for evaluation
    *segments: tuples of (tuple of (float, float), float)
        Set weights values for segments of the data.
        The first entry specify the bound of the segment.
        The second entry specify the weight of the segment.
        Use np.inf for unbounded segments.
    default_weight: float, optional
        The default value of the weighting function.
        Defaults to be 1.

    Returns
    -------
    weight: np.ndarray
        The weighting function as specfied.
    """

    weight = np.ones_like(x) * default_weight

    for seg in segments:
        lower = seg[0][0]
        upper = seg[0][1]
        weight_val = seg[1]
        if lower > upper:
            _ = lower
            lower = upper
            upper = _
        mask_bool = (x >= lower) & (x <= upper)
        mask_value = mask_bool * weight_val
        weight *= np.logical_not(mask_bool)
        weight += mask_value

    return weight

def noise2zpk(f, noise_data, x0=None, bounds=None, max_order=20, weight=None):
    """ Noise spectrum regression using zpk defined transfer function.

    Parameters
    ----------
    f: numpy.ndarray
        Frequency axis of the noise spectrum. In Hz.
    noise_data: numpy.ndarray
        The amplitude/amplitude spectral density of the noise.
    x0: numpy.ndarray, optional
        Initial guess for of the noise model parameters. If not specified,
        zeros and poles will be default to logrithmic center of f, gain
        will be defaulted to noise_data[0]
    bounds: tuple of (float, float), optional
        The bounds for the zeros and poles in unit of Hz.
        This will default to (min(f) - 1 decade, max(f + 1 decade),
        if not specified. The bounds for the gain goes
        to the last entry and is defaulted to be
        (noise_data[0]*1e-6, noise_data[0]*1e6)
    max_order: int, optional
        The maximum number of zeros and poles. Note that the number
        of zeros and poles are the same in this regression.
        Defaults to be 20, maximum allowable order in foton.
    weight: numpy.ndarray, optional
        Weightings in frequency domain that will be multiplied to the
        residues before summing. This can be used to filter unwanted data
        or to emphasize particular frequency regions.

    Returns
    -------
    noise_zpk: control.xferfcn.TransferFunction
        The regressed transfer function with magnitude profile
        matching the noise spectrum specified.

    Notes
    -----
    The reason why the numbers of zeros and poles are the same here
    is because the magnitude have to be bounded, i.e. flat at
    very low and very high frequencies, for H2 and H-infinity synthesis
    to work. This doesn't really matter if the flattness only happens at
    irrelevant frequencies. We can always modify the final filter by
    removing the corresponding zeros/poles that make the filter flat.
    """

    if x0 is None:
        log_center = (np.log10(max(f)) + np.log10(min(f))) / 2
        log_center = 10**log_center
        # x0 = np.ones(max_order*2) * log_center
        # x0 = np.ones(max_order*2) * min(f)
        x0 = np.random.choice(np.logspace(np.log10(min(f)), np.log10(max(f)), len(f)), max_order*2)
        x0 = np.append(x0, noise_data[0])
#     print(len(x0))
    if bounds is None:
        bounds = [(min(f)*0.1, max(f)*10)]*max_order*2
    else:
        bounds = [(min(bounds), max(bounds))]*max_order*2
    bounds.append([noise_data[0]*1e-12, noise_data[0]*1e12])
#     print(len(bounds))
    if weight is None:
        weight = np.ones_like(noise_data)

    options = {
        'xtol':1e-7,
        'ftol':1e-8,
    }
    res = minimize(_cost, x0=x0, args=(f, noise_data, weight), bounds=bounds, method='Powell', options={'disp':True, **options})
    # res = differential_evolution(_cost, args=(f, noise_data, weight), bounds=bounds, disp=True, tol=1e-6, workers=-1, mutation=(0,1))
#     print(bounds)
#     print(res.x)
    noise_zpk = _args2zpk(res.x)
    return noise_zpk

def _args2zpk(args):
    """ Convert a list of arguments to zpk transfer function.

    Parameters
    ----------
    args: list of floats
        Length must be odd. The last number is gain. The first half of
        the rest are zeros, and the second half are poles.
    returns
    -------
    control.xferfcn.TransferFunction
        The transfer function defined with the arguments.
    """

    zeros = args[0:int(len(args)/2)]
    poles = args[int(len(args)/2):len(args)-1]
    gain = args[-1]
    return(zpk(zeros, poles, gain))

def _cost(args, f, noise_data, weight):
    zpk_fit = _args2zpk(args)
    mag_fit = np.abs(zpk_fit.horner(2*np.pi*1j*f)[0][0])
    residue = np.sum(((mag_fit-noise_data)  * weight)**2)
    return residue

def vinagre_weight(omega, normalize=True, log=True):
    """Vinagre's weight [1]_, with options to normalize and use logrithmic.

    Parameters
    ----------
    omega: array
        Frequency.
    normalize: boolean, optional
        Normalize the weights. Defaults to True.
    log: boolean, optional
        Return the logrithmic version (Basically returns the log weight and
        shift it vertically to zero minimum.).

    Returns
    -------
    weight: array
        The Vinagre's weight.

    Referernce
    ----------
    .. [1]
        Valério, Duarte & Ortigueira, Manuel & Costa, José. (2008).
        Identifying a Transfer Function From a Frequency Response.
        Journal of Computational and Nonlinear Dynamics -
        J COMPUT NONLINEAR DYN. 3. 7-1077. 10.1115/1.2833906.
    """
    weight=np.ones_like(omega)
    for p in range(len(omega)):
        if p == 0:
            weight[p] = (omega[1]-omega[0])/(2*omega[0]**2)
        elif p == len(omega)-1:
            weight[p] = (omega[p]-omega[p-1])/(2*omega[p]**2)
        else:
            weight[p] = (omega[p+1]-omega[p-1])/(2*omega[p]**2)
    if log:
        weight = np.log10(weight) - min(np.log10(weight))
    if normalize:
        weight /= max(weight)
    return weight


def coherence_weight(coh, threshold, invert=False):
    """ Make binary weight from coherence

    Pararmeters
    -----------
    coh: array
        The coherence.
    threshold: float
        The threshold from 0 to 1
    invert: boolean, optional
        If not invert, the weight is 1 when coh is greater than threshold and
        zero otherwise. If invert, the weight is inverted.

    Returns
    -------
    weight: array
        The coherence weight.
    """
    weight = np.ones_like(coh)
    weight = coh>threshold
    if invert:
        weight = weight==0
    return weight.astype(int)


def one_on_f_weight(f, normalize=True):
    """ Literally 1/f weight. Useful for fitting linspace data in logspace.

    Parameters
    ----------
    f: array
        Frequency
    normalize: boolean, optional
        Normalized the weight to [0, 1].
        Defaults to True.

    Returns
    -------
    weight: array
        The 1/f weight.
    """
    weight = 1/f
    if normalize:
        weight /= max(weight)
    return(weight)

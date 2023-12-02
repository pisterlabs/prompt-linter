#!/usr/bin/env python3
"""
========
Overview
========
Python library of filtering functions.

=========
Changelog
=========
v1.0 20220316 Yu Morioshita
 - Original implementation
"""

import sys
import warnings
import numpy as np
import statsmodels.api as sm
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft


# %%
def rolling_window2d(A, win_sz):
    """
    Return rolling window of input ndarray.

    Inputs:
        A      : Input 2D ndarray (N x M)
        win_sz : Window size of rolling window (W)

    Return:
        Arol   : 2D Rolling window (N-W+1) x (M-W+1) x W x W

    Note: Edge (W-1) is cut in retuern. To get the return with the same size of A in 1st/2nd dimension, expand A with the W/2 beforehand.

    """
    shape = (A.shape[0]-win_sz+1, A.shape[1]-win_sz+1, win_sz, win_sz)
    strides = A.strides+A.strides
    return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides).copy()


# %% nanmedian2d
def nanmedian2d_simple(A, win_sz=7, preserve_nan_flag=True):
    """
    2D median filter with taking into account nan.

    Inputs:
        A : Input ndarray
        win_sz : Window size of 2D median filter (Default: 7).
                 Must be odd number.
        preserve_nan_flag : If True, Amed contains nan where A is nan. If False, points where A is nan are filled with filtered value. (Default: True)

    Return:
        Amed : Filtered ndarray
    """
    # Check win_sz
    if not np.mod(win_sz-1, 2) == 0:
        print("\nERROR:", file=sys.stderr, end='')
        print("\nwin_sz ({}) must be odd number!!".format(win_sz))
        return 1
    win_sz_half = int((win_sz-1)/2)

    # Simple iteration
    length, width = A.shape  # read dimension
    Amed = np.zeros_like(A)

    for l in range(length):
        l_start = l - win_sz_half if l >= win_sz_half else 0
        for w in range(width):
            w_start = w - win_sz_half if w >= win_sz_half else 0

            Amed[l, w] = np.nanmedian(
                A[l_start:l+win_sz_half+1, w_start:w+win_sz_half+1].ravel())
            # not +1 but +2??

        if np.mod(l, 1000) == 0:
            print('Finished {0} / {1}'.format(l, length))

    # Fill back nan
    if preserve_nan_flag:
        Amed[np.isnan(A)] = np.nan

    return Amed


# %% nanmedian2d
def nanmedian2d(A, win_sz=7, preserve_nan_flag=True, rm_ratio=0.5):
    """
    2D median filter with taking into account nan. Fast by using rolling window.

    Inputs:
      A : Input ndarray
      win_sz : Window size of 2D median filter (Default: 7).
               Must be odd number.
      preserve_nan_flag
             : If True, Amed contains nan where A is nan.
               If False, points where A is nan are filled with filtered value.
               (Default: True)
      rm_ratio : Remove a sparse pixel which have smaller number of valid pixels
                 than this rate in a window (Default: 0.5)

    Return:
        Amed : Filtered ndarray
    """
    # Check win_sz
    if not np.mod(win_sz-1, 2) == 0:
        print("\nERROR:", file=sys.stderr, end='')
        print("\nwin_sz ({}) must be odd number!!".format(win_sz))
        return 1
    win_sz_half = int((win_sz-1)/2)

    # Get rolling window matrix filled with nan at edge
    length, width = A.shape  # read dimension
    Aexp = np.full((length+win_sz-1, width+win_sz-1), np.nan, dtype='float32')
    Aexp[win_sz_half:-win_sz_half, win_sz_half:-win_sz_half] = A
    Arol = rolling_window2d(Aexp, win_sz)

    # Nanmedian rolling window matrix
    Arol = Arol.reshape(
        (Arol.shape[0], Arol.shape[1], Arol.shape[2]*Arol.shape[3]))
    Amed = np.nanmedian(Arol, axis=2)

    # Remove sparse pixels and fill back nan
    if preserve_nan_flag:
        Anum = np.sum(~np.isnan(Arol), axis=2)
        n_pt_win = win_sz*win_sz
        Amed[Anum < n_pt_win*rm_ratio] = np.nan
        Amed[np.isnan(A)] = np.nan

    return Amed


# %%
def fillhole0(data):
    """
    Fill hole of 0 by average of surrounding 8 pixels.

    Inputs:
        data : Input ndarray (nodata filled with 0)

    Returns:
        data_filled : Best fit plain with the same demention as A

    """
    length, width = data.shape
    data_ex = np.zeros((length+2, width+2), dtype=np.float32)
    data_ex[1:length+1, 1:width+1] = data
    n_data_ex = np.int16(data_ex != 0)  # 1 if exist, 0 if no data

    # Average 8 srrounding pixels. [1, 1] is center
    pixels = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
    _data = np.zeros_like(data)
    _n_data = np.zeros_like(data)

    for pixel in pixels:
        # Adding data and number of data
        _data = _data + data_ex[pixel[0]:length +
                                pixel[0], pixel[1]:width+pixel[1]]
        _n_data = _n_data + n_data_ex[pixel[0]:length+pixel[0], pixel[1]:width+pixel[1]]

    _n_data[_n_data == 0] = 1  # avoid 0 division
    _data = _data/_n_data

    # Fill hole
    data[data == 0] = _data[data == 0]

    return data


# %% fit2d
def fit2d(A, w=None, deg="1"):
    """
    Estimate best fit plain with indicated degree of polynomial.

    Inputs:
        A : Input ndarray (can include nan)
        w : Wieight (1/std**2) for each element of A (with the same dimention as A)
        deg : degree of polynomial of fitting plain
         - 0   -> a (no ramp, just unbias)
         - 1   -> a+bx+cy (ramp)
         - bl  -> a+bx+cy+dxy (biliner)
         - 2   -> a+bx+cy+dxy+ex**2_fy**2 (2d polynomial)

    Returns:
        Afit : Best fit plain with the same demention as A
        m    : set of parameters of best fit plain (a,b,c...)

    """

    # Make design matrix G
    length, width = A.shape  # read dimension
    Xgrid, Ygrid = np.meshgrid(
        np.arange(width), np.arange(length))  # mesh grid

    if str(deg) == "0":
        G = np.stack((np.ones((length*width)))).T
    elif str(deg) == "1":
        G = np.stack((np.ones((length*width)),
                     Xgrid.flatten(), Ygrid.flatten())).T
    elif str(deg) == "bl":
        G = np.stack((np.ones((length*width)), Xgrid.flatten(),
                     Ygrid.flatten(), Xgrid.flatten()*Ygrid.flatten())).T
    elif str(deg) == "2":
        G = np.stack((np.ones((length*width)), Xgrid.flatten(), Ygrid.flatten(),
                     Xgrid.flatten()*Ygrid.flatten(), Xgrid.flatten()**2, Ygrid.flatten()**2)).T
    else:
        print('\nERROR: Not proper deg ({}) is used\n'.format(deg), file=sys.stderr)
        return False

#    ### Weight G and A.flatten() by w.flatten()
#    ### if phi = Gm with Q, m = inv(G.T*inv(Q)*G)*G.T*inv(Q)*phi
#    ### inv(Q) = 1./Q = w because Q have no covariance (only diagonal) in this case!
#    ### i.e, G.T -> w.flatten()*G.T and phi -> w.flatten()*phi
#    ### I dont understand why sqrt(w) but it is correct according to comparison with  statsmodels
#    if w is None:
#        w = np.array([1])
#    Gw = G*np.sqrt(np.float64(w.flatten()[:,np.newaxis]))
#    A1w = A.flatten()*np.sqrt(np.float64(w.flatten()))
#
#    ### Least square and calc best fit plain
#    m = np.linalg.lstsq(Gw,A1w,rcond=None)[0]
#    Afit = np.float32(np.dot(G,m).reshape((length,width)))

    # Handle nan by 0 padding and 0 weight
    # Not drop in sm because cannot return Afit
    if np.any(np.isnan(A)):
        bool_nan = np.isnan(A)
        A = A.copy()  # to avoid change original value in main
        A[bool_nan] = 0
        if w is None:
            w = np.ones_like(A)
        w = w.copy()  # to avoid change original value in main
        w[bool_nan] = 0

    # Invert
    if w is None:  # Ordinary LS
        results = sm.OLS(A.ravel(), G).fit()
    else:  # Weighted LS
        results = sm.WLS(A.ravel(), G, weights=w.ravel()).fit()

    m = results.params
    Afit = np.float32(results.predict().reshape((length, width)))

    return Afit, m


# %% fit2d_dem
def fit2d_dem(A, dem, w=None, deg="0"):
    """
    Estimate best fit topography-correlated phase (with a 2D plain).

    Inputs:
        A  : Input ndarray (can include nan)
        dem: DEM with the same demention as A (not include nan)
        w  : Wieight (1/std**2) for each element of A (with the same dimention as A)
        deg : degree of polynomial of fitting plain
         - 0  -> ah+b (no deramp)
         - 1  -> ah+b+cx+dy (ramp)
         - bl -> ah+b+cx+dy+exy (biliner)
         - 2  -> ah+b+cx+dy+exy+fx**2_gy**2 (2d polynomial)

    Returns:
        Afit : Best fit phase with the same demention as A
        m    : set of parameters of best fit plain (a,b,c...)

    """

    # Make design matrix G
    length, width = A.shape  # read dimension
    Xgrid, Ygrid = np.meshgrid(
        np.arange(width), np.arange(length))  # mesh grid

    if str(deg) == "0":
        G = np.stack((dem.flatten(), np.ones((length*width)))).T
    elif str(deg) == "1":
        G = np.stack((dem.flatten(), np.ones((length*width)),
                     Xgrid.flatten(), Ygrid.flatten())).T
    elif str(deg) == "bl":
        G = np.stack((dem.flatten(), np.ones((length*width)), Xgrid.flatten(),
                     Ygrid.flatten(), Xgrid.flatten()*Ygrid.flatten())).T
    elif str(deg) == "2":
        G = np.stack((dem.flatten(), np.ones((length*width)), Xgrid.flatten(), Ygrid.flatten(),
                     Xgrid.flatten()*Ygrid.flatten(), Xgrid.flatten()**2, Ygrid.flatten()**2)).T
    else:
        print('\nERROR: Not proper deg ({}) is used\n'.format(deg), file=sys.stderr)
        return False

    # Handle nan by 0 padding and 0 weight
    # Not drop in sm because cannot return Afit
    if np.any(np.isnan(A)):
        bool_nan = np.isnan(A)
        A = A.copy()  # to avoid change original value in main
        A[bool_nan] = 0
        if w is None:
            w = np.ones_like(A)
        w = w.copy()  # to avoid change original value in main
        w[bool_nan] = 0

    # 0 weight at dem=0
    if np.any(dem == 0):
        bool_dem0 = (dem == 0)
        if w is None:
            w = np.ones_like(A)
        w = w.copy()  # to avoid change original value in main
        w[bool_dem0] = 0

    # Invert
    if w is None:  # Ordinary LS
        results = sm.OLS(A.ravel(), G).fit()
    else:  # Weighted LS
        results = sm.WLS(A.ravel(), G, weights=w.ravel()).fit()

    m = results.params
    Afit = np.float32(results.predict().reshape((length, width)))

    return Afit, m


# %% fit2d_pt
def fit2d_pt(A, xy, w=None, deg="1"):
    """
    Estimate best fit plain with indicated degree of polynomial.

    Inputs:
        A  : Input N vector (can include nan)
        xy : Nx2 Point locations
        w  : Wieight (1/std**2) for each element of A
        deg : degree of polynomial of fitting plain
         - 1   -> a+bx+cy (ramp)
         - bl  -> a+bx+cy+dxy (biliner)
         - 2   -> a+bx+cy+dxy+ex**2_fy**2 (2d polynomial)

    Returns:
        Afit : Best fit plain with the same demention as A
        m    : set of parameters of best fit plain (a,b,c...)

    """

    # Make design matrix G
    n_pt = len(A)

    if str(deg) == "1":
        G = np.stack((np.ones((n_pt)), xy[:, 0], xy[:, 1])).T
    elif str(deg) == "bl":
        G = np.stack(
            (np.ones((n_pt)), xy[:, 0], xy[:, 1], xy[:, 0]*xy[:, 1])).T
    elif str(deg) == "2":
        G = np.stack((np.ones(
            (n_pt)), xy[:, 0], xy[:, 1], xy[:, 0]*xy[:, 1], xy[:, 0]**2, xy[:, 1]**2)).T
    else:
        print('\nERROR: Not proper deg ({}) is used\n'.format(deg), file=sys.stderr)
        return False

    # Handle nan by 0 padding and 0 weight
    # Not drop in sm because cannot return Afit
    if np.any(np.isnan(A)):
        bool_nan = np.isnan(A)
        A = A.copy()  # to avoid change original value in main
        A[bool_nan] = 0
        if w is None:
            w = np.ones_like(A)
        w = w.copy()  # to avoid change original value in main
        w[bool_nan] = 0

    # Invert
    if w is None:  # Ordinary LS
        results = sm.OLS(A, G).fit()
    else:  # Weighted LS
        results = sm.WLS(A, G, weights=w).fit()

    m = results.params
    Afit = np.float32(results.predict())

    return Afit, m


# %%
def gausfil_fcx(fcx, x_stddev, y_stddev, coh=1, preserve_nan_flag=False, fft_flag=True):
    '''
    Filter float complex data (wrapped phase) by 2D Gaussian kernel.

    Input fcx is ndarray of complex64.

    Filtered output is ndarray of complex64.

    [x|y]_stddev are standard deviation of the Gaussian in [x|y].
    Size of Gaussian kernel is 8 * stddev.

    If coherence file (ndarray with float32, same size as fcx) is indicated, it is used as weight (if not, equal weight).
    The weight is calculated by 1/sigma^2=2*c^2/(1-c^2) [Spaans et al., 2016].

    If preserve_nan_flag is True, filtered result at fcx=0 is also 0.
    Otherwise (False), the filtered result at fcx=0 is interpolated and amplitude is filled with epsilon.

    If fft_flag is True, convolve_fft is used, much faster for large kernel.
    Otherwise convolve is used.
    '''

    # Settings
    fill_value = np.nan  # not sure this affects result
    boundary_flag = 'extend'  # smooth edge, only for convolve, not convolve_fft
#    allow_huge = False
    allow_huge = True

    # Calc weight from coherence
    if not np.isscalar(coh):
        weight = 2*coh**2/(1-coh**2)
    else:
        weight = 1

    # Fill 0 of fcx with nan for correct filter
#    fcx[fcx==0] = np.nan
    # Caution! Substituting to element of list affect out side of function!
    # Substituting object itself doesnt affect out side of function.

    # Normalize amplitude and multiply weight
    fcx_w = np.exp(1j*np.angle(fcx))*weight

    # Make kernel
    kernel = Gaussian2DKernel(x_stddev, y_stddev)

    # Separate into real ang imag
    real = fcx_w.real.copy()
    imag = fcx_w.imag.copy()

    # Fill 0 of real and imag respectively with nan for correct filter
    # nan pad for fcx return nan+0j, which couse goast phase by 0 imag
    real[fcx == 0] = np.nan
    imag[fcx == 0] = np.nan

    if fft_flag:
        with warnings.catch_warnings():  # To silence warning
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', FutureWarning)
            real_filt = convolve_fft(
                real, kernel, preserve_nan=preserve_nan_flag, fill_value=fill_value, allow_huge=allow_huge)
            imag_filt = convolve_fft(
                imag, kernel, preserve_nan=preserve_nan_flag, fill_value=fill_value, allow_huge=allow_huge)
    else:
        real_filt = convolve(
            real, kernel, boundary=boundary_flag, preserve_nan=preserve_nan_flag)
        imag_filt = convolve(
            imag, kernel, boundary=boundary_flag, preserve_nan=preserve_nan_flag)

    fcx_filt = np.complex64(real_filt+1j*imag_filt)

    # Retrieve imput amplitude
    amp = np.abs(fcx)
    if not preserve_nan_flag:
        amp[amp == 0] = sys.float_info.epsilon  # to keep filtered phase

#    fcx_filt = fcx_filt/np.abs(fcx_filt)*amp
    fcx_filt = np.exp(1j*np.angle(fcx_filt))*amp

    fcx_filt[np.isnan(fcx_filt)] = 0  # fill nan with 0 at outisde of kernel

    return fcx_filt

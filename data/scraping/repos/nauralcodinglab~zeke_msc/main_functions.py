import numpy as np
import sys
from scipy.integrate import simps
from scipy.signal import coherence
from cell_params import params
from lba_params import params as params2

# MUST CHANGE '3' IN CELL cell_params3 and lba_params3 TO THE APPROPRIATE NUMBER FOR THE DESIRED FIGURE

# Functions ------------------------------------------------------------------------------------------------------------


def sigmoid0(x):
    """
    no explanation
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def se_kernel(g_bar, ta_rise, ta_decay, N, dt=1):
    """
    Superposition of exponentials synaptic kernel (see Ch. 3.1 Neural Dynamics eqn 3.3). In this function we only have
    one decay timescale.
    :param g_bar: scalar, scale factor (pS)
    :param ta_rise: scalar, time constant for depolarization (ms)
    :param ta_decay: scalar, time constant for repolarization (ms)
    :param N: integer, length of kernel in time steps (timesteps)
    :param dt: scalar, length of timestep (ms)
    :return: (1 x length) array, synaptic kernel
    """

    t = np.arange(0, N, dt)

    return g_bar * (1 - np.exp(-t/ta_rise)) * np.exp(-t/ta_decay)


def get_kernel2(beta, tau, dt=1, k='exponential', sig_tau=1):
    """
    calculates kernel vector
    :param beta: param
    :param tau: length of kernel (and spiketrain) in time units; tau = n_k*dt
    :param dt: time step (same units as above) for spiketrain and kernel
    :param k: kernel function. Can be 'exponential' or 'sigmoid'
    :param sig_tau: time constant for sigmoid kernel (hyperparameter)
    :return:
    """

    if k == 'exponential':
        return np.insert(np.exp(- beta * np.arange(0, tau, dt)), 0, 0)[:-1]
    elif k == 'sigmoid':
        return np.insert(sigmoid0((-(np.arange(0, tau, dt) - beta))/sig_tau), 0, 0)[:-1]
    else:
        print('error get_kernel2: invalid value for k')
        sys.exit()


def decoder(spikes, thresh, stp='depression'):

    """
    Takes in an iid array of n spiketrain from method4() and decodes it using an STP rule represented by a
    step function. Assumes heaviside step function as w(\cdot). ASSUMES THRESHOLD IS THE SAME!!
    :param spikes:    [n_cells x L] numpy array storing input spiketrain,
    :param thresh: threshold for weight function in timesteps,
    :param stp:    string, depression if it decodes spikes above ISI threshold, facilitation if it decodes spikes below,
    :return: ebd:  Need to return it to work, apparently...(?)
    """

    n = spikes.shape[0]
    n_t = thresh.shape[0]
    extra = np.max(thresh)

    # Decode events
    if np.any(spikes != 0):
        spikes = np.insert(spikes, 0, np.zeros((extra + 5, n)), axis=-1)
        spikes[:, 0] = 1
        spikes = spikes.flatten()

        times = np.nonzero(spikes)[0]              # Get spike times in units of time steps
        diffs = np.insert(np.diff(times), 0, 0)                    # Get ISIs in units of time steps

        # Decode events
        if stp == 'both':
            e = np.tile(spikes, (n_t, 1))
            b = np.tile(spikes, (n_t, 1))
            del spikes
            for i in range(n_t):
                not_event = times[diffs < thresh[i]]
                e[i, not_event] = 0                        # STD rule removes spikes with pre-ISI less than thresh
                b[i, :] = b[i, :] ^ e[i, :]        # STF rule removes spikes with pre-ISI greater than thresh

            e = np.array(np.split(e, n, axis=-1))[:, :, extra + 5:]
            b = np.array(np.split(b, n, axis=-1))[:, :, extra + 5:]

            return np.sum(e, axis=0), np.sum(b, axis=0)

        else:
            spikes = np.tile(spikes, (n_t, 1))
            if stp == 'depression':
                for i in range(n_t):
                    not_event = times[diffs < thresh[i]]
                    spikes[i, not_event] = 0             # STD rule removes spikes with pre-ISI less than thresh
            elif stp == 'facilitation':
                for i in range(n_t):
                    not_burst = times[diffs > thresh[i]]
                    spikes[i, not_burst] = 0             # STF rule removes spikes with pre-ISI greater than thresh
            else:
                sys.exit('stp rule not understood')

            spikes = np.array(np.split(spikes, n, axis=-1))[:, :, extra + 5:]

            return np.sum(spikes, axis=0)


def decoder2(spikes, theta, N, dt=1, sig_tau=1):

    """
    Takes in an iid array of n spiketrain from method4, 6() and decodes it using an STP rule represented by a weight function
    with parameters theta. CURRENTLY HARDCODED FOR SIGMOID
    :param spikes: [n_cells x L] numpy array storing input spiketrain,
    :param theta: params ya'll. (a, tau, b)
    :param N: length of kernel in timesteps
    :param dt: time step
    :param sig_tau sigmoid time constant
    :param type: determines if decoding burst or event
    :return: decoded signal (STP rule applied and summed)
    """

    spikes = spikes.transpose()
    kappa = get_kernel2(theta[1], N * dt, k='sigmoid', sig_tau=sig_tau) #exp_kernel(theta[0], theta[1], theta[2], N, dt)
    weights = np.zeros(spikes.shape)

    for ix in range(spikes.shape[1]):
        weights[:, ix] = np.convolve(spikes[:, ix], kappa, mode='full')[:-(N - 1)]

    spikes = sigmoid0(theta[0] * weights + theta[2]) * spikes

    return np.sum(spikes.transpose(), axis=0)


def decode_trial(spikes, thresh, which='event'):
    """
    Decodes spikes inside trial_loop function
    Produces A (2 x L) and U_e,b (thresh x L) where L is length of simulation in timesteps.
    :param spikes: (2 x n_cells x L) array of spikes
    :param thresh: array, event, burst decoding thresholds to try in units of timesteps
    :param which: 'burst', or 'event'
    """

    if which == 'event':
        a_e = np.sum(spikes[0, :, :], axis=0)
        u_e = decoder(np.any(spikes, axis=0), thresh, stp='depression')

        return a_e.astype(float), u_e.astype(float)
    elif which == 'burst':
        a_b = np.sum(spikes[-1, :, :], axis=0)
        u_b = decoder(np.any(spikes, axis=0), thresh, stp='facilitation')

        return a_b.astype(float), u_b.astype(float)
    else:
        print('error decode_trial: which not understood')
        sys.exit()


def decode_trial2_bc(spikes, thresh_e, thresh_b, s_t):
    """
    decode_trial2 but for both channels at once
    Produces A (2 x L) and U_e,b (thresh x L) where L is length of simulation in timesteps.
    :param spikes: (2 x n_cells x L) array of spikes
    :param thresh_e: array, event, burst decoding thresholds to try in units of timesteps
    :param thresh_b: array, event, burst decoding thresholds to try in units of timesteps
    :param s_t: decoding time constant
    """
    if thresh_e.shape[0] != thresh_b.shape[0]:
        print('error decode_trial2c: require same number of thresholds for e, b')
        sys.exit()

    nt = thresh_e.shape[0]
    u_e = np.zeros((spikes.shape[-1], nt))
    u_b = np.zeros((spikes.shape[-1], nt))
    a = np.sum(spikes, axis=1).astype(float)

    for ix in range(nt):
        u_e[:, ix] = decoder2(np.any(spikes[:2, :, :], axis=0), thresh_e[ix, :], N=650, sig_tau=s_t)
        u_b[:, ix] = decoder2(np.any(spikes[:2, :, :], axis=0), thresh_b[ix, :], N=650, sig_tau=s_t)

    return a[0, :], a[-1, :], u_e.transpose(), u_b.transpose()


def decode_trial_bc(spikes, thresh):
    """
    decode trial but for two channels at once
    Produces A (2 x L) and U_e,b (thresh x L) where L is length of simulation in timesteps.
    :param spikes: (2 x n_cells x L) array of spikes
    :param thresh: array, event AND burst decoding thresholds to try in units of timesteps
    """

    a = np.sum(spikes, axis=1)
    u_e, u_b = decoder(np.any(spikes, axis=0), thresh, stp='both')

    return a[0, :].astype(float), a[-1, :].astype(float), u_e.astype(float), u_b.astype(float)


def decode_trial2(spikes, thresh, sig_tau=1, B=False):
    """
    Decodes spikes for variable length stuff. CURRENTLY N IS HARDCODED
    Produces A (2 x L) and U_e,b (thresh x L) where L is length of simulation in timesteps.
    :param spikes: (2 x n_cells x L) array of spikes
    :param thresh: n_t x 3 array, event AND burst decoding thresholds to try in units of timesteps
    :param sig_tau sigmoid time constant
    :param B: set to True if decoding bursts
    :return: perfectly decoded event/burst activities (L length arrays) and imperfectly decoded (nt x L arrays)
    """

    nt = thresh.shape[0]
    u = np.zeros((spikes.shape[-1], nt))
    for ix in range(nt):
        u[:, ix] = decoder2(np.any(spikes[:2, :, :], axis=0), thresh[ix, :], N=650, sig_tau=sig_tau)

    if B:
        return np.sum(spikes[-1, :, :], axis=0).astype(float), u.transpose()
    else:
        return np.sum(spikes[0, :, :], axis=0).astype(float), u.transpose()


def spikes_to_potential(spikes, sk):
    """
    takes array of spike counts and converts them to a time series of conductance or voltage changes based on
    a synaptic kernel
    :param spikes: len(time) array of spikecounts
    :param sk: (1 x n_kernel) array that is the synaptic kernel
    :param nsk: int, length of sk kernel
    :return: (trials x time) array of synaptic potential/conductance, or whatever
    """
    array = np.zeros((spikes.shape))
    nsk = len(sk)

    for ix in range(array.shape[0]):
        array[ix, :] = np.convolve(spikes[ix, :], sk, mode='full')[:-(nsk - 1)]

    return array


def syn_filter_align(activity, syn_kernel, E_ps, n, lag, scale_bs=1, B=False):
    """
    filters and organizes activities (A, or U type things) into downstream stuff (u_e, u_b). Does not calc fraction
    :param activity: (thresh x L) array
    :param syn_kernel: (1 x n_sf) array (mV???)
    :param E_ps: scalar, post synaptic resting membrane potential (mV)
    :param n: number of cells
    :param lag: num of time steps that event signal lags burst signal
    :param scale_bs: scale factor for burst signal used in burst/event quotient. Keep at 1 if for events
    :param B: set to True if burst channel
    :return: u_e (n_et x L) array of synaptic time series for each decoding threshold used
    """

    if len(activity.shape) == 1:
        activity = np.expand_dims(activity, axis=0)

    if B:
        if lag == 0:
            u = (spikes_to_potential(activity, syn_kernel) * scale_bs) / n + E_ps
        else:
            u = (spikes_to_potential(activity[:, lag:], syn_kernel) * scale_bs) / n + E_ps
    else:
        if lag == 0:
            u = spikes_to_potential(activity, syn_kernel) / n + E_ps
        else:
            u = spikes_to_potential(activity[:, :-lag], syn_kernel) / n + E_ps

    return u


def get_lb(coh, f, max_freq):
    """
    calculates lower bound on mutual info using the coherence method
    :param coh: should be 2d
    :param f:
    :param max_freq: for integral
    :return:
    """

    iw = - np.log2(1 - coh)

    i = simps(iw[:, :max_freq], f[:max_freq], even='avg', axis=-1)

    return i, iw


def lower_bound(signal1, signal2, fs, f, nps, nov, FREQ_MAX):
    """
    :param signal1:
    :param signal2:
    :param fs:
    :param f:
    :param nps:
    :param nov:
    :param FREQ_MAX:
    :return:
    """

    coh = coherence(signal1, signal2, fs=fs, nperseg=nps, noverlap=nov, axis=-1)[1]

    l, lw = get_lb(coh, f, FREQ_MAX)

    return l, lw


def lba_ec(S, event_sig, thresh, s_t, burn, n_eta, n_cells, lag=6, decoder=2):
    """
    lower bound analysis event channel. Calculates lower bound from spiking data
    :param S:
    :param event_sig:
    :param thresh: threshold for events
    :param s_t:
    :param burn:
    :param n_eta:
    :param n_cells:
    :param lag:
    :param decoder:
    :return:
    """

    # GET PARAMS
    dt = params['dt']
    E_ps = params2['E_ps']
    fs = params2['fs']
    nps = params2['nps']
    nov = params2['nov']
    FREQ_MAX_e = params2['FREQ_MAX_e']
    g_bar = params2['g_bar']
    ta_rise = params2['ta_rise']
    ta_decay = params2['ta_decay']
    t_sf = params2['t_syn']

    # DEFINE VARIABLES
    f = np.arange(0, 500 + 1 / (nps / fs), 1 / (nps / fs))
    sf = se_kernel(g_bar, ta_rise, ta_decay, t_sf // dt, dt=dt)

    # DECODE SPIKETRAINS
    if decoder == 2:
        ae, ue = decode_trial2(S, thresh, sig_tau=s_t, B=False)
    elif decoder == 1:
        ae, ue = decode_trial(S, thresh, which='event')
    else:
        print('error syn_filter_align: invalid value for decoder')
        sys.exit()

    # CALCULATE INFORMATION LOWER BOUND
    if len(ue.shape) == 1:
        ue = np.expand_dims(ue, axis=0)

    ue = np.concatenate((ue, np.expand_dims(ae, axis=0)), axis=0)

    u_e = syn_filter_align(ue[:, burn:-n_eta], sf, E_ps, n_cells, lag, B=False)

    g_e = np.tile(event_sig[:-lag], (u_e.shape[0], 1))

    le, lwe = lower_bound(g_e, u_e, fs, f, nps, nov, FREQ_MAX_e)  # Only integrate up to 200Hz

    return le, lwe, u_e


def lba_bc(S, burst_sig, thresh, s_t, burn, n_eta, n_cells, u_e, lag=6, decoder=2):
    """
    lower bound analysis burst channel. Calculates lower bound from spiking data
    :param S:
    :param burst_sig:
    :param thresh: threshold for bursts
    :param s_t:
    :param burn:
    :param n_eta:
    :param n_cells:
    :param u_e: to divide by to get fraction
    :param lag:
    :param decoder:
    :return:
    """

    # GET PARAMS
    dt = params['dt']
    scale_bs = params2['scale_bs']
    E_ps = params2['E_ps']
    fs = params2['fs']
    nps = params2['nps']
    nov = params2['nov']
    FREQ_MAX_b = params2['FREQ_MAX_b']
    g_bar = params2['g_bar']
    ta_rise = params2['ta_rise']
    ta_decay = params2['ta_decay']
    t_sf = params2['t_syn']

    # DEFINE VARIABLES
    f = np.arange(0, 500 + 1 / (nps / fs), 1 / (nps / fs))
    sf = se_kernel(g_bar, ta_rise, ta_decay, t_sf // dt, dt=dt)

    # DECODE SPIKETRAINS
    if decoder == 2:
        ab, ub, = decode_trial2(S, thresh, sig_tau=s_t, B=True)
    elif decoder == 1:
        ab, ub = decode_trial(S, thresh, which='burst')
    else:
        print('error syn_filter_align: invalid value for decoder')
        sys.exit()

    # CALCULATE INFORMATION LOWER BOUND
    if len(ub.shape) == 1:
        ub = np.expand_dims(ub, axis=0)

    ub = np.concatenate((ub, np.expand_dims(ab, axis=0)), axis=0)

    u_b = syn_filter_align(ub[:, burn:-n_eta], sf, E_ps, n_cells, lag, scale_bs=scale_bs, B=True)

    u_p = u_b / u_e

    g_p = np.tile(burst_sig[:-lag], (u_p.shape[0], 1))

    lp, lwp = lower_bound(g_p, u_p, fs, f, nps, nov, FREQ_MAX_b)  # Only integrate up to 100Hz

    return lp, lwp


def lba_ao(S, event_sig, burst_sig, burn, n_eta, n_cells, lag=6):
    """
    lower bound analysis activity only. Calculates lower bound from spiking data
    :param S:
    :param burst_sig:
    :param event_sig
    :param burn:
    :param n_eta:
    :param n_cells:
    :param lag:
    :return:
    """

    # GET PARAMS
    dt = params['dt']
    scale_bs = params2['scale_bs']
    E_ps = params2['E_ps']
    fs = params2['fs']
    nps = params2['nps']
    nov = params2['nov']
    FREQ_MAX_e = params2['FREQ_MAX_e']
    FREQ_MAX_b = params2['FREQ_MAX_b']
    g_bar = params2['g_bar']
    ta_rise = params2['ta_rise']
    ta_decay = params2['ta_decay']
    t_sf = params2['t_syn']

    # DEFINE VARIABLES
    f = np.arange(0, 500 + 1 / (nps / fs), 1 / (nps / fs))
    sf = se_kernel(g_bar, ta_rise, ta_decay, t_sf // dt, dt=dt)

    # DECODE SPIKETRAINS
    a = np.sum(S, axis=1).astype(float)
    ae = a[0, :]
    ab = a[-1, :]

    # CALCULATE INFORMATION LOWER BOUND
    ae = np.expand_dims(ae, axis=0)
    ab = np.expand_dims(ab, axis=0)

    a_e = syn_filter_align(ae[:, burn:-n_eta], sf, E_ps, n_cells, lag, B=False)
    a_b = syn_filter_align(ab[:, burn:-n_eta], sf, E_ps, n_cells, scale_bs=scale_bs, lag=lag, B=True)

    a_p = a_b / a_e

    g_e = event_sig[np.newaxis, :-lag]
    g_p = burst_sig[np.newaxis, :-lag]

    le, lwe = lower_bound(g_e, a_e, fs, f, nps, nov, FREQ_MAX_e)  # Only integrate up to 200Hz
    lp, lwp = lower_bound(g_p, a_p, fs, f, nps, nov, FREQ_MAX_b)  # Only integrate up to 100Hz

    return le, lp, lwe, lwp


def lba(S, event_sig, burst_sig, burn, n_eta, n_cells, thresh, thresh2=None, s_t=1, lag=6, decoder=2, FR=None):
    """
    lower bound analysis activity only. Calculates lower bound from spiking data
    :param S:
    :param burst_sig:
    :param event_sig
    :param burn:
    :param n_eta:
    :param n_cells:
    :param thresh: event and burst if decoder = 1; event only if decoder = 2
    :param thresh2: burst thresh if decoder = 2
    :param s_t: only required if decoder = 2
    :param lag:
    :param decoder:
    :return:
    """

    # GET PARAMS
    dt = params['dt']
    scale_bs = params2['scale_bs']
    E_ps = params2['E_ps']
    fs = params2['fs']
    nps = params2['nps']
    nov = params2['nov']
    if FR is None:
        FREQ_MAX_e = params2['FREQ_MAX_e']
        FREQ_MAX_b = params2['FREQ_MAX_b']
    else:
        FREQ_MAX_e = FR[0]
        FREQ_MAX_b = FR[1]

    g_bar = params2['g_bar']
    ta_rise = params2['ta_rise']
    ta_decay = params2['ta_decay']
    t_sf = params2['t_syn']

    # DEFINE VARIABLES
    f = np.arange(0, 500 + 1 / (nps / fs), 1 / (nps / fs))
    sf = se_kernel(g_bar, ta_rise, ta_decay, t_sf // dt, dt=dt)

    # DECODE SPIKETRAINS
    if decoder == 1:
        ae, ab, ue, ub = decode_trial_bc(S, thresh)
    elif decoder == 2:
        ae, ab, ue, ub = decode_trial2_bc(S, thresh, thresh2, s_t)
    else:
        print('error lba: invalid decoder value')
        sys.exit()

    # CALCULATE INFORMATION LOWER BOUND
    ue = np.concatenate((ue, np.expand_dims(ae, axis=0)), axis=0)
    ub = np.concatenate((ub, np.expand_dims(ab, axis=0)), axis=0)

    u_e = syn_filter_align(ue[:, burn:-n_eta], sf, E_ps, n_cells, lag=lag, B=False)
    u_b = syn_filter_align(ub[:, burn:-n_eta], sf, E_ps, n_cells, scale_bs=scale_bs, lag=lag, B=True)

    u_p = u_b / u_e

    g_e = np.tile(event_sig[:-lag], (u_e.shape[0], 1))
    g_p = np.tile(burst_sig[:-lag], (u_p.shape[0], 1))

    le, lwe = lower_bound(g_e, u_e, fs, f, nps, nov, FREQ_MAX_e)  # Only integrate up to 200Hz
    lp, lwp = lower_bound(g_p, u_p, fs, f, nps, nov, FREQ_MAX_b)  # Only integrate up to 100Hz

    return le, lp, lwe, lwp



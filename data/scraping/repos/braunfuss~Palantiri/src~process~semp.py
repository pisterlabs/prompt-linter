import os
import sys
import time
import numpy as num
from threading import Thread
from obspy.signal.headers import clibsignal
from obspy import Stream, Trace
from palantiri.common.ConfigFile import ConfigObj, OriginCfg, SynthCfg, FilterCfg
import numpy as np
from palantiri.common import Basic, Logfile
import ctypes as C
from pyrocko import trace as trld
from pyrocko.marker import PhaseMarker
from pyrocko import obspy_compat
import math
from scipy.signal import coherence
from collections import OrderedDict
from palantiri.process import music
from pyrocko import io, trace, util

trace_txt = 'trace.txt'
travel_txt = 'travel.txt'
latv_txt = 'latv.txt'
lonv_txt = 'lonv.txt'
semb_txt = 'semb.txt'


def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)


def hilbert(x, N=None):
    '''
    Return the hilbert transform of x of length N.
    (from scipy.signal, but changed to use fft and ifft from numpy.fft)
    '''
    x = num.asarray(x)
    if N is None:
        N = len(x)
    if N <= 0:
        raise ValueError("N must be positive.")
        x = num.real(x)
    Xf = num.fft.fft(x, N, axis=0)
    h = num.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    if len(x.shape) > 1:
        h = h[:, num.newaxis]
    x = num.fft.ifft(Xf*h)
    return x


def xcorr(tr1, tr2, shift_len, full_xcorr=False):

    from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
    if min(len(tr1), len(tr2)) - 2 * shift_len <= 0:
        msg = "shift_len too large. The underlying C code would silently " + \
              "use shift_len/2 which we want to avoid."
        raise ValueError(msg)
    tr1 = np.ascontiguousarray(tr1, np.float32)
    tr2 = np.ascontiguousarray(tr2, np.float32)
    corp = np.empty(2 * shift_len + 1, dtype=np.float64, order='C')

    shift = C.c_int()
    coe_p = C.c_double()

    res = clibsignal.X_corr(tr1, tr2, corp, shift_len, len(tr1), len(tr2),
                            C.byref(shift), C.byref(coe_p))
    if res:
        raise MemoryError

    if full_xcorr:
        return shift.value, coe_p.value, corp
    else:
        return shift.value, coe_p.value


class MyThread(Thread):

    def __init__(self, nostat, nsamp, i, nstep, dimX,
                 dimY, mint, new_freq, minSampleCount):
        Thread.__init__(self)

        self.nostat = nostat
        self.nsamp = nsamp
        self.i = i
        self.nstep = nstep
        self.dimX = dimX
        self.dimY = dimY
        self.mint = mint
        self.new_freq = new_freq
        self.minSampleCount = minSampleCount


def toMatrix(npVector, nColumns):

    t = npVector.tolist()[0]
    n = nColumns
    mat = []

    for i in range(int(len(t) / n)):
        pos1 = i * n
        pos2 = pos1 + n
        slice = t[pos1:pos2]
        assert len(slice) == nColumns
        mat.append(slice)

    return mat


def semblance(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
              new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
              trace_1, calcStreamMap, time, Config, Origin, refshifts, nstats,
              bs_weights=None, flag_rpe=False, boot_shifts=None):

        cfg = Config
        origin = OriginCfg(Origin)
        cfg_f = FilterCfg(Config)

        if cfg.config_filter.dynamic_filter is False:
            if cfg.config.bp_freq is True:
               return semblance_py_freq(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY,
                                   mint, new_frequence, minSampleCount, latv_1,
                                   lonv_1, traveltime_1, trace_1, calcStreamMap,
                                   time, cfg, refshifts, nstats, bs_weights=bs_weights)
            if cfg.config.bp_coh is True:
               return semblance_py_coherence(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY,
                                   mint, new_frequence, minSampleCount, latv_1,
                                   lonv_1, traveltime_1, trace_1, calcStreamMap,
                                   time, cfg, refshifts, nstats, bs_weights=bs_weights)

            if cfg.config_geometry.dimz!= 0:
                return semblance_py_cube(ncpus, nostat, nsamp, ntimes, nstep,
                                         dimX, dimY, mint, new_frequence,
                                         minSampleCount, latv_1, lonv_1,
                                         traveltime_1, trace_1, calcStreamMap,
                                         time, cfg, refshifts,
                                         bs_weights=bs_weights)

            if cfg.config.bp_music is True:
                return music_wrapper(ncpus, nostat, nsamp, ntimes, nstep, dimX,
                                     dimY, mint, new_frequence, minSampleCount,
                                     latv_1, lonv_1, traveltime_1, trace_1,
                                     calcStreamMap, time, cfg, refshifts,
                                     nstats,
                                     bs_weights=bs_weights)

            if flag_rpe is True:
               return semblance_py(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY,
                                   mint, new_frequence, minSampleCount, latv_1,
                                   lonv_1, traveltime_1, trace_1, calcStreamMap,
                                   time, cfg, refshifts, nstats, bs_weights=bs_weights,
                                   flag_rpe=flag_rpe)

            else:
               return semblance_py(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY,
                                   mint, new_frequence, minSampleCount, latv_1,
                                   lonv_1, traveltime_1, trace_1, calcStreamMap,
                                   time, cfg, refshifts, nstats, bs_weights=bs_weights,
                                   flag_rpe=flag_rpe, boot_shifts=boot_shifts)
        else:
           return semblance_py_dynamic_cf(ncpus, nostat, nsamp, ntimes, nstep,
                                          dimX, dimY, mint, new_frequence,
                                          minSampleCount, latv_1, lonv_1,
                                          traveltime_1, trace_1, calcStreamMap,
                                          time, origin, cfg_f)


def t2ind_fast(t, tdelta, snap=round):
    return int(int((t/tdelta)*(10**0))/(10.**0))


def t2ind(t, tdelta, snap=round):
    return int(snap(t/tdelta))


def semblance_py_dynamic_cf(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY,
                            mint, new_frequence, minSampleCount, latv_1, lonv_1,
                            traveltime_1, trace_1, calcStreamMap, time,
                            origin, FilterCfg):

    obspy_compat.plant()
    trs_orgs = []
    for tr in calcStreamMap:
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(tr_org.ydata)))
        trs_orgs.append(tr_org)
    trace = toMatrix(trace_1, minSampleCount)
    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY)

    latv = latv_1.tolist()
    lonv = lonv_1.tolist()

    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    snap=(round, round)
    backSemb = np.ndarray(shape=(ntimes, dimX*dimY), dtype=float)
    for i in range(ntimes):
        #  loop over grid points
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0

        for j in range(dimX * dimY):
            semb = 0
            nomin = 0
            denom = 0
            sums_cc = 0
            sums = 0
            shifted = []
            relstart = []
            relstarts = nostat
            cc_data = []
            tt = []

            for k in range(nostat):
                relstart = traveltime[k][j]
                tr = trs_orgs[k]
                tmin = time+relstart+(i*nstep)-mint
                tmax = time+relstart+(i*nstep)-mint+nsamp
                try:
                    ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
                    iend = min(
                        tr.data_len(),
                        t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))
                except:
                    print('Loaded traveltime grid wrong!')

                data = tr.ydata[ibeg:iend]
                try:
                    sums += num.gradient(abs(data))
                except:
                    pass
                relstarts -=(relstart)

            sum = abs(num.sum(((sums))))
            denom = sum**2
            nomin = sum
            semb = sum

            backSemb[i][j] = sum
            if semb > sembmax:

               sembmax  = semb
               sembmaxX = latv[j]
               sembmaxY = lonv[j]

        Logfile.add('max semblance: ' + str(sembmax) + ' at lat/lon: ' +
                     str(sembmaxX)+','+ str(sembmaxY))

    backSemb = backSemb/num.max(num.max(backSemb))
    return abs(backSemb)


def semblance_py(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
                 new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
                 trace_1, calcStreamMap, time, cfg, refshifts, nstats,
                 bs_weights=None, flag_rpe=False, output=False,
                 boot_shifts=None):

    trs_orgs = []
    snap = (round, round)
    if cfg.config_weight.combine_all is True:
        combine = True
    else:
        combine = False
    if cfg.config_weight.bootstrap_array_weights is True:
        do_bs_weights = True
    else:
        do_bs_weights = False
    do_weight_by_array = False
    if do_weight_by_array:
        k = 0
        stats_done = 0
        for k in range(0, len(nstats)):
            for stats in range(0, nstats[k]):
            #    list(calcStreamMap.keys())[stats].data = list(calcStreamMap.keys())[stats].data/np.max(list(calcStreamMap.keys())[0].data)
                stats_done = stats_done + nstats[k]

        k = 0
        s_index = 0
        for tr in calcStreamMap:
            tr_org = calcStreamMap[tr][0]
            trs_orgs.append(tr_org)

        for tr in calcStreamMap:
            tr_org = calcStreamMap[tr][0]
            datas = trs_orgs[0:s_index].ydata
            if k <= nstats[s_index]:
                k = k+1
        #        tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(datas)))

            if k == nstats[s_index]:
                s_index = s_index+1
            calcStreamMap[tr] = obspy_compat.to_obspy_trace(tr_org)

            stats_done = stats_done + nstats[k]
    trs_orgs = []
    for tr in sorted(calcStreamMap):
        tr_org = calcStreamMap[tr]
        trs_orgs.append(tr_org)
    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY)
    latv = latv_1.tolist()
    lonv = lonv_1.tolist()
    index_begins = OrderedDict()
    index_steps = []
    index_window = []
    for j in range(dimX * dimY):
        markers = []

        for k in range(nostat):
            if cfg.config_weight.bootstrap_array_weights is True:
                relstart = traveltime[k][j] + boot_shifts[k] + num.random.uniform(-1*cfg.config_weight.shift_max/10.,
                                 cfg.config_weight.shift_max/10.)
            else:
                relstart = traveltime[k][j]
            tr = trs_orgs[k]
            try:
                tmin = time+relstart-mint+refshifts[k]
                tmax = time+relstart-mint+nsamp+refshifts[k]
            except:
                tmin = time+relstart-mint
                tmax = time+relstart-mint+nsamp

        #    m = PhaseMarker(tmin=tmin,
    #                        tmax=tmax,
    #                        phasename='P',
    #                        nslc_ids=(tr.nslc_id,))
#            markers.append(m)

            ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
            index_begins[str(j)+str(k)] = [ibeg, tmin]

            iend = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))

            iend_step = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin+nstep, tr.deltat, snap[1]))
            index_steps.append(iend_step-iend)

            index_window.append(iend-ibeg)
            # for debug:
    #    trld.snuffle(trs_orgs, markers=markers)

    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    trs_orgs = []
    k = 0

    for tr in sorted(calcStreamMap):

        tr_org = calcStreamMap[tr]
        if combine is True:
            # some trickery to make all waveforms have same polarity, while still
            # considering constructive/destructive interferences. This is needed
            # when combing all waveforms/arrays from the world at once(only then)
            # for a single array with polarity issues we recommend fixing polarity.
            # advantage of the following is that nothing needs to be known about the
            # mechanism.

    #        tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(tr_org.ydata)))
    #        tr_org.ydata = abs(tr_org.ydata)

    #        tr_org.ydata = num.ediff1d(tr_org.ydata)
            if max(index_steps) % 2 == 1:
                tr_org.ydata = abs(tr_org.ydata)
    #            tr_org.ydata = num.ediff1d(tr_org.ydata)
    #    if cfg.config_weight.shift_by_phase_pws is True:
    #                cfx = num.fft.fft(tr_org.ydata)
    #                sums_schimmel = sums_schimmel + (cfx/(abs(cfx)) * num.exp(1j*2*num.pi*cfx)))
    #                print('calculate pws')
        trs_orgs.append(tr_org)
    if nsamp == 0:
        nsamp = 1
    backSemb = np.ndarray(shape=(ntimes, dimX*dimY), dtype=float)
    for i in range(ntimes):
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0
        for j in range(dimX * dimY):
            semb = 0.
            nomin = 0
            denom = 0
            sums = num.zeros(max(index_steps))
            sums = 0.
            relstart = []
            relstarts = nostat
            sums_schimmel = 0

            for k in range(nostat):
                relstart = traveltime[k][j]
                tr = trs_orgs[k]

                ibeg = index_begins[str(j)+str(k)][0]+i*index_steps[j+k]
                iend = index_begins[str(j)+str(k)][0]+index_window[j+k]+i*index_steps[j+k]
                data = tr.ydata[ibeg:iend]
                if cfg.config_weight.shift_by_phase_pws is True:
                    cfx = num.fft.fft(data)
                    sums_schimmel = sums_schimmel + (cfx/(abs(cfx)) * num.exp(1j*2*num.pi*cfx))
                try:
                    if do_bs_weights is True and combine is True:
                        sums += data*bs_weights[k]
                    else:
                        sums = sums+data
                except ValueError:
                        try:
                            if num.shape(data) < num.shape(sums):
                                data = tr.ydata[ibeg:iend+1]
                            else:
                                data = tr.ydata[ibeg:iend-1]
                            sums = sums+data
                        except:
                            sums = sums

                relstarts -= relstart
            sums_schimmel = num.real(sums_schimmel)**2.
            sums_copy = sums
        #    data = sums_schimmel
        #    t1 = trace.Trace(
        #        station='TEST', channel='Z', deltat=0.5, tmin=0., ydata=data)

            if cfg.config_weight.shift_by_phase_pws is True:
                for k in range(nostat):
                    relstart = traveltime[k][j]
                    tr = trs_orgs[k]
                    ibeg = index_begins[str(j)+str(k)][0]+i*index_steps[j+k]
                    iend = index_begins[str(j)+str(k)][0]+index_window[j+k]+i*index_steps[j+k]
                    data = tr.ydata[ibeg:iend]
                    data_copy = data.copy()
                    cfx = num.fft.fft(data) * sums_schimmel
                    data = num.fft.ifft(cfx)
                    try:
                        if do_bs_weights is True:
                            sums += data*bs_weights[k]
                        else:
                            sums = sums+data
                    except ValueError:
                            if num.shape(data) < num.shape(sums):
                                data = tr.ydata[ibeg:iend+1]
                            else:
                                data = tr.ydata[ibeg:iend-1]
                            sums = sums+data
            data = sums_copy
            basetime = util.str_to_time("2016-11-25 14:24:30.000")
            data = num.real(sums)
            sum = abs(num.sum(sums))
            if combine is True:
                sum = (1./nostat)* ((abs(num.sum((sums)))**2) /num.sum(sums))
            semb = sum

            backSemb[i][j] = sum
            if semb > sembmax:
                sembmax = semb
                sembmaxX = latv[j]
                sembmaxY = lonv[j]
                #backSemb[i][:] = backSemb[i][:]/num.max(backSemb[i][:])
        if output is True:
            Logfile.add('max semblance: ' + str(sembmax) + ' at lat/lon: ' +
                        str(sembmaxX) + ',' + str(sembmaxY))
    if flag_rpe is False:
        backSemb = backSemb/num.max(num.max(backSemb))

    return backSemb


def music_wrapper(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
                 new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
                 trace_1, calcStreamMap, time, cfg, refshifts, nstats,
                 bs_weights=None):
    obspy_compat.plant()
    trs_orgs = []
    snap = (round, round)
    if cfg.config_weight.combine_all is True:
        combine = True
    else:
        combine = False
    if cfg.config_weight.bootstrap_array_weights is True:
        do_bs_weights = True
    else:
        do_bs_weights = False

    do_weight_by_array = False
    if do_weight_by_array:
        k = 0
        stats_done = 0
        for k in range(0, len(nstats)):
            for stats in range(0, nstats[k]):
                list(calcStreamMap.keys())[stats].data = list(calcStreamMap.keys())[stats].data/np.max(list(calcStreamMap.keys())[0].data)
                stats_done = stats_done + nstats[k]

        k = 0
        s_index = 0
        for tr in calcStreamMap:
            tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
            trs_orgs.append(tr_org)

        for tr in calcStreamMap:
            tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
            datas = trs_orgs[0:s_index].ydata
            if k <= nstats[s_index]:
                k = k+1
                tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(datas)))

            if k == nstats[s_index]:
                s_index = s_index+1
            calcStreamMap[tr] = obspy_compat.to_obspy_trace(tr_org)

            stats_done = stats_done + nstats[k]
    trs_orgs = []
    for tr in sorted(calcStreamMap):
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        trs_orgs.append(tr_org)
        array_lats, array_lons = calcStreamMap[tr].lat, calcStreamMap[tr].lon
    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY)
    latv = latv_1.tolist()
    lonv = lonv_1.tolist()
    from collections import OrderedDict
    index_begins = OrderedDict()

    index_steps = []
    index_window = []
    for j in range(dimX * dimY):
        markers = []

        for k in range(nostat):
            relstart = traveltime[k][j]
            tr = trs_orgs[k]

            try:
                tmin = time+relstart-mint-refshifts[k]
                tmax = time+relstart-mint+nsamp-refshifts[k]
            except IndexError:
                tmin = time+relstart-mint
                tmax = time+relstart-mint+nsamp


                m = PhaseMarker(tmin=tmin,
                                tmax=tmax,
                                phasename='P',
                                nslc_ids=(tr.nslc_id,))
                markers.append(m)

            ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
            index_begins[str(j)+str(k)]= [ibeg, tmin]

            iend = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))

            iend_step = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin+nstep, tr.deltat, snap[1]))
            index_steps.append(iend_step-iend)

            index_window.append(iend-ibeg)
    #    trld.snuffle(trs_orgs, markers=markers)


    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    trs_orgs = []
    k = 0
    for tr in sorted(calcStreamMap):
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        trs_orgs.append(tr_org)

    if nsamp == 0:
        nsamp = 1
    backSemb = np.ndarray(shape=(ntimes, dimX*dimY), dtype=float)
    for i in range(ntimes):
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0
        for j in range(dimX * dimY):
            semb = 0.
            nomin = 0
            denom = 0
            sums = num.zeros(max(index_steps))
            sums = 0.
            relstart = []
            relstarts = nostat
            sums_schimmel = 0

            for k in range(nostat):
                relstart = traveltime[k][j]
                tr = trs_orgs[k]
                ibeg = index_begins[str(j)+str(k)][0]+i*index_steps[j+k]
                iend = index_begins[str(j)+str(k)][0]+index_window[j+k]+i*index_steps[j+k]
                data = tr.ydata[ibeg:iend]
                # normalize:
                #data = data / np.sqrt(np.mean(np.square(data)))
                try:
                    if do_bs_weights is True:
                        sums += data*bs_weights[k]
                    else:
                        sums = sums+data
                except ValueError:
                        if num.shape(data) < num.shape(sums):
                            data = tr.ydata[ibeg:iend+1]
                        else:
                            data = tr.ydata[ibeg:iend-1]
                        sums = sums+data

                relstarts -= relstart
            sums_schimmel = abs(sums_schimmel)**2.

            sum = abs(num.sum(sums))

            semb = sum

            backSemb[i][j] = sum
            if semb > sembmax:
                sembmax  = semb
                sembmaxX = latv[j]
                sembmaxY = lonv[j]
        Logfile.add('max semblance: ' + str(sembmax) + ' at lat/lon: ' +
                    str(sembmaxX) + ',' + str(sembmaxY))
    return backSemb



def semblance_py_coherence(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
                 new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
                 trace_1, calcStreamMap, time, cfg, refshifts, nstats,
                 bs_weights=None):
    obspy_compat.plant()
    trs_orgs = []
    snap = (round, round)
    if cfg.config_weight.combine_all is True:
        combine = True
    else:
        combine = False
    if cfg.config_weight.bootstrap_array_weights is True:
        do_bs_weights = True
    else:
        do_bs_weights = False

    do_weight_by_array = True
    if do_weight_by_array:
        tr_bases = []
        tr_bases_data = []
        k = 0
        ks = 0
        s_index = 0
        for tr in calcStreamMap:
            tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        #    tr_org.ydata = abs(tr_org.ydata)

        #    tr_org.ydata = num.ediff1d(tr_org.ydata, to_end=tr_org.ydata[-1])
            if k == 1:
                tr_bases.append(tr_org)

            tr_org.set_location('%s' % s_index)
            if k == nstats[s_index]:
                s_index = s_index+1
                k = 0
            if k < nstats[s_index]:
                k = k+1
            calcStreamMap[tr] = obspy_compat.to_obspy_trace(tr_org)
            ks = ks +1
    trs_orgs = []
    for tr in sorted(calcStreamMap):
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        trs_orgs.append(tr_org)
    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY)
    latv = latv_1.tolist()
    lonv = lonv_1.tolist()
    from collections import OrderedDict
    index_begins = OrderedDict()

    index_steps = []
    index_window = []
    for j in range(dimX * dimY):
        markers = []

        for k in range(nostat):
            relstart = traveltime[k][j]
            tr = trs_orgs[k]

            try:
                tmin = time+relstart-mint-refshifts[k]
                tmax = time+relstart-mint+nsamp-refshifts[k]
            except IndexError:
                tmin = time+relstart-mint
                tmax = time+relstart-mint+nsamp


                m = PhaseMarker(tmin=tmin,
                                tmax=tmax,
                                phasename='P',
                                nslc_ids=(tr.nslc_id,))
                markers.append(m)

            ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
            index_begins[str(j)+str(k)]= [ibeg, tmin]

            iend = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))

            iend_step = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin+nstep, tr.deltat, snap[1]))
            index_steps.append(iend_step-iend)

            index_window.append(iend-ibeg)
    #    trld.snuffle(trs_orgs, markers=markers)


    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    trs_orgs = []
    do_trad = False
    for tr in sorted(calcStreamMap):
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
    #    if combine is True:
    #        tr_org.ydata = abs(tr_org.ydata)

    #        tr_org.ydata = num.ediff1d(tr_org.ydata, to_end=tr_org.ydata[-1])

#            tr_org.ydata = tr_org.ydata/num.max(tr_org.ydata)
#            tr_org.ydata = abs(tr_org.ydata)
#                tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(tr_org.ydata)))


        trs_orgs.append(tr_org)

    if nsamp == 0:
        nsamp = 1
    backSemb = np.ndarray(shape=(ntimes, dimX*dimY), dtype=float)
    for i in range(ntimes):
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0
        for j in range(dimX * dimY):
            semb = 0.
            nomin = 0
            denom = 0
            sums = num.zeros(max(index_steps))
            sums = 0.
            relstart = []
            relstarts = nostat
            data_comp = False
            for k in range(nostat):
                relstart = traveltime[k][j]
                tr = trs_orgs[k]
                ibeg = index_begins[str(j)+str(k)][0]+i*index_steps[j+k]
                iend = index_begins[str(j)+str(k)][0]+index_window[j+k]+i*index_steps[j+k]
                data = tr.ydata[ibeg:iend]
                data = data / np.sqrt(np.mean(np.square(data)))
                if combine is True:
                        ind = int(tr_org.location)
                        data_comp = tr_bases[ind].ydata[ibeg:iend]
                #        data_comp = data

                else:
                    if data_comp is False:
                        data_comp = data
                cfx = num.fft.fft(data)
                cfy = num.fft.fft(data_comp)

                # Get cross spectrum
                cross = cfx.conj()*cfy
        #        cross = num.correlate(data, data_comp)
                #f, coh = coherence(data, data_comp, fs=tr.deltat)
                sums = sums+num.sum(abs(cross))

                relstarts -= relstart
            sum = abs(num.sum(sums))

            semb = sum

            backSemb[i][j] = sum
            if semb > sembmax:
                sembmax  = semb   # search for maximum and position of maximum on semblance
                                 # grid for given time step
                sembmaxX = latv[j]
                sembmaxY = lonv[j]
        Logfile.add('max semblance: ' + str(sembmax) + ' at lat/lon: ' +
                    str(sembmaxX) + ','+ str(sembmaxY))

    backSemb = backSemb/num.max(num.max(backSemb))
    return backSemb

def semblance_py_freq(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
                 new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
                 trace_1, calcStreamMap, time, cfg, refshifts, nstats,
                 bs_weights=None):
    obspy_compat.plant()
    trs_orgs = []
    snap = (round, round)
    if cfg.config_weight.combine_all is True:
        combine = True
    else:
        combine = False
    if cfg.config_weight.bootstrap_array_weights is True:
        do_bs_weights = True
    else:
        do_bs_weights = False
    trs_data = []

    for tr in sorted(calcStreamMap):
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        trs_orgs.append(tr_org)

    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY)
    latv = latv_1.tolist()
    lonv = lonv_1.tolist()

    from collections import OrderedDict
    index_begins = OrderedDict()

    index_steps = []
    index_window = []


    for j in range(dimX * dimY):
        for k in range(nostat):
            relstart = traveltime[k][j]
            tr = trs_orgs[k]

            try:
                tmin = time+relstart-mint-refshifts[k]
                tmax = time+relstart-mint+nsamp-refshifts[k]
            except IndexError:
                tmin = time+relstart-mint
                tmax = time+relstart-mint+nsamp

            ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
            index_begins[str(j)+str(k)]= [ibeg, tmin]

            iend = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))

            iend_step = min(
                tr.data_len(),
                t2ind_fast(tmax-tr.tmin+nstep, tr.deltat, snap[1]))
            index_steps.append(iend_step-iend)
            index_window.append(iend-ibeg)


    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    if nsamp == 0:
        nsamp = 1
    backSemb = np.ndarray(shape=(ntimes, dimX*dimY), dtype=float)
    for i in range(ntimes):
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0
        for j in range(dimX * dimY):
            semb = 0.
            nomin = 0
            denom = 0
            sums = 1.
            relstart = []
            relstarts = nostat
            ws = []
            # or normalize group

            #for k in range(nostat):
            #    relstart = traveltime[k][j]
            #    tr = trs_orgs[k]
            #    ibeg = index_begins[str(j)+str(k)][0]+i*index_steps[j+k]
            #    iend = index_begins[str(j)+str(k)][0]+index_window[j+k]+i*index_steps[j+k]
            #    data_tr = tr.ydata[ibeg:iend]
            #    w = 1. / np.sqrt(np.mean(np.square(tr_org.ydata)))
            #    ws.append(w)

            for k in range(nostat):
                relstart = traveltime[k][j]
                tr = trs_orgs[k]
                ibeg = index_begins[str(j)+str(k)][0]+i*index_steps[j+k]
                iend = index_begins[str(j)+str(k)][0]+index_window[j+k]+i*index_steps[j+k]
                data_tr = tr.ydata[ibeg:iend]
                fydata = num.fft.rfft(data_tr, data_tr.size)
                df = 1./(tr.deltat)
                fxdata = num.arange(len(fydata))*df
                w = 1. / np.sqrt(np.mean(np.square(tr_org.ydata)))
                data = fydata * (num.exp(relstart)*num.imag(2*math.pi*fxdata))*w

                if do_bs_weights is True:
                    sums *= data
                else:
                    sums *= data

            backSemb[i][j] = num.sum(sums)

            if semb > sembmax:
                sembmax  = semb   # search for maximum and position of maximum on semblance
                                 # grid for given time step
                sembmaxX = latv[j]
                sembmaxY = lonv[j]
        Logfile.add('max semblance: ' + str(sembmax) + ' at lat/lon: ' +
                    str(sembmaxX) + ','+ str(sembmaxY))

    return backSemb


def semblance_py_cube(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
                 new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
                 trace_1, calcStreamMap, time, cfg, refshifts,
                 bs_weights=None):
    obspy_compat.plant()
    trs_orgs = []
    snap = (round, round)
    if cfg.config_weight.combine_all is True:
        combine = True
    else:
        combine = False
    if cfg.config_weight.bootstrap_array_weights is True:
        do_bs_weights = True
    else:
        do_bs_weights = False
    for tr in sorted(calcStreamMap):
        tr_org = obspy_compat.to_pyrocko_trace(calcStreamMap[tr])
        tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(tr_org.ydata)))
        if combine is True:
            # some trickery to make all waveforms have same polarity, while still
            # considering constructive/destructive interferences. This is needed
            # when combing all waveforms/arrays from the world at once(only then)
            # for a single array with polarity issues we recommend fixing polarity.
            # advantage of the following is that nothing needs to be known about the
            # mechanism.
            tr_org.ydata = abs(tr_org.ydata)
            tr_org.ydata = num.diff(tr_org.ydata)
        trs_orgs.append(tr_org)

    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY * cfg.Int('dimz'))
    latv = latv_1.tolist()
    lonv = lonv_1.tolist()
    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    if nsamp == 0:
        nsamp = 1
    backSemb = np.ndarray(shape=(ntimes, dimX*dimY), dtype=float)
    for i in range(ntimes):
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0
        for j in range(dimX * dimY):
            semb = 0
            nomin = 0
            denom = 0
            if combine is True:
                sums = 0
            else:
                sums = 0
            relstart = []
            relstarts = nostat

            for k in range(nostat):
                relstart = traveltime[k][j]
                tr = trs_orgs[k]
                try:
                    tmin = time+relstart+(i*nstep)-mint-refshifts[k]
                    tmax = time+relstart+(i*nstep)-mint+nsamp-refshifts[k]
                except IndexError:
                    tmin = time+relstart+(i*nstep)-mint
                    tmax = time+relstart+(i*nstep)-mint+nsamp
                try:
                    ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
                    iend = min(
                        tr.data_len(),
                        t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))
                except Exception:
                    print('Loaded traveltime grid wrong!')

                data = tr.ydata[ibeg:iend]

                if combine is True:
                    if do_bs_weights is True:
                        sums += (data)*bs_weights[k]
                    else:
                        sums += (data)

                else:
                    sums += (data)

                relstarts -= relstart

            sum = abs(num.sum(((sums))))
            semb = sum

            backSemb[i][j] = sum
            if semb > sembmax:
                sembmax  = semb   # search for maximum and position of maximum on semblance
                                 # grid for given time step
                sembmaxX = latv[j]
                sembmaxY = lonv[j]
        Logfile.add('max semblance: ' + str(sembmax) + ' at lat/lon: ' +
                     str(sembmaxX)+','+ str(sembmaxY))

    backSemb = backSemb
    return abs(backSemb)

def semblance_py_fixed(ncpus, nostat, nsamp, ntimes, nstep, dimX, dimY, mint,
                 new_frequence, minSampleCount, latv_1, lonv_1, traveltime_1,
                 trace_1, calcStreamMap, time, cfg, refshifts,nstats,
                 bs_weights=None, flag_rpe=True):
    trs_orgs = []
    snap = (round, round)
    if cfg.config_weight.combine_all is True:
        combine = True
    else:
        combine = False
    if cfg.config_weight.bootstrap_array_weights is True:
        do_bs_weights = True
    else:
        do_bs_weights = False

    index_mean_x = dimX/2
    index_mean_y = dimY/2
    j_mean = int((dimX * dimY)/2)
    for tr in sorted(calcStreamMap):
        tr_org = calcStreamMap[tr]
        tr_org.ydata = tr_org.ydata / np.sqrt(np.mean(np.square(tr_org.ydata)))
        if combine is True:
            # some trickery to make all waveforms have same polarity, while still
            # considering constructive/destructive interferences. This is needed
            # when combing all waveforms/arrays from the world at once(only then)
            # for a single array with polarity issues we recommend fixing polarity.
            # advantage of the following is that nothing needs to be known about the
            # mechanism.
            tr_org.ydata = abs(tr_org.ydata)
            tr_org.ydata = num.diff(tr_org.ydata)
        trs_orgs.append(tr_org)

    traveltime = []
    traveltime = toMatrix(traveltime_1, dimX * dimY)
    latv = latv_1.tolist()
    lonv = lonv_1.tolist()
    '''
    Basic.writeMatrix(trace_txt,  trace, nostat, minSampleCount, '%e')
    Basic.writeMatrix(travel_txt, traveltime, nostat, dimX * dimY, '%e')
    Basic.writeVector(latv_txt,   latv, '%e')
    Basic.writeVector(lonv_txt,   lonv, '%e')
    '''
    if nsamp == 0:
        nsamp = 1
    backSemb = np.ndarray(shape=(ntimes, 1), dtype=float)
    for i in range(ntimes):
        sembmax = 0
        sembmaxX = 0
        sembmaxY = 0
        j = j_mean
        semb = 0
        nomin = 0
        denom = 0
        if combine is True:
            sums = 1
        else:
            sums = 0
        relstart = []
        relstarts = nostat

        for k in range(nostat):
            relstart = traveltime[k][j]
            tr = trs_orgs[k]
            try:
                tmin = time+relstart+(i*nstep)-mint-refshifts[k]
                tmax = time+relstart+(i*nstep)-mint+nsamp-refshifts[k]
            except IndexError:
                tmin = time+relstart+(i*nstep)-mint
                tmax = time+relstart+(i*nstep)-mint+nsamp
            try:
                ibeg = max(0, t2ind_fast(tmin-tr.tmin, tr.deltat, snap[0]))
                iend = min(
                    tr.data_len(),
                    t2ind_fast(tmax-tr.tmin, tr.deltat, snap[1]))
            except Exception:
                print('Loaded traveltime grid wrong!')

            data = tr.ydata[ibeg:iend]

            if combine is True:
                if do_bs_weights is True:
                    sums *= (data)*bs_weights[k]
                else:
                    sums *= (data)

            else:
                sums += (data)

            relstarts -= relstart

            sum = abs(num.sum(((sums))))
            semb = sum

            backSemb[i] = sum

            if semb > sembmax:
                sembmax  = semb   # search for maximum and position of maximum on semblance
                                 # grid for given time step
                sembmaxX = latv[j]
                sembmaxY = lonv[j]

    return abs(sembmax)


def execsemblance(nostat, nsamp, i, nstep, dimX,dimY, mint, new_freq, minSampleCount) :

    f = [nostat, nsamp, i, nstep, dimX,dimY, mint, new_freq, minSampleCount]
    args  = Basic.floatToString(f, delim= ',')
    prog  = sys.executable + ' ' + __file__
    cmd   = prog  + ' ' + args

    Logfile.add('--------------------------------------------', cmd)
    result = Basic.systemCmd(cmd)
    Logfile.addLines(result)
    backSemb = Basic.readVector(semb_txt)
    return backSemb

def execsemblance2() :

    for i in range(len(sys.argv)):
        print(sys.argv[i])

    params = Basic.stringToFloat(sys.argv[1])
    [nostat, nsamp, i, nstep, dimX, dimY, mint, new_freq, minSampleCount] = params
    backSemb = startsemblance(int(nostat), int(nsamp), int(i), int(nstep),
                              int(dimX), int(dimY),  mint, new_freq,
                              int(minSampleCount), False)

    Basic.writeVector(semb_txt, backSemb)


def startsemblance(nostat, nsamp, i, nstep, dimX,dimY, mint, new_freq, minSampleCount, isParent = True):

    backSemb = []

    if isParent:
        backSemb = execsemblance(nostat, nsamp, i, nstep, dimX, dimY, mint,
                                 new_freq, minSampleCount)

    else:
       trace = Basic.readMatrix(trace_txt,  nostat, minSampleCount, '%e')
       traveltime = Basic.readMatrix(travel_txt, nostat, dimX * dimY, '%e')
       latv = Basic.readVector(latv_txt, '%e')
       lonv = Basic.readVector(lonv_txt, '%e')

    for j in range(dimX * dimY):
      semb  = 0
      nomin = 0
      denom = 0

      for l in range(int(nsamp)):
         sum = 0

         for k in range(nostat):
            relstart_samples = int((traveltime[k][j] - mint) * new_freq + 0.5) + i * nstep

            val   =  trace[k][relstart_samples+l]
            sum   += val
            denom +=(val * val)

         nomin += sum * sum;
         semb  = nomin /(float(nostat) * denom)

      backSemb.append(semb)


    return backSemb


if __name__ == "__main__":
   execsemblance2()

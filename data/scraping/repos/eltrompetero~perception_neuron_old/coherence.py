# ================================================================================================ # 
# Functions for calculating performance measures.
# Authors: Eddie Lee, edl56@cornell.edu
#          Ted Esposito
# ================================================================================================ # 


import numpy as np
from scipy.signal import coherence
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from warnings import warn
from gaussian_process.regressor import GaussianProcessRegressor
import multiprocess as mp



def precompute_coherence_nulls(v,t0,windowDuration,pool,
        sampling_rate=30,n_iters=100):
    """
    This is unnecessary! As noted in Tango NBIII 2017-10-30.
    Calculate coherence values for trajectory with many samples of white noise. 
    
    This uses multiprocess to speed up calculation.
    
    Parameters
    ----------
    v : function
        Given times, return signal.
    t0 : ndarray
        Times at which windows begin for calculating nulls.
    windowDuration : float
        Window duration in seconds
    pool : multiprocess.Pool
        Pool for parallelizing computation.
        
    Returns
    -------
    One tuple for each x,y,z axis with
    f : ndarray
        Frequencies at which coherence was calculated
    coh_mean : ndarray
        Mean of coherence over random noise samples.
    coh_std : ndarray
        Std of coherence over random noise samples.
    """
    import multiprocess as mp
    
    def f(t0):
        # Data to analyize.
        t = t0+np.arange(windowDuration*sampling_rate)/sampling_rate
        v_ = v(t)
        
        # Coherence null values for each axis independently.
        Cnullx,Cnully,Cnullz = [],[],[]
        for i in range(n_iters):
            fx,cwtcohx = cwt_coherence_auto_nskip(v_[:,0],np.random.normal(size=len(v_)),
                                         sampling_period=1/sampling_rate,period_multiple=3)
            fy,cwtcohy = cwt_coherence_auto_nskip(v_[:,1],np.random.normal(size=len(v_)),
                                         sampling_period=1/sampling_rate,period_multiple=3)
            fz,cwtcohz = cwt_coherence_auto_nskip(v_[:,2],np.random.normal(size=len(v_)),
                                         sampling_period=1/sampling_rate,period_multiple=3)
            Cnullx.append( cwtcohx )
            Cnully.append( cwtcohy )
            Cnullz.append( cwtcohz )
        Cnullx = np.vstack(Cnullx)
        Cnully = np.vstack(Cnully)
        Cnullz = np.vstack(Cnullz)
        
        mux,stdx = Cnullx.mean(0),Cnullx.std(0)
        muy,stdy = Cnully.mean(0),Cnully.std(0)
        muz,stdz = Cnullz.mean(0),Cnullz.std(0)
        
        return fx,fy,fz,mux,muy,muz,stdx,stdy,stdz

    fx,fy,fz,cohmux,cohmuy,cohmuz,cohstdx,cohstdy,cohstdz = list(zip(*pool.map(f,t0)))
    
    return ( (fx[0],np.vstack(cohmux),np.vstack(cohstdx)),
             (fy[0],np.vstack(cohmuy),np.vstack(cohstdy)),
             (fz[0],np.vstack(cohmuz),np.vstack(cohstdz)) )

def check_coherence_with_null(ref,sample,threshold,
                              sampling_rate=30):
    """
    Given subject's trajectory compare it with the given null and return the fraction of
    frequencies at which the subject exceeds the white noise null which is just a flat cutoff.
    
    Parameters
    ----------
    ref : ndarray
        Reference signal against which to compare the sample. This determines the noise
        threshold.
    sample : ndarray
        Sample signal to compare against reference signal.
    threshold : float
        This determines the constant with which to multiply the power spectrum of the reference
        signal to determine the null cutoff.

    Returns
    -------
    performanceMetric : float
    """
    assert 0<=threshold<=1
    
    # Calculate coherence between given signals.
    f,cwtcoh = cwt_coherence_auto_nskip(ref,sample,
                                        sampling_period=1/sampling_rate,
                                        period_multiple=3)
    
    # Ignore nans
    notnanix = np.isnan(cwtcoh)==0
    betterPerfFreqIx = cwtcoh[notnanix]>threshold
    
    return ( betterPerfFreqIx ).mean()

def phase_coherence(x,y):
    """
    Parameters
    ----------
    x : ndarray
    y : ndarray
    S : ndarray
        Smoothing filter for 2d convolution.

    Returns
    -------
    Phase coherence
    """
    xcwt,f = pywt.cwt(x,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)
    ycwt,f = pywt.cwt(y,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)

    smoothx = np.abs(xcwt)
    smoothy = np.abs(ycwt)
    smoothxy = xcwt*ycwt.conjugate()

    smoothcoh = smoothxy.mean(1) / ( smoothx*smoothy ).mean(1)
    return f,smoothcoh

def tf_phase_coherence(x,y,S):
    """
    Parameters
    ----------
    x : ndarray
    y : ndarray
    S : ndarray
        Smoothing filter for 2d convolution.

    Returns
    -------
    """
    from scipy.signal import convolve2d

    xcwt,f = pywt.cwt(x,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)
    ycwt,f = pywt.cwt(y,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)

    smoothx = convolve2d(np.abs(xcwt)**2,S,mode='same')
    smoothy = convolve2d(np.abs(ycwt)**2,S,mode='same')
    smoothxy = convolve2d(xcwt*ycwt.conjugate(),S,mode='same')
    
    smoothcoh = smoothxy.mean(1) / np.sqrt(( smoothx*smoothy ).mean(1))
    return f,smoothcoh

def tf_coherence(x,y,S):
    """
    Parameters
    ----------
    x : ndarray
    y : ndarray
    S : ndarray
        Smoothing filter for 2d convolution.
    """
    from scipy.signal import convolve2d

    xcwt,f = pywt.cwt(x,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)
    ycwt,f = pywt.cwt(y,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)

    smoothx = convolve2d(np.abs(xcwt)**2,S,mode='same')
    smoothy = convolve2d(np.abs(ycwt)**2,S,mode='same')
    smoothxy = convolve2d(xcwt*ycwt.conjugate(),S,mode='same')

    smoothcoh = smoothxy/np.sqrt(smoothx*smoothy)
    return f,smoothcoh

def coherence_before_vis(subcwt,avcwt,f,vis,dt,min_freq=0,max_freq=10):
    """
    Coherence using the wavelet transform for dt seconds around the visibility turning back on.
    
    Parameters
    ----------
    subcwt
    avcwt
    f : ndarray
        Frequencies.
    vis
    dt : float
        temporal distance from the start of a new visibility section. Positive value is for
        before visibility starts.
    min_freq : float,0
    max_freq : float,10

    Returns
    -------
    Average coherence between (min_freq,max_freq).
    """
    # Get indices of points near the end of the invisibility window.
    dtprev = int(dt*60)
    visStartIx = np.where(np.diff(vis)==1)[0]-dtprev
    visStartIx = visStartIx[(visStartIx>=0)&(visStartIx<len(vis))]

    Psub = ( np.abs(subcwt[:,visStartIx])**2 ).mean(-1)
    Pav = ( np.abs(avcwt[:,visStartIx])**2 ).mean(-1)
    Pcross = ( subcwt[:,visStartIx]*avcwt[:,visStartIx].conjugate() ).mean(-1)
    
    coh = np.abs(Pcross)**2/Psub/Pav
    freqix = (f>=min_freq)&(f<=max_freq)

    # Errors.
    #print ( Psub/( np.abs(subcwt[:,visStartIx])**2 ).std(-1) )[freqix]
    #print ( Pav/( np.abs(avcwt[:,visStartIx])**2 ).std(-1) )[freqix]
    #print ( np.abs(Pcross)**2/(np.abs( subcwt[:,visStartIx]*avcwt[:,visStartIx].conjugate()
    #    )**2).std(-1) )[freqix]

    avgC = np.trapz(coh[freqix],x=f[freqix]) / (f[freqix].max()-f[freqix].min())
    if avgC<0:
        return -avgC
    return avgC

def cwt_coherence(x,y,nskip,
                  scale=np.logspace(0,2,100),
                  sampling_period=1/60,
                  **kwargs):
    """
    Use the continuous wavelet transform to measure coherence.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    nskip : int
        Number of indices to skip when averaging across spectra for coherence. This is to reduce
        correlation between samples when averaging.
    scale : list
        Scale of continuous wavelets.
    sampling_period : float,1/60
        Used to choose scales.
    **kwargs
        for pywt.cwt()

    Returns
    -------
    f : ndarray
    coherence : ndarray
    """
    import pywt
    assert len(x)==len(y)
    xcwt,f = pywt.cwt(x,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    ycwt,f = pywt.cwt(y,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    
    # Get indices of points with some overlap.
    selectix = np.arange(nskip,len(x),nskip,dtype=int)

    Psub = ( np.abs(xcwt[:,selectix])**2 ).mean(-1)
    Pav = ( np.abs(ycwt[:,selectix])**2 ).mean(-1)
    Pcross = ( xcwt[:,selectix]*ycwt[:,selectix].conjugate() ).mean(-1)
    coh = np.abs(Pcross)**2/Psub/Pav
    
    # Skip low frequencies that have periods longer the duration of the window.
    fCutoff = f<(1/(len(x)*sampling_period))

    return f[f>=fCutoff],coh[f>=fCutoff]

def cwt_coherence_auto_nskip(x,y,
                             scale=np.logspace(0,2,100),
                             sampling_period=1/60,
                             period_multiple=1,
                             **kwargs):
    """
    Use the continuous wavelet transform to measure coherence but automatically choose the amount to
    subsample separately for each frequency when averaging. The subsampling is determined by nskip
    which only takes a sample every period of the relevant frequency.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    nskip : int
        Number of indices to skip when averaging across spectra for coherence. This is to reduce
        correlation between samples when averaging.
    scale : list
        Scale of continuous wavelets.
    sampling_period : float,1/60
        Used to choose scales.
    **kwargs
        for pywt.cwt()

    Returns
    -------
    f : ndarray
    coherence : ndarray
    """
    import pywt
    assert len(x)==len(y)
    xcwt,f = pywt.cwt(x,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    ycwt,f = pywt.cwt(y,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    
    Psub = np.zeros(len(f))
    Pav = np.zeros(len(f))
    Pcross = np.zeros(len(f),dtype=complex)
    #PsubStd = np.zeros(len(f))
    #PavStd = np.zeros(len(f))
    #PcrossStd = np.zeros(len(f))

    # For each freq, skip roughly by a period.
    for fIx,f_ in enumerate(f):
        nskip = int(1/f_/sampling_period)
        if nskip>(len(x)//3):
            Psub[fIx] = np.nan
            Pav[fIx] = np.nan
            Pcross[fIx] = np.nan
        else:
            selectix = np.arange(nskip,len(x),nskip,dtype=int)
            #print f_,len(selectix)

            Psub[fIx] = ( np.abs(xcwt[fIx,selectix])**2 ).mean(-1)
            Pav[fIx] = ( np.abs(ycwt[fIx,selectix])**2 ).mean(-1)
            Pcross[fIx] = ( xcwt[fIx,selectix]*ycwt[fIx,selectix].conjugate() ).mean(-1)

            #PsubStd[fIx] = ( np.abs(xcwt[fIx,selectix])**2 ).std(-1)
            #PavStd[fIx] = ( np.abs(ycwt[fIx,selectix])**2 ).std(-1)
            #PcrossStd[fIx] = ( xcwt[fIx,selectix]*ycwt[fIx,selectix].conjugate() ).std(-1)

    coh = np.abs(Pcross)**2/Psub/Pav
    #stds = (PsubStd,PavStd,PcrossStd)

    # Skip low frequencies that have periods longer the duration of the window.
    fCutoffIx = f>(period_multiple/(len(x)*sampling_period))

    return f[fCutoffIx],coh[fCutoffIx]

def max_coh_time_shift(subv,temv,
                       dtgrid=np.linspace(0,1,100),
                       mx_freq=10,
                       sampling_rate=60,
                       window_width=2,
                       disp=False,
                       ax=None):
    """
    Find the global time shift that maximizes the coherence between two signals.
    
    Parameters
    ----------
    subv : ndarray
        Subject time series. If multiple cols, each col is taken to be a data point and the average
        coherence is maximized.
    temv : ndarray
        Template time series.
    dtgrid : ndarray,np.linspace(0,1,100)
    window_width : float,2
        Window duration for computing coherence in terms of seconds.
    disp : bool,False
    ax : AxesSubplot,None
        
    Returns
    -------
    dt : float
        Time shift in seconds that maximizes scipy coherence. Time shift is relative to subject
        time, i.e.  negative shift is shifting subject back in time and positive is shifting subject
        forwards in time. If subject is tracking template, then dt>0.
    maxcoh : float
        Coherence max.
    """
    from scipy.signal import coherence
        
    # Convert dtgrid to index shifts.
    dtgrid = np.unique(np.around(dtgrid*sampling_rate).astype(int))
    if subv.ndim==1:
        coh = np.zeros(len(dtgrid))
    else:
        coh = np.zeros((len(dtgrid),subv.shape[1]))
    window_width = int(sampling_rate*window_width)
    
    def _calc_coh(subv,temv):
        for i,dt in enumerate(dtgrid):
            if dt<0:
                f,c = coherence(subv[-dt:],temv[:dt],fs=sampling_rate,nperseg=window_width)
            elif dt>0:
                f,c = coherence(subv[:-dt],temv[dt:],fs=sampling_rate,nperseg=window_width)
            else:
                f,c = coherence(subv,temv,fs=sampling_rate,nperseg=window_width)
            coh[i] = abs(c)[f<mx_freq].mean()
        return coh
        
    if subv.ndim==1:
        coh = _calc_coh(subv,temv)
    else:
        coh = np.vstack([_calc_coh(subv[:,i],temv[:,i]) for i in range(subv.shape[1])]).mean(1)
    shiftix = np.argmax(coh)
    
    if disp:
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(dtgrid/sampling_rate,coh,'o')
        ax.set(xlabel='dt',ylabel='coherence')
        
    return dtgrid[shiftix]/sampling_rate,coh[shiftix]



# ======= #
# Classes #
# ======= #
class DTWPerformance(object):
    def __init__(self,inner_prod_threshold=.3,
                 norm_dv_threshold=.1,
                 norm_dv_ratio_threshold=np.log2(1.5),
                 dt_threshold=.5,
                 dwt_kwargs={'dist':2}):
        """
        Class for using fast DWT to compare two temporal trajectories and return a performance metric.
        Performance is the fraction of time warped trajectories that the individuals remain within some
        threshold preset.
        
        Parameters
        ----------
        inner_prod_threshold : float,.5
            Max deviation allowed for one minus normalized dot product between vectors.
        norm_dv_threshold : float,.1
            Max difference in speeds allowed. Units of m/s.
        norm_dv_ratio_threshold : float,1
            Max ratio in speeds allowed in units of log2.
        dt_threshold : float,.5
            Max deviation for time allowed.
        """
        self.dwtSettings = dwt_kwargs
        self.innerThreshold = inner_prod_threshold
        self.normdvThreshold = norm_dv_threshold
        self.normdvRatioThreshold = norm_dv_ratio_threshold
        self.dtThreshold = dt_threshold

    def time_average(self,x,y,dt=1.,strict=False,bds=[0,np.inf]):
        """
        Measure performance as the fraction of time you are within the thresholds.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        dt : float,1
            Sampling rate for x and y.
        bds : list,[0,inf]
            Lower and upper bounds for times at which to truncate the data according to x before calculating
            performance.

        Returns
        -------
        performance : float
            Fraction of the length of given trajectories that are within set thresholds.
        """
        from numpy.linalg import norm
        from fastdtw import fastdtw
        from warnings import warn

        dist,path = fastdtw(x,y,**self.dwtSettings)
        try:
            path = np.vstack(path)
        except ValueError:
            warn("fastdtw could not align. Possible because subject data is flat.")
            path=list(range(len(x)))
        keepIx=((path[:,0]*dt)>=bds[0])&((path[:,0]*dt)<=bds[1])

        normx = norm(x[path[:,0]],axis=1)+np.nextafter(0,1)
        normy = norm(y[path[:,1]],axis=1)+np.nextafter(0,1)
        # Dot product between the two vectors.
        inner = (x[path[:,0]]*y[path[:,1]]).sum(1) / normx / normy
        # Relative norms.
        normDiff = np.abs(normx-normy)
        normRatio = np.abs(np.log2(normx)-np.log2(normy))
        dt = np.diff(path,axis=1) * dt

        # Calculate performance metric.
        # In strict case, dt must be within cutoff at all times to get a nonzero performance value.
        # Otherwise, we just take the average time during which subject is within all three norm, inner
        # product, and dt cutoffs.
        if strict:
            if (np.abs(dt)<self.dtThreshold).all():
                performance = ((1-inner)<self.innerThreshold)[keepIx].mean()
            else:
                performance = 0.
        else:
            performance = ( #((normDiff<self.normdvThreshold)|(normRatio<self.normdvRatioThreshold)) &
                            ((1-inner)<self.innerThreshold) & 
                            (np.abs(dt)<self.dtThreshold) )[keepIx].mean()
        
        return performance

    def time_average_binary(self,x,y,dt=1.,bds=[0,np.inf],path=None):
        """
        Measure performance as the fraction of time you are within the thresholds using the two
        Success and Failure states identified in the paper. Use Laplace counting to regularize the
        values.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        dt : float,1
            Sampling rate for x and y.
        bds : list,[0,inf]
            Lower and upper bounds for times at which to truncate the data according to x before calculating
            performance.

        Returns
        -------
        performance : float
            Fraction of the length of given trajectories that are within set thresholds.
        """
        from numpy.linalg import norm
        from fastdtw import fastdtw
        from warnings import warn
        
        if path is None:
            dist,path = fastdtw(x,y,**self.dwtSettings)
            try:
                path = np.vstack(path)
            except ValueError:
                warn("fastdtw could not align. Possible because subject data is flat.")
                path=list(range(len(x)))
        keepIx=((path[:,0]*dt)>=bds[0])&((path[:,0]*dt)<=bds[1])
        dt = np.diff(path,axis=1) * dt

        # Calculate performance metric.
        return ( ((np.abs(dt)<self.dtThreshold))[keepIx].sum()+1 ) / (keepIx.sum()+2)

    def raw(self,x,y,dt=1.):
        """
        Performance as measured by the similarity of time warped trajectories. If time warping is too big,
        then performance is 0.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        dt : float,1
            Sampling rate for x and y.

        Returns
        -------
        performance : float
        """
        from numpy.linalg import norm
        from fastdtw import fastdtw

        dist,path = fastdtw(x,y,**self.dwtSettings)
        path = np.vstack(path)

        # Calculate dot product between the two vectors.
        inner = ( (x[path[:,0]]*y[path[:,1]]).sum(1) / 
                  (norm(x[path[:,0]],axis=1)*norm(y[path[:,1]],axis=1)+np.nextafter(0,1)) )
        dt = np.diff(path,axis=1) * dt

        # Calculate performance metric.
        if (np.abs(dt)<self.dtThreshold).all():
            performance = inner.mean()
            if performance<0:
                performance = 0
        else:
            performance = 0.
        return performance
#end DTWPerformance



class CoherenceEvaluator(object):
    '''
    update() evaluates the average coherence over the given time.
    These assume V_person and V_avatar are pre-aligned and have the same length.
    '''
    def __init__(self,maxfreq,sample_freq=60,window_length=90):
        '''
        Parameters
        ----------
        maxfreq : float
            Max frequency up to which to average coherence.
        sampleFreq : float,60
            Sampling frequency
        windowLength : int,90
            Number ofdata points to use in window for coherence calculation.

        Subfields
        ---------
        coherence
        coherences
        '''
        self.maxfreq = maxfreq
        self.sampleFreq = sample_freq
        self.windowLength = window_length
        
        self.v = None
        self.v_av = None
        
        self.coherence = 0
        self.coherences = np.empty(0)
        self.performanceValues = np.empty(0)
        
    def getCoherence(self):
        return self.coherence
    
    def getAverageCoherence(self):
        '''
        For GPR: returns average coherence over a full trial. Coherences should
        then be reset for the new trial.
        '''
        return np.mean(self.coherences)
    
    def resetCoherences(self):
        self.coherences = np.empty(0)
    
    def getPerformanceValues(self):
        return self.performanceValues
    
    def getAveragePerformance(self):
        return np.mean(self.performanceValues)
    
    def resetPerformance(self):
        self.performanceValues = np.empty(0)
        
    def evaluateCoherence(self,v1,v2,use_cwt=True):
        '''
        Returns average coherence between current v and v_av data vectors.

        Parameters
        ----------
        v1 : ndarray
            Vector.
        v2 : ndarray
            Vector.

        Returns
        -------
        avg_coh : float
        '''
        assert len(v1)==len(v2)
            
        if not use_cwt:
            # Scipy.signal's coherence implementation.
            self.f,self.C = coherence(v1,v2,
                                      fs=self.sampleFreq,
                                      nperseg=self.windowLength,
                                      nfft=2 * self.windowLength,
                                      noverlap=self.windowLength//4)
        else:
            self.f,self.C = cwt_coherence(v1,v2,1,sampling_period=1/self.sampleFreq)
            self.C *= -1 

        # Evaluate Average Coherence by the Trapezoid rule.
        freqIx = (self.f>0)&(self.f<self.maxfreq)
        avg_coherence = np.trapz(self.C[freqIx],x=self.f[freqIx]) / (self.f[freqIx].max()-self.f[freqIx].min())
        
        if np.isnan(avg_coherence): avg_coherence = 0.
        return avg_coherence
    
    def evaluatePerformance(self):
        '''
        Evaluates average coherence against threshold value, and writes binary
        value to target output file.

        Returns
        -------
        performance
        '''
        performance = 0
        avg_coherence = self.evaluateCoherence()
        
        if avg_coherence >= self.performanceThreshold:
            performance = 1
        
        self.coherences = np.append(self.coherences,avg_coherence)
        self.performanceValues = np.append(self.performanceValues,performance)
        return performance
# end CoherenceEvaluator



class GPR(object):
    def __init__(self,
                 alpha = .2,
                 mean_performance=np.log(1),
                 theta=.5,
                 length_scale=np.array([1.,.2]),
                 tmin=0.5,tmax=2,tstep=0.1,
                 fmin=0.1,fmax=1.,fstep=0.1):
        '''
        Wrapper around GPR class to perform useful functions for HandSyncExperiment.

        Parameters
        ----------
        alpha : float
            Uncertainty in diagonal matrix for GPR kernel.
        mean_performance : float,.5
            By default, the sigmoid is centered around 0, the mean of the Gaussian process, corresponding to
            perf=0.5. However, we should center the sigmoid around the mean value of y which is specified
            here. Since the GPR is trained in the logistic space, the offset is given by the logistic offset.
            The mean is automatically accounted for under the hood, so you don't have to worry about adding or
            subtracting it in the interface.
        theta : float
            Coefficient for kernel.
        length_scale : ndarray
            (duration_scale,fraction_scale) Remember that duration is the angle about the origin restricted to
            be between [0,pi] and fraction is the radius. The GPR learns once (r,theta) has been mapped to the
            Cartesian coordinate system.
        tmin : float,0.5
            minimum window time
        tmax : float,2
        tstep : float,0.1
        fmin : float
            minimum visibility fraction.
        fmax : float
        fstep : float

        Members
        -------
        fractions : ndarray
            Fraction of time stimulus is visible.
        durations : ndarray
            Duration of window.
        meshPoints : ndarray
            List of grid points (duration,fraction) over which performance was measured.
        performanceData : ndarray
            List of performance data points that have been used to update GPR.
        performanceGrid : ndarray
            Unraveled grid of performance predictions on mesh of duration and fraction.
        '''
        assert (type(length_scale) is np.ndarray) and len(length_scale)==2

        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.fmin = fmin
        self.fmax = fmax
        self.fstep = fstep
        
        self.length_scale = length_scale
        self.kernel = self.handsync_experiment_kernel(length_scale,theta)
        self.alpha = alpha
        self.theta = theta
        
        self.durations = np.zeros(0)
        self.fractions = np.zeros(0)
        self.performanceData = np.zeros(0)
        self.mean_performance = mean_performance
        
        # Create two grids for t and f.
        self.tRange = np.arange(self.tmin,self.tmax+self.tstep/2,self.tstep)
        self.fRange = np.arange(self.fmin,self.fmax+self.fstep/2,self.fstep)
        self.meshPoints = np.meshgrid(self.tRange,self.fRange)
        # Flatten t and f grids and stack them into an Nx2 array.
        self.meshPoints = np.vstack([x.ravel() for x in self.meshPoints]).T
        
        self.gp = GaussianProcessRegressor(self.kernel,alpha**-2)
        self.performanceGrid = 0  # [-inf,inf]
        self.std_pred = 0

        self.pointsToAvoid = []
   
    def predict(self,mesh=None):
        """
        Fits the GPR to all data points and saves the predicted values with errors. The mean in the target
        perf values is accounted for here.

        Updates self.performanceGrid and self.std_pred with the latest prediction.

        If you want to just query the model, you should access self.gp directly.

        Parameters
        ----------
        mesh : ndarray,None
            Points at which to evaluate GPR. Should be (samples,2) with first column durations and the second
            fraction of visible window.

        Returns
        -------
        perf : ndarray
            Performance grid.
        perfErr : ndarray
            Performance estimated standard deviation.
        """
        if mesh is None:
            mesh = self.meshPoints

        self.gp.fit( np.vstack((self.durations,self.fractions)).T,self.performanceData-self.mean_performance )
        self.performanceGrid, self.std_pred = self.gp.predict(mesh,return_std=True)
        self.performanceGrid += self.mean_performance

        return self.performanceGrid.copy(),self.std_pred.copy()

    def grad(self,eps=1e-5):
        '''
        Estimates the gradient at each point of the mesh.

        Parameters
        ----------
        eps : float,1e-5

        Returns
        -------
        grad : ndarray
            Dimensions (n_tRange,n_fRange,2). Last dimension corresponds to the gradient along each diemnsion
            of the input.
        '''
        grad = np.zeros((len(self.meshPoints),2))
        X1 = self.meshPoints.copy()
        X0 = self.meshPoints.copy()

        X1[:,0] += eps 
        X0[:,0] -= eps 
        grad[:,0] = ( self.gp.predict(X1)-self.gp.predict(X0) )/(2*eps)
        X1[:,0] -= eps
        X0[:,0] += eps

        X1[:,1] += eps
        X0[:,1] -= eps
        grad[:,1] = ( self.gp.predict(X1)-self.gp.predict(X0) )/(2*eps)
        
        shape = len(self.fRange),len(self.tRange)
        grad = np.concatenate((grad[:,0].reshape(shape)[:,:,None],grad[:,1].reshape(shape)[:,:,None]),2)
        return grad
        
    def max_uncertainty(self,explore_new=True):
        """
        Returns next_duration,next_fraction as the point where the variance of the GPR is max
        Currently finds maximum uncertainty, and then returns a point with that uncertainty as the
        update value.

        Avoids values that have already been measured and fraction=1.

        Parameters
        ----------
        explore_new : bool,True 
        
        Returns
        -------
        next_window_duration : float
        next_vis_fraction : float
        """
        sortIx=np.argsort(self.std_pred)[::-1]
        
        if explore_new:
            next_fraction=1
            counter=0
            while next_fraction==1 or ((next_duration in self.durations) and (next_fraction in self.fractions)):
                next_duration=self.meshPoints[sortIx[counter]][0]
                next_fraction=self.meshPoints[sortIx[counter]][1]
                counter+=1
        else:
            next_fraction=1
            counter=0
            while next_fraction==1:
                next_duration=self.meshPoints[sortIx[counter]][0]
                next_fraction=self.meshPoints[sortIx[counter]][1]
                counter+=1
        
        return next_duration,next_fraction
        
    def select_contour(self,pstar,choose_algo='',algo_kwargs={}):
        """
        Select the point in the field with mean closest to desired performance value in the [0,1] space.

        Option to avoid points so that we don't pick the same points over and over again.

        Parameters
        ----------
        pstar : ndarray
            Performance value around which to choose points. This is in the [-inf,inf] stretched space.
        choose_algo : str,''
            If 'avoid', do not sample points in self.pointsToAvoid. Typically, this will be a stored list of
            points already sampled.
            If 'err', weight choice by the uncertainty. Choose the point that minimize the distance to pstar
            with max error. First, thresholds points by distance to pstar. Must specify 'threshold' and
            'std_scale'.
        algo_kwargs : dict,{}
            Any keyword args needed by algorithm for choosing the next point.

        Returns
        -------
        duration : float
        fraction : float
        """
        if type(pstar) is int:
            pstar = float(pstar)
        if type(pstar) is float:
            pstar = np.array([pstar])
        
        if choose_algo=='avoid':
            ix = []
            for pstar_ in pstar:
                sortix = np.argsort( np.abs(self.logistic(self.performanceGrid)-pstar_) )
                counter = 0
                while counter<len(sortix):
                    if not any(np.array_equal(self.meshPoints[sortix[counter]],x)
                               for x in self.pointsToAvoid):
                        ix.append(sortix[counter])
                        counter = len(sortix)
                    counter += 1
        elif choose_algo=='err':
            assert not algo_kwargs.get('threshold',None) is None
            assert not algo_kwargs.get('std_scale',None) is None

            ix = []
            for pstar_ in pstar:
                dist = np.abs(self.logistic(self.performanceGrid)-pstar_)
                thresholdIx = dist<algo_kwargs['threshold']
                dist[thresholdIx==0] += 1e30
                ix.append( np.argmin(dist-algo_kwargs['std_scale']*self.std_pred) )
        else:
            ix = [np.argmin(np.abs(self.performanceGrid-pstar_)) for pstar_ in pstar]
        return self.meshPoints[ix,0],self.meshPoints[ix,1]
    
    def update(self,new_performance,window_dur,vis_fraction):
        '''
        This is called to add new data point to prediction.

        Parameters
        ----------
        new_performance : float
        window_dur : float
        vis_fraction : float
        '''
        self.performanceData = np.append(self.performanceData,new_performance)
        self.durations = np.append(self.durations,window_dur)
        self.fractions = np.append(self.fractions,vis_fraction)
        
        self.predict()

    def _search_hyperparams(self,initial_guess,n_restarts=1,min_alpha_bound=1e-3):
        """Find the hyperparameters that maximize the log likelihood of the data.

        This doesn't seem well-behaved with the length_scale parameters so those are not optimized.

        Parameters
        ----------
        n_restarts : int,1
        min_alpha_bound : float,1e-3
        """
        from scipy.optimize import minimize

        def train_new_gpr(params):
            length_scale=params[:2]
            alpha=params[2]
            mean_performance=params[3]
            theta=params[4]

            kernel=self.handsync_experiment_kernel(length_scale,theta)
            gp=GaussianProcessRegressor(kernel,alpha**-2)
            gp.fit( np.vstack((self.durations,self.fractions)).T,
                    self.performanceData-mean_performance )
            return gp

        def f(params):
            """First parameter is std, second is mean, third is coefficient for kernel."""
            assert len(params)==3
            if params[0]<min_alpha_bound:
                return 1e30
            if params[2]<=0:
                return 1e30
            params=np.concatenate((self.length_scale,params))
            gp=train_new_gpr(params)
            return -gp.log_likelihood()
        
        soln=[]
        soln.append( minimize(f,initial_guess) )
        for i in range(1,n_restarts):
            initial_guess=np.array([np.random.exponential(),np.random.normal(),np.random.exponential()])
            soln.append( minimize(f,initial_guess) )

        if len(soln)>1:
            minLikIx=np.argmin([s['fun'] for s in soln])
            soln=[soln[minLikIx]]
        return soln[0]

    def optimize_hyperparams(self,initial_guess=None,verbose=False):
        """Find the hyperparameters that optimize the log likelihood and reset the kernel and the
        GPR landscape.

        Parameters
        ----------
        initial_guess : ndarray
            (alpha,mean,theta)
        verbose : bool,False
        """
        from datetime import datetime

        if initial_guess is None:
            initial_guess=np.array([self.alpha,self.mean_performance,self.theta])
        else:
            assert len(initial_guess)==3
        
        t0=datetime.now()
        soln=self._search_hyperparams(initial_guess)
        if verbose:
            print("Optimal hyperparameters are\nalpha=%1.2f, mu=%1.2f"%(soln['x'][0],soln['x'][1]))
        self.alpha,self.mean_performance,self.theta=soln['x']
        
        # Refresh kernel.
        self.kernel=self.handsync_experiment_kernel(self.length_scale,self.theta)
        self.gp=GaussianProcessRegressor(self.kernel,self.alpha**-2)
        self.predict()

        if verbose:
            return "GPR hyperparameter optimization took %1.2f s."%(datetime.now()-t0).total_seconds()

    @staticmethod
    def _scale_erf(x,mu,std):
        from scipy.special import erf
        return .5 + erf( (x-mu)/std/np.sqrt(2) )/std/2
   
    @staticmethod
    def ilogistic(x):
        return -np.log(1/x-1)
    
    @staticmethod
    def logistic(x):
        return 1/(1+np.exp(-x))

    def handsync_experiment_kernel(self,length_scales,theta):
        """
        Calculates the RBF kernel for one pair of (window_duration,visible_fraction). Specifically for the
        handsync experiment.
        
        Custom kernel converts (duration,fraction) radial coordinate system pair into Cartesian coordinates
        and then calculates distance to calculate GPR in this space.

        Parameters
        ----------
        length_scales : ndarray
            2 element vector specifying (duration,fraction)
        theta : float
            Coefficient in front of kernel.

        Returns
        -------
        kernel : function
        """
        from numpy.linalg import norm
        assert length_scales[0]>=(self.tmax/np.pi)
        delta = define_delta(1,width=.0)
        
        def kernel(tfx,tfy,length_scales=length_scales):
            xy0 = np.array([ (1-tfx[1])/length_scales[1]*np.cos(tfx[0]/length_scales[0]),
                             (1-tfx[1])/length_scales[1]*np.sin(tfx[0]/length_scales[0]) ])
            xy1 = np.array([ (1-tfy[1])/length_scales[1]*np.cos(tfy[0]/length_scales[0]),
                             (1-tfy[1])/length_scales[1]*np.sin(tfy[0]/length_scales[0]) ])

            return np.exp( -((xy0-xy1)**2).sum() )*theta
        return kernel
#end GPR



class GPREllipsoid(GPR):
    def __init__(self,*args,**kwargs):
        """
        Modeling performance landscape to live on the surface of an ellipsoid. For now, it is
        assumed that it is a sphere because optimization with the major and minor axes seems
        difficult.
        But that may have had to do with a bad form of the kernel.
        """
        from geographiclib.geodesic import Geodesic
        super(GPREllipsoid,self).__init__(*args,**kwargs)
        self.DEFAULT_LENGTH_SCALE=100
        self._geodesic=Geodesic(self.DEFAULT_LENGTH_SCALE,0)
        
        if self.alpha<1:
            self.alpha=1  # make this big to improve hyperparameter search
        self.length_scale=self.DEFAULT_LENGTH_SCALE
        self.dist_power=1.
        self._update_kernel(self.theta,self.length_scale)

    def _search_hyperparams_no_length_scale(self,n_restarts=1,
                                            initial_guess=None,
                                            alpha_bds=(0,np.inf),
                                            coeff_bds=(0,np.inf)):
        """Find the hyperparameters alpha, mu, theta that maximize the log likelihood of the data.
        These are the noise std, mean performance, and kernel coefficient.

        Parameters
        ----------
        n_restarts : int,1
        initial_guess : list
        alpha_bds : tuple
            Lower and upper bounds on alpha.
        """
        from scipy.optimize import minimize
        if initial_guess is None:
            initial_guess=np.array([self.alpha,self.mean_performance,self.theta])

        def train_new_gpr(params):
            alpha,mean_performance,coeff=params
            
            kernel=self.define_kernel(coeff,self.length_scale)
            gp=GaussianProcessRegressor(kernel,alpha**-2)
            gp.fit( np.vstack((self.durations,self.fractions)).T,self.performanceData-mean_performance )
            return gp

        def f(params):
            if not alpha_bds[0]<params[0]<alpha_bds[1]: return 1e30
            if not coeff_bds[0]<params[2]<coeff_bds[1]: return 1e30

            gp=train_new_gpr(params)
            return -gp.log_likelihood()
        
        # Parameters are noise std, mean perf
        if n_restarts>1:
            initial_guess=np.vstack((initial_guess,
                                    np.vstack((np.random.exponential(size=n_restarts-1),
                                               np.random.normal(size=n_restarts-1),
                                               np.random.exponential(size=n_restarts-1))).T ))
            pool=mp.Pool(mp.cpu_count())
            soln=pool.map( lambda x:minimize(f,x),initial_guess )
            pool.close()
        else:
            soln=[minimize(f,initial_guess)]

        if len(soln)>1:
            minNegLikIx=np.argmin([s['fun'] for s in soln])
            soln=[soln[minNegLikIx]]
        return soln[0]

    def _search_hyperparams(self,n_restarts=1,
                            initial_guess=None,
                            alpha_bds=(1e-3,np.inf),
                            coeff_bds=(0,np.inf),
                            min_ocv=False):
        """Find the hyperparameters that maximize the log likelihood of the data including length
        scale parameters on the surface of ellipsoid.
        
        Must run several times for good results with minimizing length scale parameters.

        Parameters
        ----------
        n_restarts : int,1
        initial_guess : list,None

        Returns
        -------
        soln : dict
            As returned by scipy.optimize.minimize.
        """
        from scipy.optimize import minimize
        if initial_guess is None:
            initial_guess=np.array([self.alpha,self.mean_performance,self.theta,self.length_scale])

        def train_new_gpr(params):
            alpha,mean_performance,coeff,length_scale=params
            
            kernel=self.define_kernel(coeff,length_scale)
            gp=GaussianProcessRegressor(kernel,alpha**-2)
            gp.fit( np.vstack((self.durations,self.fractions)).T,self.performanceData-mean_performance )
            return gp

        def f(params):
            if not alpha_bds[0]<params[0]<alpha_bds[1]: return 1e30
            if not coeff_bds[0]<params[2]<coeff_bds[1]: return 1e30

            # Bound length_scale to be above certain value.
            if params[3]<=10: return 1e30

            gp=train_new_gpr(params)
            predMu,predStd=gp.predict(gp.X,return_std=True)
            if np.isnan(predStd).any():
                return 1e30
            try:
                if min_ocv:
                    return gp.ocv_error()
                return -gp.log_likelihood()
            except AssertionError:
                # This is printed when the determinant of the covariance matrix is not positive.
                print("Bad parameter values %f, %f, %f, %f"%tuple(params))
                return 1e30
        
        # Parameters are noise std, mean perf, equatorial radius, oblateness.
        if n_restarts>1:
            initial_guess=np.vstack((initial_guess,
                np.vstack((np.random.exponential(size=n_restarts-1),
                           np.random.normal(size=n_restarts-1),
                           np.random.exponential(size=n_restarts-1,),
                           np.random.exponential(size=n_restarts-1,scale=self.DEFAULT_LENGTH_SCALE)+10)).T ))
            pool=mp.Pool(mp.cpu_count())
            soln=pool.map( lambda x:minimize(f,x),initial_guess )
            pool.close()
        else:
            soln=[minimize(f,initial_guess)]

        if len(soln)>1:
            minNegLikIx=np.argmin([s['fun'] for s in soln])
            soln=[soln[minNegLikIx]]
        return soln[0]

    def _search_hyperparams_no_mean(self,n_restarts=1,
                                    initial_guess=None,
                                    alpha_bds=(1e-3,np.inf),
                                    coeff_bds=(0,np.inf),
                                    min_ocv=False):
        """Find the hyperparameters that maximize the log likelihood of the data while fixing the
        mean.
        
        Parameters
        ----------
        n_restarts : int,1
        initial_guess : list,None

        Returns
        -------
        soln : dict
            As returned by scipy.optimize.minimize.
        """
        from scipy.optimize import minimize
        if initial_guess is None:
            initial_guess=np.array([self.alpha,self.theta,self.length_scale])
        else: assert len(initial_guess)==3
        mean_performance=self.mean_performance

        def train_new_gpr(params):
            alpha,coeff,length_scale=params
            
            kernel=self.define_kernel(coeff,length_scale)
            gp=GaussianProcessRegressor(kernel,alpha**-2)
            gp.fit( np.vstack((self.durations,self.fractions)).T,self.performanceData-mean_performance )
            return gp

        def f(params):
            if not alpha_bds[0]<params[0]<alpha_bds[1]: return 1e30
            if not coeff_bds[0]<params[1]<coeff_bds[1]: return 1e30

            # Bound length_scale to be above certain value.
            if params[2]<=10: return 1e30

            gp=train_new_gpr(params)
            predMu,predStd=gp.predict(gp.X,return_std=True)
            if np.isnan(predStd).any():
                return 1e30
            try:
                if min_ocv:
                    return gp.ocv_error()
                return -gp.log_likelihood()
            except AssertionError:
                # This is printed when the determinant of the covariance matrix is not positive.
                print("Bad parameter values %f, %f, %f"%tuple(params))
                return 1e30
        
        # Parameters are noise std, mean perf, equatorial radius, oblateness.
        if n_restarts>1:
            initial_guess=np.vstack((initial_guess,
                np.vstack((np.random.exponential(size=n_restarts-1),
                           np.random.exponential(size=n_restarts-1,),
                           np.random.exponential(size=n_restarts-1,scale=self.DEFAULT_LENGTH_SCALE)+10)).T ))
            pool=mp.Pool(mp.cpu_count())
            soln=pool.map( lambda x:minimize(f,x),initial_guess )
            pool.close()
        else:
            soln=[minimize(f,initial_guess)]

        if len(soln)>1:
            minNegLikIx=np.argmin([s['fun'] for s in soln])
            soln=[soln[minNegLikIx]]
        return soln[0]

    def optimize_hyperparams(self,verbose=False,
                             exclude_parameter=None,
                             initial_guess=None,
                             n_restarts=4,
                             use_ocv=False):
        """Find the hyperparameters that optimize the log likelihood and reset the kernel and the
        GPR landscape.

        Parameters
        ----------
        verbose : bool,False
        exclude_parameter : str,None
            Name of the parameter to exclude from optimization. Can be 'length_scale',
            'mean_performance'.
        initial_guess : ndarray,None
        n_restarts : int,4
        use_ocv : bool,False
            If True, minimize the OCV error instead of maximizing log likelihood.

        Returns
        -------
        logLikelihood : float
            Log likelihood of the data given the found parameters.
        """
        # Optimize all parameters.
        if exclude_parameter is None:
            if initial_guess is None:
                initial_guess=[self.alpha,self.mean_performance,self.theta,self.length_scale]

            soln=self._search_hyperparams(n_restarts=n_restarts,
                                          initial_guess=initial_guess,
                                          alpha_bds=(2e-1,1e3),
                                          coeff_bds=(.1,10),
                                          min_ocv=use_ocv)
            if verbose:
                print(( "Optimal hyperparameters are\n"+
                       "alpha=%1.2f, mu=%1.2f, coeff=%1.2f, length_scale=%1.2f"%tuple(soln['x']) ))
            self.alpha,self.mean_performance,self.theta,self.length_scale=soln['x']
        # Do not optimize mean performance.
        elif exclude_parameter=='mean_performance':
            if initial_guess is None:
                initial_guess=[self.alpha,self.theta,self.length_scale]

            soln=self._search_hyperparams_no_mean(n_restarts=n_restarts,
                                                  initial_guess=initial_guess,
                                                  alpha_bds=(2e-1,1e3),
                                                  coeff_bds=(.1,10),
                                                  min_ocv=use_ocv)
            if verbose:
                print(( "Optimal hyperparameters are\n"+
                       "alpha=%1.2f, coeff=%1.2f, length_scale=%1.2f"%tuple(soln['x']) ))
            self.alpha,self.theta,self.length_scale=soln['x']
        # Do not optimize length scale."
        elif exclude_parameter=='length_scale':
            if initial_guess is None:
                initial_guess=[self.alpha,self.mean_performance,self.theta,self.length_scale]

            soln=self._search_hyperparams_no_length_scale(n_restarts=n_restarts,
                                                          initial_guess=initial_guess,
                                                          alpha_bds=(2e-1,1e3),
                                                          coeff_bds=(.1,10))
            if verbose:
                print("Optimal hyperparameters are\nalpha=%1.2f, mu=%1.2f"%tuple(soln['x']))
            self.alpha,self.mean_performance,self.theta=soln['x']

        else: raise Exception("Unrecognized parameter to exclude.")

        # Refresh kernel.
        self._update_kernel(self.theta,self.length_scale)
        self.predict()

        return soln['fun']

    def print_parameters(self):
        print("Noise parameter alpha = %1.2f"%self.alpha)
        print("Mean performance mu = %1.2f"%self.mean_performance)
        print("Kernel coeff theta = %1.2f"%self.theta)
        print("Kernel length scale el = %1.2f"%self.length_scale)
        print("Kernel exponent = %1.2f"%self.dist_power)
    
    @staticmethod
    def _kernel(_geodesic,tmin,tmax,coeff,length_scale,dist_power):
        """Return kernel function as defined with given parameters.
        """
        assert tmax>tmin
        assert length_scale>0

        def kernel_function(tfx,tfy):
            """Takes in pairs (t,f) where t is duration and f is fraction."""
            # Account for cases where f=1.
            if tfx[0]==0:
                lon0=0
            else:
                lon0=(tfx[0]-.5)*180/(tmax-tmin)

            if tfy[0]==0:
                lon1=0
            else:
                lon1=(tfy[0]-.5)*180/(tmax-tmin)

            lat0=(tfx[1]-.5)*180
            lat1=(tfy[1]-.5)*180
            return coeff*np.exp( -_geodesic.Inverse(lat0,lon0,lat1,lon1)['s12']**dist_power/length_scale )
        return kernel_function

    def define_kernel(self,coeff,length_scale,dist_power=None):
        """Define new Geodesic within given parameters and wrap it nicely.

        Parameters
        ----------
        coeff : float
            Coefficient in front of kernel.
        length_scale : float
            Length scale used in the kernel.
        """
        dist_power=dist_power or self.dist_power
        return self._kernel(self._geodesic,self.tmin,self.tmax,coeff,length_scale,dist_power)

    def _update_kernel(self,coeff,length_scale,dist_power=None):
        """Update instance Geodesic kernel parameters and wrap it nicely.

        Performance grid is not updated. Must run self.predict() if you wish to do that.

        Parameters
        ----------
        coeff : float
            Coefficient in front of kernel.
        length_scale : float
            Length scale used in the kernel.
        """
        dist_power=dist_power or self.dist_power
        self.kernel=self._kernel( self._geodesic,self.tmin,self.tmax,coeff,length_scale,dist_power )
        self.gp=GaussianProcessRegressor( self.kernel,self.alpha**-2 )
#end GPREllipsoid



def define_delta(x,width=0.):
    """
    Return delta function or Gaussian approximation with given width.

    Parameters
    ----------
    x : float
        delta(x-y)
    width : float,0.

    Returns
    -------
    function
    """
    if width>0:
        return lambda xp:np.exp(-(xp-x)**2/2/width**2)
    return lambda xp:0. if xp!=x else 1.

def handsync_experiment_kernel(self,length_scales):
        """
        For backwards compatibility. See GPR. 
        """
        warn("Only for backwards compatibility.")
        from numpy.linalg import norm
        delta = define_delta(1,width=.00)
        
        def kernel(tfx,tfy,length_scales=length_scales):
            xy0 = np.array([ (1-tfx[1])/length_scales[1]*np.cos(tfx[0]/length_scales[0]),
                             (1-tfx[1])/length_scales[1]*np.sin(tfx[0]/length_scales[0]) ])
            xy1 = np.array([ (1-tfy[1])/length_scales[1]*np.cos(tfy[0]/length_scales[0]),
                             (1-tfy[1])/length_scales[1]*np.sin(tfy[0]/length_scales[0]) ])

            return np.exp( -((xy0-xy1)**2).sum() )
        return kernel


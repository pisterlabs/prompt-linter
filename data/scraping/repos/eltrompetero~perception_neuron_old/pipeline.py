# Module for quick access to analysis pipelines.
# These are functions that iterate over multiple individuals or trials.
# 2017-03-31

import pickle
import numpy as np
import os
from .data_access import *
from .axis_neuron import *
from .utils import *
from .filter import *
from numpy import pi


def distance_dtw(spec_list,trial_type,trials,
                  precision=.1,
                  firstix=0,
                  disp=1):
    """
    Calculate dtw statistics over all given trials for given window specs.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    precision : float,.1
    firstix : int
    disp : bool,True

    Returns
    -------
    distmat : ndarray
    dtmat : ndarray
        Cols are [avg_dt,std_dt,min,max]
    """
    # In last dim, first three cols correspond to xyz dimensions and last col is norm.
    distmat = np.zeros((len(trials),len(spec_list),4))
    dtmat = np.zeros((len(trials),len(spec_list),4))
    
    for itrial,trial in enumerate(trials):
        for specix,spec in enumerate(spec_list):
            # Get subject and template velocities.
            output = _compare_dtw(trial,[spec]*2,[trial_type]*2,[precision]*2,
                                  firstix=firstix,disp=disp)
                        
            if not output is None:
                distmat[itrial,specix,:] = np.sqrt( (output[0]**2).mean(0) )
                dt = np.diff(output[1],1)/30
                dtmat[itrial,specix,:] = dt.mean(),dt.std(),dt.min(),dt.max()
            else:
                distmat[itrial,specix,:] = np.nan 
                dtmat[itrial,specix,:] = np.nan 
    return distmat,dtmat

def _compare_dtw(trial,windows,trial_types,precisions,
                       firstix=0,disp=True,
                       template_only=False):
    """
    Compare coherence for the given windows specified for the subject and for the template.

    Parameters
    ----------
    trial : VRTrial
    windows : list of tuples
    trial_types : list of strings
    precisions : list of precision
    firstix : int,0
    disp : bool,True
    template_only : bool,False

    Returns
    -------
    dist : ndarray
        (n_time,4) Cols are x, y, z then cosine of angle between velocity vectors.
    path : ndarray
    """
    from fastdtw import fastdtw
    try:
        if template_only:
            sspec,t,subjectv = trial.template_by_window_spec([windows[0]],
                                                           trial_types[0],
                                                           precisions[0]
                                                          )[firstix]
        else:
            sspec,t,subjectv = trial.subject_by_window_spec([windows[0]],
                                                             trial_types[0],
                                                             precisions[0]
                                                            )[firstix]
        tspec,t,templatev = trial.template_by_window_spec([windows[1]],
                                                           trial_types[1],
                                                           precisions[1]
                                                          )[firstix]
        # Ensure that data sets are of the same size.
        #t = t[-1200:]
        #subjectv = subjectv[-1200:]
        #templatev = templatev[-1200:]
        #assert (len(subjectv)==1200) and (len(templatev)==1200)

        if disp:
            print("Subject: (%1.1f,%1.1f), Template: (%1.1f,%1.1f)"%(sspec[0],sspec[1],
                                                                     tspec[0],tspec[1]))
        
        if len(subjectv)==0:
            return
        
        # Run dtw.
        dtwdist,path = fastdtw(subjectv,templatev)
        path = np.vstack(path)
        dist = np.zeros((len(path),4))
        for dimIx in range(3):
            dist[:,dimIx] = subjectv[path[:,0],dimIx]-templatev[path[:,1],dimIx]
        dist[:,3] = ( (subjectv[path[:,0]]*templatev[path[:,1]]).sum(1) /
                      np.linalg.norm(subjectv[path[:,0]],axis=1)/np.linalg.norm(templatev[path[:,1]],axis=1) )
        return dist,path
    except Exception as err:
        if disp:
            print("No data for window spec (%1.1f,%1.1f)."%(windows[0][0],
                                                            windows[0][1]))
        return

def pipeline_phase_lag(v1,v2,dt,
                       maxshift=60,
                       windowlength=100,
                       v_threshold=.03,
                       measure='dot',
                       save='temp.p'):
    """
    Find phase lag for each dimension separately and for the vector including all dimensions together.
    
    Params:
    -------
    v1,v2,dt
    maxshift (int=60)
    windowlength (int=100)
    v_threshold (float=.03)
    save (str='temp.p')
    """
    import pickle as pickle

    phasexyz,overlapcostxyz = [],[]
    for i in range(3):
        p,o = phase_lag(v1[:,i],v2[:,i],maxshift,windowlength,
                        measure=measure,dt=dt)
        phasexyz.append(p)
        overlapcostxyz.append(o)
    phase,overlapcost = phase_lag(v1,v2,maxshift,windowlength,
                                  measure=measure,dt=dt)
    
    if save:
        print("Pickling results as %s"%save)
        pickle.dump({'phase':phase,'overlapcost':overlapcost,
                     'phasexyz':phasexyz,'overlapcostxyz':overlapcostxyz,
                     'maxshift':maxshift,'windowlength':windowlength,
                     'measure':measure,
                     'v1':v1,'v2':v2},
                    open(save,'wb'),-1)
    return phasexyz,phase,overlapcostxyz,overlapcost

def max_coherence(windowSpec,trials):
    """
    Time shifted max coherence across all trials.
    """
    maxcoh = np.zeros((len(trials),4))
    
    for trialno,trial in enumerate(trials):
        subjectVel = trial.subject_by_window_spec([windowSpec],'hand',(.05,.11))
        templateVel = trial.template_by_window_spec([windowSpec],'hand',(.05,.11))
        visWindow = trial.visibility_by_window_spec([windowSpec],'hand',(.05,.11))


        # Load data.
        if len(subjectVel[0][2])>0:
            subt = subjectVel[0][1]
            subv = subjectVel[0][2]
            subt = subt[:len(subv)]

            temt = templateVel[0][1]
            temv = templateVel[0][2]

            mnlen = min([len(subt),len(temt)])
            temt,temv = temt[:mnlen],temv[:mnlen]
            vist = visWindow[0][1]

            # coherence as graphs are time shifted.
            cohdtShifts = np.zeros(3)
            for i in range(3):
                cohdtShifts[i] = max_coh_time_shift(subv[:,i],temv[:,i],
                                                 disp=False,dtgrid=np.linspace(-1,1,100))

            firstix = 0
            maxcoh[trialno],_ = coherence([subjectVel[0][0]],'hand',[trial],
                                           firstix=firstix,
                                           precision=(.05,.1),
                                           offset=int(-cohdtShifts[1:].mean()*60))
        else:
            maxcoh[trialno] = np.nan
    return maxcoh

def coherence(spec_list,trial_type,trials,
              mx_freq=10,
              precision=.1,
              firstix=0,
              offset=None,
              cwt=False,
              disp=1):
    """
    Calculate average coherence over all given trials for given window specs.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    mx_freq : int,10
        Maximum frequency over which to average coherence
    precision : float,.1
    firstix : int
    offset : int,None
        Number of indices to offset the subject and template time series. If offset>0, we skip the
        first offset elements from subject. If offset<0, -offset elements are removed from the
        subject.
    disp : bool,True

    Returns
    -------
    cohmat : ndarray
        Coherence statistic for all trials and all given window specs for xyz and vel norm.
    cohmaterr : ndarray
        Standard error of the mean of coherence statistic across trials.
    """
    # In last dim, first three cols correspond to xyz dimensions and last col is norm.
    cohmat = np.zeros((len(trials),len(spec_list),4))
    
    for itrial,trial in enumerate(trials):
        for specix,spec in enumerate(spec_list):
            # Get subject and template velocities.
            cohOutput = _compare_coherence(trial,[spec]*2,[trial_type]*2,[precision]*2,mx_freq,
                                           firstix=firstix,disp=disp,offset=offset,cwt=cwt)
                        
            if not cohOutput is None:
                f,cohmat[itrial,specix,:] = cohOutput
            else:
                cohmat[itrial,specix,:] = np.nan
    
    return cohmat,np.nanstd(cohmat,axis=0)/np.sqrt((np.isnan(cohmat[:,0,0])==0).sum())

def coherence_null_visible(spec_list,trial_type,trials,
                          mx_freq=10,
                          precision=.1,
                          firstix=0,
                          offset=None,
                          disp=1):
    """
    Calculate coherence between trials and flashing visibility window.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    mx_freq : int,10
        Maximum frequency over which to average coherence
    precision : float,.1
    firstix : int,0
    offset : int,None
        Number of indices to offset the subject and template time series. If offset>0, we skip the
        first offset elements from subject. If offset<0, -offset elements are removed from the
        subject.
    disp : bool,1

    Returns
    -------
    cohmat : ndarray
        Average coherence statistic over all trials. First three cols correspond to xyz dimensions
        and last col is norm.
    cohmaterr : ndarray
        Standard error of the mean of coherence statistic down the columns.
    """
    # First three cols correspond to xyz dimensions and last col is norm.
    cohmat = np.zeros((len(trials),len(spec_list),4))
    
    for itrial,trial in enumerate(trials):
        for specix,spec in enumerate(spec_list):
            cohOutput = _compare_coherence_vis(trial,spec,trial_type,
                                               precision,mx_freq,
                                               firstix,disp,offset)
           
            if not cohOutput is None:
                f,cohmat[itrial,specix,:] = cohOutput
            else:
                cohmat[itrial,specix,:] = np.nan
    return cohmat,np.nanstd(cohmat,axis=0)/np.sqrt((np.isnan(cohmat[:,0,0])==0).sum())

def coherence_null_time_shift(spec_list,trial_type,trials,
                              mx_freq=10,
                              precision=.1,
                              firstix=0,
                              offset=None,
                              disp=1):
    """
    Calculate coherence between trials and time shifted version of signal.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    mx_freq : int,10
        Maximum frequency over which to average coherence
    precision : float,.1
    firstix : int
    offset : int,None
        Number of indices to offset the subject and template time series. If offset>0, we skip the
        first offset elements from subject. If offset<0, -offset elements are removed from the
        subject.
    disp : bool,True

    Returns
    -------
    cohmat : ndarray
        Average coherence statistic over all trials. First three cols correspond to xyz dimensions
        and last col is norm.
    cohmaterr : ndarray
        Standard error of the mean of coherence statistic.
    """
    # First three cols correspond to xyz dimensions and last col is norm.
    cohmat = np.zeros((len(trials),len(spec_list),4))
    
    for itrial,trial in enumerate(trials):
        for specix,spec in enumerate(spec_list):
            cohOutput = _compare_coherence(trial,[spec,spec],[trial_type]*2,
                                           [precision]*2,mx_freq,
                                           firstix,disp,offset,template_only=True)
           
            if not cohOutput is None:
                f,cohmat[itrial,specix,:] = cohOutput
            else:
                cohmat[itrial,specix,:] = np.nan
    
    return cohmat,np.nanstd(cohmat,axis=0)/np.sqrt((np.isnan(cohmat[:,0,0])==0).sum())

def coherence_null(spec_list,trial_type,trials,test_signal,
                   mx_freq=10,
                   precision=.1,
                   firstix=0,
                   offset=None,
                   disp=1):
    """
    Calculate coherence between trials and an arbitrary signal.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    mx_freq : int,10
        Maximum frequency over which to average coherence
    precision : float,.1
    firstix : int,0
    offset : int,None
        Number of indices to offset the subject and template time series. If offset>0, we skip the
        first offset elements from subject. If offset<0, -offset elements are removed from the
        subject.
    disp : bool,True

    Returns
    -------
    cohmat : ndarray
        Average coherence statistic over all trials. First three cols correspond to xyz dimensions
        and last col is norm.
    cohmaterr : ndarray
        Standard error of the mean of coherence statistic.
    """
    # First three cols correspond to xyz dimensions and last col is norm.
    cohmat = np.zeros((len(trials),len(spec_list),4))

    for itrial,trial in enumerate(trials):
        for specix,spec in enumerate(spec_list):
            cohOutput = _compare_coherence_given_vel(trial,spec,trial_type,
                                                     precision,test_signal,
                                                     mx_freq,
                                                     firstix=firstix,disp=disp,offset=offset)
           
            if not cohOutput is None:
                f,cohmat[itrial,specix,:] = cohOutput
            else:
                cohmat[itrial,specix,:] = np.nan

    return cohmat,np.nanstd(cohmat,axis=0)/np.sqrt((np.isnan(cohmat[:,0,0])==0).sum())

def _coherence_null(ignore_spec,trial_type,trials,
                   mx_freq=10,
                   precision=.1,
                   firstix=0,
                   offset=None,
                   disp=1):
    """
    Calculate average coherence over all given trials excluding the particular one of interest.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    mx_freq : int,10
        Maximum frequency over which to average coherence
    precision : float,.1
    firstix : int
    offset : int,None
        Number of indices to offset the subject and template time series. If offset>0, we skip the
        first offset elements from subject. If offset<0, -offset elements are removed from the
        subject.

    Returns
    -------
    cohmat : ndarray
        Average coherence statistic over all trials. First three cols correspond to xyz dimensions
        and last col is norm.
    cohmaterr : ndarray
        Standard error of the mean of coherence statistic.
    """
    # First three cols correspond to xyz dimensions and last col is norm.
    cohmat = []
    cohmaterr = []

    for itrial,trial in enumerate(trials):
        counter = 0 
        spec_list = [w[0] for w in trial.windowsByPart[trial_type] if not
                isclose(w[0],ignore_spec,precision)]
        cohmat.append( np.zeros((len(spec_list),4)) )
        for specix,spec in enumerate(spec_list):
            cohOutput = _compare_coherence(trial,[ignore_spec,spec],[trial_type]*2,
                                           [precision]*2,mx_freq,
                                           firstix,disp,offset)
                        
            if not cohOutput is None:
                f,cxy = cohOutput
                for dimIx,c in enumerate(cxy):
                    cohmat[-1][specix,dimIx] += c
            counter += 1

        ntrialmat = (cohmat[-1]!=0).sum(0)  # number of trials available for each window spec
                                        # used for normalization
        cohmat[-1][cohmat[-1]==0] = np.nan
        cohmaterr.append( np.nanstd(cohmat[-1],axis=0) )
        cohmat[-1] = np.nansum(cohmat[-1],axis=0)
        cohmat[-1] /= ntrialmat  # averaged over number of data points
        cohmaterr[-1] /= np.sqrt(ntrialmat)  # standard error of the mean
    return cohmat,cohmaterr

def coherence_null_null(ignore_spec,trial_type,trials,
                        mx_freq=10,
                        precision=.1,
                        firstix=0,
                        offset=None,
                        disp=1):
    """
    Calculate coherence by comparing one window with all the other windows. The original idea was
    that this would be a good null for checking performance of individuals, but this turns out not
    to be so useful because the avatar's motion is pretty similar across some windows. This means
    that this is not such a good null.

    Instead a better idea might be to just make two different models for nulls: the same trajectory
    displaced by a time delay and a periodic jerking motion.

    Parameters
    ----------
    spec_list : list
        List of twoples (invisible_fraction,window_duration).
    trial_type : str
        Trial type. 'hand' or 'avatar'
    trials : list of VRTrial instances
    mx_freq : int,10
        Maximum frequency over which to average coherence
    precision : float,.1
    firstix : int
    offset : int,None
        Number of indices to offset the subject and template time series. If offset>0, we skip the
        first offset elements from subject. If offset<0, -offset elements are removed from the
        subject.

    Returns
    -------
    cohmat : ndarray
        Average coherence statistic over all trials. First three cols correspond to xyz dimensions
        and last col is norm.
    cohmaterr : ndarray
        Standard error of the mean of coherence statistic.
    """
    # First three cols correspond to xyz dimensions and last col is norm.
    cohmat = []
    cohmaterr = []

    for itrial,trial in enumerate(trials):
        counter = 0 
        spec_list = [w[0] for w in trial.windowsByPart[trial_type] if not
                isclose(w[0],ignore_spec,precision)]
        cohmat.append( np.zeros((len(spec_list),4)) )
        for specix,spec in enumerate(spec_list):
            cohOutput = _compare_coherence(trial,[ignore_spec,spec],[trial_type]*2,
                                           [precision]*2,mx_freq,
                                           firstix,disp,offset,template_only=True)
                        
            if not cohOutput is None:
                f,cxy = cohOutput
                for dimIx,c in enumerate(cxy):
                    cohmat[-1][specix,dimIx] += c
            counter += 1

        ntrialmat = (cohmat[-1]!=0).sum(0)  # number of trials available for each window spec
                                        # used for normalization
        cohmat[-1][cohmat[-1]==0] = np.nan
        cohmaterr.append( np.nanstd(cohmat[-1],axis=0) )
        cohmat[-1] = np.nansum(cohmat[-1],axis=0)
        cohmat[-1] /= ntrialmat  # averaged over number of data points
        cohmaterr[-1] /= np.sqrt(ntrialmat)  # standard error of the mean
    return cohmat,cohmaterr


def isclose(spec1,spec2,precision):
    """
    Compare two different specs.
    """
    if not type(precision) is tuple:
        precision = (precision,precision)

    if (abs(spec1[0]-spec2[0])<=precision[0]) and (abs(spec1[1]-spec2[1])<=precision[1]):
        return True
    return False


def _compare_coherence(trial,windows,trial_types,precisions,mx_freq,
                       firstix=0,disp=True,offset=None,
                       template_only=False,cwt=False):
    """
    Compare coherence for the given windows specified for the subject and for the template.

    Parameters
    ----------
    trial : VRTrial
    windows : list of tuples
    trial_types : list of strings
    precisions : list of precision
    mx_freq : float
    firstix : int,0
    disp : bool,True
    offset : int,None
    template_only : bool,False
    cwt : bool,False
        If True, compute coherence using continuous wavelet transform.

    Returns
    -------
    """
    from scipy.signal import coherence
    try:
        if template_only:
            sspec,t,subjectv = trial.template_by_window_spec([windows[0]],
                                                           trial_types[0],
                                                           precisions[0]
                                                          )[firstix]
        else:
            sspec,t,subjectv = trial.subject_by_window_spec([windows[0]],
                                                             trial_types[0],
                                                             precisions[0]
                                                            )[firstix]
        tspec,t,templatev = trial.template_by_window_spec([windows[1]],
                                                           trial_types[1],
                                                           precisions[1]
                                                          )[firstix]
        if not offset is None:
            if offset>0:
                t,subjectv = t[-1200:][:-offset],subjectv[-1200:][offset:]
                templatev = templatev[-1200:][:-offset]
            elif offset<0:
                t = t[-1200:][:offset]
                subjectv = subjectv[-1200:][:offset],
                templatev = templatev[-1200:][-offset:]
        else:
            t = t[-1200:]
            subjectv = subjectv[-1200:]
            templatev = templatev[-1200:]
        assert (len(subjectv)==1200) and (len(templatev)==1200)

        if disp:
            print("Subject: (%1.1f,%1.1f), Template: (%1.1f,%1.1f)"%(sspec[0],sspec[1],
                                                                     tspec[0],tspec[1]))
        
        if len(subjectv)==0:
            return
        
        # Calculate coherence for each dimension.
        cxy = np.zeros(4)  # averaged coherence
        noverlap,nperseg = 30,90
        nfft = nperseg*2
        for dimIx in range(3):
            if cwt:
                f,cxy_ = cwt_coherence(subjectv[:,dimIx],templatev[:,dimIx],noverlap)
                cxy_ *= -1
            else:
                f,cxy_ = coherence(subjectv[:,dimIx],templatev[:,dimIx],
                                   fs=60,nperseg=nperseg,noverlap=noverlap,nfft=nfft)

            cxy[dimIx] = np.trapz( cxy_[f<mx_freq],x=f[f<mx_freq] )/(f[f<mx_freq].max()-f[f<mx_freq].min())
        
        # Coherence for velocity magnitude.
        if cwt:
            f,cxy_ = cwt_coherence(subjectv[:,dimIx],templatev[:,dimIx],noverlap)
            cxy_ *= -1
        else:
            f,cxy_ = coherence(np.linalg.norm(subjectv,axis=1),
                               np.linalg.norm(templatev,axis=1),
                               fs=60,nperseg=nperseg,noverlap=noverlap,nfft=nfft)
        cxy[3] = np.trapz( cxy_[f<mx_freq],x=f[f<mx_freq] )/(f[f<mx_freq].max()-f[f<mx_freq].min())
        return f,cxy
    except Exception as err:
        if disp:
            print("No data for window spec (%1.1f,%1.1f)."%(windows[0][0],
                                                            windows[0][1]))
        return

def _compare_coherence_vis(trial,window,trial_type,precision,mx_freq,
                           firstix=0,disp=True,offset=None):
    """
    Compare coherence.

    Parameters
    ----------
    trial : VRTrial
    window
    trial_type
    precision
    mx_freq
    firstix : int,0
    disp : bool,True
    offset : int,None
    """
    from scipy.signal import coherence
    try:
        sspec,t,visibility = trial.visibility_by_window_spec([window],
                                                            trial_type,
                                                            precision
                                                           )[firstix]
        tspec,t,templatev = trial.template_by_window_spec([window],
                                                            trial_type,
                                                            precision
                                                           )[firstix]
        if not offset is None:
            if offset>0:
                t,visibility = t[-1200:][:-offset],visibility[-1200:][offset:]
                templatev = templatev[-1200:][:-offset]
            elif offset<0:
                t = t[-1200:][:offset]
                visibility = visibility[-1200:][:offset],
                templatev = templatev[-1200:][-offset:]
        else:
            t = t[-1200:]
            visibility = visibility[-1200:]
            templatev = templatev[-1200:]
        assert (len(visibility)==1200) and (len(templatev)==1200)

        if disp:
            print("Subject: (%1.1f,%1.1f), Template: (%1.1f,%1.1f)"%(sspec[0],sspec[1],
                                                                     tspec[0],tspec[1]))
        
        if len(visibility)==0:
            return

        # Calculate coherence for each dimension.
        cxy = np.zeros(4)
        for dimIx in range(3):
            f,cxy_ = coherence(visibility,templatev[:,dimIx],
                                fs=60,nperseg=120)
            cxy[dimIx] = np.trapz( cxy_[f<mx_freq],x=f[f<mx_freq] )/(f[f<mx_freq].max()-f[f<mx_freq].min())

        # Coherence for velocity magnitude.
        f,cxy_ = coherence(visibility,
                           np.linalg.norm(templatev,axis=1),
                           fs=60,nperseg=120)
        cxy[3] = np.trapz( cxy_[f<mx_freq],x=f[f<mx_freq] )/(f[f<mx_freq].max()-f[f<mx_freq].min())
        return f,cxy
    except Exception as err:
        if disp:
            print("No data for window spec (%1.1f,%1.1f)."%window)
        return

def _compare_coherence_given_vel(trial,window,trial_type,precision,test_signal,mx_freq,
                                 firstix=0,disp=True,offset=None):
    """
    Compare coherence for the given windows specified for the subject and for the template.

    Parameters
    ----------
    trial : VRTrial
    windows
    trial_types
    precisions
    """
    from scipy.signal import coherence
    assert len(test_signal)>=1200
    try:
        subjectv = test_signal
        tspec,t,templatev = trial.template_by_window_spec([window],
                                                            trial_type,
                                                            precision
                                                           )[firstix]
        if not offset is None:
            if offset>0:
                t,subjectv = t[-1200:][:-offset],subjectv[-1200:][offset:]
                templatev = templatev[-1200:][:-offset]
            elif offset<0:
                t = t[-1200:][:offset]
                subjectv = subjectv[-1200:][:offset],
                templatev = templatev[-1200:][-offset:]
        else:
            t = t[-1200:]
            subjectv = subjectv[-1200:]
            templatev = templatev[-1200:]
        assert (len(subjectv)==1200) and (len(templatev)==1200)

        if disp:
            print("Template: (%1.1f,%1.1f)"%(tspec[0],tspec[1]))
        
        if len(subjectv)==0:
            return
        
        # Calculate coherence for each dimension.
        cxy = np.zeros(4)
        for dimIx in range(3):
            f,cxy_ = coherence(subjectv[:,dimIx],templatev[:,dimIx],
                               fs=60,nperseg=120)
            cxy[dimIx] = np.trapz( cxy_[f<mx_freq],x=f[f<mx_freq] )/(f[f<mx_freq].max()-f[f<mx_freq].min())

        # Coherence for velocity magnitude.
        f,cxy_ = coherence(np.linalg.norm(subjectv,axis=1),
                           np.linalg.norm(templatev,axis=1),
                           fs=60,nperseg=120)
        cxy[3] = np.trapz( cxy_[f<mx_freq],x=f[f<mx_freq] )/(f[f<mx_freq].max()-f[f<mx_freq].min())
        return f,cxy
    except Exception as err:
        if disp:
            print("No data for window spec (%1.1f,%1.1f)."%(window[0],
                                                            window[1]))
        return

def extract_motionbuilder_test(hand,
                               clear_pickle=False,
                               reverse_time=False):
    """
    Load model motion data. Assuming the play rate is a constant 1/60 Hz as has been set in MotionBuilder when
    exported. Returned data is put into standard global coordinate frame: x-axis is the axis between the two
    subjects where positive is towards the front, y is the side to side, and z is up and down such that
    positive y is determined by following the right hand rule.
    
    These are pickled csv files that were exported from Mokka after preprocessing in Motionbuilder. Note that
    the coordinate system in Motionbuilder and Mokka are different.

    NOTE: Directory where animation data is stored is hard-coded.
    
    Parameters
    ----------
    hand : str
        Hand of the model.
    fname : str,'Eddie_Grid_Model_%s_Anim_Export_Take_001'
        Name of file with %s to replace with handedness.
    reverse_time : bool,False
        Read data backwards from end.

    Returns
    -------
    mbV : scipy.interpolate.interp1d
        Returns (n_samples,3) dimensional matrix.
    mbT : ndarray of float
        Number of seconds since the beginning of the avatar motion file.
    """
    from datetime import datetime,timedelta
    import pickle as pickle
    from scipy.interpolate import interp1d
    assert hand=='Left' or hand=='Right'

    fname='mb_test'
    dr = ( os.path.expanduser('~')+'/Dropbox/Research/tango/data/UE4_Experiments/'+
           'Simple_MB_Test' )

    if (not os.path.exists('%s/%s.p'%(dr,fname))) or clear_pickle:
        from .axis_neuron import load_csv
        mbdf = load_csv('%s/%s.csv'%(dr,fname))
        mbdf.to_pickle('%s/%s.p'%(dr,fname))

    mbdf = pickle.load(open('%s/%s.p'%(dr,fname),'rb'))
    mbT = mbdf['Time'].values.astype(float)
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%hand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s
    mbV[:,:] = mbV[:,[1,0,2]]
    mbV[:,1] *= -1

    if reverse_time:
        mbV = mbV[::-1]

    mbV = interp1d(mbT,mbV,axis=0,assume_sorted=True,copy=False)

    return mbV,mbT

def extract_motionbuilder_model3_3(hand,
                                   dr=( os.path.expanduser('~')+'/Dropbox/Research/tango/data/UE4_Experiments/'+
                                        'Animations/Eddie_Grid_Model' ),
                                   fname='Eddie_Grid_Model_%s_Anim_Export_Take_001',
                                   reverse_time=False):
    """
    Load model motion data. Assuming the play rate is a constant 1/60 Hz as has been set in MotionBuilder when
    exported. Returned data is put into standard global coordinate frame: x-axis is the axis between the two
    subjects where positive is towards the front, y is the side to side, and z is up and down such that
    positive y is determined by following the right hand rule.
    
    These are pickled csv files that were exported from Mokka after preprocessing in Motionbuilder. Note that
    the coordinate system in Motionbuilder and Mokka are different.

    Parameters
    ----------
    hand : str
        Hand of the model. Must be 'Left' or 'Right'.
    fname : str,'Eddie_Grid_Model_%s_Anim_Export_Truncate_Take_001'
        Name of file with %s to replace with handedness.
    reverse_time : bool,False
        Read data backwards from end. This option is used when the avatar's motion is played in reverse.

    Returns
    -------
    mbV : scipy.interpolate.interp1d
        Returns (n_samples,3) dimensional matrix.
    mbT : ndarray of float
        Number of seconds since the beginning of the avatar motion file.
    """
    from datetime import datetime,timedelta
    import pickle as pickle
    from scipy.interpolate import interp1d
    assert hand=='Left' or hand=='Right'

    fname = fname%hand
    
    # Create pickle if it doesn't already exist.
    if not os.path.exists('%s/%s.p'%(dr,fname)):
        from .axis_neuron import load_csv
        mbdf = load_csv('%s/%s.csv'%(dr,fname))
        mbdf.to_pickle('%s/%s.p'%(dr,fname))

    mbdf = pickle.load(open('%s/%s.p'%(dr,fname),'rb'))
    mbT = mbdf['Time'].values.astype(float)
    print("MB start and end times: %1.2f and %1.2f"%(mbT[0],mbT[-1]))
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%hand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s
    mbV[:,:] = mbV[:,[1,0,2]]

    if reverse_time:
        # When you reverse time, you must also reverse the velocities.
        mbV = -mbV[::-1]

    # Put these in the standard global coordinate system such that avatars are facing +x direction. See Tango
    # III pg 45.
    if hand=='Left':
        mbV[:,0] *= -1
    else:
        # With right hand, the avatar starts facing the opposite direction so she is already facing the
        # same direction as the original y-axis.
        mbV[:,1] *= -1

    # y-axis needs to be reflected to put into same chirality as subject
    mbV[:,1] *= -1
    
    mbV = interp1d(mbT,mbV,axis=0,assume_sorted=True,copy=False)

    return mbV,mbT

def extract_motionbuilder_Eddie_Grid_Model_2(hand,
                       dr=( os.path.expanduser('~')+'/Dropbox/Research/tango/data/UE4_Experiments/'+
                            'Animations/Eddie_Grid_Model_2' ),
                       fname='Eddie_Grid_Model_2_%s_Take_001',
                       reverse_time=False):
    """
    See extract_motionbuilder_model3_3 for notes. The only difference here is that the orientation
    of the avatar in Mokka is in the -x direction for both left and right hands. 

    Parameters
    ----------
    hand : str
        Hand of the model. Must be 'Left' or 'Right'.
    dr : str,( os.path.expanduser('~')+'/Dropbox/Research/tango/data/UE4_Experiments/'+
                                       'Animations/Eddie_Grid_Model_2' )
    fname : str,'Eddie_Grid_Model_%s_Anim_Export_Take_001'
        Name of file with %s to replace with handedness.
    reverse_time : bool,False
        Read data backwards from end. This option is used when the avatar's motion is played in reverse.

    Returns
    -------
    mbV : scipy.interpolate.interp1d
        Returns (n_samples,3) dimensional matrix.
    mbT : ndarray of float
        Number of seconds since the beginning of the avatar motion file.
    """
    from datetime import datetime,timedelta
    import pickle as pickle
    from scipy.interpolate import interp1d
    assert hand=='Left' or hand=='Right'

    fname = fname%hand
    
    # Create pickle if it doesn't already exist.
    if not os.path.exists('%s/%s.p'%(dr,fname)):
        from .axis_neuron import load_csv
        mbdf = load_csv('%s/%s.csv'%(dr,fname))
        mbdf.to_pickle('%s/%s.p'%(dr,fname))

    mbdf = pickle.load(open('%s/%s.p'%(dr,fname),'rb'))
    mbT = mbdf['Time'].values.astype(float)
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%hand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s

    if reverse_time:
        # When you reverse time, you must also reverse the velocities.
        mbV = -mbV[::-1]

    # Put these in the standard global coordinate system such that avatars are facing +x direction. See Tango
    # III pg 45. Only 180 degree rotation about z-axis is required here.
    mbV[:,:2] *= -1

    # y-axis needs to be reflected to put into same chirality as subject
    mbV[:,1] *= -1
    
    mbV = interp1d(mbT,mbV,axis=0,assume_sorted=True,copy=False)

    return mbV,mbT


def extract_motionbuilder_model3(hand,
                                 fname='Eddie_Grid_Model_%s_Anim_Export_Take_001',
                                 reverse_time=False):
    """
    Load model motion data. Assuming the play rate is a constant 1/60 Hz as has been set in MotionBuilder when
    exported. Returned data is put into standard global coordinate frame: x-axis is the axis between the two
    subjects where positive is towards the front, y is the side to side, and z is up and down such that
    positive y is determined by following the right hand rule.
    
    These are pickled csv files that were exported from Mokka after preprocessing in Motionbuilder. Note that
    the coordinate system in Motionbuilder and Mokka are different.

    NOTE: Directory where animation data is stored is hard-coded.
    
    Parameters
    ----------
    hand : str
        Hand of the model.
    fname : str,'Eddie_Grid_Model_%s_Anim_Export_Take_001'
        Name of file with %s to replace with handedness.
    reverse_time : bool,False
        Read data backwards from end.

    Returns
    -------
    mbV : scipy.interpolate.interp1d
        Returns (n_samples,3) dimensional matrix.
    mbT : ndarray of float
        Number of seconds since the beginning of the avatar motion file.
    """
    from datetime import datetime,timedelta
    import pickle as pickle
    from scipy.interpolate import interp1d
    assert hand=='Left' or hand=='Right'

    dr = ( os.path.expanduser('~')+'/Dropbox/Research/tango/data/UE4_Experiments/'+
           'Animations/Eddie_Grid_Model' )
    fname = fname%hand
    
    # Create pickle if it doesn't already exist.
    if not os.path.exists('%s/%s.p'%(dr,fname)):
        from .axis_neuron import load_csv
        mbdf = load_csv('%s/%s.csv'%(dr,fname))
        mbdf.to_pickle('%s/%s.p'%(dr,fname))

    mbdf = pickle.load(open('%s/%s.p'%(dr,fname),'rb'))
    mbT = mbdf['Time'].values.astype(float)
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%hand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s
    mbV[:,:] = mbV[:,[1,0,2]]

    if reverse_time:
        mbV = -mbV[::-1]

    # Put these in the standard global coordinate system such that avatars are facing +x direction. See Tango
    # III pg 45.
    if hand=='Left':
        mbV[:,0] *= -1
    else:
        # With right hand, the avatar starts facing the opposite direction so she is already facing the
        # same direction as the original y-axis.
        mbV[:,1] *= -1

    # y-axis needs to be reflected to put into same chirality as subject
    mbV[:,1] *= -1
    
    mbV = interp1d(mbT,mbV,axis=0,assume_sorted=True,copy=False)

    return mbV,mbT

def extract_motionbuilder_model2(trial_type,visible_start,modelhand,return_time=True):
    """
    Load model motion data. Assuming the play rate is a constant 1/60 Hz as has been set in MotionBuilder when
    exported. Returned data is put into standard global coordinate frame: x-axis is the axis between the two
    subjects where positive is towards the front, y is the side to side, and z is up and down such that
    positive y is determined by following the right hand rule.
    
    These are pickled csv files that were exported from Mokka after preprocessing in Motionbuilder. Note that
    the coordinate system in Motionbuilder and Mokka are different.

    NOTE: Directory where animation data is stored is hard-coded.

    Returns
    -------
    trial_type (str)
    visible_start (datetime.datetime)
    modelhand (str)
    """
    from datetime import datetime,timedelta
    from workspace.utils import load_pickle

    dr = ( os.path.expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/'+
           'Animations' )
    fname = {'hand0':'Eddie_Smooth_Model_%s_12000.p'%modelhand,
             'hand':'Eddie_Smooth_Model_%s.p'%modelhand,
             'avatar0':'Eddie_Smooth_Model_%s_6000.p'%modelhand,
             'avatar':'Eddie_Smooth_Model_%s_12000.p'%modelhand}[trial_type]
    load_pickle('%s/%s'%(dr,fname))
    mbT = mbdf['Time'].values.astype(float)
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%modelhand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s
    # Put these in the standard global coordinate system as explained in function description.
    mbV[:,:] = mbV[:,[1,0,2]]
    mbV[:,1] *= -1

    if return_time:
        mbT = np.array([timedelta(seconds=t)+visible_start for t in mbT])
        return mbT,mbV
    return mbV

def extract_motionbuilder_model(trial_type,visible_start,modelhand):
    """
    Load model motion data. Assuming the play rate is a constant 1/60 Hz. Returned data is put into standard
    global coordinate frame.
    
    Directory where animation data is stored is hard-coded.

    Params:
    -------
    trial_type (str)
    visible_start (datetime.datetime)
    modelhand (str)
    """
    from datetime import datetime,timedelta
    from workspace.utils import load_pickle

    dr = os.path.expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/Animations'
    fname = {'hand':'Eddie_%s_Hand_Model_19000_Recorded.p'%modelhand,
             'arm':'Eddie_%s_Hand_Model_Recorded.p'%modelhand,
             'avatar':'Freya_(F)_Eddie_(L)_%s_Anim_Recorded.p'%modelhand}[trial_type]
    load_pickle('%s/%s'%(dr,fname))
    mbT = mbdf['Time'].values.astype(float)
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%modelhand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s

    mbT = np.array([timedelta(seconds=t)+visible_start for t in mbT])

    # Put these in the standard global coordinate system.
    mbV[:,:] = mbV[:,[1,0,2]]
    mbV[:,1] *= -1
    return mbT,mbV

def quick_load(fileix,dt=1/120,negate_x=False,negate_y=False,disp=True):
    """
    Quick and dirty method for loading filtered velocities from hand trials. 

    Params:
    -------
    fileix
    dt (float=1/120)
    negate_x (bool=True)
    negate_y (bool=True)
    disp (bool=True
    
    Returns:
    --------
    T
    v1
    v2
    """
    bodypartix = hand_ix(fileix)
    fname = get_fnames()[fileix]
    if type(fname) is tuple:
        fname,date = fname
    else:
        date = None

    data = pickle.load(open('%s/%s.p'%(get_dr(fname,date),fname),'rb'))
    leaderX,leaderV,leaderA = data['leaderX'],data['leaderV'],data['leaderA']
    followerX,followerV,followerA = data['followerX'],data['followerV'],data['followerA']
    
    T = np.arange(len(leaderX[0]))*dt

    x1,x2 = leaderX[bodypartix],followerX[bodypartix]
    v1,v2 = leaderV[bodypartix],followerV[bodypartix]
    a1,a2 = leaderA[bodypartix],followerA[bodypartix]
    
    if negate_x:
        v2[:,0] *= -1
    if negate_y:
        v2[:,1] *= -1
    if disp:
        print(np.corrcoef(v1[:,0],v2[:,0])[0,1])
        print(np.corrcoef(v1[:,1],v2[:,1])[0,1])

    # Detrending necessary in some cases.
    if fileix in [51,52,53]:
        detrend(v2[:,2],inplace=True)
    return T,v1,v2

def pipeline_phase_calc(fileixs=[],
                        trajs=[],
                        dr='phase_files',
                        file_names='temp_phase',
                        down_sample=False,
                        **phase_calc_kwargs):
    """
    Pipeline loading pickled lowpass filtered data and running phase extraction after bandpass filtering.
    Pickles list of tuples phases, list of tuples vs (containing filtered velocities), and array fs
    (frequencies that were bandpassed).  

    If fileix's are given, those are loaded then processed. Otherwise, the data given is processed.
    
    Params:
    -------
    fileix (list)
        List of ints of files to load.
    trajs (list of tuples)
        (T,v1,v2) or ( T,(v1,v2,...,) )
    sample_freq (str='120')
        '120','60','30'
    bandwidth (float=.1)
        Bandwidth of bandpass filter.
    down_sample (bool=False)
        Down sample data by a factor of 2 if true.
    """
    if not os.path.isdir(dr):
        os.makedirs(dr)
    #fs = np.concatenate((np.arange(-3,0,.1),np.arange(.1,3.1,.1)))
    fs = np.arange(.1,3.1,.1)
    if type(file_names) is str:
        file_names = [file_names+'_%d'%i for i in range(len(trajs))]
    
    if len(fileixs)>0:
        counter = 0
        for fileix in fileixs:
            print("Starting file %d..."%fileix)
            T,v1,v2 = quick_load(fileix,dt=1/int(phase_calc_kwargs['sample_freq']))
            if down_sample:
                T = T[::2]
                v1 = v1[::2]
                v2 = v2[::2]

            phases,vs = phase_calc(fs,v1,v2,**phase_calc_kwargs) 

            pickle.dump({'phases':phases,'vs':vs,'fs':fs},open('%s/%s.p'%(dr,file_names[counter]),'wb'),-1)
            print("Done with file %d."%fileix)
            counter += 1
    else:
        counter = 0
        for T,v in trajs:
            assert T.ndim==1 and all([i.ndim==1 for i in v])
            if down_sample:
                T = T[::2]
                v = [i[::2] for i in v]
            
            phases,vs = [],[]
            for i in v:
                phases_,vs_ = phase_calc(fs,i,**phase_calc_kwargs) 
                phases.append(phases_)
                vs.append(vs_)
            
            pickle.dump({'phases':phases,'vs':vs,'fs':fs},
                        open('%s/%s.p'%(dr,file_names[counter]),'wb'),-1)
            print("Done with file %d."%counter)
            counter += 1

def phase_calc(fs,v1,v2=None,
               sample_freq='120',
               bandwidth=.1,
               down_sample=False):
    """
    Params:
    -------
    fs (ndarray)
        Frequencies at which to bandpass.
    v1 (ndarray)
        1d arrays.
    v2 (ndarray=None)
    sample_freq (str='120')
        '120','60','30'
    bandwidth (float=.1)
        Bandwidth of bandpass filter.
    down_sample (bool=False)
        Down sample data by a factor of 2 if true.

    Returns:
    --------
    phases (list of ndarrays)
    vs (list of ndarrays)
        Bandpass filtered velocities.
    """
    from scipy.signal import hilbert

    if str(sample_freq)=='120':
        windowLength,filtwidth = 501,50
    elif str(sample_freq=='60'):
        windowLength,filtwidth = 251,25
    elif str(sample_freq=='30'):
        windowLength,filtwidth = 125,13
    else:
        raise NotImplementedError
    
    phases = []
    vs = []
    for f in fs:
        v1_ = moving_freq_filt(v1,window=windowLength,
                               window_type=('gaussian',filtwidth),
                               filter_type='single',
                               sample_freq=int(sample_freq),
                               pass_freq=f,
                               bandwidth=bandwidth,
                               axis=0)
        h1 = hilbert(v1_,axis=0)
        phase1 = np.angle(h1)

        if not v2 is None:
            v2_ = moving_freq_filt(v2,window=windowLength,
                                   window_type=('gaussian',filtwidth),
                                   filter_type='single',
                                   sample_freq=int(sample_freq),
                                   pass_freq=f,
                                   bandwidth=bandwidth,
                                   axis=0)
            h2 = hilbert(v2_,axis=0)
        
            phase2 = np.angle(h2)
            phases.append((phase1,phase2))
            vs.append((v1_,v2_))
        else:
            phases.append(phase1)
            vs.append(v1_)
    
    return phases,vs

def filter_hand_trials(filesToFilter,dt=1/60,
        extract_calc_kwargs={'rotate_to_face':False,
                             'remove_hip_drift':True,
                             'dotruncate':5},
        filterparams='default',
        suffix=''):
    """
    Shortcut for filtering hand trials data by just giving file number.
    2017-03-19
    
    Params:
    -------
    filesToFilter (list)
    dt (float=1/60)
    extract_calc_kwargs (dict)
    filterparams (str='default')
        Choose between 'default' and '120'. Filter parameters for butterworth filter as in utils.smooth()
    """
    from .filter import smooth
    import pickle as pickle
    
    bodyparts = [['RightHand','LeftHand'],
                 ['LeftHand','RightHand']]

    for fileix in filesToFilter:
        # Read position, velocity and acceleration data from files.
        fname = get_fnames()[fileix]
        if type(fname) is tuple:
            fname,date = fname
        else:
            date = None
        T,leaderX,leaderV,leaderA,followerX,followerV,followerA = extract_calc(fname,
                                                                   get_dr(fname,date),
                                                                   bodyparts,
                                                                   dt,
                                                                   rotation_angle=global_rotation(fileix),
                                                                   **extract_calc_kwargs)

        for x in leaderX:
            x-=x.mean(0)
        for x in followerX:
            x-=x.mean(0)
        
        # Butterworth filter data and pickle it.
        for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:] = smooth(x,filterparams=filterparams)[:]
            v[:] = smooth(v,filterparams=filterparams)[:]
            a[:] = smooth(a,filterparams=filterparams)[:]
        for x,v,a in zip(followerX,followerV,followerA):
            x[:] = smooth(x,filterparams=filterparams)[:]
            v[:] = smooth(v,filterparams=filterparams)[:]
            a[:] = smooth(a,filterparams=filterparams)[:]
        
        # Save into same directory as calc file.
        savedr = '%s/%s%s.p'%(get_dr(fname,date),fname,suffix)
        print("Saving as %s"%savedr)
        pickle.dump({'T':T,
                     'leaderX':leaderX,'followerX':followerX,
                     'leaderV':leaderV,'followerV':followerV,
                     'leaderA':leaderA,'followerA':followerA},
                    open(savedr,'wb'),-1)


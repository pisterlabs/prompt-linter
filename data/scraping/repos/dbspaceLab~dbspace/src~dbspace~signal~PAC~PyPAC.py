# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:11:22 2015

@author: rali

Researcher: REHMAN ALI
Principal Investigator: ROBERT BUTERA
Graduate Collaborator: VINEET TIRUVADI
Neurolab at Georgia Institute of Technology
"""

import numpy as np
from scipy import signal as sig
from scipy.stats import entropy
from scipy.signal import coherence, welch
import matplotlib.pyplot as plt
from matplotlib import cm
from nitime.analysis.spectral import MorletWaveletAnalyzer
from nitime import timeseries as ts

# MUST INSTALL nitime: easy_install nitime

def CFCfilt(signal,freqForAmp,freqForPhase,fs,passbandRipl):
	""" CFCFILT Returns a matrix of bandpass filtered LFP signals
		USAGE: oscillations = CFCfilt(signal,freqForAmp,freqForPhase,fs,passbandRipl)
		signal is the input LFP to be bandpassed, fs is the sampling rate
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		passbandRipl is on a linear scale (not decibel): its preferred value is 0.02
		oscillations is a matrix of complex-valued time-series (3D array):
			rows correspond to frequency for phase
			columns correspond to frequency for amplitude """
	
	" Setting-up empty 3D array for bandpassed time-series "		
	numTimeSamples = np.size(signal);
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	oscillations = np.zeros((frqPhaseSize,frqAmpSize,numTimeSamples),dtype=np.complex64);
	
	" Linear Ripple to Decibel Ripple Conversion "
	Rp = 40*np.log10((1+passbandRipl)/(1-passbandRipl));
	
	" Variable Band-pass Filtering "
	" First index varies bandwidth (frequency for phase) "
	" Second index varies center frequency (frequency for amplitude) "
	for jj in np.arange(frqPhaseSize):
		for kk in np.arange(frqAmpSize):
			freq = freqForAmp[kk]; # Center Frequency
			delf = freqForPhase[jj]; # Bandwidth
			if freq > 1.8*delf:
				freqBand = np.array([freq-1.2*delf, freq+1.2*delf])/(fs/2);
				bb, aa = sig.cheby1(3,Rp,freqBand,btype='bandpass');
			else:
				bb, aa = sig.cheby1(3,Rp,(freq+1.2*delf)/(fs/2));
			oscillation = sig.filtfilt(bb,aa,signal);
			oscillations[jj,kk,:] = sig.hilbert(oscillation);
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(freq));
	return oscillations;
			
def morletCWT(signal,fs,frequencies,sd_rel,sd):
	""" MORLETCWT Determines Complex Morlet Wavelet CWT Coefficients for Signal
		USAGE: coefs = morletCWT(signal,Fb,Fc,frequencies)
		signal is the input signal, fs is sampling rate (Hz)
		frequencies are the center frequencies for the CWT 
		sd_rel is filter bandwidth as a fraction of the center frequencies (default 0.2)
		sd is a list of sd_rel for each center frequency in frequencies """
	
	signalObject = ts.TimeSeries(signal,sampling_rate=fs);
	cwtMorletObject = MorletWaveletAnalyzer(signalObject,freqs=frequencies,sd_rel=sd_rel,sd=sd);
	return np.array(cwtMorletObject.analytic);
	
def preCFCProc(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl):
	"""	PRECFCPROC Uses variable-bandwidth bandpass filtering on signal
		USAGE: oscilsForAmp, oscilsForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters for phase (typically 4.5 Hz)
		passbandRipl is on a linear scale (not decibel): its preferred value is 0.02 """

	oscilsAmpMod = CFCfilt(sigForAmp,freqForAmp,freqForPhase,fs,passbandRipl);
	oscilsForPhase = CFCfilt(sigForPhase,freqForPhase,np.array([bw]),fs,passbandRipl);
	return oscilsAmpMod, oscilsForPhase;
    
def preCWTProc(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,sd_rel_phase,sd_rel_amp):
    """ PRECWTProc Determines Complex Morlet Wavelet CWT Coefficients for Signals
        USAGE: coefsForAmp, coefsForPhase = preCWTProc(sigForAmp,sigForPhase,
            freqForAmp,freqForPhase,fs,sd_rel,sd_ffa,sd_ffp)
        sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz)
        sd_rel_phase is filter bandwidth given as a fraction of the freqForPhase (default 0.2)
        sd_rel_amp is filter bandwidth given as a fraction of freqForPhase in order 
            to enforce variable-bandwidth bandpass filtering around freqForAmp 
			(freqForAmp +/- sd_rel_amp * freqForPhase)"""

    coefsForPhase = morletCWT(sigForPhase,fs,freqForPhase,sd_rel_phase,None);
    coefsForAmp = np.zeros((sigForAmp.size,freqForAmp.size,freqForPhase.size),dtype=np.complex64);
    for kk in np.arange(freqForPhase.size):
	    sd_ffa = 2*(freqForPhase[kk]/freqForAmp)*sd_rel_amp;
	    coefsForAmp[:,:,kk] = morletCWT(sigForAmp,fs,freqForAmp,None,sd_ffa);
	    print("Completed: Frequency for Phase [Hz] = "+str(freqForPhase[kk]));
    return coefsForAmp, coefsForPhase
	
def comodShow(freqForAmp,freqForPhase,MIs):
	""" COMODSHOW Displays the comodulogram
		USAGE: comodShow(freqForAmp,freqForPhase,MIs)
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
        MIs is the modulation indices (columns--freqForPhase; rows--freqForAmp) """
		
	dfp = np.mean(np.diff(freqForPhase)); 
	dfa = np.mean(np.diff(freqForAmp));
	comodplt = plt.imshow(MIs.transpose(), interpolation='nearest', 
			extent=[np.min(freqForPhase)-dfp/2,np.max(freqForPhase)+dfp/2,
					np.min(freqForAmp)-dfa/2,np.max(freqForAmp)+dfa/2],
					origin = 'lower', cmap = cm.jet, aspect = 'auto');
	plt.xlabel('Frequency for Phase [Hz]'); 
	plt.ylabel('Frequency for Amplitude [Hz]');
	plt.colorbar();
	return comodplt;
	
def GenLinMod(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
	""" GENLINMOD Calculates comulolograms based on Generalized Linear Model
		USAGE: MIs = GenLinMod(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		oscAmpMod is a cell matrix of time-series oscillations bandpassed 
			around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
		oscForPhase is a row cell vector of time-series oscillations bandpassed
			around freqForPhase with some small bandwidth """
	
	" Applying Generalized Linear Model-based CFC to Oscillations Data "
	ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(oscAmpMod[rr,cc,:]);
			phaseOsc = np.angle(oscForPhase[0,rr,:]);
			X = np.matrix(np.column_stack((np.cos(phaseOsc),
					np.sin(phaseOsc), np.ones(np.size(phaseOsc)))));
			B = np.linalg.inv((np.transpose(X)*X))* \
					np.transpose(X)*np.transpose(np.matrix(ampOsc));
			ampOscTrend = X*B;
			ampOscResid = ampOsc.flatten()-np.array(ampOscTrend).flatten();
			rsq = 1-np.var(ampOscResid)/np.var(ampOsc);
			ModCorr[rr,cc] = np.sqrt(rsq);
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	MIs = np.arctanh(ModCorr);
	return MIs;

def GenLinModCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase):
	""" GENLINMODCWT Calculates comulolograms based on Generalized Linear Model
		USAGE: MIs = GenLinModCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase)
        coefsForAmp is a cell matrix of time-series oscillations bandpassed
			around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
        coefsForPhase is a row cell vector of time-series oscillations bandpassed
            around freqForPhase with some small bandwidth """
	
	" Applying Generalized Linear Model-based CFC to Oscillations Data "
	ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
    
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(coefsForAmp[:,cc,rr]);
			phaseOsc = np.angle(coefsForPhase[:,rr]);
			X = np.matrix(np.column_stack((np.cos(phaseOsc),
				np.sin(phaseOsc), np.ones(np.size(phaseOsc)))));
			B = np.linalg.inv((np.transpose(X)*X))* \
				np.transpose(X)*np.transpose(np.matrix(ampOsc));
			ampOscTrend = X*B;
			ampOscResid = ampOsc.flatten()-np.array(ampOscTrend).flatten();
			rsq = 1-np.var(ampOscResid)/np.var(ampOsc);
			ModCorr[rr,cc] = np.sqrt(rsq);
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	MIs = np.arctanh(ModCorr);
	return MIs;
	
def GLMcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
	"""	GLMCOMOD Generates a Generalized-Linear-Model Based Comodulogram
		USAGE: MIs, comodplt = GLMcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
		passbandRipl is on a linear scale (not decibel): its preferred value is 0.02 """

	oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
	MIs = GenLinMod(oscAmpMod,oscForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Generalized Linear Model (GLM)");
	return MIs, comodplt;

def GLMcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,sd_rel_phase=0.2,sd_rel_amp=40):
	"""	GLMCOMODCWT Generates a Generalized-Linear-Model Based Comodulogram
		USAGE: MIs, comodplt = GLMcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz)
        sd_rel_phase is filter bandwidth given as a fraction of the freqForPhase (default 0.2)
        sd_rel_amp is filter bandwidth given as a fraction of freqForPhase in order 
            to enforce variable-bandwidth bandpass filtering around freqForAmp 
			(freqForAmp +/- sd_rel_amp * freqForPhase)"""

	coefsForAmp, coefsForPhase = preCWTProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,sd_rel_phase,sd_rel_amp)
	MIs = GenLinModCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Generalized Linear Model (GLM)");
	return MIs, comodplt;

def EnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
	""" ENVSIGCORR Calculates comulolograms based on envelope-to-signal correlation
		USAGE: MIs = EnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		oscAmpMod is a cell matrix of time-series oscillations bandpassed 
    		around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
		oscForPhase is a row cell vector of time-series oscillations bandpassed
            around freqForPhase with some small bandwidth """
      
	" Applying Envelope-to-Signal-Correlation based CFC to Oscillations Data "
	ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(oscAmpMod[rr,cc,:]);
			phaseOsc = np.real(oscForPhase[0,rr,:]);
			ModCorr[rr,cc] = np.corrcoef(ampOsc,phaseOsc)[0,1];
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	MIs = np.arctanh(np.abs(ModCorr));
	return MIs;

def EnvSigCorrCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase):
	""" ENVSIGCORR Calculates comulolograms based on envelope-to-signal correlation
		USAGE: MIs = EnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		coefsForAmp is a cell matrix of time-series oscillations bandpassed
			around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
        coefsForPhase is a row cell vector of time-series oscillations bandpassed
            around freqForPhase with some small bandwidth """
      
	" Applying Envelope-to-Signal-Correlation based CFC to Oscillations Data "
	ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(coefsForAmp[:,cc,rr]);
			phaseOsc = np.angle(coefsForPhase[:,rr]);
			ModCorr[rr,cc] = np.corrcoef(ampOsc,phaseOsc)[0,1];
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	MIs = np.arctanh(np.abs(ModCorr));
	return MIs;
 
def ESCcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
	"""	ESCCOMOD Generates a Envelope-to-Signal Correlation-Based Comodulogram
		USAGE: MIs, comodplt = ESCcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
		passbandRipl is on a linear scale (not decibel): its preferred value is 0.02 """

	oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
	MIs = EnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Envelope-to-Signal Correlation (ESC)");
	return MIs, comodplt;

def ESCcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,sd_rel_phase=0.2,sd_rel_amp=40):
	"""	ESCCOMODCWT Generates a Envelope-to-Signal Correlation-Based Comodulogram
		USAGE: MIs, comodplt = ESCcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz)
        sd_rel_phase is filter bandwidth given as a fraction of the freqForPhase (default 0.2)
        sd_rel_amp is filter bandwidth given as a fraction of freqForPhase in order 
            to enforce variable-bandwidth bandpass filtering around freqForAmp 
			(freqForAmp +/- sd_rel_amp * freqForPhase)"""

	coefsForAmp, coefsForPhase = preCWTProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,sd_rel_phase,sd_rel_amp)
	MIs = EnvSigCorrCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Envelope-to-Signal Correlation (ESC)");
	return MIs, comodplt;

def NormEnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
	""" NORMENVSIGCORR Calculates comulolograms based on normalized envelope-to-signal correlation
		USAGE: MIs = NormEnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		oscAmpMod is a cell matrix of time-series oscillations bandpassed 
			around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
		oscForPhase is a row cell vector of time-series oscillations bandpassed
			around freqForPhase with some small bandwidth """

	" Applying Envelope-to-Signal-Correlation based CFC to Oscillations Data "
	ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(oscAmpMod[rr,cc,:]);
			phaseOsc = np.angle(oscForPhase[0,rr,:]);
			ModCorr[rr,cc] = np.corrcoef(ampOsc,np.cos(phaseOsc))[0,1];
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	MIs = np.arctanh(np.abs(ModCorr));
	return MIs;

def NormEnvSigCorrCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase):
	""" NORMENVSIGCORRCWT Calculates comulolograms based on normalized envelope-to-signal correlation
		USAGE: MIs = NormEnvSigCorrCWT(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		coefsForAmp is a cell matrix of time-series oscillations bandpassed
			around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
        coefsForPhase is a row cell vector of time-series oscillations bandpassed
            around freqForPhase with some small bandwidth """

	" Applying Envelope-to-Signal-Correlation based CFC to Oscillations Data "
	ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(coefsForAmp[:,cc,rr]);
			phaseOsc = np.angle(coefsForPhase[:,rr]);
			ModCorr[rr,cc] = np.corrcoef(ampOsc,np.cos(phaseOsc))[0,1];
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	MIs = np.arctanh(np.abs(ModCorr));
	return MIs;

def NESCcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
	"""	NESCCOMOD Generates a Normalized ESC-Based Comodulogram
		USAGE: MIs, comodplt = NESCcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
		passbandRipl is on a linear scale (not decibel): its preferred value is 0.02 """

	oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
	MIs = NormEnvSigCorr(oscAmpMod,oscForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Normalized Envelope-to-Signal Correlation (NESC)");
	return MIs, comodplt;
	
def NESCcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,sd_rel_phase=0.2,sd_rel_amp=40):
	"""	NESCCOMODCWT Generates a Envelope-to-Signal Correlation-Based Comodulogram
		USAGE: MIs, comodplt = NESCcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz)
        sd_rel_phase is filter bandwidth given as a fraction of the freqForPhase (default 0.2)
        sd_rel_amp is filter bandwidth given as a fraction of freqForPhase in order 
            to enforce variable-bandwidth bandpass filtering around freqForAmp 
			(freqForAmp +/- sd_rel_amp * freqForPhase)"""

	coefsForAmp, coefsForPhase = preCWTProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,sd_rel_phase,sd_rel_amp)
	MIs = NormEnvSigCorrCWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Normalized Envelope-to-Signal Correlation (NESC)");
	return MIs, comodplt;
	
def PrinCompAnal(MultChannIn):
	"""	PRINCOMPANAL Outputs Principal Component Analysis of Multichannel Input                                                                       
		USAGE: [PrinVals PrinComps] = PrinCompAnal(MultChannIn)
		Each row of MultChannIn is a separate channel, each column represents a
       		synchronous sampling of all channels at a time point.
		PrinVals is a column vector containing the eigenvalues of the covariance matrix for MultChannIn.
		PrinComps is a matrix whose columns are the principal components of the MultChannIn data."""
	
	MultChannInCov = np.cov(MultChannIn);
	PrinVals, PrinComps = np.linalg.eig(MultChannInCov);
	return PrinVals, PrinComps
	
def zScoredMVL(TwoChannIn):
	""" ZSCOREDMVL Give a z-score to the mean vector based on PCA
		USAGE: zScore = zScoredMVL(MultChannIn)
		Each row of MultChannIn is a separate channel, each column represents a
       		synchronous sampling of all channels at a time point.
		zScore is the z-score of the mean column vector in MultChannIn. """
	
	XPrinVals, XPrinComps = PrinCompAnal(TwoChannIn);
	meanVect = np.array([np.mean(TwoChannIn[0,:]), np.mean(TwoChannIn[1,:])]);
	theta = np.arccos(np.dot(meanVect,XPrinComps[0,:])/np.linalg.norm(meanVect));
	R = np.sqrt((np.sqrt(XPrinVals[0])*np.cos(theta))**2+(np.sqrt(XPrinVals[1])*np.sin(theta))**2);
	zScore = np.linalg.norm(meanVect)/R;	
	return zScore
	
def zScoredMV_PCA(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
	""" ZSCOREDMVCFC Calculates and displays the CFC Comulolograms based on inputs
		USAGE: [MIs MVLs] = ZScoredMVCFC(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		MIs is the comodulogram based on the z-scored mean vector
		MVLs is the comodulogram based on Canolty's mean vector length (MVL)
		oscAmpMod is a cell matrix of time-series oscillations bandpassed 
    		around freqForAmp with bandwidth specified by freqForPhase.
		oscForPhase is a row cell vector of time-series oscillations bandpassed
    		around freqForPhase with some small bandwidth. """
	
	" Applying Envelope-to-Signal-Correlation based CFC to Oscillation Data "
	MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	MVLs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(oscAmpMod[rr,cc,:]);
			phaseOsc = np.angle(oscForPhase[0,rr,:]);
			phasor = ampOsc*np.exp(1j*phaseOsc);
			MVLs[rr,cc] = np.abs(np.mean(phasor));
			phasorComponents = np.row_stack((np.real(phasor), np.imag(phasor)));
			MIs[rr,cc] = zScoredMVL(phasorComponents);
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	return MIs, MVLs;
	
def zScoredMV_PCA_CWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase):
	""" ZSCOREDMVCFC Calculates and displays the CFC Comulolograms based on inputs
		USAGE: [MIs MVLs] = ZScoredMVCFC(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
		MIs is the comodulogram based on the z-scored mean vector
		MVLs is the comodulogram based on Canolty's mean vector length (MVL)
		coefsForAmp is a cell matrix of time-series oscillations bandpassed
			around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
        coefsForPhase is a row cell vector of time-series oscillations bandpassed
            around freqForPhase with some small bandwidth """
	
	" Applying Envelope-to-Signal-Correlation based CFC to Oscillation Data "
	MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	MVLs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
	
	" Phase will change each row. Amplitude will change each column "
	frqAmpSize = np.size(freqForAmp);
	frqPhaseSize = np.size(freqForPhase);
	for cc in np.arange(frqAmpSize):
		for rr in np.arange(frqPhaseSize):
			ampOsc = np.abs(coefsForAmp[:,cc,rr]);
			phaseOsc = np.angle(coefsForPhase[:,rr]);
			phasor = ampOsc*np.exp(1j*phaseOsc);
			MVLs[rr,cc] = np.abs(np.mean(phasor));
			phasorComponents = np.row_stack((np.real(phasor), np.imag(phasor)));
			MIs[rr,cc] = zScoredMVL(phasorComponents);
			delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
			print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
				", Frequency for Amplitude [Hz] = "+str(ctrfreq));
	return MIs, MVLs;    
	
def zScoreMVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
	""" ZSCOREMVCOMOD Generates a Mean Vector Length-Based Comodulogram
		USAGE: MIs = zScoreMVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
		passbandRipl is on a linear scale (not decibel): its preferred value is 0.02
		option is either 'MVL' or 'Z-Score':
			"MVL" gives the mean vector length based on Canolty's Work
			"Z-Score" gives z-score of mean vector based on PCA """
	
	oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
	MIs, MVLs = zScoredMV_PCA(oscAmpMod,oscForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Principal Component Analysis (PCA)");
	return MIs, MVLs, comodplt;
	
def zScoreMVcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,sd_rel_phase=0.2,sd_rel_amp=40):
	""" ZSCOREMVCOMODCWT Generates a Mean Vector Length-Based Comodulogram
		USAGE: MIs = zScoreMVcomodCWT(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
		sigForAmp is the input LFP to be analyzed for amplitude
		sigForPhase is the input LFP to be analyzed for phase
		freqForAmp is a vector of center frequencies (frequency for amplitude)
		freqForPhase is a vector of frequency for phase controlling bandwidth
		fs is sampling rate (Hz)
        sd_rel_phase is filter bandwidth given as a fraction of the freqForPhase (default 0.2)
        sd_rel_amp is filter bandwidth given as a fraction of freqForPhase in order 
            to enforce variable-bandwidth bandpass filtering around freqForAmp 
			(freqForAmp +/- sd_rel_amp * freqForPhase)"""
	
	coefsForAmp, coefsForPhase = preCWTProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,sd_rel_phase,sd_rel_amp)
	MIs, MVLs = zScoredMV_PCA_CWT(coefsForAmp,coefsForPhase,freqForAmp,freqForPhase);
	comodplt = comodShow(freqForAmp,freqForPhase,MIs);
	plt.title("Principal Component Analysis (PCA)");
	return MIs, MVLs, comodplt;

def KullLiebDiv(P, Q = None):
    """ KULLLEIBDIV Calculates the Kullback-Liebler Divergence of the probability vector P from the probability vector Q.
        USAGE: KLDiv = KullLiebDiv(P,Q)
       	If Q is not given, it is assumed to be the uniform distribution of the length of P.
       	This function accepts two inputs methods: KLDiv(P,Q) or KLDiv(P) with Q implied as aforementioned. """
       	
    if Q == None: KLDiv = np.log(np.size(P)) - entropy(P);
    else: KLDiv = entropy(P,Q);
    return KLDiv;

def KLDivModIndex(P):
    """	KLDIVMODINDEX determines the MI for an input Probability Vector P
        USAGE: MI = KLDivModIndex(P)
        Divides the Kullback-Liebler Divergence of P with respect to the uniform distribution
           	by natural log of the length of P which bounds the output between 0 and 1. """

    return KullLiebDiv(P)/np.size(P);

def KullLeibBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n):
    """ KULLLEIBBIN Calculates and displays the CFC Comulolograms based on inputs
        USAGE: MIs = KullLeibBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n,option)
        oscAmpMod is a cell matrix of time-series oscillations bandpassed 
       		around freqForAmp with bandwidth specified by freqForPhase.
       	oscForPhase is a row cell vector of time-series oscillations bandpassed 
       		around freqForPhase with some small bandwidth.
       	n is the number phasebins for the Kullback-Liebler Modulation Index. """

    " Applying Kullback-Leibler Divergence-based CFC to Oscillation Data "
    phaseBins = np.linspace(-np.pi,np.pi,n+1); highFreqAmplitude = np.zeros(n);
    MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));

    " Phases will change each row. Amplitudes will change each column "
    frqAmpSize = np.size(freqForAmp); frqPhaseSize = np.size(freqForPhase);
    for cc in np.arange(frqAmpSize):
        for rr in np.arange(frqPhaseSize):
       		amplitudes = np.abs(oscAmpMod[rr,cc,:]);
       		phases = np.angle(oscForPhase[0,rr,:]);
       		for kk in np.arange(n):
           		amps = amplitudes[(phases > phaseBins[kk]) & (phases <= phaseBins[kk+1])];
           		highFreqAmplitude[kk] = np.mean(amps);
       		MIs[rr,cc] = KLDivModIndex(highFreqAmplitude);
       		delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
       		print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
           		", Frequency for Amplitude [Hz] = "+str(ctrfreq));
    return MIs;

def KLDivMIcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02,n=36):
    """ KLDIVMICOMOD Generates a Kullback-Liebler-Based Comodulogram
        USAGE: MIs = KLDivMIcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,n,option)
        sigForAmp is the input LFP to be analyzed for amplitude
        sigForPhase is the input LFP to be analyzed for phase
        freqForAmp is a vector of center frequencies (frequency for amplitude)
        freqForPhase is a vector of frequency for phase controlling bandwidth
        fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
        passbandRipl is on a linear scale (not decibel): its preferred value is 0.02
        n is the number phasebins for the Kullback-Liebler Modulation Index(MI). """

    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
    MIs = KullLeibBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n);
    comodplt = comodShow(freqForAmp,freqForPhase,MIs);
    plt.title("Kullback-Liebler Divergence (KLDiv)");
    return MIs, comodplt;

def HeightsRatioBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n,method):
    """ HEIGHTSRATIOBIN Calculates and displays the CFC Comulolograms based on Heights Ratio
        MIs = HeightsRatioBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n,method,option)
        oscAmpMod is a cell matrix of time-series oscillations bandpassed
        	around freqForAmp with bandwidth specified by freqForPhase.
        oscForPhase is a row cell vector of time-series oscillations bandpassed
        	around freqForPhase with some small bandwidth
        n is the number phasebins for the Heights-Ratio Modulation Index(MI)
        method: there are 3 ways of doing this
       		1) 'Lakatos' -- h_max/h_min
       		2) 'Tort' -- (h_max - h_min)/h_max;
       		3) 'AM Radio' --- (h_max - h_min)/(h_max + h_min) """

    " Applying Kullback-Leibler Divergence-based CFC to Oscillation Data "
    phaseBins = np.linspace(-np.pi,np.pi,n+1); highFreqAmplitude = np.zeros(n);
    MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
    
    " Phases will change each row. Amplitudes will change each column "
    frqAmpSize = np.size(freqForAmp); frqPhaseSize = np.size(freqForPhase);
    for cc in np.arange(frqAmpSize):
       	for rr in np.arange(frqPhaseSize):
            amplitudes = np.abs(oscAmpMod[rr,cc,:]);
            phases = np.angle(oscForPhase[0,rr,:]);
            for kk in np.arange(n):
                amps = amplitudes[(phases > phaseBins[kk]) & (phases <= phaseBins[kk+1])];
                highFreqAmplitude[kk] = np.mean(amps);
            if method == 'AM Radio':
                MIs[rr,cc] = (max(highFreqAmplitude)-min(highFreqAmplitude)) \
           			/ (max(highFreqAmplitude)+min(highFreqAmplitude));
            if method == 'Tort':
                MIs[rr,cc] = (max(highFreqAmplitude)-min(highFreqAmplitude)) \
           			/ (max(highFreqAmplitude));
            if method == 'Lakatos':
                MIs[rr,cc] = (max(highFreqAmplitude))/(min(highFreqAmplitude));
            delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
            print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
	            ", Frequency for Amplitude [Hz] = "+str(ctrfreq));
    return MIs;

def HRcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02,n=36,method='AM Radio'):
    """ HRCOMOD Generates a Heights Ratio-Based Comodulogram
        USAGE: MIs = HRcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,n,method,option)
        sigForAmp is the input LFP to be analyzed for amplitude
        sigForPhase is the input LFP to be analyzed for phase
        freqForAmp is a vector of center frequencies (frequency for amplitude)
       	freqForPhase is a vector of frequency for phase controlling bandwidth
        fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
        passbandRipl is on a linear scale (not decibel): its preferred value is 0.02
       	n is the number phasebins for the Heights-Ratio Modulation Index (MI)
        method: there are 3 ways of doing this
			1) 'Lakatos' -- h_max/h_min
       		2) 'Tort' -- (h_max - h_min)/h_max;
       		3) 'AM Radio' --- (h_max - h_min)/(h_max + h_min) """
    
    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
    MIs = HeightsRatioBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n,method);
    comodplt = comodShow(freqForAmp,freqForPhase,MIs);
    plt.title("Heights Ratio (HR)");
    return MIs, comodplt;

def PhaseLocVal(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
    """ PHASELOCVAL Calculates comulolograms based on Phase Locking Value
        USAGE: MIs = PhaseLocVal(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
        oscAmpMod is a cell matrix of time-series oscillations bandpassed
        	around freqForAmp (vector) with bandwidth specified by freqForPhase (vector).
        oscForPhase is a row cell vector of time-series oscillations bandpassed
        	around freqForPhase with some small bandwidth """
            
    " Applying Generalized Linear Model-based CFC to Oscillations Data "
    PLVs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
                    
    " Phase will change each row. Amplitude will change each column "
    frqAmpSize = np.size(freqForAmp);
    frqPhaseSize = np.size(freqForPhase);
    for cc in np.arange(frqAmpSize):
       	for rr in np.arange(frqPhaseSize):
       	    ampOsc = np.abs(oscAmpMod[rr,cc,:]);
       	    phaseOsc = np.angle(oscForPhase[0,rr,:]);
            ampOscPhase = np.angle(sig.hilbert(ampOsc));
       	    PLVs[rr,cc] = np.abs(np.mean(np.exp(1j*(phaseOsc - ampOscPhase))));
       	    delf = freqForPhase[rr]; ctrfreq = freqForAmp[cc];
       	    print("Completed: Frequency for Phase [Hz] = "+str(delf)+ \
           	    ", Frequency for Amplitude [Hz] = "+str(ctrfreq));
    MIs = np.arcsin(2*PLVs-1);
    return MIs;

def PLVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
    """	PLVCOMOD Generates a Phase-Locking-Value Based Comodulogram
       	USAGE: MIs, comodplt = PLVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl,option)
       	sigForAmp is the input LFP to be analyzed for amplitude
       	sigForPhase is the input LFP to be analyzed for phase
       	freqForAmp is a vector of center frequencies (frequency for amplitude)
       	freqForPhase is a vector of frequency for phase controlling bandwidth
       	fs is sampling rate (Hz), bw is the bandwidth of the bandpass filters typically (4.5 Hz)
    	passbandRipl is on a linear scale (not decibel): its preferred value is 0.02 """
            
    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
	    freqForPhase,fs,bw=bw,passbandRipl=passbandRipl);
    MIs = PhaseLocVal(oscAmpMod,oscForPhase,freqForAmp,freqForPhase);
    comodplt = comodShow(freqForAmp,freqForPhase,MIs);
    plt.title("Phase Locking Value (PLV)");
    return MIs, comodplt;

def mscohere(x,y,Fs,f,window):
    """ MSCOHERE Returns the Magnitude-Squared Coherence at Specified Frequencies:
        USAGE = Cxy = mscohere(x,y,Fs,f,window)
        x and y are input signals
       	Fs is the sampling frequency in [Hz]
       	f are the specified frequencies over which Cxy is calculated 
       	window is the windowing function (i.e. 'flattop', 'blackmanharris', 'hamming', 'hanning', etc.) """   	
    
    nfft = int(2**(np.floor(np.log2(np.size(y)))-1));
    noverlap = np.floor(nfft*0.99);
    ff, Cxy = coherence(x,y,fs=Fs,nperseg=nfft,noverlap=noverlap,window=window);
    return np.exp(np.interp(f,ff,np.log(Cxy)));

def CVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,passbandRipl=0.02,window='flattop',bw=None):        
    """ CVCOMODULOGRAM Generates a Coherence Value-Based Comodulogram
        USAGE: MIs = CVcomodulogram(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,passbandRipl,option)
        sigForAmp is the input LFP to be analyzed for amplitude
        sigForPhase is the input LFP to be analyzed for phase
        freqForAmp is a vector of center frequencies (frequency for amplitude)
        freqForPhase is a vector of frequency for phase controlling bandwidth
        fs is sampling rate (Hz)
        passbandRipl is on a linear scale (not decibel): its preferred value is 0.02 
        window is the windowing function (i.e. 'flattop', 'blackmanharris', 'hamming', 'hanning', etc.) 
        bw is the bandwidth of the filter for sigForAmp (default is max of freqForPhase)"""
        
    if bw == None: bandwidth = max(freqForPhase);
    else: bandwidth = bw;
    oscAmpMod = CFCfilt(sigForAmp,freqForAmp,np.array([bandwidth]),fs,passbandRipl);
    CVs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
    for cc in np.arange(np.size(freqForAmp)):
       	ampOsc = np.abs(oscAmpMod[-1,cc,:]); ctrfreq = freqForAmp[cc];
       	CVs[:,cc] = mscohere(ampOsc,sigForPhase,fs,freqForPhase,window);
       	print("Completed: Frequency for Amplitude [Hz] = "+str(ctrfreq));
    MIs = np.arctanh(CVs);
        
    comodplt = comodShow(freqForAmp,freqForPhase,MIs);
    plt.title("Coherence Value (CV)");
    return MIs, comodplt;

def powSpecDens(x,Fs,f,window):
    """ POWSPECDENS Returns the Magnitude-Squared Coherence at Specified Frequencies:
    	USAGE = Cxy = powSpecDens(x,Fs,f,window)
    	x and y are input signals
    	Fs is the sampling frequency in [Hz]
    	f are the specified frequencies over which Cxy is calculated 
    	window is the windowing function (i.e. 'flattop', 'blackmanharris', 'hamming', 'hanning', etc.) """

    nfft = int(2**(np.floor(np.log2(np.size(x)))-1)); noverlap = np.floor(nfft*0.99); # nfft - 1;
    ff, Pxx = welch(x,fs=Fs,nperseg=nfft,noverlap=noverlap,window=window);
    return np.exp(np.interp(f,ff,np.log(Pxx)));

def PSDcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,passbandRipl=0.02,window='flattop',bw=None):
    """ PSDCOMODULOGRAM Generates a Power Spectral Density-Based Comodulogram
        USAGE: MIs = CVcomodulogram(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,passbandRipl,option)
        sigForAmp is the input LFP to be analyzed for amplitude
        sigForPhase is the input LFP to be analyzed for phase
        freqForAmp is a vector of center frequencies (frequency for amplitude)
        freqForPhase is a vector of frequency for phase controlling bandwidth
       	fs is sampling rate (Hz)
       	passbandRipl is on a linear scale (not decibel): its preferred value is 0.02
       	window is the windowing function (i.e. 'flattop', 'blackmanharris', 'hamming', 'hanning', etc.)
       	bw is the bandwidth of the filter for sigForAmp (default is max of freqForPhase)"""
    
    if bw == None: bandwidth = max(freqForPhase);
    else: bandwidth = bw;
    oscAmpMod = CFCfilt(sigForAmp,freqForAmp,np.array([bandwidth]),fs,passbandRipl);
    MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)));
    for cc in np.arange(np.size(freqForAmp)):
       	ampOsc = np.abs(oscAmpMod[-1,cc,:]); ctrfreq = freqForAmp[cc];
       	MIs[:,cc] = powSpecDens(ampOsc,fs,freqForPhase,window);
       	print("Completed: Frequency for Amplitude [Hz] = "+str(ctrfreq));

    comodplt = comodShow(freqForAmp,freqForPhase,MIs);
    plt.title("Power Spectral Density (PSD)");
    return MIs, comodplt;

# CHANGE CWT BASED COMODULOGRAM AS FOLLOWS:
# CURRENTLY CWT USES NO VARIABLE-BANDWIDTH CONSIDERATIONS
# make sd_ffa depend on freqForPhase to somehow enforce this
# Later Translate to MATLAB

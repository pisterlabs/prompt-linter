#!/usr/bin/python

"""
Speech Transmission Index (STI) from speech waveforms (real speech)

Copyright (C) 2011 Jon Polom <jmpolom@wayne.edu>
Licensed under the GNU General Public License
"""

from datetime import date, datetime
from matplotlib.mlab import cohere,psd
from numpy import append,array,clip,log10,nonzero,ones,power,reshape
from numpy import searchsorted,shape,sqrt,sum,vstack,zeros
from numpy.ma import masked_array
from scipy.io import wavfile
from scipy.signal import butter,firwin,decimate,lfilter
from sys import stdout
from warnings import catch_warnings,simplefilter

__author__ = "Jonathan Polom <jmpolom@wayne.edu>"
__date__ = date(2011, 04, 22)
__version__ = "0.5"

def thirdOctaves(minFreq, maxFreq):
    """
    Calculates a list of frequencies spaced 1/3 octave apart in hertz
    between minFreq and maxFreq
    
    Input
    -----
    * minFreq : float or int
    
        Must be non-zero and non-negative

    * maxFreq : float or int
    
        Must be non-zero and non-negative
    
    Output
    ------
    * freqs : ndarray
    """

    if minFreq <= 0 or maxFreq <= 0:
        raise ValueError("minFreq and maxFreq must be non-zero and non-negative")
    else:
        maxFreq = float(maxFreq)

        f = float(minFreq)
        freqs = array([f])

        while f < maxFreq:
            f = f * 10**0.1
            freqs = append(freqs, f)

        return freqs

def fftWindowSize(freqRes, hz):
    """
    Calculate power of 2 window length for FFT to achieve specified frequency
    resolution. Useful for power spectra and coherence calculations.
    
    Input
    -----
    * freqRes : float
    
        Desired frequency resolution in hertz
    
    * hz : int
    
        Sample rate, in hertz, of signal undergoing FFT
    
    Output
    ------
    * window : int
    """
    
    freqRes = float(freqRes)         # make sure frequency res is a float
    pwr = 1                          # initial power of 2 to try
    res = hz / float(2**pwr) # calculate frequency resolution
    
    while res > freqRes:
        pwr += 1
        res = hz / float(2**pwr)
    
    return 2**pwr

def downsampleBands(audio, hz, downsampleFactor):
    """
    Downsample audio by integer factor
    
    Input
    -----
    * audio : array-like
    
        Array of original audio samples
    
    * hz : float or int
    
        Original audio sample rate in hertz
    
    * downsampleFactor : int
    
        Factor to downsample audio by, if desired
    
    Output
    ------
    * dsAudio : ndarray
    
        Downsampled audio array
    
    * hz : int
    
        Downsampled audio sample rate in hertz
    """

    # calculate downsampled audio rate in hertz
    downsampleFactor = int(downsampleFactor)        # factor must be integer
    hz = int(hz / downsampleFactor)
    
    for band in audio:
        ds = decimate(band, downsampleFactor, ftype='fir')
        
        try:
            dsAudio = append(dsAudio, ds)
        except:
            dsAudio = ds
    
    return dsAudio, hz

def octaveBandFilter(audio, hz, 
                     octaveBands=[125, 250, 500, 1000, 2000, 4000, 8000],
                     butterOrd=6, hammingTime=16.6):

    """
    Octave band filter raw audio. The audio is filtered through butterworth
    filters of order 6 (by default), squared to obtain the envelope and finally
    low-pass filtered using a 'hammingTime' length Hamming filter at 25 Hz.
    
    Input
    -----
    * audio : array-like
    
        Array of raw audio samples
    
    * hz : float or int
    
        Audio sample rate in hertz
    
    * octaveBands : array-like
    
        list or array of octave band center frequencies
    
    * butterOrd : int
    
        butterworth filter order
    
    * hammingTime : float or int
    
        Hamming window length, in milliseconds relative to audio sample rate
    
    Output
    ------
    * octaveBandAudio : ndarray
    
        Octave band filtered audio
    
    * hz : float or int
    
        Filtered audio sample rate
    """

    print "Butterworth filter order:",butterOrd
    print "Hamming filter length:   ",hammingTime,"milliseconds"
    print "Audio sample rate:       ",hz

    # calculate the nyquist frequency
    nyquist = hz * 0.5

    # length of Hamming window for FIR low-pass at 25 Hz
    hammingLength = (hammingTime / 1000.0) * hz

    # process each octave band
    for f in octaveBands:
        bands = str(octaveBands[:octaveBands.index(f) + 1]).strip('[]')
        statusStr = "Octave band filtering audio at: " + bands
        unitStr = "Hz ".rjust(80 - len(statusStr))
        stdout.write(statusStr)
        stdout.write(unitStr)
        stdout.write('\r')
        stdout.flush()
    
        # filter the output at the octave band f
        f1 = f / sqrt(2)
        f2 = f * sqrt(2)

        # for some odd reason the band-pass butterworth doesn't work right
        # when the filter order is high (above 3). likely a SciPy issue?
        # also, butter likes to complain about possibly useless results when
        # calculating filter coefficients for high order (above 4) low-pass
        # filters with relatively low knee frequencies (relative to nyquist F).
        # perhaps I just don't know how digital butterworth filters work and
        # their limitations but I think this is odd.
        # the issue described here will be sent to their mailing list
        if f < max(octaveBands):
            with catch_warnings():      # suppress the spurious warnings given
                simplefilter('ignore')  # under certain conditions
                b1,a1 = butter(butterOrd, f1/nyquist, btype='high')
                b2,a2 = butter(butterOrd, f2/nyquist, btype='low')
            
            filtOut = lfilter(b1, a1, audio)   # high-pass raw audio at f1
            filtOut = lfilter(b2, a2, filtOut) # low-pass after high-pass at f1
        else:
            with catch_warnings():
                simplefilter('ignore')
                b1,a1 = butter(butterOrd, f/nyquist, btype='high')
            filtOut = lfilter(b1, a1, audio)

        filtOut = array(filtOut)**2
        b = firwin(hammingLength, 25.0, nyq=nyquist)
        filtOut = lfilter(b, 1, filtOut)
        filtOut = filtOut * -1.0

        # stack-up octave band filtered audio
        try:
            octaveBandAudio = vstack((octaveBandAudio, filtOut))
        except:
            octaveBandAudio = filtOut

    print
    return octaveBandAudio

def octaveBandSpectra(filteredAudioBands, hz, fftRes=0.06):
    """
    Calculate octave band power spectras
    
    Input
    -----
    * filteredAudioBands : array-like
    
        Octave band filtered audio
    
    * hz : float or int
    
        Audio sample rate in hertz. Must be the same for clean and dirty audio
    
    * fftRes : float or int
    
        Desired FFT frequency resolution
    
    Output
    ------
    * spectras : ndarray
    
        Power spectra values
    
    * fftfreqs : ndarray
    
        Frequencies for FFT points
    """

    # FFT window size for PSD calculation: 32768 for ~0.06 Hz res at 2 kHz
    psdWindow = fftWindowSize(fftRes, hz)
    
    print "Calculating octave band power spectras",
    print "(FFT length:",psdWindow,"samples)"

    for band in filteredAudioBands:        
        spectra, freqs = psd(band, NFFT=psdWindow, Fs=hz)
        spectra = reshape(spectra, len(freqs))  # change to row vector
        spectra = spectra / max(spectra)        # scale to [0,1]
        
        # stack-up octave band spectras
        try:
            spectras = vstack((spectras, spectra))
            fftfreqs = vstack((fftfreqs, freqs))
        except:
            spectras = spectra
            fftfreqs = freqs
        
    return spectras, fftfreqs

def octaveBandCoherence(degrAudioBands, refAudioBands,
                        hz, fftRes=0.122):
    """
    Calculate coherence between clean and degraded octave band audio
    
    Input
    -----
    * degrAudioBands : array-like
    
        Degraded octave band audio
    
    * refAudioBands : array-like
    
        Reference (clean) octave band audio
    
    * hz : float or int
    
        Audio sample rate. Must be common between clean and dirty audio
    
    * fftRes : float or int
    
        Desired FFT frequency resolution
    
    Output
    ------
    * coherences : ndarray
    
        Coherence values
    
    * fftfreqs : ndarray
    
        Frequencies for FFT points
    """

    # FFT window size for PSD calculation: 32768 for ~0.06 Hz res at 2 kHz
    # Beware that 'cohere' isn't as forgiving as 'psd' with FFT lengths 
    # larger than half the length of the signal
    psdWindow = fftWindowSize(fftRes, hz)
    
    print "Calculating degraded and reference audio coherence",
    print "(FFT length:",psdWindow,"samples)"

    for i,band in enumerate(degrAudioBands):
        with catch_warnings():      # catch and ignore spurious warnings
            simplefilter('ignore')  # due to some irrelevant divide by 0's
            coherence, freqs = cohere(band, refAudioBands[i], 
                                      NFFT=psdWindow, Fs=hz)
        
        # stack-up octave band spectras
        try:
            coherences = vstack((coherences, coherence))
            fftfreqs = vstack((fftfreqs, freqs))
        except:
            coherences = coherence
            fftfreqs = freqs
        
    return coherences, fftfreqs

def thirdOctaveRootSum(spectras, fftfreqs, minFreq=0.25, maxFreq=25.0):
    """
    Calculates square root of sum of spectra over 1/3 octave bands
    
    Input
    -----
    * spectras : array-like
    
        Array or list of octave band spectras
    
    * fftfreqs : array-like
    
        Array or list of octave band FFT frequencies
    
    * minFreq : float
    
        Min frequency in 1/3 octave bands
    
    * maxFreq : float
    
        Max frequency in 1/3 octave bands
    
    Output
    ------
    * thirdOctaveRootSums : ndarray
    
        Square root of spectra sums over 1/3 octave intervals
    """

    print "Calculating 1/3 octave square-rooted sums from",
    print minFreq,"to",maxFreq,"Hz"

    thirdOctaveBands = thirdOctaves(minFreq, maxFreq)
        
    # loop over the spectras contained in 'spectras' and calculate 1/3 oct MTF
    for i,spectra in enumerate(spectras):
        freqs = fftfreqs[i]                # get fft frequencies for spectra

        # calculate the third octave sums
        for f13 in thirdOctaveBands:
            f131 = f13 / power(2, 1.0/6.0) # band start
            f132 = f13 * power(2, 1.0/6.0) # band end
            
            li = searchsorted(freqs, f131)
            ui = searchsorted(freqs, f132) + 1
            
            s = sum(spectra[li:ui]) # sum the spectral components in band
            s = sqrt(s)             # take square root of summed components
            
            try:
                sums = append(sums, s)
            except:
                sums = array([s])

        # stack-up third octave modulation transfer functions
        try:
            thirdOctaveSums = vstack((thirdOctaveSums, sums))
        except:
            thirdOctaveSums = sums
        
        # remove temp 'sum' and 'counts' variables for next octave band
        del(sums)

    return thirdOctaveSums

def thirdOctaveRMS(spectras, fftfreqs, minFreq=0.25, maxFreq=25.0):
    """
    Calculates RMS value of spectra over 1/3 octave bands
    
    Input
    -----
    * spectras : array-like
    
        Array or list of octave band spectras
    
    * fftfreqs : array-like
    
        Array or list of octave band FFT frequencies
    
    * minFreq : float
    
        Min frequency in 1/3 octave bands
    
    * maxFreq : float
    
        Max frequency in 1/3 octave bands
    
    Output
    ------
    * thirdOctaveRMSValues : ndarray
    
        RMS value of spectra over 1/3 octave intervals
    """

    print "Calculating 1/3 octave RMS values from",
    print minFreq,"to",maxFreq,"Hz"

    thirdOctaveBands = thirdOctaves(minFreq, maxFreq)
        
    # loop over the spectras contained in 'spectras' and calculate 1/3 oct MTF
    for i,spectra in enumerate(spectras):
        freqs = fftfreqs[i]                # get fft frequencies for spectra

        # calculate the third octave sums
        for f13 in thirdOctaveBands:
            f131 = f13 / power(2, 1.0/6.0) # band start
            f132 = f13 * power(2, 1.0/6.0) # band end
            
            li = searchsorted(freqs, f131)
            ui = searchsorted(freqs, f132) + 1
            
            s = sum(spectra[li:ui]**2)  # sum the spectral components in band
            s = s / len(spectra[li:ui]) # divide by length of sum
            s = sqrt(s)                 # square root
            
            try:
                sums = append(sums, s)
            except:
                sums = array([s])

        # stack-up third octave modulation transfer functions
        try:
            thirdOctaveRMSValues = vstack((thirdOctaveRMSValues, sums))
        except:
            thirdOctaveRMSValues = sums
        
        # remove temp 'sum' and 'counts' variables for next octave band
        del(sums)

    return thirdOctaveRMSValues

def sti(modulations, coherences, minCoherence=0.8):
    """
    Calculate the speech transmission index from third octave modulation
    indices. The indices are truncated after coherence between clean and dirty
    audio falls below 'minCoherence' or 0.8, by default.
    
    Input
    -----
    * modulations : array-like
    
        Modulation indices spaced at 1/3 octaves within each octave band
    
    * coherences : array-like
    
        Coherence between clean and dirty octave band filtered audio
    
    * minCoherence : float
    
        The minimum coherence to include a mod index in the STI computation
    
    Output
    ------
    * index : float
    
        The speech transmission index (STI)
    """
    
    # create masking array of zeroes
    snrMask = zeros(modulations.shape, dtype=int)
    
    # sort through coherence array and mask corresponding SNRs where coherence
    # values fall below 'minCoherence' (0.8 in most cases and by default)
    for i,band in enumerate(coherences):
        lessThanMin = nonzero(band < minCoherence)[0]
        if len(lessThanMin) >= 1:
            discardAfter = min(lessThanMin)
            snrMask[i][discardAfter:] = ones((len(snrMask[i][discardAfter:])))
    
    modulations = clip(modulations, 0, 0.99)      # clip to [0, 0.99] (max: ~1)
    snr = 10*log10(modulations/(1 - modulations)) # estimate SNR
    snr = clip(snr, -15, 15)                      # clip to [-15,15]
    snr = masked_array(snr, mask=snrMask)         # exclude values from sum
    snrCounts = (snr / snr).sum(axis=1)           # count SNRs
    snrCounts = snrCounts.data                    # remove masking
    octaveBandSNR = snr.sum(axis=1) / snrCounts   # calc average SNR
    alpha = 7 * (snrCounts / snrCounts.sum())     # calc alpha weight

    # octave band weighting factors, Steeneken and Houtgast (1985)
    w = [0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143]
    
    # calculate the STI measure
    snrp = alpha * w * octaveBandSNR
    snrp = snrp.sum()
    index = (snrp + 15) / 30.0
    
    print "Speech Transmission Index (STI):",index
    return index

def stiFromAudio(reference, degraded, hz, calcref=False, downsample=None,
                 name="unnamed"):
    """
    Calculate the speech transmission index (STI) from clean and dirty
    (ie: distorted) audio samples. The clean and dirty audio samples must have
    a common sample rate for successful use of this function.
    
    Input
    -----
    * reference : array-like
    
        Clean reference audio sample as an array of floating-point values
    
    * degraded : array-like
    
        Degraded audio sample as an array, or array of arrays for multiple
        samples, of floating-point values
    
    * hz : int
    
        Audio sample rate in hertz
    
    * calcref : boolean
    
        Calculate STI for reference signal alone
    
    * downsample : int or None
    
        Downsampling integer factor
    
    * name : string
    
        Name of sample set, for output tracking in larger runs
    
    Output
    ------
    * sti : array-like or float
    
        The calculated speech transmission index (STI) value(s)
    """
    
    # put single sample degraded array into another array so the loop works
    if type(degraded) is not type([]):
        degraded = [degraded]
    
    print "-"*80
    print "Speech Transmission Index (STI) from speech waveforms".center(80)
    print "-"*80
    print
    print "Sample set:             ",name
    print "Number of samples:      ",len(degraded)
    print "Date/time:              ",datetime.now().isoformat()
    print "Calculate reference STI:",
    if calcref:
        print "yes"
    else:
        print "no"
    print
    print " Reference Speech ".center(80,'*')
    
    refOctaveBands = octaveBandFilter(reference, hz)
    refRate = hz

    # downsampling, if desired
    if type(downsample) is type(1):
        refOctaveBands, refRate = downsampleBands(refOctaveBands, refRate,
                                                  downsample)
    
    # calculate STI for reference sample, if boolean set
    if calcref:
        # STI calc procedure
        spectras, sfreqs = octaveBandSpectra(refOctaveBands, refRate)
        coherences, cfreqs = octaveBandCoherence(refOctaveBands, refOctaveBands,
                                                 refRate)
        thirdOctaveMTF = thirdOctaveRootSum(spectras, sfreqs)
        thirdOctaveCoherences = thirdOctaveRMS(coherences, cfreqs)
        
        # add to interim array for MTFs and coherences
        try:
            thirdOctaveTemps.append([thirdOctaveMTF, thirdOctaveCoherences])
        except:
            thirdOctaveTemps = [[thirdOctaveMTF, thirdOctaveCoherences]]
    
    print
    
    # loop over degraded audio samples and calculate STIs
    for j,sample in enumerate(degraded):
        print " Degraded Speech: Sample {0} ".format(j + 1).center(80,'*')
        degrOctaveBands = octaveBandFilter(sample, hz)
        degrRate = hz
        
        # downsampling, if desired
        if type(downsample) is type(1):
            degrOctaveBands, degrRate = downsampleBands(degrOctaveBands, 
                                                        degrRate, downsample)
        
        # STI calc procedure
        spectras, sfreqs = octaveBandSpectra(degrOctaveBands, degrRate)
        coherences, cfreqs = octaveBandCoherence(refOctaveBands,
                                                 degrOctaveBands, refRate)
        thirdOctaveMTF = thirdOctaveRootSum(spectras, sfreqs)
        thirdOctaveCoherences = thirdOctaveRMS(coherences, cfreqs)

        # add to interim array for MTFs and coherences
        try:
            thirdOctaveTemps.append([thirdOctaveMTF, thirdOctaveCoherences])
        except:
            thirdOctaveTemps = [[thirdOctaveMTF, thirdOctaveCoherences]]

        print
    
    # calculate the STI values
    print " Speech Transmission Index ".center(80,'*')
    for i in range(0,len(thirdOctaveTemps)):
        sampleSTI = sti(thirdOctaveTemps[i][0], thirdOctaveTemps[i][1])
        
        # add to STI output array
        try:
            stiValues.append(sampleSTI)
        except:
            stiValues = [sampleSTI]
    
    # unpack single value
    if len(stiValues) == 1:
        stiValues = stiValues[0]
    
    print    
    return stiValues

def readwav(path):
    """
    Reads Microsoft WAV format audio files, scales integer sample values and
    to [0,1]. Returns a tuple consisting of scaled WAV samples and sample rate
    in hertz.
    
    Input
    -----
    * path : string
    
        Valid system path to file
    
    Output
    ------
    * audio : array-like
    
        Array of scaled sampled
    
    * rate : int
    
        Audio sample rate in hertz
    """
    wav = wavfile.read(path)
    
    rate = wav[0]
    audio = array(wav[1])
    
    scale = float(max(audio))
    audio = audio / scale
    
    return audio, rate

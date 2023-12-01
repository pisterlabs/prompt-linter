import math
import numpy as np
import ipdb
import scipy.signal as sig
import scipy.fft as fft

from utils import * 
from coherence_double_talk_detector import CoherenceDoubleTalkDetector


def BFDF(X,H,S):
    """
    A Block-Frequency-domain filter implementation.

    Parameters
    ----------
    X : ndarray (in STFT-domain)
        shape: (N_block x N_DFT)
        Input signal.
    H : ndarray (in STFT-domain)
        shape: (1 x N_DFT)
        Impulse response.
    S : int
        Shift sample size (block length / # of shifts)

    Returns (yields)
    ----------
    y : filter output
    """
   
    Y = np.zeros_like(X)

    for i in range(X.shape[0]): #a better way to write this?
        Yi = np.sum(H*X[i,:],axis=0)
        Y[i,:] = Yi
    y = fft.ifft(Y)
    y = y[:,S:].ravel()[:,None]

    return y.real

def FDAF_OS(x, d, M=2400, S=1200, alpha=0.85, delta=1e-8, mu=0.3, open_loop_threshold=0.8, closed_loop_threshold=0.95, freeze_index=None):
    """
    A Frequency-domain adaptive filter based on overlap-add method.

    Parameters
    ----------
    x : ndarray
        Far end signal a.k.a. the sound played from the speaker
    d : ndarray
        Near end signal a.k.a. microphone signal
    M : int
        Block size
    S : int
        Number of shifts.
    alpha: number
        The forgetting factor
    delta: number
        Regularization parameter
    mu: number
        step size
    open_loop_threshold: number
        Determines the threshold for the open-loop coherence value below which double-talk is detected
    closed_loop_threshold: number
        Determines the threshold for the closed-loop coherence value below which double-talk is detected. Should be higher
        than its open-loop equivalent.

    Returns (yields)
    ----------
    W : filter
    E : filter output (error)
    """

    x_ = get_shifted_blocks(x,M,S)
    X = fft.fft(x_,n=M)
    
    H = np.zeros((1,M))

    y = np.zeros_like(x)
    e = np.zeros_like(y)
    p = np.zeros((1,M))

    k = np.zeros((S,S))
    kp = np.diagflat(np.ones(S))
    k = np.concatenate((k,kp)).T
    kp = np.zeros((1,M))
    kp[:,:S] = 1
    g = np.diagflat(kp)

    nb_iterations = len(X)-3

    dtd = CoherenceDoubleTalkDetector(block_length=S, lambda_coherence=0.8)
    open_loop_rhos = np.zeros((nb_iterations,))
    closed_loop_rhos = np.zeros((nb_iterations,))
    adapt_flag = np.ones((nb_iterations,))

    for i in range(nb_iterations): #per block

        Xm = np.diagflat(X[i,:])

        Y = H@Xm
        yk = (k@(fft.ifft(Y).T)).real
        y[S*(i+1):S*(i+2)] = yk
        e[S*(i+1):S*(i+2)] = d[S*(i+1):S*(i+2)] - yk

        open_loop_rhos[i], closed_loop_rhos[i] = dtd.is_double_talk(x[S*(i+1):S*(i+2)], d[S*(i+1):S*(i+2)], y[S*(i+1):S*(i+2)])
        if freeze_index is None:
            do_not_adapt = open_loop_rhos[i] < open_loop_threshold and closed_loop_rhos[i] < closed_loop_threshold and i > 20
        else:
            do_not_adapt = (freeze_index[:,0]<=i*M).any() \
                            and (i*M<freeze_index[:,1]).any()
        
        if do_not_adapt:
            adapt_flag[i] = 0
            continue

        #adaptation
        e_ = k.T@e[S*(i+1):S*(i+2)]

        E = fft.fft(e_,axis=0,n=M) #check 1 more
        #stepsize computation
            #PSD estimate

        p = (1-alpha)*p + alpha*(np.abs(np.diag(Xm))**2)

        mu_a = mu * np.diagflat(np.reciprocal(p+delta))
        
        #filter update
        H_upd = 2 * fft.fft(\
            g @ fft.ifft(\
                mu_a @ (np.conj(Xm).T @ E),axis=0 ,n=M),axis=0,n=M)

        H = H + H_upd.T

    return e, y, H, p, open_loop_rhos, closed_loop_rhos, adapt_flag

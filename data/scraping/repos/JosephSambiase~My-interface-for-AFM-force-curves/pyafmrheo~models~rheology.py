import numpy as np
from scipy.signal import coherence
from scipy.fft import fft, fftfreq

def model_pyramid(G, wc, half_angle, freq, fi, bcoef, poisson_ratio):
    # Model for pyramidal indenter.
    # Reference: https://pubmed.ncbi.nlm.nih.gov/12609908/
    # Equation (5)
    div = 3 * wc * np.tan(np.radians(half_angle))
    coeff = (1.0 - poisson_ratio) / div
    if fi < 0:
        Piezo_corr = np.exp(1j * np.radians(fi))
    else:
        Piezo_corr = np.exp(-1j * np.radians(fi))
    G_corr = coeff * 2 * np.pi * 1j * freq * bcoef
    G_complex =  G * Piezo_corr * coeff - G_corr
    return G_complex.real, G_complex.imag


def model_paraboloid(G, wc, tip_radius, freq, fi, bcoef, poisson_ratio):
    # Model for paraboloid indenter.
    # Reference: https://pubmed.ncbi.nlm.nih.gov/16196611/
    # Equation (18)
    div = 4 * np.sqrt(tip_radius * wc)
    coeff = (1.0 - poisson_ratio) / div
    if fi < 0:
        Piezo_corr = np.exp(1j * np.radians(fi))
    else:
        Piezo_corr = np.exp(-1j * np.radians(fi))
    G_corr = coeff * 2 * np.pi * 1j * freq * bcoef
    G_complex =  G * Piezo_corr * coeff - G_corr
    return G_complex.real, G_complex.imag


single_freq_models = {
    "paraboloid": model_paraboloid,
    "pyramid": model_pyramid
}


def TransferFunction(input_signal, output_signal, fs, frequency=None, nfft=None, freq_tol=0.0001):
    # Define nfft
    if not nfft:
        nfft = len(output_signal)
    # Compute deltat from sampling frequency
    deltat = 1/fs
    # Compute frequency vector
    W = fftfreq(nfft, d=deltat)
    # Compute fft of both signals
    input_signal_hat = fft(input_signal, nfft)
    output_signal_hat = fft(output_signal, nfft)
    # Compute transfer function
    G = output_signal_hat / input_signal_hat
    # Compute coherence
    coherence_params = {"fs": fs, "nperseg":nfft, "noverlap":0, "nfft":nfft, "detrend":False}
    _, gamma2 = coherence(input_signal_hat, output_signal_hat, **coherence_params)
    if frequency:
        # Compute index where to find the frequency
        idx = frequency / (1 / (deltat * nfft))
        idx = int(np.round(idx))
        # Check if the idx is at the right frequency
        if not abs(frequency - W[idx]) <= freq_tol:
            print(f"The frequency found at index {W[idx]} does not match with the frequency applied {frequency}")
        
        return W[idx], G[idx], gamma2[idx], input_signal_hat[idx], output_signal_hat[idx]
    
    else:
        return W, G, gamma2, input_signal_hat, output_signal_hat


def ComputePiezoLag(zheight, deflection, fs, freq, nfft=None, freq_tol=0.0001):

    # Compute transfer function
    _, G, gamma2, zheight_hat, deflection_hat =\
         TransferFunction(zheight, deflection, fs, frequency=freq, nfft=nfft, freq_tol=freq_tol)
    
    # Get phase shift in degrees
    fi = np.angle(G, deg=True)

    # Get amplitude quotient
    amp_def = np.abs(deflection_hat)
    amp_height = np.abs(zheight_hat)
    amp_quotient = amp_def / amp_height

    return fi, amp_quotient, gamma2


def ComputeComplexModulusFFT(
    deflection, zheight, poc, k, fs, freq, ind_shape, tip_parameter,
    wc, poisson_ratio=0.5, fi=0, amp_quotient=1, bcoef=0, nfft=None, freq_tol=0.0001
):  

    # Correct zheight based on amplitude quotient
    # obtained from piezo characterization routine
    zheight = zheight * amp_quotient

    # Get indentation and force
    indentation = zheight - deflection - (poc[0] - poc[1])
    force = deflection * k - (poc[1] * k)

    # Compute transfer function
    _, G, gamma2, _, _ =\
         TransferFunction(indentation, force, fs, frequency=freq, nfft=nfft, freq_tol=freq_tol)
    
    # Compute G'and G''
    model_func = single_freq_models[ind_shape]
    G_storage, G_loss = model_func(G, wc, tip_parameter, freq, fi, bcoef, poisson_ratio)

    return G_storage, G_loss, gamma2

def ComputeComplexModulusSine(
    A_defl, A_ind, wc, dPhi, freq, ind_shape, tip_parameter,
    k, fi=0, amp_quotient=1, bcoef=0, poisson_ratio=0.5
):  

    # Correct indentation amplitude based on amplitude quotient
    # obtained from piezo characterization routine
    A_ind = A_ind * amp_quotient

    # Based on indenter geometry, compute G*
    if ind_shape == "cone":
        # Geometry dependent params
        n = 2
        coeff = 2/np.pi * np.tan(np.radians(tip_parameter))
        # Compute G* correcting for vdrag
        G = complex(k * A_defl / A_ind  * np.cos(dPhi), k * A_defl / A_ind * np.sin(dPhi) -  2 * np.pi * freq * bcoef)
        # Correct G* using factor based on phase shift
        G = G * np.exp(-1j * np.radians(fi))
        # Scale G* properly
        return G  * ((1 - poisson_ratio**2) / (n * coeff * wc**(n-1)))
    elif ind_shape == "paraboloid":
        # Geometry dependent params
        n = 3/2
        coeff = 4 / 3 * np.sqrt(tip_parameter)
        # Compute G* correcting for vdrag
        G = complex(k * A_defl / A_ind  * np.cos(dPhi), k * A_defl / A_ind * np.sin(dPhi) -  2 * np.pi * freq * bcoef )
        # Correct G* using factor based on phase shift
        G = G * np.exp(-1j * np.radians(fi))
        # Scale G* properly
        return G  * ((1 - poisson_ratio**2) / (n * coeff * wc**(n-1)))
    elif ind_shape == "pyramid":
        # Geometry dependent params
        n = 2
        coeff = 4 / (3 * np.sqrt(3)) * np.tan(np.radians(tip_parameter))
        # Compute G* correcting for vdrag
        G = complex(k * A_defl / A_ind  * np.cos(dPhi), k * A_defl / A_ind * np.sin(dPhi) -  2 * np.pi * freq * bcoef)
        # Correct G* using factor based on phase shift
        G = G * np.exp(-1j * np.radians(fi))
        # Scale G* properly
        return G  * ((1 - poisson_ratio**2) / (n * coeff * wc**(n-1)))


def ComputeBh(
    deflection, zheight, poc, k, fs, freq, fi=0, amp_quotient=1, nfft=None, freq_tol=0.0001
):  

    # Correct zheight based on amplitude quotient
    # obtained from piezo characterization routine
    zheight = zheight * amp_quotient

    # Get indentation and force
    indentation = zheight - deflection - (poc[0] - poc[1])
    force = deflection * k - (poc[1] * k)

    # Compute Hd(f)=F(f)/δ(f)e-φ
    _, G, gamma2, _, _ =\
         TransferFunction(indentation, force, fs, frequency=freq, nfft=nfft, freq_tol=freq_tol)
    
    Hd = G * np.exp(-1 * np.radians(fi))

    # Caluculate correction factor B(h)=Hd/(2πif)
    Bh = np.imag(Hd) / (2 * np.pi * freq)

    return Bh, Hd, gamma2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from  scipy import fftpack

def band_hilbert(x):
    N = len(x)
    Xf = fftpack.fft(x, N)


    w = fftpack.fftfreq(N, 1/fs)
    Xf[np.abs(w) <8] = 0
    Xf[np.abs(w) > 12] = 0
    #plt.plot(w, np.abs(Xf))
    #plt.show()

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return fftpack.ifft(Xf * h), Xf*h/np.sqrt(len(Xf))

np.random.seed(4)
duration = 10
fs = 1000.0
samples = int(fs*duration)
t = np.arange(samples) / fs
signal  = np.sin(2.0*np.pi*10.0*t - 0) + np.random.normal(size = samples)*0.0
ph = np.concatenate([[np.random.randint(-2, 2)]*int(fs) for k in range(duration)])
print(ph[samples//2]+10.)
signal2 = np.sin(2.0*np.pi*(10.+ph)*t+1)+ np.random.normal(size = samples)*0.0
# phase = np.linspace(-1, 1, 1000)
x_list = []
y_list = []
s_smth = []
n_window = 8
k_smth = 0.99

from pynfb.signal_processing.filters import Coherence
coherence = Coherence(500, fs, (8, 12))
for tt in range(n_window, samples//n_window):
    time = tt * n_window
    analytic_signal, xf = band_hilbert(signal[time-n_window: time])
    analytic_signal2, xf2 = band_hilbert(signal2[time-n_window: time])
    #coh = np.dot(xf, xf2.conj())/np.sqrt(np.abs(np.dot(xf, xf.conj())*np.dot(xf2, xf2.conj())))
    #x_list.append(np.imag(np.dot(analytic_signal2, analytic_signal.conj()))/np.sqrt(np.abs(np.dot(analytic_signal, analytic_signal.conj())*np.dot(analytic_signal2, analytic_signal2.conj()))))
    coh = coherence.apply(np.vstack([signal[time-n_window: time], signal2[time-n_window: time]]).T)
    y_list.append((coh * np.ones(n_window)))
    s_smth.append((coh  * np.ones(n_window)))

y_list = np.concatenate(y_list)
s_smth = np.concatenate(s_smth)

#print(np.array(x_list)/np.array(y_list))

f, ax = plt.subplots(3, sharex=True)
#ax[0].plot(t[n_window:-n_window], x_list)
ax[0].plot( y_list)
ax[0].plot(s_smth)
ax[0].legend(['Im', 'Abs'])
ax[0].set_ylabel('Coh')
ax[1].set_ylabel('$\Delta w$')
ax[1].plot( ph[n_window:-n_window])
ax[2].set_ylabel('Signals')
ax[2].plot(signal)
ax[2].plot(signal2)
ax[2].legend(['Signal1', 'Signal2'])
plt.show()
#analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)

plt.show()



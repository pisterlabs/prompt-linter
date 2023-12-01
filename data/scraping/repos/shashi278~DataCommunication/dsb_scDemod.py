import numpy as np
from numpy import *
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ampDemod import CoherentDetectionDemod

if __name__ == '__main__':
	labels= {
	'title': '====Coherent Demodulation(DSB-SC)====',
	'xlabel': 'Time(Sec)',
	'ylabel': 'Amplitude',
	'subtitle1': 'Modulated Signal: m(t)*sin(Wc*t)' ,
	'subtitle2': 'Vx= Modulated*Local oscillator',
	'subtitle3': 'Message Signal'
	}
	
	cdd= CoherentDetectionDemod(Ac= 3, Am= 2, fc= 5, fm= 1)
	#message signal
	m= lambda t: cdd.Am*sin(2*pi*cdd.fm*t)
	
	#modulated signal
	am= lambda t: (0+m(t))*sin(2*pi*cdd.fc*t)
	
	cdd.plot(*cdd.createSignals(am), labels)
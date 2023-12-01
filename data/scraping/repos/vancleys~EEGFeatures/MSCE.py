# Vancley Simão
# vancleys@gmail.com


# Magnitude Squared Coherence Estimate

from scipy.signal import coherence

def MSCE( x,y ):

	resp = coherence(x,y,fs=128)

	return resp

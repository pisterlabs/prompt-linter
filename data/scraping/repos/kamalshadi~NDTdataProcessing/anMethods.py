import numpy as num
import pylab as pl
import math
import cmath as cx
from collections import deque
from operator import itemgetter
from signalSystem import *
from myParser import *
from CDF import *
import scipy.signal
from matplotlib.mlab import cohere



def correlate(z,y,mode='p'):
	# p for pearson, c for pcc and s for spectral coherence
	l=len(y)
	if len(z)!=len(y):
		print 'Error'
		return
	if mode=='p':
		a=num.correlate(dNorm(z),dNorm(y),'valid')
		return a/len(y)
	elif mode=='c' :
		za=scipy.signal.hilbert(z)
		ya=scipy.signal.hilbert(y)
		phi1=[cx.phase(x) for x in za]
		phi2=[cx.phase(x) for x in ya]
		a=0
		for i in range(l):
			a=a+(abs(cx.exp(1j*phi1[i])+cx.exp(1j*phi2[i]))-\
			abs(cx.exp(1j*phi1[i])-cx.exp(1j*phi2[i])))/2
		return a/len(y)
	elif mode=='s':
		sc,f=cohere(z,y)
		return sc,f
	return
		
	

def spectrum(z,fs=1,p=False):
	# this function compute and plot spectrum and power spectrum of x
	a=num.fft.fft(z)
	f=[i for i in range(len(a))]
	ts=1.0/float(fs)
	freq = num.fft.fftfreq(len(a))
	#~ freq=num.fft.fftshift(freq1)
	#~ S=num.fft.fftshift(a)
	
	if p:
		S1=[abs(x) for x in a]
		S=[20*math.log10(x) for x in S1]
		#~ S=num.fft.fftshift(S2)
		pl.plot(freq,S,'r')
		pl.xlabel('Normalized frequency',fontsize=20)
		pl.ylabel('PSD',fontsize=20)
		#~ pl.xlim(-.1,.1)
		pl.show()
	return [freq,S]
	
def correlation(z,y,ind=0):
	#ind is how much y lead z
	l1=len(z)
	l2=len(y)
	lmn=min(l1,l2)
	lmx=max(l1,l2)
	slide_f=l1-ind-1
	slide_b=l2+ind-1
	l=slide_f+slide_b+1
	if ind>0:
		y=[0.0]*ind+y
	if ind<0:
		z=[0.0]*(-1*ind)+z
	if len(z)>len(y):
		y=y+[0.0]*(len(z)-len(y))
	if len(y)>len(z):
		z=z+[0.0]*(len(y)-len(z))
	lag=[x-slide_b for x in range(l)]
	cc=[]
	for k in lag:
		w1=list(z)
		w2=list(y)		
		if (k<0):
			w2=w2[(-1*k):-1]+[w2[-1]]
			w2=w2+[0.0]*(-1*k)
			cc.append(correlate(w1,w2))
		if k==0:
			cc.append(correlate(w1,w2))
		if k>0 :
			w2=[0.0]*k+w2
			del w2[-k:-1]
			del w2[-1]
			cc.append(correlate(w1,w2))
	return [lag,cc]
	
def comb(y,delay):
	if delay == 1 :
		return sigPower(y)
	l=len(y)
	x=range(delay)
	i=0
	chunk=int(num.ceil(float(l)/float(delay)))
	ind=[0]*chunk
	while True:
		ind[i] = [o+i*delay for o in range(delay)]
		i=i+1
		if i>=len(ind):
			break
	p = l%delay
	if p != 0 :
		y=y+[0]*(delay-p)
	ct = [0]*chunk
	for q in range(chunk):
		ct[q]=list(itemgetter(*ind[q])(y))
	out = [0]*len(ct[0])
	for w in ct:
		for h in range(len(w)) :
			out[h]=out[h]+float(w[h])/float(chunk)
	return sigPower(out)
			

def SCF(y,a,fs=1):
	# a is cylic frequecy
	N=len(y)
	u=[y[x]*cx.exp(math.pi*(-1j)*a*float(x)/float(fs)) for x in range(N)]
	v=[(y[x]*cx.exp(math.pi*(1j)*a*float(x)/float(fs))).conjugate() for x in range(N)]
	lag,cc=correlation(u+[0]*5000,v+[0]*5000)
	pl.plot(lag,[abs(w) for w in cc])
	freq,F = spectrum(cc,fs)
	return [freq,F]
	
def SCF2(y,a,fs=1):
	# a is cylic frequecy
	N=len(y)
	f,z = spectrum(y,fs)
	rot = a
	zq  = deque(z)
	zq.rotate(rot)
	z_shifted = list(zq)
	zsc_p = [x.conjugate() for x in z_shifted]
	zq  = deque(z)
	zq.rotate(-rot)
	zs_n = list(zq)
	lag,cc=correlation(zs_n+[0]*2000,zsc_p+[0]*2000)
	b=[num.abs(x) for x in cc]
	pl.plot(lag,b)
	pl.show()
	#~ pl.plot(lag,[abs(w) for w in cc])
	#~ freq,F = spectrum(cc,fs)
	#~ return abs(a)
	
def combf(z,maxDelay,s='b'):
	#~ noise = list(num.random.normal(0,1,len(z)))
	dV=range(1,maxDelay)
	p=[0]*len(dV)
	for delay in dV:
		p[delay-1]=comb(z,delay)
	pl.plot(dV,p,s)
	
def slidingCorr(ts,vs,S=0,mode='p'):
	# ts is 2*N list of two time series
	# vs is 2*N list of two value series
	# mode is 'p' for pearson 'c' for phase coherency and 's' for spectral coherency
	aa=[]
	if S==0 :
		p=pairing(ts[0],vs[0],ts[1],vs[1],0.5)
		u,v=[x[0] for x in p],[x[1] for x in p]
		r=correlate(dNorm(u),dNorm(v),mode)
		return [r,len(p)]
	else :
		pp=[]
		a=0
		b=S
		while True: 
			t_s1,v_s1=slice(ts[0],vs[0],a,b)
			t_s2,v_s2=slice(ts[1],vs[1],a,b)
			if t_s1==[] or t_s2==[] :
				break
			a=b
			b=b+S
			p=pairing(t_s1,v_s1,t_s2,v_s2,0.5)
			aa.append(len(p))
			print len(p)
			u,v=[x[0] for x in p],[x[1] for x in p]
			pp.append(float(correlate(dNorm(u),dNorm(v),mode)))
	return [pp,num.mean(aa)]

def slice(t,v,a,b):
	# t is a time vector
	# v is value vector
	# a,b are start and end of slice
	vs=[v[x] for x in range(len(t)) if (t[x]>=a and t[x]<b)]
	ts=[t[x] for x in range(len(t)) if (t[x]>=a and t[x]<b)]
	return [ts,vs]

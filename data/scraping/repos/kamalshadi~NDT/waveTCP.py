#!/usr/bin/env python

import csv
import os
import sys
import subprocess
import statvfs
import pylab as pl
import pywt as wx
import numpy as num
from matplotlib.mlab import PCA
import math
from scipy.linalg import eigh
from scipy.stats import norm,mstats
from scipy.signal import hilbert
import cmath as cx
from matplotlib.mlab import cohere
from myStat import *


def dNorm(x):
	s=num.std(x)
	u=num.mean(x)
	return [(xx-u)/s for xx in x]

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
		za=hilbert(z)
		ya=hilbert(y)
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

def featureNorm(fv,nonLin=False):
	mn=num.mean(fv,0)
	std=num.std(fv,0)
	fvn=(num.array(fv)-mn)/std
	if nonLin:
		fvn=1.0/(1+num.exp(-1*fvn))
	return fvn
	

def db(x):
	print type(x)
	if type(x)!=list:
		return 20*math.log10(abs(x))
	else:
		return [20*math.log10(abs(xx)) for xx in x]
	

def usage():
    return """
Summary:
./geoUoS -p False/True -f <filename> 
locate the UoS in city resolution
"""
		

def parse_args():
    from optparse import OptionParser
    parser = OptionParser(usage=usage())
    parser.add_option("-d", "--dirc", dest="dirc", default=None, 
                      help="Required: sub_directory in Dump")
    parser.add_option("-u", "--uos", dest="uos", default="1", 
                      help="Required: filename for geo data")
        
    
                       
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    (options, args) = parser.parse_args()
    if options.dirc is None:
        print "Error: Please provide --dir to read data \n \
        (do not include D- prefix)"
        sys.exit(1)

    return (options, args)
    
    
def order(v,w):
	a=zip(v,w)
	a.sort()
	l=zip(*a)
	v=list(l[0])
	w=list(l[1])
	return [v,w]

if __name__ == '__main__':
	(options, args) = parse_args()
	dirc=options.dirc
	uos=options.uos
	ad="Dump/D-"+dirc+"/uos_"+uos
	i=0
	dic={}
	with open(ad,'r') as f:
		val=csv.reader(f,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
		fv1=[]
		for i,line in enumerate(val):
			if i==0:
				l=len(line)
				i=1
				continue
			else:
				cIP=line[0]
				server=line[-1]
				log=int(line[1])
				t=[float(xx)/1e6 for xx in line[2].strip('"').split(',')]
				rtt=[float(xx) for xx in line[3].strip('"').split(',')]
				cwnd=[float(xx) for xx in line[4].strip('"').split(',')]
				cong=max([float(xx) for xx in line[5].strip('"').split(',')])
				ro=correlate(rtt,cwnd)
				acked=int(line[8])
				down=float(acked)/(1e6*max(t))
				t,w=order(t,zip(rtt,cwnd))
				d=5 # level of wavelet decomposition
				rtt,cwnd=[list(xx) for xx in zip(*w)]
				wc = wx.wavedec(cwnd, 'haar', level=d)
				wr = wx.wavedec(rtt, 'haar', level=d)
				#~ fig=pl.figure()
				#~ pl.subplot(2,1,1)
				#~ pl.plot(range(len(cwnd)),cwnd)
				#~ pl.subplot(2,1,2)
				#~ pl.plot(range(len(rtt)),rtt)
				#~ pl.title(str(ro))
				#~ pl.suptitle(str(down))
				f1=[]
				#~ pl.show()
				for i in range(1,d+1):
					f1=f1+[num.std(wc[i]),num.mean(wc[i]),max(wc[i])-min(wc[i])]
					f1=f1+[num.std(wr[i]),num.mean(wr[i]),max(wr[i])-min(wr[i])]
				fv1.append(f1)
	fv=num.array(fv1)
	fvn=featureNorm(fv,nonLin=False)
	y=PCA(fvn)
	s=y.fracs
	#~ print len(s)
	#~ print len(s[0])
	pl.plot(range(len(s)),list(s))
	pl.xlabel('Dimension')
	pl.ylabel('Significance')
	pl.show()
				

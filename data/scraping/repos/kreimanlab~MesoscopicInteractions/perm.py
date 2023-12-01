# Jiarui "Jerry" Wang :: jwang04@g.harvard.edu
# Kreiman Lab :: klab.tch.harvard.edu
# Last edited: March 30, 2018 [14:12:53]

DIR_OUTPUT = './results'
DIR_H5 = '../data/h5_notch20'

# This parameter is the window size for coherence calculations
#   - Please check that this number is the same as in graph.py
WINDOW = 10 # seconds
# This parameter is the minimum time delay for coherence permutations
DELAY = 1 # seconds, default: 250 seconds, debug: 1 second

from envelope import envelope
from envelope import envelope_gpu
from coherence import coherence2
from coherence import coherence2_gpu
import pyopencl as cl
from pyopencl import array
import math
import numpy
import random
import scipy
from scipy import stats
from scipy.signal import butter, lfilter, hilbert
import h5py
import time
import matplotlib.pyplot as plt
import socket
import sys
from pathlib import Path
import reikna
import os.path

OPENCL_KERNEL = 'corr2perm.cl'
CHECK_OUTPUT = False
FILT_ORDER = 3
#N_PERM = 100000

def butter_bandpass(lowcut, highcut, fs, order=FILT_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=FILT_ORDER):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def printDeviceInfo(device):
    print('[!] Using device: ' + device.name)
    print('\tDriver version: ' + str(device.driver_version))
    print('\tVendor: ' + device.vendor)
    print('\tVersion: ' + device.version)
    #print('\tExtensions: ' + str(device.extensions.strip().split(' ')))
    print('\tGlobal memory (GB): {0:0.3f}'.format(device.global_mem_size/1e9))
    print('\tLocal memory (KB): {0:0.3f}'.format(device.local_mem_size/1e3))
    print('\tAddress bits: ' + str(device.address_bits))
    print('\tMax work item dims: ' + str(device.max_work_item_dimensions))
    print('\tMax work group size: ' + str(device.max_work_group_size))
    print('\tMax compute units: ' + str(device.max_compute_units))

if __name__ == "__main__":

    # Define constants
    DEL_S = 0.5
    DEL_E = 3
    THE_S = 3
    THE_E = 8
    ALP_S = 8
    ALP_E = 12
    BET_S = 12
    BET_E = 30
    GAM_S = 30 # Hz
    GAM_E = 100 # Hz
    HG_S = 70
    HG_E = 200
    #N_PERM = 10000

    if (len(sys.argv) == 5):
        if (sys.argv[1] == 'mSu'):
            PLI = 50
        else:
            PLI = 60
        print('[*] Using patient: '+sys.argv[1]+', power line frequency: '+str(PLI)+' Hz')
    else:
        print('[!] Usage: python3 perm.py patientNumber metric nperm deviceNumber')
        exit()

    #METRIC = 'sg'
    METRIC = sys.argv[2]
    N_PERM = int(sys.argv[3])

    plat_n = 0

    OUT_DIR = DIR_OUTPUT
    h5fname = DIR_H5 + '/' + sys.argv[1] + '.h5'
    artfname = DIR_H5 + '/art_nosz/' + sys.argv[1] + '_art.h5'

    # Check for file
    if (not Path(h5fname).is_file()):
        print("[!] Error: perm.py "+' '.join(sys.argv[1:])+": h5eeg file not found.")
        print("\tpath: "+h5fname)
        exit()
    
    # gpu init
    USE_CPU = False
    if (sys.argv[4] == 'cpu'):
        USE_CPU = True
    clMetrics = ['s','p','sd','st','sa','sb','sg','shg']
    if ((METRIC in clMetrics) and (not USE_CPU)):   
        platform = cl.get_platforms()[plat_n]
        device = platform.get_devices()[int(sys.argv[4])]
        printDeviceInfo(device)
        context = cl.Context([device])
    # reikna init
    envMetrics = ['sd','st','sa','sb','sg','shg','sc','pc']
    if ((METRIC in envMetrics) and (not USE_CPU)):   
        dev_n = int(sys.argv[4])
        api = reikna.cluda.any_api()
        plat = api.get_platforms()[plat_n]
        p0_dev = plat.get_devices()
        dev = p0_dev[dev_n]
        thr = api.Thread(dev)

    # Read h5eeg
    sid = h5fname.split('/')[-1].split('.h5')[0]
    h5f = h5py.File(h5fname,'r')
    #arts = h5f['/h5eeg/artifacts']
    #width = int(arts.attrs['width'][0])
    n_chan = int(h5f['/h5eeg/eeg'].attrs['n_chan'][0])
    n_samples = int(h5f['/h5eeg/eeg'].attrs['n_samples'][0])
    fs = h5f['/h5eeg/eeg'].attrs['rate'][0]

    # Read artifact
    artf = h5py.File(artfname,'r')
    arts = artf['/artifacts_v1']
    width = int(arts.attrs['width'][0])

    w = WINDOW # seconds (6/250 makes r_row = 5)
    #alpha = 120 # seconds
    alpha = DELAY # seconds
    #r_rows = int(w*fs)
    r_rows = int(w * round(fs))
    print('[*] r_rows: ' + str(r_rows))
    print('[*] Using file: {0}\n[*] n_chan: {1}\n[*] n_samples: {2}'.format(h5fname,n_chan,n_samples))
    bip = h5f['/h5eeg/eeg'].attrs['bip']
    e1 = numpy.array(bip[0,:], numpy.float32)
    e2 = numpy.array(bip[1,:], numpy.float32)
    #h5f.close()

    # Compile
    if ((METRIC in clMetrics) and (not USE_CPU)):
        # Append array size to source code (since C and thus OpenCL doesn't
        #   support variable length arrays)
        src = ''.join(open(OPENCL_KERNEL,'r').readlines())
        src = '#define R_ROWS ' + str(r_rows) + '\n' + src
        program = cl.Program(context,src).build()
        queue = cl.CommandQueue(context)
        mem_flags = cl.mem_flags
    # reikna compile
    #envMetrics = ['sd','st','sa','sb','sg','shg']
    #if ((METRIC in envMetrics) and (not USE_CPU)):   
    #    ffti = reikna.fft.FFT(numpy.ones((r_rows,len(e1)),numpy.complex64), axes=(0,))
    #    cffti = ffti.compile(thr)

    # DEBUG
    #n_samples = r_rows*10

    # Get channel combinations
    n_comb = int(0.5*len(e1)*(len(e1)-1))
    chan1 = numpy.zeros((1,n_comb))
    chan2 = numpy.zeros((1,n_comb))
    count = 0
    for i in range(len(e1)-1):
        for j in range((i+1),len(e1)):
            chan1[0,count] = i
            chan2[0,count] = j
            count = count + 1
    chan1 = numpy.array(chan1, numpy.float32)
    chan2 = numpy.array(chan2, numpy.float32)

    # Make final matrices
    if ((METRIC == 'sc') or (METRIC == 'pc')):
        N_BANDS = 6
        Rf = numpy.zeros((N_PERM*N_BANDS,n_comb), numpy.float32)

        if (not USE_CPU):
            Phf = numpy.zeros((N_PERM*N_BANDS,n_comb), numpy.float32)
    else:
        Rf = numpy.zeros((N_PERM,n_comb), numpy.float32)

    parCount = 0
    #for parIdx in range(0,n_samples,r_rows):
    print('[*] Starting permutations..')
    print('\tNote: if nothing happens there may be too many artifacts to find continuous chunks')
    secure_random = random.SystemRandom()
    Starts = numpy.zeros((2,N_PERM))

    # without replacement bank
    bank_rIdx = []
    bank_rIdx2 = []
    for parIdx in range(N_PERM):
        t_par0 = time.time()

        # Choose permutation indices
        choose = True
        while choose:
            chIdx = secure_random.randrange(0,len(arts))
            ch = arts[chIdx:int(chIdx+(fs/width)*(w+1))] # +1 to trim off end case
            chTest = ch[:,0]
            endI = int(ch[0][1])+r_rows
            if ((numpy.count_nonzero(chTest) == 0) and (endI < n_samples) and (int(ch[0][1]) not in bank_rIdx)):
                r_idx = int(ch[0][1])

                # without replacement
                bank_rIdx.append(r_idx)

                choose = False

        choose  = True
        while choose:
            chIdx2 = secure_random.randrange(0,len(arts))
            ch2 = arts[chIdx2:int(chIdx2+(fs/width)*(w+1))] # +1 to trim off end case
            chTest2 = ch2[:,0]
            endI2 = int(ch2[0][1])+r_rows
            midpoint1 = endI + (endI - r_idx)/2
            midpoint2 = endI2 + (endI2 - int(ch2[0][1]) )/2
            alphaSAT = abs(midpoint1 - midpoint2) > (alpha * fs)
            if (((numpy.count_nonzero(chTest2) == 0) and (endI2 < n_samples)) and (alphaSAT) and (int(ch2[0][1]) not in bank_rIdx2)):
                r_idx2 = int(ch2[0][1])

                # without replacement
                bank_rIdx2.append(r_idx2)

                choose = False
        print('[{0}] Random: {1}-{2}, {3}-{4}'.format(n_samples,r_idx,endI,r_idx2,endI2))

        # Sample number offset
        #r_idx = parIdx
        #if (r_idx+r_rows >= n_samples):
        #    r_idx = n_samples-r_rows

        Starts[0,parIdx] = r_idx
        Starts[1,parIdx] = r_idx2
        t_read0 = time.time()
        X = h5f['/h5eeg/eeg'][r_idx:(r_idx+r_rows),0:(0+n_chan)]
        Y = h5f['/h5eeg/eeg'][r_idx2:(r_idx2+r_rows),0:(0+n_chan)]
        t_read = time.time() - t_read0
        Mbytes = 2*r_rows*n_chan*4/1000000

        # EEG referencing and pre-processing
        Xbip = numpy.zeros((r_rows,len(e1)))
        Ybip = numpy.zeros((r_rows,len(e1)))
        #XbipGam = numpy.zeros((r_rows,len(e1)))
        print('[{0},{1}/{2} {3} of {4}] Read {5:0.3f} MB/s. Starting..'.format(r_idx,r_idx2,n_samples,parCount+1,N_PERM,Mbytes/t_read))
        for i in range(len(e1)):

            # Calculate bipolar and rank
            x = (X[:,int(e1[i]-1)] - X[:,int(e2[i]-1)])
            y = (Y[:,int(e1[i]-1)] - Y[:,int(e2[i]-1)])

            doPartial = False
            doCoherence = False
            doEnvelope = False
            if (METRIC == 's'):
                Xbip[:,i] = scipy.stats.rankdata(x)
                Ybip[:,i] = scipy.stats.rankdata(y)
            elif (METRIC == 'p'):
                Xbip[:,i] = x
                Ybip[:,i] = y
            elif (METRIC == 'sc'):
                Xbip[:,i] = scipy.stats.rankdata(x)
                Ybip[:,i] = scipy.stats.rankdata(y)
                doCoherence = True
            elif (METRIC == 'pc'):
                Xbip[:,i] = x
                Ybip[:,i] = y
                doCoherence = True
            elif (METRIC == 'sP'):
                Xbip[:,i] = scipy.stats.rankdata(x)
                Ybip[:,i] = scipy.stats.rankdata(y)
                doPartial = True
            elif (METRIC == 'pP'):
                Xbip[:,i] = x
                Ybip[:,i] = y
                doPartial = True
            elif (METRIC == 'sd'):
                # Delta envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, DEL_S, DEL_E, fs, order=FILT_ORDER)
                Ybip[:,i] = butter_bandpass_filter(y, DEL_S, DEL_E, fs, order=FILT_ORDER)
            elif (METRIC == 'st'):
                # Theta envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, THE_S, THE_E, fs, order=FILT_ORDER)
                Ybip[:,i] = butter_bandpass_filter(y, THE_S, THE_E, fs, order=FILT_ORDER)
            elif (METRIC == 'sa'):
                # Alpha envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, ALP_S, ALP_E, fs, order=FILT_ORDER)
                Ybip[:,i] = butter_bandpass_filter(y, ALP_S, ALP_E, fs, order=FILT_ORDER)
            elif (METRIC == 'sb'):
                # Beta envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, BET_S, BET_E, fs, order=FILT_ORDER)
                Ybip[:,i] = butter_bandpass_filter(y, BET_S, BET_E, fs, order=FILT_ORDER)
            elif (METRIC == 'sg'):
                # Gamma envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, GAM_S, GAM_E, fs, order=FILT_ORDER)
                Ybip[:,i] = butter_bandpass_filter(y, GAM_S, GAM_E, fs, order=FILT_ORDER)
            elif (METRIC == 'shg'):
                doEnvelope = True
                # High Gamma envelope
                Xbip[:,i] = butter_bandpass_filter(x, HG_S, HG_E, fs, order=FILT_ORDER)
                Ybip[:,i] = butter_bandpass_filter(y, HG_S, HG_E, fs, order=FILT_ORDER)
            else:
                if (i == 0):
                    print('[*] Metric: '+METRIC+' not found. Using Pearson corr.')
                Xbip[:,i] = x
                Ybip[:,i] = y

        if (doPartial):
            r_cols = int(len(e1))
            R = numpy.zeros(n_comb, numpy.float32)
            XYbip = numpy.concatenate((Xbip,Ybip), axis=1)
            Ci = numpy.linalg.inv(numpy.cov(numpy.transpose(XYbip)))
            for ic in range(n_comb):
                i = int(chan1[0][ic])
                j = int(chan2[0][ic])
                R[ic] = - Ci[i][j+r_cols] / math.sqrt( Ci[i][i] * Ci[j+r_cols][j+r_cols] )
            #print(numpy.mean(R))
            #print(numpy.var(R))
            #print(R)
            #exit()
        elif (doCoherence):
            r_cols = int(len(e1))
            if (USE_CPU):
                r_coh = coherence2(Xbip,Ybip,int(fs))
            else:
                # resolution 0.5 Hz
                #r_coh,ph_coh = coherence2_gpu(Xbip,Ybip,int(round(2*fs)),int(round(fs)),thr,PLI)
                # resolution 1 Hz
                r_coh,ph_coh = coherence2_gpu(Xbip,Ybip,int(round(fs)),int(round(fs)),thr,PLI)
                # resolution 2 Hz
                #r_coh,ph_coh = coherence2_gpu(Xbip,Ybip,int(round(fs/2)),int(round(fs)),thr,PLI)
                # resolution 5 Hz
                #r_coh,ph_coh = coherence2_gpu(Xbip,Ybip,int(round(fs/5)),int(round(fs)),thr,PLI)
        else:
            r_cols = int(len(e1))

            # Pre-correlation processing
            if (doEnvelope):
                if (USE_CPU):
                    Xbip = envelope(Xbip)
                    Ybip = envelope(Ybip)
                    XYbip = numpy.concatenate((Xbip,Ybip), axis=1)
                else:
                    XYbip = numpy.concatenate((Xbip,Ybip), axis=1)
                    XYbip = envelope_gpu(XYbip, thr)
                    #XYbip = envelope_gpu(XYbip, plat_n, dev_n)
                    #Xbip = XYbip[0:r_cols]
                    #Ybip = XYbip[r_cols:]
                # rank data
                for i in range(2*r_cols):
                    if (i < r_cols):
                        Xbip[:,i] = scipy.stats.rankdata(XYbip[:,i])
                    else:
                        Ybip[:,i-r_cols] = scipy.stats.rankdata(XYbip[:,i])


            if (USE_CPU):
                R = numpy.zeros(n_comb, numpy.float32)
                XYbip = numpy.concatenate((Xbip,Ybip), axis=1)
                C = numpy.cov(XYbip,rowvar=False)
                Cd = numpy.sqrt(numpy.diagonal(C))
                for ic in range(n_comb):
                    i = int(chan1[0,ic])
                    j = int(chan2[0,ic])
                    R[ic] = C[i,j+r_cols] / (Cd[i] * Cd[j])
            else:
                X = numpy.array(Xbip, numpy.float32)
                Y = numpy.array(Ybip, numpy.float32)
                # --------------------------------------------------------------------------
                # INPUTS:
                #   X - r_rows by r_cols
                #   Y - r_rows by r_cols
                #   chan1 - 1 by n_comb
                #   chan2 - 1 by n_comb
                # OUTPUTS:
                #   R - 1 by n_comb
                # --------------------------------------------------------------------------
                
                ## Step #8. Allocate device memory and move input data from the host to the device memory.
                matrix_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=X)
                matrix2_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=Y)
                vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=chan1)
                vector2_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=chan2)
                R = numpy.zeros(n_comb, numpy.float32)
                destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, R.nbytes)

                #src2 = ''.join(open('rank.cl','r').readlines())
                #src2 = '#define R_ROWS ' + str(r_rows) + '\n' + src
                # Compile
                #program2 = cl.Program(context,src2).build()
              
                # working:
                #R2 = numpy.zeros(r_cols, numpy.float32)
                #print('r_cols: '+str(r_cols))
                #program.rank(queue, R2.shape, None, matrix_buf, matrix2_buf, numpy.int32(r_rows), numpy.int32(r_cols))

                program.corr2perm(queue, R.shape, None, matrix_buf, matrix2_buf, vector_buf, vector2_buf, destination_buf, numpy.int32(r_rows), numpy.int32(r_cols))
                 
                cl.enqueue_copy(queue, R, destination_buf)
        
        # Aggregate
        if (doCoherence):
            for i in range(N_BANDS):
                Rf[(parCount*N_BANDS) + i,:] = r_coh[i,:]

                if (not USE_CPU):
                    Phf[(parCount*N_BANDS) + i,:] = ph_coh[i,:]
        else:
            Rf[parCount,:] = numpy.reshape(R,(1,n_comb))

        t_par = time.time()
        print('[{0:0.4} s] ETA: {1:0.3f} hrs, {2}:{3},{4}'.format(t_par-t_par0,(N_PERM-parCount)*(t_par-t_par0)/3600, sid, n_chan, METRIC))

        if (CHECK_OUTPUT):
            if (doCoherence):
                print(Rf[0:6,0:5])
                print(Rf[6:12,0:5])
                print(Rf[12:18,0:5])
                print('...')
            else:
                print(Rf)
        parCount = parCount + 1

    # Save
    #artf.close()
    if (doCoherence):
        bandn = ['Delta','Theta','Alpha','Beta','Gamma','Broadband']
        for i in range(N_BANDS):
            ofname = OUT_DIR + '/' + sid+'_perm-'+METRIC+bandn[i]+'-'+str(N_PERM)+'.h5'
            Rcoh = numpy.zeros((N_PERM,n_comb), numpy.float32)
            for j in range(N_PERM):
                Rcoh[j,:] = Rf[j*N_BANDS + i,:]

            print(Rcoh)
            print('[*] Saving to: '+ofname)
            h5fout = h5py.File(ofname,'w')
            h5fout.create_dataset('R', data=Rcoh)

            if (not USE_CPU):
                PHcoh = numpy.zeros((N_PERM,n_comb))
                for j in range(N_PERM):
                    PHcoh[j,:] = Phf[j*N_BANDS + i,:]
            h5fout.create_dataset('PH', data=PHcoh)

            h5fout.create_dataset('starts', data=Starts)
            h5fout.create_dataset('chan1', data=chan1)
            h5fout.create_dataset('chan2', data=chan2)
            h5fout.create_dataset('r_rows', data=r_rows)
            h5fout.create_dataset('r_cols', data=r_cols)
            h5fout.create_dataset('w', data=w)
            h5fout.create_dataset('alpha', data=alpha)
            h5fout.flush()
            h5fout.close()
    else:
        ofname = OUT_DIR + '/' + sid+'_perm-'+METRIC+'-'+str(N_PERM)+'.h5'
        print(Rf)
        print('[*] Saving to: '+ofname)
        h5fout = h5py.File(ofname,'w')
        h5fout.create_dataset('R', data=Rf)
        h5fout.create_dataset('starts', data=Starts)
        h5fout.create_dataset('chan1', data=chan1)
        h5fout.create_dataset('chan2', data=chan2)
        h5fout.create_dataset('r_rows', data=r_rows)
        h5fout.create_dataset('r_cols', data=r_cols)
        h5fout.create_dataset('w', data=w)
        h5fout.create_dataset('alpha', data=alpha)
        h5fout.flush()
        h5fout.close()

    print('[!] Done.')

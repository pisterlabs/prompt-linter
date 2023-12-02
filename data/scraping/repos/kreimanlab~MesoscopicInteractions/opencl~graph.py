# Jiarui "Jerry" Wang :: jwang04@g.harvard.edu
# Kreiman Lab :: klab.tch.harvard.edu
# Last edited: March 30, 2018 [14:13:06]

DIR_OUTPUT = './results'
DIR_H5 = '../data/h5_notch20'

# This parameter is the window size for coherence calculations
#   - Please check that this number is the same as in perm.py
WINDOW = 10 # seconds

from envelope import envelope
from envelope import envelope_gpu
from coherence import coherence
from coherence import coherence_gpu
import pyopencl as cl
from pyopencl import array
import math
import numpy
import scipy
from scipy import stats
from scipy.signal import butter, lfilter, hilbert
import h5py
import time
import matplotlib.pyplot as plt
import sys
import socket
from pathlib import Path
import reikna
import os.path

DEBUG = False
FILT_ORDER = 3

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

    if (len(sys.argv) == 4):
        if (sys.argv[1] == 'mSu'):
            PLI = 50
        else:
            PLI = 60
        print('[*] Using patient: '+sys.argv[1]+', power line frequency: '+str(PLI)+' Hz')

    else:
        print('[!] Usage: python3 graph.py patientNumber metric deviceNumber')
        exit()
    METRIC = sys.argv[2]

    OUT_DIR = DIR_OUTPUT
    h5fname = DIR_H5 + '/' + sys.argv[1] + '.h5'
    plat_n = 0
    
    # Check for file
    if (not Path(h5fname).is_file()):
        print("[!] Error: graph.py "+' '.join(sys.argv[1:])+": h5eeg file not found.")
        exit()

    USE_CPU = False
    if (sys.argv[3] == 'cpu'):
        USE_CPU = True

    clMetrics = ['s','p','sd','st','sa','sb','sg','shg']
    if ((METRIC in clMetrics) and (not USE_CPU)):
        platform = cl.get_platforms()[plat_n]
        device = platform.get_devices()[int(sys.argv[3])]
        printDeviceInfo(device)
        context = cl.Context([device])
        program = cl.Program(context, """
            __kernel void corr2graph(__global const float *matrix,
            __global const float *vector, __global const float *vector2, 
            __global float *result, const int r_rows, const int r_cols)
            {
                int gid = get_global_id(0);
                int chan1 = vector[gid];
                int chan2 = vector2[gid];
                float x_i;
                float y_i;
                float r = 0;
                float sum_x = 0;
                float sum_y = 0;
                float sum_xx = 0;
                float sum_yy = 0;
                float sum_xy = 0;
                for (int i = 0; i < r_rows; i++) {
                    x_i = matrix[i * r_cols + chan1];
                    y_i = matrix[i * r_cols + chan2];
                    sum_x += x_i;
                    sum_y += y_i;
                    sum_xx += x_i * x_i;
                    sum_yy += y_i * y_i;
                    sum_xy += x_i * y_i;
                }
                r = ( (r_rows*sum_xy) - (sum_x*sum_y) )/sqrt( (r_rows*sum_xx - sum_x*sum_x)*(r_rows*sum_yy - sum_y*sum_y) );
                result[gid] = r;
            }
            """).build()
        queue = cl.CommandQueue(context)
        mem_flags = cl.mem_flags

    # reikna init
    envMetrics = ['sd','st','sa','sb','sg','shg','sc','pc']
    if ((METRIC in envMetrics) and (not USE_CPU)):
        dev_n = int(sys.argv[3])
        api = reikna.cluda.any_api()
        plat = api.get_platforms()[plat_n]
        p0_dev = plat.get_devices()
        dev = p0_dev[dev_n]
        thr = api.Thread(dev)

    # Read h5eeg
    #h5fname = '/home/klab/data/h5eeg/artifact_removed/test/opencl/sub4.h5'
    sid = h5fname.split('/')[-1].split('.h5')[0]
    h5f = h5py.File(h5fname,'r')
    arts = h5f['/h5eeg/artifacts']
    n_chan = int(h5f['/h5eeg/eeg'].attrs['n_chan'][0])
    n_samples = int(h5f['/h5eeg/eeg'].attrs['n_samples'][0])
    fs = h5f['/h5eeg/eeg'].attrs['rate'][0]
    w = WINDOW # seconds
    #r_rows = int(w*fs)
    r_rows = int(w * round(fs))
    print('[*] Using file: {0}\n[*] n_chan: {1}\n[*] n_samples: {2}'.format(h5fname,n_chan,n_samples))
    bip = h5f['/h5eeg/eeg'].attrs['bip']
    e1 = numpy.array(bip[0,:], numpy.float32)
    e2 = numpy.array(bip[1,:], numpy.float32)

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
    N_GRAPH = math.ceil(n_samples/r_rows)
    if ((METRIC == 'sc') or (METRIC == 'pc')):
        N_BANDS = 6
        Rf = numpy.zeros((N_GRAPH*N_BANDS,n_comb), numpy.float32)

        if (not USE_CPU):
            Phf = numpy.zeros((N_GRAPH*N_BANDS,n_comb), numpy.float32)
    else:
        Rf = numpy.zeros((N_GRAPH,n_comb), numpy.float32)

    parCount = 0
    for parIdx in range(0,n_samples,r_rows):
        t_par0 = time.time()
        # Sample number offset
        r_idx = parIdx
        if (r_idx+r_rows >= n_samples):
            r_idx = n_samples-r_rows
        t_read0 = time.time()
        X = h5f['/h5eeg/eeg'][r_idx:(r_idx+r_rows),0:(0+n_chan)]
        t_read = time.time() - t_read0
        Mbytes = r_rows*n_chan*4/1000000

        # EEG referencing and pre-processing
        Xbip = numpy.zeros((r_rows,len(e1)))
        #XbipGam = numpy.zeros((r_rows,len(e1)))
        print('[{0}-{1}/{2} {3} of {4}] Read {5:0.3f} MB/s. Starting..'.format(r_idx,r_idx+r_rows,n_samples,parCount,math.ceil(n_samples/r_rows),Mbytes/t_read))
        for i in range(len(e1)):

            # Calculate bipolar and rank
            x = (X[:,int(e1[i]-1)] - X[:,int(e2[i]-1)])

            doPartial = False
            doCoherence = False
            doEnvelope = False
            if (METRIC == 's'):
                Xbip[:,i] = scipy.stats.rankdata(x)
            elif (METRIC == 'p'):
                Xbip[:,i] = x
            elif (METRIC == 'sc'):
                Xbip[:,i] = scipy.stats.rankdata(x)
                doCoherence = True
            elif (METRIC == 'pc'):
                Xbip[:,i] = x
                doCoherence = True
            elif (METRIC == 'sP'):
                Xbip[:,i] = scipy.stats.rankdata(x)
                doPartial = True
            elif (METRIC == 'pP'):
                Xbip[:,i] = x
                doPartial = True
            elif (METRIC == 'sd'):
                # Delta envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, DEL_S, DEL_E, fs, order=FILT_ORDER)
            elif (METRIC == 'st'):
                # Theta envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, THE_S, THE_E, fs, order=FILT_ORDER)
            elif (METRIC == 'sa'):
                # Alpha envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, ALP_S, ALP_E, fs, order=FILT_ORDER)
            elif (METRIC == 'sb'):
                # Beta envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, BET_S, BET_E, fs, order=FILT_ORDER)
            elif (METRIC == 'sg'):
                # Gamma envelope
                doEnvelope = True
                Xbip[:,i] = butter_bandpass_filter(x, GAM_S, GAM_E, fs, order=FILT_ORDER)
            elif (METRIC == 'shg'):
                # High Gamma envelope
                doEnvelope = True
                #Xbip[:,i] = scipy.stats.rankdata( abs( hilbert( butter_bandpass_filter(x, HG_S, HG_E, fs, order=FILT_ORDER) ) ) )
                Xbip[:,i] = butter_bandpass_filter(x, HG_S, HG_E, fs, order=FILT_ORDER)
            else:
                if (i == 0):
                    print('[*] Metric: '+METRIC+' not found. Using Pearson corr.')
                Xbip[:,i] = x

        if (doPartial):
            r_cols = int(len(e1))
            R = numpy.zeros(n_comb, numpy.float32)
            Ci = numpy.linalg.inv(numpy.cov(numpy.transpose(Xbip)))
            for ic in range(n_comb):
                i = int(chan1[0][ic])
                j = int(chan2[0][ic])
                R[ic] = - Ci[i][j] / math.sqrt( Ci[i][i] * Ci[j][j] )
        elif (doCoherence):
            r_cols = int(len(e1))
            if (USE_CPU):
                r_coh = coherence(Xbip,int(fs))
            else:
                #r_coh,ph_coh = coherence_gpu(Xbip,int(round(2*fs)),int(round(fs)),thr,PLI)
                r_coh,ph_coh = coherence_gpu(Xbip,int(round(fs/5)),int(round(fs)),thr,PLI)
        else:
            r_cols = int(len(e1))

            # Pre-correlation processing
            if (doEnvelope):
                if (USE_CPU):
                    Xbip = envelope(Xbip)
                else:
                    #plat_n = 0
                    #dev_n = int(sys.argv[3])
                    #Xbip = envelope_gpu(Xbip, plat_n, dev_n)
                    Xbip = envelope_gpu(Xbip, thr)
                # rank data
                for i in range(r_cols):
                    Xbip[:,i] = scipy.stats.rankdata(Xbip[:,i])

            if (USE_CPU):
                C = numpy.cov(Xbip,rowvar=False)
                Cd = numpy.sqrt(numpy.diagonal(C))
                R = numpy.zeros(n_comb, numpy.float32)
                for i in range(n_comb):
                    c1 = int(chan1[0,i])
                    c2 = int(chan2[0,i])
                    R[i] = C[c1,c2]/(Cd[c1] * Cd[c2])
            else:
                X = numpy.array(Xbip, numpy.float32)
                # --------------------------------------------------------------------------
                # INPUTS:
                #   X - r_rows by r_cols
                #   chan1 - 1 by n_comb
                #   chan2 - 1 by n_comb
                # OUTPUTS:
                #   R - 1 by n_comb
                # --------------------------------------------------------------------------
                
                ## Step #8. Allocate device memory and move input data from the host to the device memory.
                vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=chan1)
                vector2_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=chan2)
                matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=X)
                R = numpy.zeros(n_comb, numpy.float32)
                destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, R.nbytes)
                 
                program.corr2graph(queue, R.shape, None, matrix_buf, vector_buf, vector2_buf, destination_buf, numpy.int32(r_rows), numpy.int32(r_cols))
                 
                cl.enqueue_copy(queue, R, destination_buf)
            
        # Aggregate
        if (doCoherence):
            for i in range(N_BANDS):
                Rf[(parCount*N_BANDS) + i,:] = r_coh[i,:]

                if (not USE_CPU):
                    Phf[(parCount*N_BANDS) + i,:] = ph_coh[i,:]
        else:
            Rf[parCount,:] = numpy.reshape(R,(1,n_comb))
        #if (parCount==9):
        #    print(Rf)
        #    exit()

        # Time estimate
        t_par = time.time()
        print('[{0:0.4} s] ETA: {1:0.3f} hrs, {2}:{3},{4}'.format(t_par-t_par0,(math.ceil(n_samples/r_rows)-parCount)*(t_par-t_par0)/3600, sid, n_chan, METRIC))
        parCount = parCount + 1

        # DEBUG
        if (DEBUG):
            if (parCount == 20):
                break

    # Save
    if (doCoherence):
        bandn = ['Delta','Theta','Alpha','Beta','Gamma','Broadband']
        for i in range(N_BANDS):
            ofname = OUT_DIR + '/' + sid+'_graph-'+METRIC+bandn[i]+'.h5'
            Rcoh = numpy.zeros((N_GRAPH,n_comb), numpy.float32)
            for j in range(N_GRAPH):
                Rcoh[j,:] = Rf[j*N_BANDS + i,:]


            print(Rcoh)
            print('[*] Saving to: '+ofname)
            h5fout = h5py.File(ofname,'w')
            h5fout.create_dataset('R', data=Rcoh)
            
            if (not USE_CPU):
                PHcoh = numpy.zeros((N_GRAPH,n_comb))
                for j in range(N_GRAPH):
                    PHcoh[j,:] = Phf[j*N_BANDS + i,:]
            #h5fout.create_dataset('PH', data=PHcoh)

            h5fout.create_dataset('chan1', data=chan1)
            h5fout.create_dataset('chan2', data=chan2)
            h5fout.create_dataset('r_rows', data=r_rows)
            h5fout.create_dataset('r_cols', data=r_cols)
            h5fout.create_dataset('w', data=w)
            h5fout.flush()
            h5fout.close()
    else:
        ofname = OUT_DIR + '/' + sid+'_graph-'+METRIC+'.h5'
        print(Rf)
        print('[*] Saving to: '+ofname)
        h5fout = h5py.File(ofname,'w')
        h5fout.create_dataset('R', data=Rf)
        h5fout.create_dataset('chan1', data=chan1)
        h5fout.create_dataset('chan2', data=chan2)
        h5fout.create_dataset('r_rows', data=r_rows)
        h5fout.create_dataset('r_cols', data=r_cols)
        h5fout.create_dataset('w', data=w)
        h5fout.flush()
        h5fout.close()
    print('[!] Done.')

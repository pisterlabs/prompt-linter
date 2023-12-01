import sciunit
import neuronunit
from neuronunit.capabilities import ProducesMembranePotential, ProducesSpikes

import pickle
import neo
import quantities as pq
import numpy as np
from scipy.signal import coherence, hanning
from scipy.fftpack import fft
from itertools import chain
from pathlib import Path, PurePath

import sys
sys.path.append("..") #not the best way to modify sys.path but anyway...
from Validation.lfpcapabilities import ProducesLocalFieldPotential, ProducesConductance
import Fonctions.math_functions as mf
import Fonctions.neuron_functions as nf
import Fonctions.crosscorrelation as crsscorr
import Fonctions.filters as filt


class CoulombModel(sciunit.Model, ProducesLocalFieldPotential, ProducesMembranePotential,
                   ProducesConductance, ProducesSpikes):
    """
    A model of LFP computation that relies on the Coulomb law.
    It also checks if positional data is available. If not it assigns positions to the neurons (randomly).
    """

    def __init__(self, name=None, network_model="VA", space_dependency=False, dimensionnality=3,
                 dimensions=np.array([0.002, 0.002, 0.]), reach=0.001,
                 electrode_positions=np.array([[0.], [0.], [0.]]), sigma=0.3):
        self.name                = name
        self.network_model       = network_model       #Voggels-Abbott for the moment
        self.space_dependency    = space_dependency    #Boolean indicating if the neurons' positions are available
        self.dimensionnality     = dimensionnality     #dimensionnality of the network - either 2 or 3 D
        self.dimensions          = dimensions          #3D-array: leght, width and height of the network (in m)
        self.reach               = reach               #reach of the LFP (in m)
        np.transpose(electrode_positions)              #to have the the coordinates along the 0 axis, as opposed to the input state
        self.electrode_positions = electrode_positions #positions of the electrodes (in m)
        self.sigma               = sigma               #parameter in the Coulomb law's formula (in S/m)
        self.directory_PUREPATH  = PurePath()
        
        ### COMPUTATION OF THE NUMBER OF SEGMENTS
        self.set_directory_path()
        if self.network_model == "T2":
            self.num_trials = 11                       #number of trials for a same experiment
        elif self.network_model == "VA":
            self.num_trials = 1
        else:
            raise ValueError("Only the T2 and the Voggels-Abott models are supported.")
        
        self.num_neurons = 0                           #number of neurons computed - will be properly initialized during LFP computation
        self.exc_counted = False
        self.inh_counted = False
        
        ### VERIFICATION IF THE ELECTRODE(S) ARE "INSIDE" THE NETWORK
        for e_pos in electrode_positions:
            if max(abs(e_pos))+self.reach > self.dimensions[0]/2.:
                raise ValueError("Wrong electrode position! Must have its reach zone in the network.")
        
        return super(CoulombModel, self).__init__(name) #MUST FINISH THIS LINE (name=name, etc.)
    
    #================================================================================================================
    #== methods related to raw available data =======================================================================
    #================================================================================================================

    def set_directory_path(self, date="20190718"):
        if self.network_model == "VA":
            parent_directory="./Exemples/Results/"
            
            directory_path = parent_directory + date
            directory_PATH = Path(directory_path)
            
            if not directory_PATH.exists():
                sys.exit("Directory does not exist!")
            self.directory_PUREPATH = PurePath(directory_path)

        elif self.network_model == "T2":
            directory_path = "./T2/ThalamoCorticalModel_data_size_____/"

            self.directory_PUREPATH = PurePath(directory_path)
        else:
            raise NotImplementedError("Only the T2 and the Voggels-Abott models are supported.")
            
    
    def get_file_path(self, segment_number="0", time="201157", neuron_type=""):
        if self.network_model == "VA":
            if neuron_type == "":
                raise ValueError("Must specify a neuron type.")
            date      = self.directory_PUREPATH.parts[-1]
            file_path = "./" + str(self.directory_PUREPATH) + "/VAbenchmarks_COBA_{0}_neuron_np1_{1}-{2}.pkl".format(neuron_type,
                                                                                                             date, time)
            file_PATH = Path(file_path)
            print(file_path + "\n\n")
            if not file_PATH.exists():
                sys.exit("File name does not exist! (Try checking the time argument.)")
        
        elif self.network_model == "T2":
            file_path = "./" + str(self.directory_PUREPATH) + "/Segment{0}.pickle".format(segment_number)

            file_PATH = Path(file_path)
            if not file_PATH.exists():
                sys.exit("File name does not exist! (Try checking segment number.)")
        else:
            raise NotImplementedError("Only the T2 and the Voggels-Abott models are supported.")
        return file_path
    

    def get_membrane_potential(self, trial=0, experiment="sin_stim"):
        """
        Returns a neo.core.analogsignal.AnalogSignal representing the membrane potential of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if self.network_model == "VA":
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_exc = analogsignal
            if self.exc_counted == False:
                self.num_neurons += vm_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_inh = analogsignal
            if self.inh_counted == False:
                self.num_neurons += vm_inh.shape[1]
                self.inh_counted  = True

            ### ALL NEURONS
            vm_array = np.concatenate(vm_exc, vm_inh, axis=1)
            vm       = neo.core.AnalogSignal(vm_array, units=vm_exc.units, t_start=vm_exc.t_start,
                                             sampling_rate=vm_exc.sampling_rate)
        else:
            if experiment == "sin_stim":
                data_int = 0
            elif experiment == "blank_stim":
                data_int = 5
            else:
                raise ValueError("The experiment argument must be either 'sin_stim' or 'blank_stim'.")
            
            ### EXCITATORY NEURONS
            seg_num     = str(10*trial+data_int+2)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file)
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_exc = analogsignal
            if self.exc_counted == False:
                self.num_neurons += vm_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            seg_num     = str(10*trial+data_int+1)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file)
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_inh = analogsignal
            if self.inh_counted == False:
                self.num_neurons += vm_inh.shape[1]
                self.inh_counted  = True
            
            ### ALL NEURONS
            vm_array = np.concatenate(vm_exc, vm_inh, axis=1)
            vm       = neo.core.AnalogSignal(vm_array, units=vm_exc.units, t_start=vm_exc.t_start,
                                             sampling_rate=vm_exc.sampling_rate)
        return vm
    

    def get_conductance(self, trial=0, experiment="sin_stim"):
        """
        Returns a neo.core.analogsignal.AnalogSignal representing the synaptic conductance of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if self.network_model == "VA":
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_exc':
                    gsyn_exc = analogsignal
            if self.exc_counted == False:
                self.num_neurons += gsyn_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_inh':
                    gsyn_inh = analogsignal
            if self.inh_counted == False:
                self.num_neurons += gsyn_inh.shape[1]
                self.inh_counted  = True

            ### ALL NEURONS
            gsyn_array = np.concatenate(gsyn_exc, gsyn_inh, axis=1)
            gsyn       = neo.core.AnalogSignal(gsyn_array, units=gsyn_exc.units, t_start=gsyn_exc.t_start,
                                               sampling_rate=gsyn_exc.sampling_rate)
        else:
            ### TO CHANGE ###
            '''
            All this has to be changed...
            The pickle files are not organised in blocks but in segments, and these segments correspond to a certain
            type of neuron in a given layer... I must hence find out which segments correspond to the same experiment
            and join them together here. Not forgetting to mention the multiple trials for the same experiment.
            '''
            if experiment == "sin_stim":
                data_int = 0
            elif experiment == "blank_stim":
                data_int = 5
            else:
                raise ValueError("The experiment argument must be either 'sin_stim' or 'blank_stim'.")
            
            seg_num     = str(trial)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file)
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn':
                    gsyn = analogsignal
        return gsyn
    
    
    def get_spike_trains(self, trial=0, experiment="sin_stim"):
        """
        Returns a list of neo.core.SpikeTrain elements representing the spike trains of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if self.network_model == "VA":
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            spiketrains_exc = seg.spiketrains
            if self.exc_counted == False:
                self.num_neurons += len(spiketrains_exc)
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            spiketrains_inh = seg.spiketrains
            if self.inh_counted == False:
                self.num_neurons += len(spiketrains_inh)
                self.inh_counted  = True

            ### ALL NEURONS
            spiketrains = spiketrains_exc + spiketrains_inh
        else:
            if experiment == "sin_stim":
                data_int = 0
            elif experiment == "blank_stim":
                data_int = 5
            else:
                raise ValueError("The experiment argument must be either 'sin_stim' or 'blank_stim'.")

            ### EXCITATORY NEURONS
            seg_num     = str(10*trial+data_int+2)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file)
            spiketrains_exc = seg.spiketrains
            if self.exc_counted == False:
                self.num_neurons += len(spiketrains_exc)
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            seg_num     = str(10*trial+data_int+1)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file)
            spiketrains_inh = seg.spiketrains
            if self.inh_counted == False:
                self.num_neurons += len(spiketrains_inh)
                self.inh_counted  = True
            
            ### ALL NEURONS
            spiketrains = spiketrains_exc + spiketrains_inh
        return spiketrains

    
    def get_spike_train(self, trial=0):
        global_spiketrain = list(chain.from_iterable(spiketrain for spiketrain in self.get_spike_trains(trial=trial)))
        global_spiketrain.sort()
        return global_spiketrain


    #================================================================================================================
    #== LFP related methods =========================================================================================
    #================================================================================================================
    
    def produce_local_field_potential(self, trial=0):
        """
        Calculates and returns the 2D-array of the LFP.
        The first dimension corresponds to the electrodes.
        """
        vm               = self.get_membrane_potential(trial=trial)
        gsyn             = self.get_conductance(trial=trial)
        neuron_positions = self.get_positions()

        num_time_points  = vm.shape[0]
        num_electrodes   = self.electrode_positions.shape[0]
        
        ones_array    = np.ones((num_electrodes, num_time_points, num_neurons))

        current_array = np.multiply(vm, gsyn)
        inv_dist      = nf.electrode_neuron_inv_dist(num_electrodes, num_neurons,
                                                     self.electrode_positions, neuron_positions,
                                                     self.reach, self.dimensionnality)

        big_current_array  = np.multiply(ones_array, current_array)
        big_inv_dist_array = np.multiply(ones_array, inv_dist)

        LFP = np.sum(big_current_array, big_inv_dist_array, axis=2)/(4*np.pi*self.sigma)

        return LFP    


    def get_positions(self):
        """
        Returns the 2D-array giving the neurons' positions.
        """
        if self.space_dependency == False:
            positions = self.assign_positions()
            if self.exc_counted == False and self.inh_counted == False:
                self.num_neurons = positions.shape[1]
        else:
            raise NotImplementedError("Must implement get_positions.")
        return positions


    def assign_positions(self):
        """
        Function that assigns positions to the neurons if they do not have.
        Only works if they have a 2D structure.
        """
        num_neurons = len(self.get_spike_trains(trial=0))
        positions   = np.multiply(self.dimensions, np.random.rand(num_neurons, self.dimensionnality)-0.5)
        return positions
    

    #================================================================================================================
    #== test related methods ========================================================================================
    #================================================================================================================
    
    def produce_vm_LFP_correlation(self, trial=0, start=600, duration=1000, dt=0.1):
        """
        Calculates the correlation between the Vm of the closest neuron to the (first) electrode and the LFP signal
        recorded at this electrode.
        Returns the correlation and the corresponding lag (in ms).
        The relevant data is supposed to be 1s long.
        """
        start_index    = int(start/dt)
        duration_index = int(duration/dt)

        vm               = self.get_membrane_potential(trial=trial)
        neuron_positions = self.get_positions()

        num_electrodes   = self.electrode_positions.shape[0]
        inv_dist         = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                        self.electrode_positions, neuron_positions,
                                                        self.reach, self.dimensionnality)[0, :]
        closest_neuron = np.argmax(inv_dist)
        selected_vm    = np.reshape(vm[start_index:start_index+duration_index+1, closest_neuron], (duration_index+1,))
        selected_LFP   = np.reshape(self.produce_local_field_potential(trial=trial)[0, start_index:start_index+duration_index+1],
                                    (duration_index+1,))
        
        corr             = crsscorr.constwindowcorrelation(selected_vm, selected_LFP)
        corr_time_points = np.arange(-duration/2, duration/2+dt, dt)
        return corr, corr_time_points
    

    def produce_vm_LFP_zerolagcorrelations(self, start=600, duration=1000, dt=0.1,
                                           trial_average=True, trial=0, withinreach=True):
        """
        Calculates the zero-lag correlations between the neurons' membrane potentials and the LFP.
        Interesting plots to do with this data can be:
        - histogram of the correlation distribution;
        - confrontation of the correlation values between a non-stimulated and stimulated state (for the same neurons).
        The trial_average boolean tells if the correlations have to be averaged over the trials.
        If not, the chosen trial is trial.
        """
        start_index    = int(start/dt)
        duration_index = int(duration/dt)

        if trial_average == True:
            trials = self.num_trials
        else:
            trials = trial
        
        self.get_positions() #just to initiate the value of self.num_neurons
        zerolagcorrelations_array = np.zeros((trials, self.num_neurons))
        for iteration_trial in range(trials):
            vm  = self.get_membrane_potential(trial=iteration_trial)
            vm  = vm[start_index:start_index+duration_index, :]
            LFP = np.reshape(self.produce_local_field_potential(trial=iteration_trial)[0, start_index:start_index+duration_index],
                             (duration_index+1,))

            def zerolagtcorrelationtoLFP(v):
                return crsscorr.zerolagcorrelation(v, LFP)
            
            ### ELIMINATION OF THE CONTRIBUTION OF NEURONS THAT ARE OUT OF THE REACH ZONE
            if withinreach:
                num_electrodes   = self.electrode_positions.shape[0]
                neuron_positions = self.get_positions()
                inv_dist         = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                                self.electrode_positions, neuron_positions,
                                                                self.reach, self.dimensionnality)[0, :]
                valid_dist_neurons = np.heaviside(inv_dist-1./self.reach, 1) #array of neurons that are within the reach
                vm                 = np.multiply(vm, valid_dist_neurons)     #vms of neurons that are out of the reach are null
            
            zerolagcorrelations_array[iteration_trial, :] = np.apply_along_axis(zerolagtcorrelationtoLFP, axis=0, arr=vm)

        zerolagcorrelations = np.average(zerolagcorrelations_array, axis=0)

        return zerolagcorrelations #if withinreach==True, neurons that are out of the reach zone have a null correlation with the LFP


    def produce_vm_LFP_meancoherence(self, trial=0, withinreach=True, start=29, duration=1000, dt=0.1):
        """
        Calculates the mean coherence between the neurons' membrane potentials and the LFP.
        returns the mean coherence, the corresponding frequencies (in Hz) and the standard deviation error for each
        coherence value.
        The relevant data is supposed to be 1s long.
        """
        start_index    = int(start/dt)
        duration_index = int(duration/dt)

        vm  = self.get_membrane_potential(trial=trial)
        vm  = vm[start_index:start_index+duration_index, :]
        LFP = np.reshape(self.produce_local_field_potential(trial=trial)[0, start_index:start_index+duration_index],
                         (duration_index+1,))
        
        ### ELIMINATION OF THE CONTRIBUTION OF NEURONS THAT ARE OUT OF THE REACH ZONE
        if withinreach:
            num_electrodes   = self.electrode_positions.shape[0]
            neuron_positions = self.get_positions()
            inv_dist         = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                            self.electrode_positions, neuron_positions,
                                                            self.reach, self.dimensionnality)[0, :]
            valid_dist_neurons = np.heaviside(inv_dist-1./self.reach, 1) #array of neurons that are within the reach
            vm                 = np.multiply(vm, valid_dist_neurons)     #vms of neurons that are out of the reach are null

        f, coherence_array  = coherence(LFP, vm, axis=0, nperseg=int(2**12), fs=1000./dt)
        meancoherence_array = np.average(coherence_array, axis=1)
        coherencestd_array  = np.std(coherence_array, axis=1)
        return meancoherence_array, f, coherencestd_array
    

    def produce_phase_lock_value(self, start=0, offset=250, duration=950, dt=0.1, 
                                 trial_average=True, trial=0, withinreach=True):
        """
        Calculates the Phase-Lock value for the spikes occuring in a (duration-offset) ms period of time.
        The neurons are supposed to be excitated by a sinusoidal input of 1s, starting offset ms before the
        selected epoch.
        Returns the Phase-Lock value and the corresponding frequencies.
        The trial_average boolean tells if the Phase-Lock value has to be averaged over the trials.
        If not, the chosen trial is trial.
        """
        if trial_average:
            trials = self.num_trials
        else:
            trials = trial
        
        fs           = 1000./dt                           #sampling frequency
        window       = 150                                #ms, size of the window in which the LFP will have its Fourier transformations
        window_index = int(window/dt)
        window_width = window_index//2
        w            = hanning(window_index+1)            #150 ms window with a 0,1 ms interval
        N_max        = 500                                #just an arbitrary value for the moment
        PLv_array    = np.zeros((trials, window_index+1),
                                dtype=float)              #multi-trial Phase-Lock value array, empty for the moment

        for iteration_trial in range(trials):
            spiketrain   = self.get_spike_train(trial=iteration_trial)
            num_spikes   = len(spiketrain)
            valid_times1 = np.heaviside(spiketrain-(start+offset)*np.ones(num_spikes), 1)   #spikes that occured after a certain time
            valid_times2 = np.heaviside((start+duration)*np.ones(num_spikes)-spiketrain, 1) #spikes that occured before the maximum admitted time
            valid_times  = np.multiply(valid_times1, valid_times2)

            selected_spikes = np.multiply(spiketrain, valid_times)
            selected_spikes = selected_spikes[selected_spikes>0]

            LFP      = self.produce_local_field_potential(trial=iteration_trial)[0, :]
            #LFP_filt = butter_lowpass_filter(LFP, 170., fs)

            N_s = min(selected_spikes.shape[0], N_max)  #security measure

            for t_index in mf.random_list(N_s, selected_spikes.shape[0], minimum=0):
                t_s       = selected_spikes[t_index]                                         #time of spike occurence
                t_s_index = int(10*t_s)                                                      #corresponding index for the arrays
                #LFP_s     = LFP_filt[t_s_index - window_width : t_s_index + window_width+1]  #LFP centered at the spike occurrence
                LFP_s     = LFP[t_s_index - window_width : t_s_index + window_width+1]       #Non-filtered version
                wLFP_s    = np.multiply(w, LFP_s)                                            #centered LFP multiplied by the Hanning window
                FT_s      = fft(wLFP_s)                                                      #Fourier transform of this weighted LFP
                nFT_s     = np.divide(FT_s, np.abs(FT_s))                                    #normalized Fourier transform
                PLv_array[iteration_trial, :] = np.add(PLv_array[iteration_trial, :], nFT_s)       #contribution to the PLv added

            PLv_array[iteration_trial, :]  = np.abs(PLv_array[iteration_trial, :])/N_s         #normalized module, according to the paper
            
        PLv  = np.average(PLv_array, axis=0)
        PLv  = PLv[:(PLv.shape[0])//2] #only the first half is relevant
        fPLv = (0.5*fs/PLv.shape[0])*np.arange(PLv.shape[0], dtype=float) #frequencies of the PLv
        return PLv, fPLv


    def produce_spike_triggered_LFP(self, start=500, duration=1000, dt=0.1, window_width=200,
                                    trial_average=True, trial=0):
        """
        Calculates the spike-triggered average of the LFP (stLFP) and arranges the results relative to the distance
        from the electrode. The distances discriminating the neurons are (in mm): 0.4, 0.8, 1.2, 1.4 and 1.6.
        Returns the stLFP for each distance interval and in a time interval around the spikes.
        The stLFP is a 2D-array with the first dimension corresponding to the distance and the second, the time.
        """
        discrim_dist     = np.array([4e-4, 8e-4, 1.2e-3, 1.6e-3])
        discrim_inv_dist = np.append(np.inf, np.power(discrim_dist, -1))
        discrim_inv_dist = np.append(discrim_dist, 0.)

        num_electrodes   = self.electrode_positions.shape[0]
        neuron_positions = self.get_positions()
        inv_dist         = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                        self.electrode_positions, neuron_positions,
                                                        self.reach, self.dimensionnality)[0, :]
        
        discrim_indexes    = [[], [], [], [], []]
        num_dist_intervals = 5
        for i in range(num_dist_intervals):
            i_normalized_inv_dist = mf.door(inv_dist, discrim_inv_dist[i+1], discrim_inv_dist[i])
            i_indexes  = np.argwhere(i_normalized_inv_dist == 1)
            discrim_indexes[i].append(i_indexes.flatten().tolist())
        
        '''
        Now I have discriminated the neurons according to their distance from the electrode (information stored in
        their indexes in the list discrim_indexes), I can separate the stLFPs according to this criteria.
        '''

        if trial_average:
            trials = self.num_trials
        else:
            trials = trial
        
        window_index = int(window_width/dt)
        window       = np.arange(-window_width/2, window_width/2+dt, dt)
        stLFP_array  = np.zeros((trials, num_dist_intervals, window_index+1))

        for iteration_trial in range(trials):
            ### LOOP ON THE TRIALS
            spiketrains = self.get_spike_trains(trial=iteration_trial)
            LFP         = self.produce_local_field_potential(trial=iteration_trial)[0, :]

            for interval_index in range(num_dist_intervals):
                ### LOOP ON THE DISTANCE INTERVALS
                average_counter = 0
                for neuron_index in discrim_indexes[interval_index]:
                    ### LOOP ON THE NEURONS WITHIN A DISTANCE INTERVAL
                    for t_s in spiketrains[neuron_index]: #maybe I can get rid of this loop... I don't like it...
                        ### LOOP ON THE SPIKES OF A GIVEN NEURON
                        t_s_index = int(t_s/dt)
                        average_counte += 1
                        stLFP_array[iteration_trial, interval_index, :] = np.add(
                                                                       stLFP_array[iteration_trial, interval_index, :],
                                                                       LFP[t_s-window_index//2:t_s_index+window_index//2+1])
                stLFP_array[iteration_trial, interval_index, :] /= average_counter
        
        stLFP = np.average(stLFP_array, axis=0) #trial-average computation
        return stLFP, window
import cohere_core.utilities.dvc_utils as dvut
from abc import ABC, abstractmethod

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Pcdi',
           'Feature',
           'ShrinkWrap',
           'PhaseMod',
           'LowPassFilter']


def set_lib(dlib):
    global devlib
    devlib = dlib
    dvut.set_lib(devlib)


class Pcdi:
    def __init__(self, params, data, dir=None):
        if 'pc_type' in params:
            self.type = params['pc_type']
        else:
            self.type = "LUCY"
        if 'pc_LUCY_iterations' in params:
            self.iterations = params['pc_LUCY_iterations']
        else:
            self.iterations = 20
        if 'pc_normalize' in params:
            self.normalize = params['pc_normalize']
        else:
            self.normalize = True
        if 'pc_LUCY_kernel' in params:
            self.kernel_area = params['pc_LUCY_kernel']
        else:
            self.kernel_area = (16, 16, 16)
        if dir is None:
            self.kernel = None
        else:
            try:
                self.kernel = devlib.load(dir + '/coherence.npy')
            except:
                self.kernel = None

        self.dims = devlib.dims(data)
        self.roi_data = dvut.crop_center(devlib.fftshift(data), self.kernel_area)
        if self.normalize:
            self.sum_roi_data = devlib.sum(devlib.square(self.roi_data))
        if self.kernel is None:
            self.kernel = devlib.full(self.kernel_area, 0.5, dtype=devlib.dtype(data))

    def set_previous(self, abs_amplitudes):
        self.roi_amplitudes_prev = dvut.crop_center(devlib.fftshift(abs_amplitudes), self.kernel_area)

    def apply_partial_coherence(self, abs_amplitudes):
        abs_amplitudes_2 = devlib.square(abs_amplitudes)
        converged_2 = devlib.fftconvolve(abs_amplitudes_2, self.kernel)
        converged = devlib.sqrt(converged_2)
        return converged

    def update_partial_coherence(self, abs_amplitudes):
        roi_amplitudes = dvut.crop_center(devlib.fftshift(abs_amplitudes), self.kernel_area)
        roi_combined_amp = 2 * roi_amplitudes - self.roi_amplitudes_prev
        if self.normalize:
            amplitudes_2 = devlib.square(roi_combined_amp)
            sum_ampl = devlib.sum(amplitudes_2)
            ratio = self.sum_roi_data / sum_ampl
            amplitudes = devlib.sqrt(amplitudes_2 * ratio)
        else:
            amplitudes = roi_combined_amp

        if self.type == "LUCY":
            self.lucy_deconvolution(devlib.square(amplitudes), devlib.square(self.roi_data),
                                    self.iterations)

    def lucy_deconvolution(self, amplitudes, data, iterations):
        data_mirror = devlib.flip(data)
        for i in range(iterations):
            conv = devlib.fftconvolve(self.kernel, data)
            conv = devlib.where(conv == 0.0, 1.0, conv)
            relative_blurr = amplitudes / conv
            self.kernel = self.kernel * devlib.fftconvolve(relative_blurr, data_mirror)
        self.kernel = devlib.real(self.kernel)
        coh_sum = devlib.sum(devlib.absolute(self.kernel))
        self.kernel = devlib.absolute(self.kernel) / coh_sum


class Feature(ABC):
    """
    Base class for features. This class creates feature objects and manages the trigger or subtrigger depending on
    configuration.
    """
    def __init__(self, sub_rows_trigs, params):
        # the self.objs will hold either one concrete feature object if configuration is for general trigger,
        # i.e. spanning across all iterations,
        # or it will hold a list of objects, each being a concrete sub-feature, appearing in order that will be
        # executed during each trigger
        self.objs = None

        # the self.f points to either self.apply_trigger_obj function if configuration is for general trigger,
        # or to self.apply_trigger_seq if sub-triggers are configured
        self.f = None

        # this function sets self.objs and self.f and creates all objects
        self.create_objs(sub_rows_trigs, params)

        self.key = None  # ex: 'SW'

        self.trig_name = None  # ex: 'shrink_wrap_trigger'

    def apply_trigger_obj(self, *args):
        """
        This function is called when general trigger is used. In this case there is only one instance of the concrete
        feature. The trigger is then executed on this instance.
        :param args: variable parameters depending on the concrete feature
        :return: depends on the concrete feature
        """
        return self.objs.apply_trigger(*args)

    def apply_trigger_seq(self, *args):
        """
        This function is called when sub-triggers are used. the self.objs is a list for this case. Every time a trigger
        is applied the first object is removed from this list and the trigger is executed on this instance.
        :param args: variable parameters depending on the concrete feature
        :return: depends on the concrete feature
        """
        sub_obj = self.objs.pop(0)
        return sub_obj.apply_trigger(*args)

    def apply_trigger(self, *args):
        """
        Thhis function is called by a trigger. It then directs the trigger by calling the self.f function.
        :param args: variable parameters depending on the concrete feature
        :return: depends on the concrete feature
        """
        # the f is either update_amp_seq function or update_amp_seq
        # The f is set in constructor depending on whether the trigger
        # was defined for the entire span of iterations or multiple
        # subtriggers were defined in algorithm sequence
        return self.f(*args)

    @abstractmethod
    def create_obj(self, *args):
        """
        This class must be overriden by subclass. It creates a feature object.
        :param args: variable parameters depending on the concrete feature
        :return: an instance of the subclass
        """
        pass

    def create_objs(self, sub_rows_trigs, params):
        """
        The params map contains value of the feature trigger. If the trigger is configured as a general one,
        i.e. a single trigger, the self.objs is set to the created instance of the feature, and the self.f
        pointer is set accordingly, i.e. to the self.apply_trigger_obj function.
        If the trigger contains several sub-triggers, the sub_row_trigs contain info about the sequence the
        subtrigger should be executed. Based on this info the unique instances are created and are added to the
        in the order the triggers should be called. The self.f function is set to the self.apply_trigger_seq
        function.
        :param sub_rows_trigs: sub-triggers info if sub-triggers are configured
        :param params: configuration parameters
        :return:
        """
        if self.key in sub_rows_trigs.keys():
            row = sub_rows_trigs[self.key][0]
            sub_trigs = sub_rows_trigs[self.key][1]
            sub_objs = {}
            for sub_t in sub_trigs:
                (beg, end, idx) = sub_t
                index = int(idx)
                sub_objs[index] = self.create_obj(params, index=index, beg=beg, end=end)
            trigs = [i-1 for i in row.tolist() if i != 0]
            # the operation of creating object might fail
            # if conditions are not met, ex: the sw_type is not supported
            # This should be already verified by a verifier, so no checking here
            self.objs = [sub_objs[idx] for idx in trigs]
            self.f = self.apply_trigger_seq
        else:
            if self.trig_name in params:
                self.objs = self.create_obj(params)
                self.f = self.apply_trigger_obj


class ShrinkWrap(Feature):
    def __init__(self, params, sub_rows_trigs):
        self.key = 'SW'
        self.trig_name = 'shrink_wrap_trigger'
        super().__init__(sub_rows_trigs, params)

    class GaussSW:
        def __init__(self, gauss_sigma, threshold):
            self.gauss_sigma = gauss_sigma
            self.threshold = threshold

        def apply_trigger(self, *args):
#            print('shrink wrap, sigma', self.gauss_sigma)
            ds_image = args[0]
            return dvut.shrink_wrap(ds_image, self.threshold, self.gauss_sigma)


    def create_obj(self, params, index=None, beg=None, end=None):
        if index is None:
            if 'shrink_wrap_type' not in params:
                sw_type = 'GAUSS'
            else:
                sw_type = params['shrink_wrap_type']
            if sw_type == 'GAUSS':
                if 'shrink_wrap_gauss_sigma' not in params:
                    sigma = 1.0
                else:
                    sigma = params['shrink_wrap_gauss_sigma']
                if 'shrink_wrap_threshold' not in params:
                    threshold = 0.1
                else:
                    threshold = params['shrink_wrap_threshold']
        else:
            if 'shrink_wrap_type' not in params or len(params['shrink_wrap_type']) - 1 < index:
                sw_type = 'GAUSS'
            else:
                sw_type = params['shrink_wrap_type'][index]
            if sw_type == 'GAUSS':
                if 'shrink_wrap_gauss_sigma' not in params or len(params['shrink_wrap_gauss_sigma']) - 1 < index:
                    sigma = 1.0
                else:
                    sigma = params['shrink_wrap_gauss_sigma'][index]
                if 'shrink_wrap_threshold' not in params or len(params['shrink_wrap_threshold']) - 1 < index:
                    threshold = 0.1
                else:
                    threshold = params['shrink_wrap_threshold'][index]
        return self.GaussSW(sigma, threshold)


class PhaseMod(Feature):
    def __init__(self, params, sub_rows_trigs):
        self.key = 'PHM'
        self.trig_name = 'phm_trigger'
        super().__init__(sub_rows_trigs, params)

    class PhasePHM:
        def __init__(self, phm_phase_min, phm_phase_max):
            self.phm_phase_min = phm_phase_min
            self.phm_phase_max = phm_phase_max

        def apply_trigger(self, *args):
#            print('phm trig, phase min', self.phm_phase_min)
            ds_image = args[0]
            phase = devlib.angle(ds_image)
            return (phase > self.phm_phase_min) & (phase < self.phm_phase_max)


    def create_obj(self, params, index=None, beg=None, end=None):
        if index is None:
            if 'phm_phase_min' not in params:
                phase_min = -1.57
            else:
                phase_min = params['phm_phase_min']
            if 'phm_phase_max' not in params:
                phase_max = 1.57
            else:
                phase_max = params['phm_phase_max']
        else:
            if 'phm_phase_min' not in params or len(params['phm_phase_min']) - 1 < index:
                phase_min = -1.57
            else:
                phase_min = params['phm_phase_min'][index]
            if 'phm_phase_max' not in params or len(params['phm_phase_max']) - 1 < index:
                phase_max = 1.57
            else:
                phase_max = params['phm_phase_max'][index]
        return self.PhasePHM(phase_min, phase_max)


class LowPassFilter(Feature):
    def __init__(self, params, sub_rows_trigs):
        self.key = 'LPF'
        self.trig_name = 'lowpass_filter_trigger'
        super().__init__(sub_rows_trigs, params)

    class Filter:
        def __init__(self, filter_range, iter_range, threshold):
            self.iter_offset = iter_range[0]
            self.filter_sigmas = [(filter_range[1] - filter_range[0]) / (iter_range[1] - iter_range[0]) *
                                  (iter - self.iter_offset) + filter_range[0]
                                  for iter in range(iter_range[0], iter_range[1])]
            self.threshold = threshold

        def apply_trigger(self, *args):
            # The filter is applied on data
            # simultaneously the sigma used in shrink wrap is changing to be reverse of the filter_sigma
            data = args[0]
            iter = args[1]
            ds_image = args[2]
            filter_sigma = self.filter_sigmas[iter - self.iter_offset]
 #           print('in apply_low_filter, filter_sigma', filter_sigma)
            return (devlib.gaussian_filter(data, filter_sigma), dvut.shrink_wrap(ds_image, self.threshold, 1/filter_sigma))


    def create_obj(self, params, index=None, beg=None, end=None):
        # in case of a all_iteration configuration the lowpass filter
        # iteration can be read from lowpass_filter_trigger
        if index is None:
            iter_range = (params['lowpass_filter_trigger'][0], params['lowpass_filter_trigger'][2])
            filter_range = params['lowpass_filter_range']
            if 'lowpass_filter_threshold' in params:
                threshold = params['lowpass_filter_threshold']
            else:
                threshold = .1
        # otherwise the iterations to apply this instance are given as arguments
        else:
            iter_range = (beg, end)
            filter_range = params['lowpass_filter_range'][index]
            if 'lowpass_filter_threshold' not in params or len(params['lowpass_filter_threshold']) - 1 < index:
                threshold = .1
            else:
                threshold = params['lowpass_filter_threshold'][index]
        if len(filter_range) == 1:
            filter_range.append(1.0)
        return self.Filter(filter_range, iter_range, threshold)



# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
cohere_core.phasing
===================

Provides phasing capabilities for the Bragg CDI data.
The software can run code utilizing different library, such as numpy and cupy. User configures the choice depending on hardware and installed software.

"""

import time
import os
from math import pi
import random
import importlib
import cohere_core.utilities.dvc_utils as dvut
import cohere_core.utilities.utils as ut
import cohere_core.utilities.config_verifier as ver
import cohere_core.controller.op_flow as of
import cohere_core.controller.features as ft

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction',
           'Rec']


def set_lib(dlib):
    global devlib
    devlib = dlib
    dvut.set_lib(devlib)
    ft.set_lib(dlib)


def get_norm(arr):
    return devlib.sum(devlib.square(devlib.absolute(arr)))


class Support:
    """
    Support class is a state holder for the support array. It initializes the support at creation time.
    Features that modify support: shrink wrap, phase modifier, lowpass filter.
    """
    def __init__(self, params, dims, dir=None):
        # create or get initial support
        if dir is None or not os.path.isfile(dir + '/support.npy'):
            initial_support_area = params['initial_support_area']
            init_support = []
            for i in range(len(initial_support_area)):
                if type(initial_support_area[0]) == int:
                    init_support.append(initial_support_area[i])
                else:
                    init_support.append(int(initial_support_area[i] * dims[i]))
            center = devlib.full(init_support, 1)
            self.support = dvut.pad_around(center, dims, 0)
        else:
            self.support = devlib.load(dir + '/support.npy')

    def get_support(self):
        return self.support

    def flip(self):
        self.support = devlib.flip(self.support)


class Rec:
    """
    cohere_core.phasing.reconstruction(self, params, data_file)

    Class, performs phasing using iterative algorithm.

    params : dict
        parameters used in reconstruction. Refer to x for parameters description
    data_file : str
        name of file containing data to be reconstructed

    """
    __all__ = []
    def __init__(self, params, data_file):
        self.iter_functions = [self.next,
                          self.lowpass_filter_trigger,
                          self.reset_resolution,
                          self.shrink_wrap_trigger,
                          self.phm_trigger,
                          self.to_reciprocal_space,
                          self.new_func_trigger,
                          self.pc_trigger,
                          self.pc_modulus,
                          self.modulus,
                          self.set_prev_pc_trigger,
                          self.to_direct_space,
                          self.er,
                          self.hio,
                          self.new_alg,
                          self.twin_trigger,
                          self.average_trigger,
                          self.progress_trigger]

        if 'init_guess' not in params:
            params['init_guess'] = 'random'
        elif params['init_guess'] == 'AI_guess':
            if 'AI_threshold' not in params:
                params['AI_threshold'] = params['shrink_wrap_threshold']
            if 'AI_sigma' not in params:
                params['AI_sigma'] = params['shrink_wrap_gauss_sigma']
        if 'reconstructions' not in params:
            params['reconstructions'] = 1
        if 'hio_beta' not in params:
            params['hio_beta'] = 0.9
        if 'initial_support_area' not in params:
            params['initial_support_area'] = (.5, .5, .5)
        if 'twin_trigger' in params and 'twin_halves' not in params:
            params['twin_halves'] = (0, 0)
        if 'pc_interval' in params and 'pc' in params['algorithm_sequence']:
            self.is_pcdi = True
        else:
            self.is_pcdi = False
        # finished setting defaults
        self.params = params
        self.data_file = data_file
        self.ds_image = None
        self.need_save_data = False
        self.saved_data = None

    def init_dev(self, device_id):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.dev = device_id
        try:
            devlib.set_device(device_id)
        except Exception as e:
            print(e)
            print('may need to restart GUI')
            return -1
        if self.data_file.endswith('tif') or self.data_file.endswith('tiff'):
            try:
                data_np = ut.read_tif(self.data_file)
                data = devlib.from_numpy(data_np)
            except Exception as e:
                print(e)
                return -1
        elif self.data_file.endswith('npy'):
            try:
                data = devlib.load(self.data_file)
            except Exception as e:
                print(e)
                return -1
        else:
            print('no data file found')
            return -1
        # in the formatted data the max is in the center, we want it in the corner, so do fft shift
        self.data = devlib.fftshift(devlib.absolute(data))
        self.dims = devlib.dims(self.data)
        print('data shape', self.dims)

        if self.need_save_data:
            self.saved_data = devlib.copy(self.data)
            self.need_save_data = False

        return 0

    def fast_ga(self, worker_qin, worker_qout):
        if self.params['low_resolution_generations'] > 0:
            self.need_save_data = True

        functions_dict = {}
        functions_dict[self.init_dev.__name__] = self.init_dev
        functions_dict[self.init.__name__] = self.init
        functions_dict[self.breed.__name__] = self.breed
        functions_dict[self.iterate.__name__] = self.iterate
        functions_dict[self.get_metric.__name__] = self.get_metric
        functions_dict[self.save_res.__name__] = self.save_res

        done = False
        while not done:
            # cmd is a tuple containing name of the function, followed by arguments
            cmd = worker_qin.get()
            if cmd == 'done':
                done = True
            else:
                if type(cmd) == str:
                    # the function does not have any arguments
                    try:
                        ret = functions_dict[cmd]()
                    except:
                        print('cought exception')
                        ret = -3
                else:
                    try:
                    # the function name is the first element in the cmd tuple, and the other elements are arguments
                        ret = functions_dict[cmd[0]](*cmd[1:])
                    except:
                        print('cought exception')
                        ret = -3

                worker_qout.put(ret)


    def init(self, dir=None, alpha_dir=None, gen=None):
        def create_feat_objects(params, feats):
            if 'shrink_wrap_trigger' in params:
                self.shrink_wrap_obj = ft.ShrinkWrap(params, feats)
            if 'phm_trigger' in params:
                self.phm_obj = ft.PhaseMod(params, feats)
            if 'lowpass_filter_trigger' in params:
                self.lowpass_filter_obj = ft.LowPassFilter(params, feats)

        if self.ds_image is not None:
            first_run = False
        elif dir is None or not os.path.isfile(dir + '/image.npy'):
            self.ds_image = devlib.random(self.dims, dtype=self.data.dtype)
            first_run = True
        else:
            self.ds_image = devlib.load(dir + '/image.npy')
            first_run = False

        self.flow_items_list = [f.__name__ for f in self.iter_functions]

        self.is_pc, flow, feats = of.get_flow_arr(self.params, self.flow_items_list, gen, first_run)
        if flow is None:
            return -1

        self.flow = []
        (op_no, self.iter_no) = flow.shape
        for i in range(self.iter_no):
            for j in range(op_no):
                if flow[j, i] == 1:
                    self.flow.append(self.iter_functions[j])

        self.aver = None
        self.iter = -1
        self.errs = []
        self.gen = gen
        self.prev_dir = dir
        self.alpha_dir = alpha_dir
        self.support_obj = Support(self.params, self.dims, dir)
        if self.is_pc:
            self.pc_obj = ft.Pcdi(self.params, self.data, dir)
        # create the object even if the feature inactive, it will be empty
        create_feat_objects(self.params, feats)

        # for the fast GA the data needs to be saved, as it would be changed by each lr generation
        # for non-fast GA the Rec object is created in each generation with the initial data
        if self.saved_data is not None:
            if self.params['low_resolution_generations'] > self.gen:
                self.data = devlib.gaussian_filter(self.saved_data, self.params['ga_lpf_sigmas'][self.gen])
            else:
                self.data = self.saved_data
        else:
            if self.gen is not None and self.params['low_resolution_generations'] > self.gen:
                self.data = devlib.gaussian_filter(self.data, self.params['ga_lpf_sigmas'][self.gen])

        if 'lowpass_filter_range' not in self.params or not first_run:
            self.iter_data = self.data
        else:
            self.iter_data = devlib.copy(self.data)

        if (first_run):
            max_data = devlib.amax(self.data)
            self.ds_image *= get_norm(self.ds_image) * max_data

            # the line below are for testing to set the initial guess to support
            # self.ds_image = devlib.full(self.dims, 1.0) + 1j * devlib.full(self.dims, 1.0)

            self.ds_image *= self.support_obj.get_support()
        return 0

    def breed(self):
        breed_mode = self.params['ga_breed_modes'][self.gen]
        if breed_mode != 'none':
            try:
                self.ds_image = dvut.breed(breed_mode, self.alpha_dir, self.ds_image)
                self.support_obj.params = dvut.shrink_wrap(self.ds_image, self.params['ga_sw_thresholds'][self.gen],
                                                           self.params['ga_sw_gauss_sigmas'][self.gen])
            except:
                print('exception during breeding')
                return -1

        return 0


    def iterate(self):
        self.iter = -1
        start_t = time.time()
        try:
            for f in self.flow:
                f()
        except:
            return -1

        print('iterate took ', (time.time() - start_t), ' sec')

        if devlib.hasnan(self.ds_image):
            print('reconstruction resulted in NaN')
            return -1

        if self.aver is not None:
            ratio = self.get_ratio(devlib.from_numpy(self.aver), devlib.absolute(self.ds_image))
            self.ds_image *= ratio / self.aver_iter

        mx = devlib.amax(devlib.absolute(self.ds_image))
        self.ds_image = self.ds_image / mx

        return 0


    def save_res(self, save_dir, only_image=False):
        from array import array

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        devlib.save(save_dir + '/image', self.ds_image)

        if only_image:
            return 0

        devlib.save(save_dir + '/support', self.support_obj.get_support())
        if self.is_pc:
            devlib.save(save_dir + '/coherence', self.pc_obj.kernel)
        errs = array('f', self.errs)

        with open(save_dir + "/errors.txt", "w+") as err_f:
            err_f.write('\n'.join(map(str, errs)))

        devlib.save(save_dir + '/errors', errs)

        metric = dvut.all_metrics(self.ds_image, self.errs)
        with open(save_dir + "/metrics.txt", "w+") as f:
            f.write(str(metric))
            # for key, value in metric.items():
            #     f.write(key + ' : ' + str(value) + '\n')

        return 0


    def get_metric(self, metric_type):
        return dvut.all_metrics(self.ds_image, self.errs)
        # return dvut.get_metric(self.ds_image, self.errs, metric_type)

    def next(self):
        self.iter = self.iter + 1

    def lowpass_filter_trigger(self):
        args = (self.data, self.iter, self.ds_image)
        (self.iter_data, self.support_obj.support) = self.lowpass_filter_obj.apply_trigger(*args)

    def reset_resolution(self):
        self.iter_data = self.data

    def shrink_wrap_trigger(self):
        args = (self.ds_image,)
        self.support_obj.support = self.shrink_wrap_obj.apply_trigger(*args)

    def phm_trigger(self):
        args = (self.ds_image,)
        self.support_obj.support *= self.phm_obj.apply_trigger(*args)

    def to_reciprocal_space(self):
        self.rs_amplitudes = devlib.ifft(self.ds_image)

    def new_func_trigger(self):
        self.params['new_param'] = 1
        print('in new_func_trigger, new_param', self.params['new_param'])

    def pc_trigger(self):
        self.pc_obj.update_partial_coherence(devlib.absolute(self.rs_amplitudes))

    def pc_modulus(self):
        abs_amplitudes = devlib.absolute(self.rs_amplitudes)
        converged = self.pc_obj.apply_partial_coherence(abs_amplitudes)
        ratio = self.get_ratio(self.iter_data, devlib.absolute(converged))
        error = get_norm(
            devlib.where(devlib.absolute(converged) != 0.0, devlib.absolute(converged) - self.iter_data, 0.0)) / get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def modulus(self):
        ratio = self.get_ratio(self.iter_data, devlib.absolute(self.rs_amplitudes))
        error = get_norm(devlib.where((self.rs_amplitudes != 0), (devlib.absolute(self.rs_amplitudes) - self.iter_data),
                                      0)) / get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def set_prev_pc_trigger(self):
        self.pc_obj.set_previous(devlib.absolute(self.rs_amplitudes))

    def to_direct_space(self):
        self.ds_image_raw = devlib.fft(self.rs_amplitudes)

    def er(self):
        self.ds_image = self.ds_image_raw * self.support_obj.get_support()

    def hio(self):
        combined_image = self.ds_image - self.ds_image_raw * self.params['hio_beta']
        support = self.support_obj.get_support()
        self.ds_image = devlib.where((support > 0), self.ds_image_raw, combined_image)

    def new_alg(self):
        self.ds_image = 2.0 * (self.ds_image_raw * self.support_obj.get_support()) - self.ds_image_raw

    def twin_trigger(self):
        # TODO this will work only for 3D array, but will the twin be used for 1D or 2D?
        # com = devlib.center_of_mass(devlib.absolute(self.ds_image))
        # sft = [int(self.dims[i] / 2 - com[i]) for i in range(len(self.dims))]
        # self.ds_image = devlib.shift(self.ds_image, sft)
        dims = devlib.dims(self.ds_image)
        half_x = int((dims[0] + 1) / 2)
        half_y = int((dims[1] + 1) / 2)
        if self.params['twin_halves'][0] == 0:
            self.ds_image[half_x:, :, :] = 0
        else:
            self.ds_image[: half_x, :, :] = 0
        if self.params['twin_halves'][1] == 0:
            self.ds_image[:, half_y:, :] = 0
        else:
            self.ds_image[:, : half_y, :] = 0

    def average_trigger(self):
        if self.aver is None:
            self.aver = devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter = 1
        else:
            self.aver = self.aver + devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter += 1

    def progress_trigger(self):
        print('------iter', self.iter, '   error ', self.errs[-1])

    def get_ratio(self, divident, divisor):
        divisor = devlib.where((divisor != 0.0), divisor, 1.0)
        ratio = divident / divisor
        return ratio


class Peak:
    """
    Holds parameters related to a peak.
    """

    def __init__(self, dir_ornt):
        (self.dir, self.orientation) = dir_ornt

    def set_data(self, G_0):
        import tifffile as tf

        self.g_vec = devlib.array([0, * self.orientation]) * G_0
        self.gdotg = devlib.array(devlib.dot(self.g_vec, self.g_vec))

        fn = self.dir + '/phasing_data/data.tif'
        data_np = tf.imread(fn.replace(os.sep, '/'))
        data = devlib.from_numpy(data_np)

        # in the formatted data the max is in the center, we want it in the corner, so do fft shift
        self.data = devlib.fftshift(devlib.absolute(data))


class CoupledRec(Rec):
    """
    Performs a coupled reconstruction of multiple Bragg peaks using iterative phase retrieval. It alternates between a
    shared object with a density and atomic displacement field and a working object with an amplitude and phase. The
    general outline of this process is as follows:
    1. Initialize the shared object with random values.
    2. Randomly select a diffraction pattern and corresponding reciprocal lattice vector G from the collected data.
    3. Set the working object to the projection of the shared object onto G.
    4. Apply standard phase retrieval techniques to the working object for a set number of iterations.
    5. Update the G-projection of the shared object to be a weighted average of its current value with the working
        object.
    6. Repeat steps 2-5.

    params : dict
        parameters used in reconstruction. Refer to x for parameters description
    data_file : str
        name of file containing data for each peak to be reconstructed

    """
    __author__ = "Nick Porter"
    __all__ = []

    def __init__(self, params, peak_dir_orient):
        super().__init__(params, None)

        self.iter_functions = self.iter_functions + [self.switch_peaks]

        if "switch_peak_trigger" not in params:
            params["switch_peak_trigger"] = [0, 10]
        if "mp_max_weight" not in params:
            params["mp_max_weight"] = 0.9
        if "mp_taper" not in params:
            params["mp_taper"] = 0.75

        self.peak_objs = [Peak(dir_ornt) for dir_ornt in peak_dir_orient]


    def init_dev(self, device_id):
        self.dev = device_id
        if device_id != -1:
            try:
                devlib.set_device(device_id)
            except Exception as e:
                print(e)
                print('may need to restart GUI')
                return -1

        G_0 = 2*pi/self.params["lattice_size"]
        for or_obj in self.peak_objs:
            or_obj.set_data(G_0)

        self.num_peaks = len(self.peak_objs)
        self.pk = 0  # index in list of current peak being reconstructed
        self.data = self.peak_objs[self.pk].data
        self.iter_data = self.data
        self.dims = self.data.shape

        if self.need_save_data:
            self.saved_data = devlib.copy(self.data)
            self.need_save_data = False

        return 0

    def init(self, img_dir=None, alpha_dir=None, gen=None):
        if super().init(img_dir, alpha_dir, gen) == -1:
            return -1

        # Define the shared image
        self.shared_image = devlib.absolute(self.ds_image[:, :, :, None]) * devlib.array([1, 1, 1, 1])
        self.rho_hat = devlib.array([1, 0, 0, 0])  # This is the density "unit vector" in the shared object.

        # Define the multipeak projection weighting and tapering
        coeff = self.params["mp_taper"] / (self.params["mp_taper"] - 1)
        self.proj_weight = devlib.square(devlib.cos(devlib.linspace(coeff*1.57, 1.57, self.iter_no).clip(0, 2)))
        self.proj_weight = self.proj_weight * self.params["mp_max_weight"]

        return 0

    def save_res(self, save_dir):
        from array import array

        self.shared_image = self.shared_image * self.support_obj.get_support()[:, :, :, None]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        devlib.save(save_dir + "/image", self.shared_image)
        devlib.save(save_dir + "/shared_density", self.shared_image[:, :, :, 0])
        devlib.save(save_dir + "/shared_u1", self.shared_image[:, :, :, 1])
        devlib.save(save_dir + "/shared_u2", self.shared_image[:, :, :, 2])
        devlib.save(save_dir + "/shared_u3", self.shared_image[:, :, :, 3])
        devlib.save(save_dir + '/support', self.support_obj.get_support())

        errs = array('f', self.errs)

        with open(save_dir + "/errors.txt", "w+") as err_f:
            err_f.write('\n'.join(map(str, errs)))

        devlib.save(save_dir + '/errors', errs)

        metric = dvut.all_metrics(self.ds_image, self.errs)
        with open(save_dir + "/metrics.txt", "w+") as f:
            f.write(str(metric))

        return 0

    def switch_peaks(self):
        self.to_shared_image()
        self.pk = random.choice([x for x in range(self.num_peaks) if x not in (self.pk,)])
        self.iter_data = self.peak_objs[self.pk].data
        self.to_working_image(self.peak_objs[self.pk])

    def to_shared_image(self):
        beta = self.proj_weight[self.iter]
        curr_peak = self.peak_objs[self.pk]
        old_image = (devlib.dot(self.shared_image, curr_peak.g_vec) / curr_peak.gdotg)[:, :, :, None] * \
                    curr_peak.g_vec + devlib.dot(self.shared_image, self.rho_hat)[:, :, :, None] * self.rho_hat
        new_image = (devlib.angle(self.ds_image) / curr_peak.gdotg)[:, :, :, None] * curr_peak.g_vec + \
                    devlib.absolute(self.ds_image)[:, :, :, None] * self.rho_hat
        self.shared_image = self.shared_image + beta * (new_image - old_image)


    def to_working_image(self, peak_obj):
        phi = devlib.dot(self.shared_image, peak_obj.g_vec)
        rho = self.shared_image[:, :, :, 0]
        self.ds_image = rho * devlib.exp(1j*phi)

    def progress_trigger(self):
        ornt = self.peak_objs[self.pk].orientation
        print(f'|  iter {self.iter:>4}  '
              f'|  [{ornt[0]:>2}, {ornt[1]:>2}, {ornt[2]:>2}]  '
              f'|  err {self.errs[-1]:0.6f}  '
              f'|  max {self.shared_image[:, :, :, 0].max():0.5g}'
              )

    def get_density(self):
        return self.shared_image[:, :, :, 0]

    def get_distortion(self):
        return self.shared_image[:, :, :, 1:].swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0)

    def flip(self):
        self.shared_image = devlib.flip(self.shared_image, axis=(0, 1, 2))
        self.shared_image[:, :, :, 1:] *= -1
        self.support_obj.flip()

    def shift_to_center(self, ind, cutoff=None):
        shift_dist = -devlib.array(ind) + (self.dims[0]//2)
        self.shared_image = devlib.shift(self.shared_image, shift_dist, axis=(0, 1, 2))
        self.support_obj.support = devlib.shift(self.support_obj.support, shift_dist)


def reconstruction(datafile, **kwargs):
    """
    Reconstructs the image from experiment data in datafile according to given parameters. The results: image.npy, support.npy, and errors.npy are saved in 'saved_dir' defined in kwargs, or if not defined, in the directory of datafile.

    example of the simplest kwargs parameters:
        - algorithm_sequence ='3*(20*ER+180*HIO)+20*ER'
        - shrink_wrap_trigger = [1, 1]
        - twin_trigger = [2]
        - progress_trigger = [0, 20]

    Parameters
    ----------
    datafile : str
        filename of phasing data. Must be either .tif format or .npy type

    kwargs : keyword arguments
        save_dir : str
            directory where results of reconstruction are saved as npy files. If not present, the reconstruction outcome will be save in the same directory where datafile is.
        processing : str
            the library used when running reconstruction. When the 'auto' option is selected the program will use the best performing library that is available, in the following order: cupy, numpy. The 'cp' option will utilize cupy, and 'np' will utilize numpy. Default is auto.
        device : list of int
            IDs of the target devices. If not defined, it will default to -1 for the OS to select device.
        algorithm_sequence : str
            Mandatory; example: "3* (20*ER + 180*HIO) + 20*ER". Defines algorithm applied in each iteration during modulus projection (ER or HIO) and during modulus (error correction or partial coherence correction). The "*" character means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: 'ER', 'ERpc', 'HIO', 'HIOpc' define algorithms used in iterations.
            If the defined algorithm contains 'pc' then during modulus operation a partial coherence correction is applied,  but only if partial coherence (pc) feature is activated. If not activated, the phasing will use error correction instead.
        hio_beta : float
             multiplier used in hio algorithm
        twin_trigger : list
             example: [2]. Defines at which iteration to cut half of the array (i.e. multiply by 0s)
        twin_halves : list
            defines which half of the array is zeroed out in x and y dimensions. If 0, the first half in that dimension is zeroed out, otherwise, the second half.
        shrink_wrap_trigger : list
            example: [1, 1]. Defines when to update support array using the parameters below.
        shrink_wrap_type : str
            supporting "GAUSS" only. Defines which algorithm to use for shrink wrap.
        shrink_wrap_threshold : float
            only point with relative intensity greater than the threshold are selected
        shrink_wrap_gauss_sigma : float
            used to calculate the Gaussian filter
        initial_support_area : list
            If the values are fractional, the support area will be calculated by multiplying by the data array dimensions. The support will be set to 1s to this dimensions centered.
        phm_trigger : list
            defines when to update support array using the parameters below by applaying phase constrain.
        phm_phase_min : float
            point with phase below this value will be removed from support area
        phm_phase_max : float
            point with phase over this value will be removed from support area
        pc_interval : int
            defines iteration interval to update coherence.
        pc_type : str
            partial coherence algorithm. 'LUCY' type is supported.
        pc_LUCY_iterations : int
            number of iterations used in Lucy algorithm
        pc_LUCY_kernel : list
            coherence kernel area.
        lowpass_filter_trigger : list
            defines when to apply lowpass filter using the parameters below.
        lowpass_filter_sw_threshold : float
            used in Gass type shrink wrap when applying lowpass filter.
        lowpass_filter_range : list
            used when applying low resolution data filter while iterating. The values are linespaced for lowpass filter iterations from first value to last. The filter is gauss with sigma of linespaced value. If only one number given, the last value will default to 1.
        average_trigger : list
            defines when to apply averaging. Negative start means it is offset from the last iteration.
        progress_trigger : list of int
            defines when to print info on the console. The info includes current iteration and error.

    """
    error_msg = ver.verify('config_rec', kwargs)
    if len(error_msg) > 0:
        print(error_msg)
        return

    if not os.path.isfile(datafile):
        print('no file found', datafile)
        return

    if 'reconstructions' in kwargs and kwargs['reconstructions'] > 1:
        print('Use cohere_core-ui package to run multiple reconstructions and GA. Proceding with single reconstruction.')
    if 'processing' in kwargs:
        pkg = kwargs['processing']
    else:
        pkg = 'auto'
    if pkg == 'auto':
        try:
            import cupy as cp
            pkg = 'cp'
        except:
            pkg = 'np'
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    else:
        print('supporting cp and np processing')
        return
    set_lib(devlib, False)

    worker = Rec(kwargs, datafile)
    if 'device' in kwargs:
        device = kwargs['device'][0]
    else:
        device = [-1]
    if worker.init_dev(device[0]) < 0:
        return

    if 'continue_dir' in kwargs:
        continue_dir = kwargs['continue_dir']
    else:
        continue_dir = None
    ret_code = worker.init(continue_dir)
    if ret_code < 0:
        return
    ret_code = worker.iterate()
    if ret_code < 0:
        return
    if 'save_dir' in kwargs:
        save_dir = kwargs['save_dir']
    else:
        save_dir, filename = os.path.split(datafile)
    if ret_code == 0:
        worker.save_res(save_dir)

import numpy as np
import os
import cohere_core.utilities.utils as ut
import math
from typing import Union

# tensorflow will use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow for trained model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.activations import sigmoid, tanh


class Mymodel:
    __model = None
    __amp_layer_model = None
    __ph_layer_model = None

    @staticmethod
    def get_model(model_file):
        """ Static access method. """
        if Mymodel.__model == None:
            Mymodel(model_file)
        return Mymodel.__amp_layer_model, Mymodel.__ph_layer_model

    def __init__(self, model_file):
        """ Virtually private constructor. """
        if Mymodel.__model != None:
            raise Exception("This class is a singleton!")
        else:
            # load trained network
            Mymodel.__model = load_model(
                model_file,
                custom_objects={
                    'tf': tf,
                    'loss_comb2_scale': loss_comb2_scale,
                    'sigmoid': sigmoid,
                    'tanh': tanh,
                    'math': math,
                    'combine_complex': combine_complex,
                    'get_mask': get_mask,
                    'ff_propagation': ff_propagation
                })
            model = Mymodel.__model
            # get the outputs from amplitude and phase layers
            Mymodel.__amp_layer_model = Model(inputs=model.input,
                                              outputs=model.get_layer('amp').output)
            Mymodel.__ph_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer('phi').output)


def threshold_by_edge(fp: np.ndarray) -> np.ndarray:
    # threshold by left edge value
    mask = np.ones_like(fp, dtype=bool)
    mask[tuple([slice(1, None)] * fp.ndim)] = 0
    zero = 1e-6
    cut = np.max(fp[mask])
    binary = np.zeros_like(fp)
    binary[(np.abs(fp) > zero) & (fp > cut)] = 1
    return binary


def select_central_object(fp: np.ndarray) -> np.ndarray:
    import scipy.ndimage as ndimage
    zero = 1e-6
    binary = np.abs(fp)
    binary[binary > zero] = 1
    binary[binary <= zero] = 0

    # cluster by connectivity
    struct = ndimage.morphology.generate_binary_structure(fp.ndim,
                                                          1).astype("uint8")
    label, nlabel = ndimage.label(binary, structure=struct)

    # select largest cluster
    select = np.argmax(np.bincount(np.ravel(label))[1:]) + 1

    binary[label != select] = 0

    fp[binary == 0] = 0
    return fp


def get_central_object_extent(fp: np.ndarray) -> list:
    fp_cut = threshold_by_edge(np.abs(fp))
    need = select_central_object(fp_cut)

    # get extend of cluster
    extent = [np.max(s) + 1 - np.min(s) for s in np.nonzero(need)]
    return extent


def get_oversample_ratio(fp: np.ndarray) -> np.ndarray:
    """ get oversample ratio
		fp = diffraction pattern
	"""
    # autocorrelation
    acp = np.fft.fftshift(np.fft.ifftn(np.abs(fp)**2.))
    aacp = np.abs(acp)

    # get extent
    blob = get_central_object_extent(aacp)

    # correct for underestimation due to thresholding
    correction = [0.025, 0.025, 0.0729][:fp.ndim]

    extent = [
        min(m, s + int(round(f * aacp.shape[i], 1)))
        for i, (s, f, m) in enumerate(zip(blob, correction, aacp.shape))
    ]

    # oversample ratio
    oversample = [
        2. * s / (e + (1 - s % 2)) for s, e in zip(aacp.shape, extent)
    ]
    return np.round(oversample, 3)


def Resize(IN, dim):
    ft = np.fft.fftshift(np.fft.fftn(IN)) / np.prod(IN.shape)

    pad_value = np.array(dim) // 2 - np.array(ft.shape) // 2
    pad = [[pad_value[0], pad_value[0]], [pad_value[1], pad_value[1]],
           [pad_value[2], pad_value[2]]]
    ft_resize = ut.adjust_dimensions(ft, pad)
    output = np.fft.ifftn(np.fft.ifftshift(ft_resize)) * np.prod(dim)
    return output


def match_oversample_diff(
    diff: np.ndarray,
    fr: Union[list, np.ndarray, None] = None,
    to: Union[list, np.ndarray, None] = None,
    shape: Union[list, np.ndarray, None] = [64, 64, 64],
):
    """ resize diff to match oversample ratios 
        diff = diffraction pattern
        fr = from oversample ratio
        to = to oversample ratio
        shape = output shape
    """
    # adjustment needed to match oversample ratio
    change = [np.round(f / t).astype('int32') for f, t in zip(fr, to)]
    change = [np.max([1, c]) for c in change]

    diff = ut.binning(diff, change)
    # crop diff to match output shape
    shape_arr = np.array(shape)
    diff_shape_arr = np.array(diff.shape)
    pad_value1 = shape_arr // 2 - diff_shape_arr // 2
    pad_value2 = shape_arr - diff_shape_arr -pad_value1
    pad = [[pad_value1[0], pad_value2[0]], [pad_value1[1], pad_value2[1]],
           [pad_value1[2], pad_value2[2]]]

    output = ut.adjust_dimensions(diff, pad)
    return output, diff.shape


def shift_com(amp, phi):
    from scipy.ndimage.measurements import center_of_mass as com
    from scipy.ndimage.interpolation import shift

    h, w, t = 64, 64, 64
    coms = com(amp)
    deltas = (int(round(h / 2 - coms[0])), int(round(w / 2 - coms[1])),
              int(round(t / 2 - coms[2])))
    amp_shift = shift(amp, shift=deltas, mode='wrap')
    phi_shift = shift(phi, shift=deltas, mode='wrap')
    return amp_shift, phi_shift


def post_process(amp, phi, th=0.1, uw=0):
    if uw == 1:
        # phi = np.unwrap(np.unwrap(np.unwrap(phi,0),1),2)
        phi = unwrap_phase(phi)

    mask = np.where(amp > th, 1, 0)
    amp_out = mask * amp
    phi_out = mask * phi

    mean_phi = np.sum(phi_out) / np.sum(mask)
    phi_out = phi_out - mean_phi

    amp_out, phi_out = shift_com(amp_out, phi_out)

    mask = np.where(amp_out > th, 1, 0)
    amp_out = mask * amp_out
    phi_out = mask * phi_out
    return amp_out, phi_out


# funcions needed in tensorflow model
@tf.function
def combine_complex(amp, phi):
    import tensorflow as tf
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output


@tf.function
def get_mask(input):
    import tensorflow as tf

    mask = tf.where(input >= 0.1, tf.ones_like(input), tf.zeros_like(input))
    return mask


@tf.function
def loss_comb2_scale(Y_true, Y_pred):
    Y_pred = Y_pred / (
        tf.math.reduce_max(Y_pred, axis=(1, 2, 3), keepdims=True) +
        1e-6) * tf.math.reduce_max(Y_true, axis=(1, 2, 3), keepdims=True)
    loss_1 = tf.math.sqrt(loss_sq(Y_true, Y_pred))
    loss_2 = loss_pcc(Y_true, Y_pred)
    a1 = 1
    a2 = 1
    loss_value = (a1 * loss_1 + a2 * loss_2) / (a1 + a2)
    return loss_value


@tf.function
def loss_sq(Y_true, Y_pred):
    top = tf.reduce_sum(tf.math.square(Y_pred - Y_true))
    bottom = tf.reduce_sum(tf.math.square(Y_true))
    loss_value = tf.sqrt(top / bottom)
    return loss_value


@tf.function
def loss_pcc(Y_true, Y_pred):
    pred = Y_pred - tf.reduce_mean(Y_pred)
    true = Y_true - tf.reduce_mean(Y_true)

    top = tf.reduce_sum(pred * true)
    bottom = tf.math.sqrt(tf.reduce_sum(pred**2) * tf.reduce_sum(true**2))
    loss_value = 1 - top / bottom
    return loss_value


@tf.function
def ff_propagation(data):
    '''
    diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        
    '''
    diff = _fourier_transform(data)

    # far-field amplitude
    intensity = tf.math.abs(diff)
    intensity = tf.cast(intensity, tf.float32)
    return intensity


@tf.function
# 3D fourier transform
def _fourier_transform(input):
    import tensorflow as tf
    # fft3d transform with channel unequal to 1
    perm_input = K.permute_dimensions(input, pattern=[4, 0, 1, 2, 3])
    perm_Fr = tf.signal.fftshift(tf.signal.fft3d(
        tf.signal.ifftshift(tf.cast(perm_input, tf.complex64),
                            axes=[-3, -2, -1])),
                                 axes=[-3, -2, -1])
    Fr = K.permute_dimensions(perm_Fr, pattern=[1, 2, 3, 4, 0])
    return Fr


def run_AI(data, model_file, dir):
    """
    Runs AI process.

    Parameters
    ----------
    data : ndarray
        data array

    model_file : str
        file name containing training model

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.
        Result of AI will be saved in dir/results_AI.

    Returns
    -------
    nothing
    """
    print('AI guess')

    # prepare data to make the oversampling ratio ~3
    wos = 3.0
    orig_os = get_oversample_ratio(data)
    # match oversampling to wos
    wanted_os = [wos, wos, wos]
    # match diff os
    new_data, inshape = match_oversample_diff(data, orig_os, wanted_os)
    new_data = new_data[np.newaxis]

    amp_layer_model, ph_layer_model = Mymodel.get_model(model_file)

    preds_amp = amp_layer_model.predict(new_data, verbose=1)

    preds_phi = ph_layer_model.predict(new_data, verbose=1)

    preds_amp, preds_phi = post_process(preds_amp[0, ..., 0],
                                        preds_phi[0, ..., 0],
                                        th=0.1,
                                        uw=0)

    pred_obj = preds_amp * np.exp(1j * preds_phi)

    # match object size with the input data
    pred_obj = Resize(pred_obj, inshape)

    pad_value = np.array(data.shape) // 2 - np.array(pred_obj.shape) // 2
    pad = [[pad_value[0], pad_value[0]], [pad_value[1], pad_value[1]],
           [pad_value[2], pad_value[2]]]
    guess = ut.adjust_dimensions(pred_obj, pad)

    np.save(dir + '/image.npy', guess)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def start_AI(pars, datafile, dir):
    """
    Starts AI process if all conditionas are met.

    Parameters
    ----------
    pars : dict
        parameters for reconstruction

    datafile : str
        file name containing data for reconstruction

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.
        Result of AI will be saved in dir/results_AI.

    Returns
    -------
    ai_dir : str
        directory where results were saved
    """
    if 'AI_trained_model' not in pars:
        print ('no AI_trained_model in config')
        return None
    if not os.path.isfile(pars['AI_trained_model']):
        print('there is no file', pars['AI_trained_model'])
        return None

    if datafile.endswith('tif') or datafile.endswith('tiff'):
        try:
            data = ut.read_tif(datafile)
        except:
            print('could not load data file', datafile)
            return None
    elif datafile.endswith('npy'):
        try:
            data = np.load(datafile)
        except:
            print('could not load data file', datafile)
            return None
    else:
        print('no data file found')
        return None

    # The results will be stored in the directory <experiment_dir>/AI_guess
    ai_dir = dir + '/results_AI'
    if os.path.exists(ai_dir):
        pass
    else:
        os.makedirs(ai_dir)

    run_AI(data, pars['AI_trained_model'], ai_dir)
    return ai_dir
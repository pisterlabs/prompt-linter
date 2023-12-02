from cohere_core.lib.cohlib import cohlib
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage

class cplib(cohlib):
    def array(obj):
        return cp.array(obj)

    def dot(arr1, arr2):
        return cp.dot(arr1, arr2)

    def set_device(dev_id):
        cp.cuda.Device(dev_id).use()

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return cp.asnumpy(arr)

    def from_numpy(arr):
        return cp.array(arr)

    def save(filename, arr):
        cp.save(filename, arr)

    def load(filename):
        return cp.load(filename)

    def dtype(arr):
        return arr.dtype

    def size(arr):
        return arr.size

    def hasnan(arr):
        return cp.any(cp.isnan(arr))

    def copy(arr):
        return cp.copy(arr)

    def random(shape, **kwargs):
        import time
        import os

        seed = np.array([time.time() * 10000 * os.getpid(), os.getpid()])
        rs = cp.random.RandomState(seed=seed)
        return cp.random.random(shape, dtype=cp.float32) + 1j * cp.random.random(shape, dtype=cp.float32)

    def fftshift(arr):
        return cp.fft.fftshift(arr)

    def ifftshift(arr):
        return cp.fft.ifftshift(arr)

    def shift(arr, sft):
        sft = [int(s) for s in sft]
        return cp.roll(arr, sft)

    def fft(arr):
        return cp.fft.fftn(arr, norm='forward')

    def ifft(arr):
        return cp.fft.ifftn(arr, norm='forward')

    def fftconvolve(arr1, arr2):
        return cupyx.scipy.ndimage.convolve(arr1, arr2)

    def where(cond, x, y):
        return cp.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.shape

    def absolute(arr):
        return cp.absolute(arr)

    def sqrt(arr):
        return cp.sqrt(arr)

    def square(arr):
        return cp.square(arr)

    def sum(arr, axis=None):
        sm = cp.sum(arr, axis)
        if axis is None:
            return sm.tolist()
        return sm

    def real(arr):
        return cp.real(arr)

    def imag(arr):
        return cp.imag(arr)

    def amax(arr):
        return cp.amax(arr)

    def argmax(arr, axis=None):
        return cp.argmax(arr, axis)

    def unravel_index(indices, shape):
        return cp.unravel_index(indices, shape)

    def maximum(arr1, arr2):
        return cp.maximum(arr1, arr2)

    def ceil(arr):
        return cp.ceil(arr)

    def fix(arr):
        return cp.fix(arr)

    def round(val):
        return cp.round(val)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return cp.angle(arr)

    def flip(arr, axis=None):
        return cp.flip(arr, axis)

    def tile(arr, rep):
        return cp.tile(arr, rep)

    def full(shape, fill_value, **kwargs):
        return cp.full(shape, fill_value)

    def expand_dims(arr, axis):
        return cp.expand_dims(arr, axis)

    def squeeze(arr):
        return cp.squeeze(arr)

    def gaussian(shape, sigma, **kwargs):
        from functools import reduce
        import operator

        n_el = reduce(operator.mul, shape)
        inarr = cp.zeros((n_el))
        inarr[int(n_el / 2)] = 1.0
        inarr = cp.reshape(inarr, shape)
        gaussian = cupyx.scipy.ndimage.gaussian_filter(inarr, sigma)
        return gaussian / cp.sum(gaussian)

    def gaussian_filter(arr, sigma, **kwargs):
        return cupyx.scipy.ndimage.gaussian_filter(arr, sigma)

    def center_of_mass(inarr):
        return cupyx.scipy.ndimage.center_of_mass(cp.absolute(inarr))

    def meshgrid(*xi):
        return cp.meshgrid(*xi)

    def exp(arr):
        return cp.exp(arr)

    def conj(arr):
        return cp.conj(arr)

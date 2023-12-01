# coding: utf-8

# Implementation of LR-GAN in Theano and Lasagne
# author : @Sentient07
# Reference : https://arxiv.org/pdf/1703.01560.pdf

import os
import cPickle
import tarfile
import time
import argparse

import numpy as np
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.layers import (Conv2DLayer, DropoutLayer, batch_norm, ReshapeLayer, InputLayer,
                            DenseLayer, NonlinearityLayer,ElemwiseSumLayer, MergeLayer,
                            TransformerLayer, MergeLayer, Layer, Gate, SliceLayer, PadLayer)
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh, identity
from lasagne.objectives import binary_crossentropy
from lasagne.layers.dense import NINLayer
from lasagne.layers.pool import GlobalPoolLayer
from lasagne.utils import as_tuple
from lasagne import init

import cv2
from IPython.display import display, Image as im
from PIL import Image
import pylab
import matplotlib.pyplot as plt


lr = lasagne.nonlinearities.LeakyRectify(leakiness=0.2)
rect = lasagne.nonlinearities.rectify
w1 = lasagne.init.Normal(0.05)
num_units = 100


class Dconv2DLayer(lasagne.layers.Layer):
    '''
    This was taken from Jän Schutler's implementation
    '''

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=rect, **kwargs):
        super(Dconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Normal(0.05),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = tensor.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)


class DotLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), nonlinearities=None, **kwargs):
        super(DotLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.nonlinearities = nonlinearities

    def get_output_for(self, input, **kwargs):
        return self.nonlinearities(tensor.dot(input, self.W))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class ComplimentLayer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(ComplimentLayer, self).__init__(incoming, **kwargs)
        self.incoming = incoming

    def get_ouput_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input):
        ones = tensor.ones(input.shape)
        return ones - input


class MinibatchLayer(lasagne.layers.Layer):
    '''
    Code borrowed from OpenAI
    '''

    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (tensor.exp(self.log_weight_scale)/tensor.sqrt(tensor.sum(tensor.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = tensor.tensordot(input, self.W, [[1], [0]])
        abs_dif = (tensor.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * tensor.eye(input.shape[0]).dimshuffle(0,'x',1))

        if init:
            mean_min_abs_dif = 0.5 * tensor.mean(tensor.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-tensor.log(mean_min_abs_dif).dimshuffle(0,'x'))]
        
        f = tensor.sum(tensor.exp(-abs_dif),axis=2)

        if init:
            mf = tensor.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return tensor.concatenate([input, f], axis=1)


class WeightNormLayer(lasagne.layers.Layer):

    '''
    Code borrowed from Tim Salliman's implementation of GAN. 
    '''

    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), train_g=False, init_stdv=1., nonlinearity=rect, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.init_stdv = init_stdv
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=train_g)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        #incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            if isinstance(incoming, Dconv2DLayer):
                W_axes_to_sum = (0,2,3)
                W_dimshuffle_args = ['x',0,'x','x']
            else:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/tensor.sqrt(1e-6 + tensor.sum(tensor.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / tensor.sqrt(1e-6 + tensor.sum(tensor.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = tensor.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            inv_stdv = self.init_stdv/tensor.sqrt(tensor.mean(tensor.square(input), self.axes_to_sum))
            input *= inv_stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m*inv_stdv), (self.g, self.g*inv_stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)

def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)


class ExposedLSTMLayer(MergeLayer):
    '''
    Edited version of LSTM layer to give the h and c vector of current state
    Author Joel Moniz
    '''

    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=tanh),
                 outgate=Gate(),
                 nonlinearity=tanh,
                 cell_init=lasagne.init.Constant(0.),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):


        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(ExposedLSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], 2*self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], 2*self.num_units

    def get_output_for(self, inputs, **kwargs):

        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = tensor.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = tensor.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = tensor.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = tensor.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            input = tensor.dot(input, W_in_stacked) + b_stacked

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = tensor.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + tensor.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = tensor.switch(mask_n, cell, cell_previous)
            hid = tensor.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = tensor.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = tensor.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = tensor.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            cell_out = cell_out[-1]
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            cell_out = cell_out.dimshuffle(1, 0, 2)
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
                cell_out = cell_out[:, ::-1]

        return tensor.concatenate([cell_out, hid_out], axis=-1)

   
class ElemwiseMergeLayer(MergeLayer):
    '''
    This layer has been borrowed from lasagne and some modifications has been made
    The special functionality of this is, it makes theano broadcast similar to numpy and the
    user has to give which of the two array has to be broadcasted. 
    I have made this broadcast only the second dimension as per my requirement. For general purpose, 
    the dimensions that has to be made broadcastable has to be specified as well

    Usage :
    When two unequal arrays(only the second dimension, or 1st in pythonic terms) are passed for merging,
    the first dimension of specified array is made broadcastable.

    ElemwiseMergeLayer([arr1, arr2, broadcastable=1]) would make arr2's first dimension broadcastable
    '''

    def __init__(self, incomings, merge_function, broadcastable=None, cropping=None, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping
        self.broadcastable = broadcastable

    def get_output_shape_for(self, input_shapes):

        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple([max(i, j) for i, j in zip(input_shapes[0], input_shapes[1])])

        return output_shape

    def get_output_for(self, inputs, **kwargs):
        # modify broadcasting pattern.
        if self.broadcastable:
            if self.broadcastable == 1:
                inputs[0] = tensor.addbroadcast(inputs[0], 1)
            elif self.broadcastable == 2:
                inputs[1] = tensor.addbroadcast(inputs[1], 1)

        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


def normalise(im):
    im = im / np.float32(127.5) - np.float32(1.)
    return im.astype(np.float32)

def denormalise(im, tanh=True):
    im = ((im + np.float32(1.0)) * np.float32(127.5)).astype(np.uint8)
    if tanh:
        im = im.transpose(0, 2, 3, 1)
    return im

def gen_bg(back_noise):
    net1 = batch_norm(DenseLayer(back_noise, 8192, W=w1))
    net1_r = ReshapeLayer(net1, ([0], 512, 4, 4))
    net2 = batch_norm(Dconv2DLayer(net1_r, 256, 3, stride=1, pad=1))
    net3 = batch_norm(Dconv2DLayer(net2, 128, 3, stride=2, pad=1))
    print ("Generator output:", net3.output_shape)
    # 8 * 8
    net4 = batch_norm(Dconv2DLayer(net3, 64, 3, stride=2, pad=1))
    print ("Generator output:", net4.output_shape)
    # 16 * 16
    net5 = Dconv2DLayer(net4, 3, 3, stride=2, pad=1, nonlinearity=tanh)
    print ("Generator output:", net5.output_shape)
    # 3 * 32 * 32
    return net5

# Foreground Generator
def gen_fc(y_t):
    net1 = batch_norm(DenseLayer(y_t, 8192, W=w1))
    net1_r = ReshapeLayer(net1, ([0], 512, 4, 4))
    net2 = batch_norm(Dconv2DLayer(net1_r, 512, 3, pad=1))
    print ("Generator output:", net2.output_shape)
    net3 = batch_norm(Dconv2DLayer(net2, 256, 3, stride=2, pad=1))
    print ("Generator output:", net3.output_shape)
    # 8 * 8
    net4 = batch_norm(Dconv2DLayer(net3, 128, 3, stride=2, pad=1))
    print ("Generator output:", net4.output_shape)
    # 16 * 16
    return net4

def gen_fi(fc_out):
    net1 = batch_norm(Dconv2DLayer(fc_out, 3, 3, stride=2, pad=1))
    return net1

# Foreground mask
def gen_fmask(fc_out):
    net1 = batch_norm(Dconv2DLayer(fc_out, 1, 3, stride=2, pad=1, nonlinearity=sigmoid))
    return net1

# Discriminator
def build_desc(inp):
    net0= InputLayer((None, 3, 32, 32), input_var=inp)
    # 32 * 32
    net1 = batch_norm(Conv2DLayer(net0, 96, 3, stride=2, pad='same', W=w1, nonlinearity=lr))
    print ("Disc output:", net1.output_shape)
    # 16 * 16
    
    net2 = batch_norm(Conv2DLayer(net1, 192, 3, stride=2, pad='same', W=w1, nonlinearity=lr))
    print ("Disc output:", net2.output_shape)
    # 8 * 8
    net3 = batch_norm(Conv2DLayer(net2, 192, 3, pad='same', W=w1, nonlinearity=lr))
    print ("Disc output:", net3.output_shape)
    # 6 * 6
    net4 = batch_norm(NINLayer(net3, 192, W=w1, nonlinearity=lr))
    net5 = batch_norm(NINLayer(net4, 192, W=w1, nonlinearity=lr))
    net6 = GlobalPoolLayer(net5)
    print ("Disc output:", net6.output_shape)

    net7 = DenseLayer(net6, 1, W=w1, nonlinearity=sigmoid)
    print ("Discriminator output:", net7.output_shape)
    return net7, net6

def construct_gen(noise_1, noise_2, batch_size=10):
    # There are two time steps considered for this model, so two LSTMs
    # Reshape noises
    noise1_rshp = noise_1.dimshuffle(0, 'x', 1)
    noise2_rshp = noise_2.dimshuffle(0, 'x', 1)
    lstm1_inp = InputLayer((None, 1, 100), input_var=noise1_rshp)
    lstm2_inp = InputLayer((None, 1, 100), input_var=noise2_rshp)

    lstm2 = ExposedLSTMLayer(lstm2_inp, 100)
    lstm2_h = SliceLayer(lstm2, indices=slice(num_units, None), axis=-1)
    lstm2_reshape = ReshapeLayer(lstm2_h, (batch_size, 100))


    print("LSTM2's output is " + str(lstm2_reshape.output_shape))

    build_bg = gen_bg(lstm1_inp)
    build_gfc = gen_fc(lstm2_reshape)
    build_gif = gen_fi(build_gfc)
    build_gfmask = gen_fmask(build_gfc)
    
    # Affine transformation and pasting with bg
    a_t = DenseLayer(lstm2_reshape, num_units=6, W=w1) #6 dim output
    m_t_hat = NonlinearityLayer(PadLayer(TransformerLayer(build_gfmask, a_t, downsample_factor=2), 8), nonlinearity=tanh)
    f_t_hat = NonlinearityLayer(PadLayer(TransformerLayer(build_gif, a_t, downsample_factor=2), 8), nonlinearity=tanh)
    
    prior = ElemwiseMergeLayer([m_t_hat, f_t_hat], merge_function=tensor.mul, broadcastable=1)
    posterior = ElemwiseMergeLayer([ComplimentLayer(m_t_hat), build_bg], merge_function=tensor.mul, broadcastable=1)

    gen_image = ElemwiseSumLayer([prior, posterior])

    return gen_image


def log_sum_exp(x, axis=1):
    m = tensor.max(x, axis=axis)
    return m + tensor.log(tensor.sum(tensor.exp(x-m.dimshuffle(0,'x')), axis=axis))


def plot_image(generated_image, epoch_val, save_path):
    fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
    # axes1.set_axis_off()
    # axes1.imshow(reshaped_image[10])
    # fig.imshow(reshaped_image[10])
    gen_trans_image = denormalise(generated_image, tanh=True)
    img_h, img_w = (32, 32)
    grid_shape = 8
    grid_pad = 5
    grid_h = img_h * grid_shape + grid_pad * (grid_shape - 1)
    grid_w = img_w * grid_shape + grid_pad * (grid_shape - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(gen_trans_image)))
            axes1[j][k].set_axis_off()
            deproc = gen_trans_image[i:i+1][0]
            axes1[j][k].imshow((deproc))
        row = (j // grid_shape) * (img_h + grid_pad)
        col = (j % grid_shape) * (img_w + grid_pad)

        img_grid[row:row+img_h, col:col+img_w, :] = deproc
    fig.savefig(str(epoch_val) + ".jpg")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1)
    parser.add_argument('--seed_data', default=1)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--initial_lr', type=float, default=0.004)
    parser.add_argument('--data_dir', type=str, default='/Users/Ramana/projects/data/cifar-100-python/train')
    parser.add_argument('--num_epochs', default=1000)
    parser.add_argument('--output_path', type=str, default='/Users/Ramana/projects/data/cifar-100-python/')

    args = parser.parse_args()
    print(args)

    # Path to where the train file is
    cifar_data = unpickle(args.data_dir)
    reshaped_image = cifar_data['data'].reshape(50000, 3, 32, 32)[np.random.randint(50000, size=1000), :, :, :]
    transposed_image = reshaped_image.transpose(0, 2, 3, 1)
    
    # Setting learning rate
    l_r = theano.shared(lasagne.utils.floatX(args.initial_lr))
    batch_number = int(round(len(reshaped_image) / args.batch_size))

    # Noise assignment
    rng = np.random.RandomState(args.seed)
    theano_rng = RandomStreams(rng.randint(2 ** 15))
    lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
    noise_dim = (args.batch_size, 100)
    noise_fg = theano_rng.normal(size=noise_dim)
    noise_bg = theano_rng.normal(size=noise_dim)
    x_inp = tensor.tensor4('x_inp', dtype='float32')

    # Build the network
    gen = construct_gen(noise_bg, noise_fg, batch_size=args.batch_size)
    disc, features = build_desc(x_inp)

    # Output of discriminator with original images. training phase, so non deterministic
    disc_out = lasagne.layers.get_output(disc, x_inp, deterministic=False)
    gen_out = lasagne.layers.get_output(gen)
    disc_over_gen = lasagne.layers.get_output(disc, gen_out)
    true_features = lasagne.layers.get_output(features, x_inp)
    fake_features = lasagne.layers.get_output(features, gen_out)
    # Loss functions. 1) Gen's 2) Disc's for predicting correctly 3) Feature matching loss
    false_loss = log_sum_exp(disc_over_gen)
    truth_loss = log_sum_exp(disc_out)
    disc_loss = -0.5 * tensor.mean(truth_loss) + 0.5 * tensor.mean(tensor.nnet.softplus(truth_loss)) + 0.5 * tensor.mean(tensor.nnet.softplus(false_loss))
    gen_loss = tensor.mean(abs(tensor.mean(true_features, axis=0) - tensor.mean(fake_features, axis=0)))

    # Fetch and Update params
    gen_params = lasagne.layers.get_all_params(gen, trainable=True)
    disc_params = lasagne.layers.get_all_params(disc, trainable=True)

    disc_updates = lasagne.updates.adam(disc_loss, disc_params, learning_rate=l_r, beta1=0.5)
    gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=l_r, beta1=0.5)


    # Compile theano functions
    print("Starting to compile functions")
    train_disc = theano.function([x_inp], [disc_loss], updates=disc_updates)
    train_gen = theano.function(inputs=[x_inp], outputs=None, updates=gen_updates)
    print("done compiling training function")
    # Functions for generating
    generate_im_func = theano.function([], gen_out)
    print("done all compilation")

    for ep in range(args.num_epochs):
        # create batches problematicand train
        # Batch size is 128, so 220
        begin = time.time()
        offset = 0
        loss = 0

        for num_iter, bn in enumerate(range(batch_number)):
            # Randomly shuffle inputs
            shuffled_inputs = reshaped_image[np.random.randint(len(reshaped_image), size=len(reshaped_image)), :, :, :]
            inputs = normalise(shuffled_inputs[offset:offset + args.batch_size])
            loss += np.array(train_disc(inputs))
            train_gen(inputs)
            offset += args.batch_size
        end = time.time()
        print("LR GAN loss is " + str(loss / np.float32(batch_number)))
        print("Finished {} of {}. Time taken {:.3f}s".format(ep + 1, args.num_epochs,  end - begin))

        # PLot the image generated for every 50epochs
        if ep % 50 == 0:
            np.savez('cifar_gen' + str(ep) + '.npz', *lasagne.layers.get_all_param_values(gen))
            np.savez('cifar_disc' + str(ep) + '.npz', *lasagne.layers.get_all_param_values(disc))
            generated_image = generate_im_func()
            plot_image(generated_image, ep, args.output_path)

        # Degrading the learning rate
        if ep >= args.num_epochs // 4:
            progress = float(ep) / args.num_epochs
            l_r.set_value(lasagne.utils.floatX(args.initial_lr*2*(1 - progress)))

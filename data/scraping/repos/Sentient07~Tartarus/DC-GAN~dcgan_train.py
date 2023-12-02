
# Implementation of Deep Convolution Generative Adversarial Network on flickr dataset
# This demo is to check the adaptability of DCGAN with higher resolution images
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers import (Conv2DLayer, MaxPool2DLayer, BatchNormLayer, DenseLayer,
                            InputLayer, Deconv2DLayer, NonlinearityLayer, batch_norm, ReshapeLayer)
from lasagne.init import Normal, Uniform
from lasagne.nonlinearities import tanh, LeakyRectify, sigmoid
import numpy as np

# Testing cell
import h5py
from IPython.display import display, Image as im
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import os
import sys
import cv2
import time
import argparse
from ..extras import normalized

# This path is very specific to my machine
PATH = '/home/neena/Ramana/Dataset/Flickr dataset/flickr30k-images/'
npy_path = '/home/neena/Ramana/Dataset/Flickr dataset/'
# The alternative loss function is followed from Improved ways of training GANs
# If any license exists, the research work is from OpenAI

def reshape(im):
        return cv2.resize(im, (256, 256))

def plot_sample_image(sample_image, normalize=False):
    cv2_im  = reshape(cv2.imread(sample_image).astype('uint8') / np.float(255.0))
    if normalized:
        cv2_im = normalized(cv2_im)
    print(cv2_im.shape)
    plt.imshow(reshaped_im)
    plt.figure(1)

def load_dataset(path):
    data_im_shp = (31783, 3, 224, 224)
    data = []
    for filename in os.listdir(PATH):
        if filename.endswith(".jpg"):
            data.append(np.rollaxis(reshape(cv2.imread(PATH + filename).astype('uint8') / np.float(255.0)), -1))

    # Saving because it's used for other experiments
    np.save(path + '256numpy.npy', np.asarray(data, dtype='float32'))


def prepare_dataset(path):

    from sklearn.model_selection import train_test_split

    load_path = '/home/neena/Ramana/Dataset/Flickr dataset/'
    load_file = '256numpy.npy'
    if not os.path.isfile(load_path + load_file):
        load_dataset(load_path)
    data = np.load(load_path + load_file, mmap_mode='r+')
    train_data, gen_data = train_test_split(data, test_size=0.1)
    return train_data, gen_data

# The binary cross entropy in Theano gives inconsistent floating points
# It always returns float64, and in a quest to save memory on GPU, I'm using this
# version for float32.
# See this for more info : https://github.com/Theano/Theano/pull/5855

def binary_crossentropy(output, target):
    from lasagne.objectives import align_targets

    output, target = align_targets(output, target)
    one = np.float32(1.0)
    return -(target * tensor.log(output) + (one - target) * tensor.log(one - output))

def build_generator(inp):
    net = InputLayer((None, 100), input_var=inp)
    net = batch_norm(DenseLayer(net, 1024))
    net = ReshapeLayer(DenseLayer(net, 4096), ([0], 1024, 2, 2))
    print ("Generator output:", net.output_shape)
    # 2 * 2
    net = batch_norm(Deconv2DLayer(net, 512, 4, stride=4))
    print ("Generator output:", net.output_shape)
    # 8 * 8
    net = batch_norm(Deconv2DLayer(net, 256, 4, stride=4))
    print ("Generator output:", net.output_shape)
    # 32 * 32
    net = batch_norm(Deconv2DLayer(net, 128, 4, stride=2, crop=1))
    print ("Generator output:", net.output_shape) 
    # 64 * 64
    net = batch_norm(Deconv2DLayer(net, 64, 4, stride=2, crop=1))
    print ("Generator output:", net.output_shape)
    # 128 * 128
    net = Deconv2DLayer(net, 3, 4, stride=2, crop=1, nonlinearity=tanh)
    print ("Generator output:", net.output_shape)
    # 3 * 256 * 256
    return net

def build_disc(inp):
    lr = LeakyRectify(leakiness=0.2)
    net= InputLayer((None, 3, 256, 256), input_var=inp)
    # 256 * 256
    net = batch_norm(Conv2DLayer(net, 64, 4, stride=2, pad=1, nonlinearity=lr))
    # 128 * 128
    net = batch_norm(Conv2DLayer(net, 128, 4, stride=2, pad=1, nonlinearity=lr))
    # 64 * 64
    net = batch_norm(Conv2DLayer(net, 256, 4, stride=2, pad=1, nonlinearity=lr))
    # 32 * 32
    net = batch_norm(Conv2DLayer(net, 512, 4, stride=4, nonlinearity=lr))
    # 8 * 8
    net = batch_norm(Conv2DLayer(net, 512, 4, stride=4, nonlinearity=lr))
    # 2 * 2
    net = batch_norm(DenseLayer(net, 4096, nonlinearity=lr))
    
    net = batch_norm(DenseLayer(net, 1024, nonlinearity=lr))

    net = DenseLayer(net, 1, nonlinearity=sigmoid)
    print ("Discriminator output:", net.output_shape)
    return net


def main(train_data, out_dir='~/', random_seed=None, batch_size=10, num_epochs=2000, initial_lr=2e-04):

    lr = theano.shared(np.float32(initial_lr))
    batch_size = 10
    batch_number = int(round(len(train_data)/batch_size))
    noise = np.zeros((batch_size, 100))
    num_epochs = 100

    noise_var = tensor.matrix('noise', dtype='float32')
    x_inp = tensor.tensor4('x_inp', dtype='float32')

    load_gen = build_generator(noise_var)
    load_disc = build_disc(x_inp)

    # Output of discriminator alone
    disc = lasagne.layers.get_output(load_disc)
    # Output of Generator combined with discriminator. This for training Generator
    disc_over_gen = lasagne.layers.get_output(load_disc, lasagne.layers.get_output(load_gen))

    # Loss functions
    generator_loss = binary_crossentropy(disc_over_gen, 1).mean()
    discriminator_loss = (binary_crossentropy(disc, 1) + binary_crossentropy(disc_over_gen, 0)).mean()
    # Update params
    gen_params = lasagne.layers.get_all_params(load_gen, trainable=True)
    disc_params = lasagne.layers.get_all_params(load_disc, trainable=True)

    # Gen updates
    gen_updates = lasagne.updates.adam(generator_loss, gen_params, beta1=np.float32(0.5), learning_rate=lr)

    overall_update = gen_updates.update(lasagne.updates.adam(
                                        discriminator_loss, disc_params, beta1=np.float32(0.5), 
                                        learning_rate=lr))

    srng = RandomStreams(seed=np.random.randint(random_seed, size=6))

    noise = srng.uniform((batch_size, 100))
    print(noise.dtype)
    # Compile theano functions
    # Functions for training
    print("Starting to compile functions")

    train_func = theano.function([x_inp, noise_var], [(disc > 0.5).mean(), (disc_over_gen < 0.5).mean()],
                                 updates=overall_update)

    print("done compiling training functions")
    # Functions for generating
    generate_im_func = theano.function([noise_var], lasagne.layers.get_output(load_gen, deterministic=True))
    print("done all compilation")

    for ep in range(num_epochs):
        # create batches and train
        # Batch size is 128, so 220
        gen_loss = []
        disc_loss = []
        begin = time.time()
        print("going into  batch iteration")
        offset = 0
        for bn in range(batch_number):
            inputs = train_data[offset:offset + batch_size]
            if not random_seed:
                noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            gan_loss.append(train_func(inputs, noise))
            offset += batch_size
        end = time.time()
        print("DC GAN loss is " + str(np.mean(gan_loss)))
        print("Finished {} of {}. Time taken {:.3f}s".format(ep + 1, num_epochs,  end - begin))

        # Decaying the learning after half of the epoch is over
        # this technique has been adapted from Jan Schutler's GAN gist
        if ep >= num_epochs // 2:
            progress = float(ep) / num_epochs
            lr.set_value(lasagne.utils.floatX(initial_lr * 2 * (1 - progress)))

    np.savez(out_dir + 'dcgan_gen.npz', *lasagne.layers.get_all_param_values(load_gen))
    np.savez(out_dir + 'dcgan_disc.npz', *lasagne.layers.get_all_param_values(load_disc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default=npy_path)
    parser.add_argument('--data_dir', type=str, default=npy_path)
    parser.add_argument('--num_epoch', type = int, default = 20)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr', type=float, default=2e-04) # learning rate for generator
    # Low batchsize as i don't have access to a good GPU
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    print(args)
    train_data, get_data = prepare_dataset(args.data_dir)
    main(train_data, args.out_dir, args.seed, args.batch_size, args.num_epochs, args.lr)

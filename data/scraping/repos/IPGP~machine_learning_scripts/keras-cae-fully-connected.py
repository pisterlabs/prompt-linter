#!/usr/bin/env python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

from coherence import get_coherence
from keras.layers import Input, Dense, MaxPooling2D
from keras.layers import UpSampling2D, Conv2D, Reshape
from keras.models import Model


# Get design matrix
X, frequency, time = get_coherence(decimate=3)
X = np.log10(np.sqrt(X.T))
X = X[:12800, :16]
print(X.shape)
x_shape = X.shape

# Tensor shape: [1 sample, image width, image height, 1 channel
# (entropy/spectral power)]
X = X.reshape([1, X.shape[0], X.shape[1], 1])

# Input layer
input_img = Input(shape=(x_shape[0], x_shape[1], 1))

nfilters1, nfilters2, nfilters3 = 3, 10, 10
filtersize1, filtersize2, filtersize3 = 3, 3, 3
nsize_dense = 3
pooling = (2, 1)

# Create graph
conv1 = Conv2D(nfilters1, filtersize1, activation='tanh', padding='same')

maxp1 = MaxPooling2D(pooling, padding='same')
conv2 = Conv2D(nfilters2, filtersize2, activation='tanh', padding='same')
maxp2 = MaxPooling2D(pooling, padding='same')
conv3 = Conv2D(nfilters3, filtersize3, activation='tanh', padding='same')
maxp3 = MaxPooling2D(pooling, padding='same')

flat = Reshape((x_shape[0]//8*x_shape[1]//1, nfilters3))
dense = Dense(nsize_dense, activation='tanh')
udense = Dense(nfilters3)
uflat = Reshape((x_shape[0]//8, x_shape[1]//1, nfilters3))

dconv4 = Conv2D(nfilters3, filtersize3, activation='tanh', padding='same')
dmaxp3 = UpSampling2D(pooling)
dconv3 = Conv2D(nfilters2, filtersize3, activation='tanh', padding='same')
dmaxp2 = UpSampling2D(pooling)
dconv2 = Conv2D(nfilters1, filtersize2, activation='tanh', padding='same')
dmaxp1 = UpSampling2D(pooling)
dconv1 = Conv2D(1, filtersize1, activation='tanh', padding='same')

x = conv1(input_img)
x = maxp1(x)
x = conv2(x)
x = maxp2(x)
x = conv3(x)
encoded = maxp3(x)
print(encoded)
h = flat(encoded)
he = dense(h)
h = udense(he)
encoded = uflat(h)
x = dconv4(encoded)
x = dmaxp3(x)
x = dconv3(x)
x = dmaxp2(x)
x = dconv2(x)
x = dmaxp1(x)
decoded = dconv1(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')
autoencoder.fit(X, X, epochs=10000, verbose=2)

conv1 = Conv2D(nfilters1, filtersize1, activation='tanh', padding='same',
               weights=autoencoder.layers[1].get_weights())
maxp1 = MaxPooling2D(pooling, padding='same')
conv2 = Conv2D(nfilters2, filtersize2, activation='tanh', padding='same',
               weights=autoencoder.layers[3].get_weights())
maxp2 = MaxPooling2D(pooling, padding='same')
conv3 = Conv2D(nfilters3, filtersize3, activation='tanh', padding='same',
               weights=autoencoder.layers[5].get_weights())
maxp3 = MaxPooling2D(pooling, padding='same')

flat = Reshape((x_shape[0]//8*x_shape[1]//1, nfilters3))
dense = Dense(nsize_dense, activation='tanh',
              weights=autoencoder.layers[8].get_weights())
udense = Dense(nfilters3,
               weights=autoencoder.layers[9].get_weights())
uflat = Reshape((x_shape[0]//8, x_shape[1]//1, nfilters3))

dconv4 = Conv2D(nfilters3, filtersize3, activation='tanh', padding='same',
                weights=autoencoder.layers[11].get_weights())
dmaxp3 = UpSampling2D(pooling)
dconv3 = Conv2D(nfilters2, filtersize3, activation='tanh', padding='same',
                weights=autoencoder.layers[13].get_weights())
dmaxp2 = UpSampling2D(pooling)
dconv2 = Conv2D(nfilters1, filtersize2, activation='tanh', padding='same',
                weights=autoencoder.layers[15].get_weights())
dmaxp1 = UpSampling2D(pooling)

x = conv1(input_img)
x = maxp1(x)
x = conv2(x)
x = maxp2(x)
x = conv3(x)
encoded = maxp3(x)
h = flat(encoded)
he = dense(h)
h = udense(he)
encoded = uflat(h)
x = dconv4(encoded)
x = dmaxp3(x)
x = dconv3(x)
x = dmaxp2(x)
x = dconv2(x)
almost = dmaxp1(x)
print(almost)

seener = Model(input_img, almost)
# seener.summary()
decoded_img = seener.predict(X)

print(decoded_img.shape)

fig, ax = plt.subplots(4, 1, figsize=(10, 10))

for i in range(3):
    ax[i].imshow(decoded_img[..., i].reshape(x_shape).T, aspect='auto',
                 origin='lower')

ax[-1].imshow(X.reshape(x_shape).T, aspect='auto', origin='lower')

plt.show()
[a.set_axis_off() for a in ax]
plt.savefig('cae-sep-keras.pdf', bbox_inches='tight')

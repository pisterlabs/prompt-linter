# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Generative Adversarial Networks (GANs)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Learn about generative and discriminative models that make up generative adversarial networks 
# MAGIC  - Apply dropout
# MAGIC  
# MAGIC ### Discriminator
# MAGIC 
# MAGIC A discriminative model, at its core is a Real/fake classifier. It takes counterfeits and real values as input, and predicts probability of counterfeit. 
# MAGIC  
# MAGIC  
# MAGIC ### Generator
# MAGIC 
# MAGIC A generative model captures the data distribution, takes noise vecotrs from latent space as input, outputs a counterfeit.
# MAGIC 
# MAGIC 
# MAGIC ### GANs
# MAGIC 
# MAGIC <img src="https://miro.medium.com/max/3000/1*t82vgL9KcDVpT4JqCb9Q4Q.png" width=1000>

# COMMAND ----------

# MAGIC %md This was a very original architecture in deep learning when it was first released by <a href="https://arxiv.org/pdf/1406.2661.pdf" target="_blank">Ian Goodfellow et al in 2014</a>. It was the first network that contains a generator and a discriminator. The two models compete against each other during training of GANs. GANs eventually generates fairly realistic synthetic images as the discriminator becomes smarter at distinguishing between real and fake images. 
# MAGIC 
# MAGIC The algorithm:
# MAGIC - G takes noise as input, outputs a counterfeit
# MAGIC - D takes counterfeits and real values as input, outputs P(counterfeit)
# MAGIC 
# MAGIC The following techniques can help **prevent overfitting**, 
# MAGIC - Alternate k steps of optimizing D and one step of optimizing G
# MAGIC - Start with k of at least 5
# MAGIC - Use *log(1 - D(G(z)))* to provide stronger, non-saturated gradients
# MAGIC 
# MAGIC <img src="https://media-exp1.licdn.com/dms/image/C5112AQGWsO2ZFbKnYQ/article-inline_image-shrink_1000_1488/0/1520192659145?e=1647475200&v=beta&t=06VrAMeZgpmcvw0K-bQV892ecuBlWJggwv045e4Jz8Q" style="width:1000px">
# MAGIC 
# MAGIC GANs can be used in generating art, deep fakes, up-scaling graphics, and astronomy research. For example, we can use GANs to generate synthetic handwritten images, resembling the MNIST dataset. 
# MAGIC 
# MAGIC <img src = "https://tensorflow.org/images/gan/dcgan.gif" width=600>
# MAGIC 
# MAGIC 
# MAGIC As a follow-up, we highly recommend this <a href="https://www.coursera.org/specializations/generative-adversarial-networks-gans?" target="_blank">GANs</a> course from coursera. There are other very interesting applications of generative models, such as <a href="https://openai.com/blog/glow/" target="_blank">Glow</a> from OpenAI and <a href="https://ai.facebook.com/blog/wav2vec-unsupervised-speech-recognition-without-supervision/" target="_blank">speech recognition</a> from Facebook AI.

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we will be using the <a href="https://github.com/zalandoresearch/fashion-mnist" target="_blank">Fashion MNIST</a> dataset that you have seen before in the SHAP for CNNs lab! <br>
# MAGIC 
# MAGIC Our goal is to create synthetic images of these clothing items using GANs.
# MAGIC 
# MAGIC <img src="https://tensorflow.org/images/fashion-mnist-sprite.png" width=500>

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

import tensorflow as tf

### Load data - we don't care about the testing sets since there is no concept of "testing" in GAN
### Our sole goal is to generate fake images that look like the real images
((X_train, y_train), (_, _)) = tf.keras.datasets.fashion_mnist.load_data()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at a sample image.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.imshow(X_train[1])

# COMMAND ----------

X_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC We need to preprocess the images first. Notice that the image shape above only has height and width, but lacks the number of channels. We need to add an extra dimension for the channel and scale the images into (-1,1). Scaling is a common image preprocessing step.

# COMMAND ----------

import numpy as np

X_train = np.expand_dims(X_train, axis=-1).astype("float32") # This is the same as using X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = (X_train - 127.5) / 127.5

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's first define a discriminator. 
# MAGIC 
# MAGIC It's a best practice to use **`LeakyReLU`** as opposed to **`ReLU`** for the discriminator. It’s similar to ReLU, but it relaxes sparsity constraints by allowing small negative activation values, rather than outputting zero activation values for negative inputs. Here is a <a href="https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu" target="_blank">resource</a> that dives into different activation functions, including ReLU and Leaky ReLU. 
# MAGIC 
# MAGIC <img src="https://miro.medium.com/max/2100/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg">

# COMMAND ----------

# MAGIC %md
# MAGIC Regarding choosing a kernel size, the best practice in GANs is to pick a number that is divisible by the stride size whenever a strided Conv2DTranspose or Conv2D is used. This is to reduce the checkerboard artifacts caused by unequal coverage of the pixel space in the generator. We will cover in the later part of the notebook what a Conv2DTranpose layer is! 
# MAGIC <br>
# MAGIC 
# MAGIC With checkerboard artifacts: <br>
# MAGIC <img src="https://distill.pub/2016/deconv-checkerboard/assets/style_artifacts.png" width="500" height="300">
# MAGIC 
# MAGIC <br>
# MAGIC Reduced checkerboard artifacts: <br>
# MAGIC <img src="https://distill.pub/2016/deconv-checkerboard/assets/style_clean.png" width="500" height="300"> 
# MAGIC 
# MAGIC Head over to this <a href="https://distill.pub/2016/deconv-checkerboard/" target="_blank">link</a> to see what "unequal coverage of the pixel space" means and play with the stride and kernel sizes! 

# COMMAND ----------

from tensorflow.keras import layers

def def_discriminator():
    """
    A discriminator is simply a binary classification model to tell if an image is real or fake
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (6, 6), padding="same", input_shape=(28,28,1))) # Note that the input shape (28, 28) here matches the pixels of the original images 
    model.add(layers.LeakyReLU()) 

    model.add(layers.Conv2D(64, (6, 6), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, (6, 6), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3)) ## Generally use between 0.3 and 0.6 
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss=tf.keras.losses.binary_crossentropy, 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LR/2, decay=LR/NUM_EPOCH)) # Half the generator's learning rate to help stabilize equilibrium
    

    return model

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's define our generator. There are two new components we haven't learned: 
# MAGIC * dropout 
# MAGIC * transposed convolutional layers 
# MAGIC 
# MAGIC Dropout is a regularization method that reduces overfitting by randomly and temporarily removing nodes during training. 
# MAGIC 
# MAGIC It works like this: <br>
# MAGIC 
# MAGIC * Apply to most type of layers (e.g. fully connected, convolutional, recurrent) and larger networks
# MAGIC * Temporarily and randomly remove nodes and their connections during each training cycle
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/nn_dropout.png)
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See the original paper here: <a href="http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf" target="_blank">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a>

# COMMAND ----------

# MAGIC %md
# MAGIC Now, onto transposed convolutional layers. <a href="https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d" target="_blank">Transposed convolutional layers</a>, also known as fractionally-strided convolution, are commonly used in GANs.
# MAGIC 
# MAGIC Transposed convolution helps us to:
# MAGIC * Accept an input from a previous layer in the network
# MAGIC * Produce an output that is larger than the input 
# MAGIC * Perform a convolution but allow us to reconstruct our target spatial resolution from before
# MAGIC 
# MAGIC This means that transposed convolutional layers combine upscaling of an image with a convolution.  
# MAGIC 
# MAGIC <img src="https://www.programmersought.com/images/174/ebc5c4c74ae847b31bc1e3a395f21b9e.png">
# MAGIC 
# MAGIC Source: https://arxiv.org/pdf/1603.07285.pdf

# COMMAND ----------

DIM = 7
DEPTH = 64

def def_generator(noise_dim):
    """
    The purpose of the generator is to generate fake/synthetic images.
    """
    input_shape = (DIM, DIM, DEPTH)

    model = tf.keras.Sequential()
    model.add(layers.Dense(DIM * DIM * DEPTH, input_dim=noise_dim))
    model.add(layers.LeakyReLU())

    ### Reshape the output of the previous layer set
    model.add(layers.Reshape(input_shape))
    # Not using a kernel size divisible by the stride for better performance in this particular case
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding="same")) 
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU())

    ### 1 represents the number of channels, for grayscale; 3 for RGB
    model.add(layers.Conv2D(1, (3, 3), activation="tanh", padding="same"))

    return model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Recall that generative adversarial networks (GANs) is composed of both the discriminator and the generator. So we are going to define our GAN using both models.

# COMMAND ----------

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def def_gan(noise_dim, discriminator, generator):
  
    ### Freeze discriminator weights so that 
    ### the feedback from the discriminator enables the generator to learn
    ### how to generate better fake images 
    discriminator.trainable = False

    gan_input = Input(shape=(noise_dim,))
    gan_output = discriminator(generator(gan_input))

    gan_model = Model(gan_input, gan_output)
    gan_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, decay=LR/NUM_EPOCH), 
                      loss=tf.keras.losses.binary_crossentropy)

    return gan_model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Finally, we are ready to train our GAN! This should take a few mins to train - if you are impatient, you can stop the training at any point and visualize it below.

# COMMAND ----------

from sklearn.utils import shuffle
tf.random.set_seed(42)

LR = 1e-3
NUM_EPOCH = 1
BATCH_SIZE = 64
NOISE_DIM = 100

### Save some random noise examples in the beginning so we can plot 
### what the generator has learned at the end of the training using these noise examples 
num_examples_to_generate = 16
benchmark_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])

### Define generator, discriminator, and GAN
discriminator = def_discriminator()
generator = def_generator(NOISE_DIM)
gan = def_gan(NOISE_DIM, discriminator, generator)

### Calculate the number of training iterations
batches_per_epoch = int(X_train.shape[0] / BATCH_SIZE)
n_steps = batches_per_epoch * NUM_EPOCH

### Shuffle once outside the loop because only doing one epoch
np.random.shuffle(X_train) 

### Training GAN starts here
for step in range(n_steps):
    ### Step 1: Generate noise vector 
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    ### Step 2: Pass noise vector through generator to generate fake images
    fake_images = generator.predict(noise)

    ### Step 3: Sample real images and mix with fake ones 
    ### Sample real images from the training data 
    real_image_batch = X_train[:BATCH_SIZE]
    real_fake_image_mix = np.concatenate([real_image_batch, fake_images])
    mix_labels = np.concatenate((np.ones(BATCH_SIZE), 
                                 np.zeros(BATCH_SIZE)), axis=0)
    mix_labels += 0.05 * np.random.random(mix_labels.shape) # A best practice: add random noise to your labels 

    ### Step 4: Train discriminator on mixed set so that it knows to distinguish between the two correctly
    dis_loss = discriminator.train_on_batch(real_fake_image_mix, mix_labels)

    ### Steps 5 and 6
    ### Step 5: Generate noise vectors again but purposely label them as "real", try to fool the discriminator
    ### Step 6: Train GAN using noise vectors labeled as "real" images 
    ### Update weights of generator based on feedback of discriminator, thus allowing us to generate more real images
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    misleading_labels = np.ones(BATCH_SIZE, dtype=int)
    gan_loss = gan.train_on_batch(noise, misleading_labels)

    print(f"Step {step}....................")
    print(f"discriminator loss: {round(dis_loss, 2)}")
    print(f"adversarial loss: {round(gan_loss, 2)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC We could now play around with the GAN with different learning rates, batch sizes, and/or number of epochs. It's an iterative process to get the best combination of hyperparameters. 
# MAGIC 
# MAGIC But for now, let's plot to see what the generative model has learned at the end of the training process! 

# COMMAND ----------

def generate_images(generator, benchmark_noise):
    """
    Generate synthetic images from the initial noise
    """

    ### Notice `training` is set to False so that all layers are in inference mode.
    predictions = generator(benchmark_noise, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        ### We then scale our image data back from the tanh range [-1, 1] to [0, 255]
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

generate_images(generator, benchmark_noise)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC GANs generally take a long time to train in order to achieve good performance. 
# MAGIC 
# MAGIC However, you can tell that our GAN has learned that the signals of the clothing items are concentrated in the center and the borders of the images are dark, just like our training images! 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC GANs are really quite complicated. Throughout this notebook, we have incorporated some best practices found in Francois Chollet's Deep Learning with Python book. However, it's worth noting that the best practices are still very much dependent on the data and the architecture. 
# MAGIC 
# MAGIC To recap, here are the best practices we have employed: <br>
# MAGIC <br>
# MAGIC * Sample random vectors from a normal distribution, rather than a uniform distribution
# MAGIC * Add dropout to the discriminator (generally between 0.3 and 0.6)
# MAGIC * Add noise to the class labels when training the discriminator
# MAGIC * Use batch normalization in the generator (this is data- and architecture-dependent, so experiment with it)
# MAGIC * Use a kernel size that is divisible by the stride size whenever a strided Conv2DTranspose or Conv2D is used in both the generator and the discriminator. (Note: We incorporated this in only the generator since it gave better performance.)
# MAGIC 
# MAGIC Additionally, if adversarial (generator) loss rises a lot while your discriminator loss falls to 0, you can try reducing the discriminator's learning rate and increasing its dropout rate. 

# COMMAND ----------

# MAGIC %md Additional interesting generator applications include:
# MAGIC 
# MAGIC - <a href="https://arxiv.org/abs/1508.06576" target="_blank">**Style Transfer to create artistic images**</a>
# MAGIC 
# MAGIC <img src="https://tensorflow.org/tutorials/generative/images/stylized-image.png" width=600>
# MAGIC 
# MAGIC 
# MAGIC - <a href="https://deepdreamgenerator.com/" target="_blank">**Deep Dream**</a>
# MAGIC 
# MAGIC <img src="https://b2h3x3f6.stackpathcdn.com/assets/landing/img/gallery/4.jpg" width=600>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>

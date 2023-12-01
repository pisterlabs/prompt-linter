#from __future__ import division
from six.moves import xrange
import numpy as np
import tensorflow as tf
import tooncol
import os
import time
import scipy.misc
import uuid

#I tested convenience functions from a number of different sources
# I (Larry) wrote dcgan_dtlayers for an unrelated research project
import du_utils
import shekkizh_utils as utils
#import dcgan_dtlayers as dtl 
#import openai_ops as oops
#import kim_ops as kops
from kim_ops import *

"""
Contains:
	-DCGAN for Deep Convolutional Adverserial Networks
	-NucConvModel for Deep Taylor Decomposition Convolutional neural networks
 
Author: Lawrence Du

"""



class DCGAN(object):
    """
    Writing this class was helped by examples from Taehoon Kim:
    	+ Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
        + License: MIT

    and Brandon Amos:
    	+ Source: http://bamos.github.io
    	+ License: MIT
    
    I have adapted DCGAN to work with my DeepTaylorDecomposition dtlayers.py wrapper.
    I also have also kept the actual session separate from the network definitions. 
    Running the actual training can be done from run_training.py
    		                                               -Larry

    """

    def __init__(self,sess,params,input_shape,toon_collection,label_dim=None,z_dim=128):

        
        self.params = params
        if input_shape[0] == -1:
            input_shape = [params.batch_size]+input_shape[1:]
        self.input_shape = input_shape #e.g. [25,64,64,3] or [None,64,64,3]
        self.batch_size = self.params.batch_size
        
        """
        seed_size is Number of random values generate z with
        If you are reusing an old classifier net, this value is the same as num_classes
        """
        self.sess=sess
        #self.seed_size=seed_size

        #self.input_name = "image_input"
        #self.z_name = "z_input"
        self.params = params

        
        self.toon_collection = toon_collection
        if self.toon_collection==None:
            self.num_train_examples = 0
        else:
            self.num_train_examples= self.toon_collection.num_examples

        self.label_dim = label_dim #None if labels are not to be used
        print "Label dim set to",self.label_dim
        self.z_dim=int(z_dim)
        
        checkpoint_dir = self.params.save_dir+os.sep+'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.checkpoint_dir = checkpoint_dir
        img_dir = self.checkpoint_dir+os.sep+'images'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        self.img_dir = img_dir

 
        
    def make_ops(self,d_method,g_method):
        """
        Define all placeholders and ops
        Note: I replaced all placeholders with a tf.train.shuffle_batch pipeline
        """
        input_pholder_shape = [self.batch_size]+self.input_shape[1:]
        self.input_pholder = tf.placeholder(tf.float32,self.input_shape,name='image_input')

        self.label_pholder = tf.placeholder(tf.float32, [self.batch_size,self.label_dim], name='label_input')
        #self.z_pholder = tf.placeholder(tf.float32, [self.batch_size,self.z_dim], name='z_input')
        self.z_pholder = tf.placeholder(tf.float32, [None,self.z_dim], name='z_input')

        #sample_b_shape = [None]+self.input_shape[1:]
        #self.sample_input_pholder = tf.placeholder(tf.float32,sample_b_shape,name='sample')


        #image_batch,label_batch = image_batch_pipeline(self.batch_size)
        
        
        """
        G,D_real, and D_fake are simply the logits passed through a sigmoid function.
        Note that D_real and D_fake should share the same weights
        """
        
        #TODO: Make random label generator, or an interface for making a custom label

        """
        Apply random l-r flip,brightness, hue, contrast and saturation manipulations
        To artificially crank dataset size up. 
        """
        
        self.adjusted_imageb = tf.map_fn(lambda img:du_utils.random_image_transforms(img),self.input_pholder)
        self.adjusted_imageb_summary = tf.summary.image('adjusted_input_image',
                                         du_utils.tanh_to_sig_scale(self.adjusted_imageb))

        self.adjusted_imageb_summary = tf.summary.image('adjusted_input_image',
                                                self.adjusted_imageb)

        self.g_method = g_method
        self.d_method = d_method
        self.D_real,self.D_real_logits = self.d_method(self.adjusted_imageb,
                                                  self.label_pholder,
                                                  reuse=False)
        self.G = self.g_method(self.z_pholder,self.label_pholder,reuse=False)
        #Note, the G output always gets put with whatever the current label is
        self.D_fake, self.D_fake_logits = self.d_method(self.G,
                                                   self.label_pholder,
                                                   reuse=True)
        #self.S = self.sampler_taehoon(self.z_pholder,self.label_pholder,reuse=True,is_phase_train=False)
        self.S = self.g_method(self.z_pholder,self.label_pholder,
                                      reuse=True)
        

        
        """ Loss function between real and ones; Loss function between fake and zeros"""

        #TODO: Check if logits need to be !=0 for sigmoid x-entropy input  (b/c log(0)=undefined)

        """Discriminator loss on real input is diff btwn real image and perfect real image"""
        self.d_real_loss= tf.reduce_mean(
                          tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits,
                                                                  tf.ones_like(self.D_real)))

        """Discriminator loss on fake input is difference btwn fake image and perfect fake image"""
        self.d_fake_loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                               self.D_fake_logits,tf.zeros_like(self.D_fake)))

        """Composite discriminator loss"""
        self.d_total_loss = self.d_real_loss + self.d_fake_loss
        
        """Generator loss is difference between logits from the fake image """
        self.g_loss = tf.reduce_mean(
                      tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
                                                              tf.ones_like(self.D_fake)))


        """Summaries"""
        self.d_real_summary = tf.summary.histogram('d_real',self.D_real)
        self.d_real_loss_summary = tf.summary.scalar('d_real_loss',self.d_real_loss)
        self.d_total_loss_summary = tf.summary.scalar('d_total_loss',self.d_total_loss)

        self.d_fake_summary = tf.summary.histogram('d_fake',self.z_pholder)
        self.d_fake_loss_summary = tf.summary.scalar('d_fake_loss',self.d_fake_loss)
        
        self.z_summary = tf.summary.histogram('z',self.z_pholder)
        #self.G_summary = tf.summary.image('Generator',du_utils.tanh_to_sig_scale(self.G))
        #self.S_summary = tf.summary.image('sampler',du_utils.tanh_to_sig_scale(self.S))
        self.G_summary = tf.summary.image('Generator',self.G)
        self.S_summary = tf.summary.image('Sampler',self.S)
       
        self.g_loss_summary = tf.summary.scalar('g_loss',self.g_loss)
        
        log_dir = self.params.save_dir+os.sep+"logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
                    
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        
        """Training vars"""
        trainable_vars = tf.trainable_variables()

        self.d_vars = [var for var in trainable_vars if var.name.startswith("discriminator")]
        self.g_vars = [var for var in trainable_vars if var.name.startswith("generator")]

        print "\n\nTrainable vars lists d:"
        for var in self.d_vars:
            print var.name
        print "\n\nTrainable vars lists g:"
        for var in self.g_vars:
            print var.name

        self.saver = tf.train.Saver()


        
    def train(self):
        """
        Training loop for DCGAN:
        input_collection needs to have pull_batch method 
        """

        d_optim = tf.train.AdamOptimizer(self.params.learning_rate,beta1=self.params.beta1).minimize(self.d_total_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.params.learning_rate,self.params.beta1).minimize(self.g_loss, var_list=self.g_vars)

        #Merging summaries allows to make the update call easier
        self.d_merged_summaries = tf.summary.merge([self.z_summary,
                                                    self.d_real_summary,
                                                    self.d_real_loss_summary, 
                                                    self.d_total_loss_summary])
        
        self.g_merged_summaries = tf.summary.merge([self.z_summary,
                                                    self.d_fake_summary,
                                                    self.d_fake_loss_summary,
                                                    self.G_summary,
                                                    self.g_loss_summary])
        
        
        #Init all variables
        tf.global_variables_initializer().run()
        #tf.initialize_all_variables()
        if self.load(self.checkpoint_dir):
            print ("Successful checkpoint load")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess,coord)
            

        """A single random sample we will use to keep track of progress"""
        batch_sample = np.random.uniform(-1,1.,[self.batch_size,
                                               self.z_dim]).astype(np.float32)
        
        start_time = time.time()
        step=0
        steps_per_epoch = self.num_train_examples//self.batch_size
        for epoch in xrange(self.params.num_epochs):
            for i in xrange(0,steps_per_epoch):

                #Generate a random noise sample with the proper dimensions 
                batch_z = np.random.uniform(-1.,1,[self.batch_size,self.z_dim]).\
                          astype(np.float32)
                #Draw sample from dataset

                                
                
                batch_images,batch_labels =self.toon_collection.next_batch(self.batch_size,
                                                                           tanh_scale=False)

            

                

                #This reshape op is used for cases where the input is a linear vector
                #batch_input = np.reshape(batch_input,self.input_shape)
                
                
                """Update discriminator network"""

                

                
                #For adjusted image summary, just check on tensorboard if the random adjustments are
                #within an acceptable range
                _,summary_str,adj_sum = self.sess.run([d_optim,self.d_merged_summaries,
                                                       self.adjusted_imageb_summary],
                                               feed_dict={self.input_pholder: batch_images,
                                                          self.label_pholder:batch_labels,
                                                          self.z_pholder:batch_z})                

                #_,summary_str = self.sess.run([d_optim,self.d_merged_summaries],
                #                               feed_dict={self.z_pholder:batch_z})                

                self.writer.add_summary(summary_str,step)
                self.writer.add_summary(adj_sum,step)



                for _ in range(2):
                    #Run generator multiple times (taken from Taehoon Kim advice)
                    """Update generator network"""
                    
                    _,summary_str= self.sess.run([g_optim,self.g_merged_summaries],
                                               feed_dict={self.z_pholder: batch_z,
                                                          self.label_pholder:batch_labels})
                
                    self.writer.add_summary(summary_str,step)

                
                
                
                
                errD_fake = self.d_fake_loss.eval({self.z_pholder:batch_z,
                                               self.label_pholder:batch_labels})
                errD_real = self.d_real_loss.eval({self.input_pholder:batch_images,
                                               self.label_pholder:batch_labels})
                errD_tot = errD_fake+errD_real
                errG = self.g_loss.eval({self.z_pholder:batch_z,self.label_pholder:batch_labels})

                """Throw in some extra training if certain thresholds are exceeded"""
                
                if errD_tot < .25:
                    _,summary_str = self.sess.run([d_optim,self.d_merged_summaries],
                                               feed_dict={self.input_pholder: batch_images,
                                                          self.label_pholder:batch_labels,
                                                          self.z_pholder:batch_z})

                    self.writer.add_summary(summary_str,step)
                if errG > 1.:
                    """Update generator network(again)""" 
                    _ ,summary_str= self.sess.run([g_optim,self.g_merged_summaries],
                                               feed_dict={self.z_pholder: batch_z,
                                                          self.label_pholder:batch_labels})
                    self.writer.add_summary(summary_str,step)
                    """Update generator network(again and again and again!!!!!!)""" 
                    _,summary_str= self.sess.run([g_optim,self.g_merged_summaries],
                                               feed_dict={self.z_pholder: batch_z,
                                                          self.label_pholder:batch_labels})
                    self.writer.add_summary(summary_str,step)
                

                if (step%500 == 0):
                #Take a look at what batch_sampler is dreaming up

                #TODO: Save these images
                    samples, summary_str = self.sess.run([self.S,self.S_summary],
                                                     feed_dict={self.z_pholder:batch_sample,
                                                                self.label_pholder:batch_labels} )
                
                    self.writer.add_summary(summary_str,step)
                    #samples should be self.input_shape
                    #Note: apply inv_transform to tanh output to get pix data to range 0-1
                
                    for imidx in range(12):
                        img_fname = (self.img_dir+os.sep+str(imidx)+'_ep'+str(epoch)+'_step'+
                                     str(step)+'.png')
                        #Rescale tanh output to [0,255.]
                        img_samp = du_utils.tanh_to_sig_scale(samples[imidx,:,:,:],255.)
                        #img_samp = samples[imidx,:,:,:]*255.
                        scipy.misc.imsave(img_fname,img_samp)
                
                        self.save(self.checkpoint_dir,step)
       
                   
                step += 1 
                
                
            #Print a message on every epoch
            print("Epoch: [%2d] step: %d time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, step,time.time() - start_time, errD_tot, errG))

             # Wait for threads to finish.
        coord.request_stop()
        coord.join(threads)



        
    
    def flask_demo(self,flask_checkpoint_dir):
        self.flask_checkpoint_dir = flask_checkpoint_dir
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(flask_checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            print "Flask demo loaded"
        else:
            print "No model loaded"
            

    def generate_random_images(self,n,image_dir):

        
        #self.flask_demo(self.flask_checkpoint_dir)
        #Use n as our batch size
        z_val = np.random.uniform(-1.,1,[n,self.z_dim]).astype(np.float32)
        z_list = np.vsplit(z_val,n)
        z_list = [z[0] for z in z_list]

        print self.sess
        print "Z_list",len(z_list)
        print "Z val",z_val.shape
    
        samples = self.sess.run([self.S],
                                feed_dict={self.z_pholder:z_val} )[0]
        fnames_list = []

        print "Samples",len(samples)
        
        for imidx in range(n):
            
            uid = str(uuid.uuid4())
            #Create image with uuid name
            img_fname = (image_dir+os.sep+uid+'.png')
            img_samp = du_utils.tanh_to_sig_scale(samples[imidx,:,:,:],255.)
            fnames_list.append(img_fname)
            scipy.misc.imsave(img_fname,img_samp)
        return fnames_list,z_list

    def latent_walk(self,z_key,z_inds,image_dir,step_size=0.05,num_steps = 8):
        """ 
        Given a 1-d numpy array, and an index or set of indices within that array,
        create copies of the input array with the selected index altered.

        :param z_key: Numpy 1d array with entries in range [-1,1] and len equal to self.z_dim
        :param z_inds: A integer or list of integers
        :param image_dir: A directory
        :param step_size: Size of steps to walk along z_index (or z-indices)
        :param num_steps: Number of steps 
        :returns: fnames_list of output files (incl directory), z_list of new keys
        :rtype:

        (Note: Hey, I tested emacs sphinx C-c M-d!) 

        """
        if type(z_inds) != list: #if only one value entered
            z_index = [z_inds]

        if z_key.shape[0] != self.z_dim:
            print ("Error, z_key shape is incorrect")
            return
        #make a num_steps x num_steps matrix with z_key in every element
        #!FIXME: cannot reshape array of size 6 into shape (6,128)
        z_vals = np.reshape(np.tile(z_key,num_steps),(num_steps,self.z_dim)) 

        for z_index in z_inds:
            step_range = step_size*num_steps
            if step_range > 2:
                print "Step range too wide"
                step_size= (2.-0.08)/num_steps
                print "Setting step size to ",step_size
            if z_index <0 or z_index > self.z_dim:
                z_index = np.random.randint(0,self.z_dim)
                print "Error z_index exceeds bounds. Setting z_index to",z_index

            center = z_key[int(z_index)]
            margin = step_range/2
            if (center-margin) <-1:
                center = -1+margin
            if center+margin>1:
                center = 1-margin


            start_walk = center-margin
            stop_walk = center+margin
            dim_steps = np.arange(start_walk,stop_walk,step_size,dtype=np.float32)

            """Create versions of the z-vec with a single dimension altered"""
            z_vals[:,z_index] = dim_steps 


        
        z_list = np.vsplit(z_vals,num_steps)
        z_list = [z[0] for z in z_list]#Need to prevent entries becoming [1,z_dim]
        
        for i in range(0,num_steps):
                   
            
            samples = self.sess.run([self.S],
                                feed_dict={self.z_pholder:z_vals} )[0]
            fnames_list = []
            for imidx in range(num_steps):
                uid = str(uuid.uuid4())
                #Create image with uuid name
                img_fname = (image_dir+os.sep+uid+'.png')
                img_samp = du_utils.tanh_to_sig_scale(samples[imidx,:,:,:],255.)
                fnames_list.append(img_fname)
                scipy.misc.imsave(img_fname,img_samp)
        return fnames_list,z_list
    
        

    def generator_openai(self,z,reuse=False):
        ##genE
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

        
    def discriminator_openai(self,input_images,reuse=False):
        ##disE
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()


    
    def discriminator_taehoon(self, image, label,reuse=False):

        """df_dim is the num output filters, which we multiply to increase going down the nn """
        self.df_dim= 64
        self.c_dim=self.input_shape[3]
        self.dfc_dim=1024
        with tf.variable_scope("discriminator") as scope:

            print "Discriminator scope name is",scope.name
            if reuse:
                scope.reuse_variables()

            self.d_bn1 = batch_norm(name='d_bn1')
            self.d_bn2 = batch_norm(name='d_bn2')
            if not self.label_dim: #If no labels
                self.d_bn3 = batch_norm(name='d_bn3')

        
            if not self.label_dim: #Don't make the network conditional
                print "Discriminator is not set to use labels"
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
          
                return tf.nn.sigmoid(h4), h4
            else:
                print "Discriminator is set to use labels"
                yb = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])
                x = conv_cond_concat(image, yb)
          
                h0 = lrelu(conv2d(x, self.c_dim + self.label_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.label_dim,
                                                       name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])      
                h1 = tf.concat_v2([h1, label], 1)
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat_v2([h2, label], 1)
                
                h3 = linear(h2, 1, 'd_h3_lin')
        
                return tf.nn.sigmoid(h3), h3



    
    def discriminator_larry(self, image, label,reuse=False):

        """df_dim is the num output filters, which we multiply to increase going down the nn """
        self.df_dim= 64
        self.c_dim=self.input_shape[3]
        self.dfc_dim=1024
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            self.d_bn1 = batch_norm(name='d_bn1')
            self.d_bn2 = batch_norm(name='d_bn2')
            if self.label_dim: #If there are labels
                self.d_bn3 = batch_norm(name='d_bn3')
                self.d_bn4 = batch_norm(name='d_bn4')
                self.d_bn5 = batch_norm(name='d_bn5')
        
            if not self.label_dim: #Don't make the network conditional
                print "Discriminator is not set to use labels"
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                #Only outputs one value                
                
                return tf.nn.sigmoid(h4), h4
            else:
                print "Discriminator is set to use labels"
                yb = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])

                x = conv_cond_concat(image, yb)
          
                h0 = lrelu(conv2d(x, self.c_dim + self.label_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.label_dim,
                                                       name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])      
                h1 = tf.concat_v2([h1, label], 1)
                print "dh1",h1.get_shape().as_list()
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat_v2([h2, label], 1)

                print "dh2",h2.get_shape().as_list()


                #h3 = linear(h2, 1, 'd_h3_lin')
              
                h3 = lrelu(self.d_bn3(linear(h2, self.dfc_dim//4, 'd_h3_lin')))
                h3 = tf.concat_v2([h3, label], 1)
                print "dh3",h3.get_shape().as_list()
                
                
                h4 = lrelu(self.d_bn4(linear(h3, self.dfc_dim//2, 'd_h4_lin')))
                h4 = tf.concat_v2([h4, label], 1)

                print "dh4",h4.get_shape().as_list()
                h5 = lrelu(self.d_bn5(linear(h4, self.dfc_dim, 'd_h5_lin')))
                h5 = tf.concat_v2([h5, label], 1)
                print "dh5",h5.get_shape().as_list()
                
                #I altered this from producing 1 value to producing the same number of values as label_dim
                h6 = linear(h5,self.label_dim,'d_h6_lin')
                
                return tf.nn.sigmoid(h6),h6


            
    def generator_larry(self, z, label,reuse=True):
        self.gf_dim=64
        self.gfc_dim =1024
        self.output_height=self.input_shape[1]
        self.output_width = self.input_shape[2]
        self.c_dim = self.input_shape[3]#num channels
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn01 = batch_norm(name='g_bn01')
        #self.g_bn02 = batch_norm(name='g_bn02')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            if not self.label_dim:

                print "Generator is not set to use labels"
                s_h, s_w = int(self.output_height),int(self.output_width)
                s_h2, s_h4, s_h8, s_h16 = \
                                          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
                s_w2, s_w4, s_w8, s_w16 = \
                                          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)
                
                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
                
                self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))
                
                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))
                
                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))
                
                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))
                
                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                
                return tf.nn.tanh(h4)
            else:
                print "Generator is set to use labels"
                ##LARRYS EXAMPLE
                """
                Note: Taehoons original conditional implementation was designed for
                mnist examples, and didn't use deconvolution as heavily, leading
                to lousy results for image data more complex than MNIST
                """
                #Note: unlike Taehoon, I match output_dims var names to each layer
                #for readability -- Larry
                s_hout,s_wout = int(self.output_height), int(self.output_width)
                s_h4, s_w4 = int(s_hout/2),int(s_wout/2) #32
                s_h3, s_w3 = int(s_hout/4),int(s_wout/4) #16
                s_h2, s_w2 = int(s_hout/8),int(s_wout/8) #8

                
                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])
                z = tf.concat_v2([z, label], 1)
                
                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat_v2([h0, label], 1)

                #I added one more linear layer --Larry
                
                h01 = tf.nn.relu(
                    self.g_bn01(linear(h0, self.gfc_dim, 'g_h01_lin')))
                h01 = tf.concat_v2([h01, label], 1)
                #Output is 1024
                
                h1 = tf.nn.relu(self.g_bn1(
                    linear(h01, s_h2*s_w2*self.gf_dim*4, 'g_h1_lin'))) #64*8*8*4
                
                #The reshaping size prior to deconv is important
                
                h1 = tf.reshape(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 4])
                h1 = conv_cond_concat(h1, yb)
                #Reshape is [b,8,8,64*4]
                
                h2 = tf.nn.relu(self.g_bn2(
                                    deconv2d(h1,
                                                  [self.batch_size, s_h3, s_w3, self.gf_dim*4],
                                                  name='g_h2')))
                h2 = conv_cond_concat(h2, yb)
                #[b,16,16,64*4]

                print "gh2",h2.get_shape().as_list()
                
                #I added two more deconvolution layers -- Larry
                h3 = tf.nn.relu(self.g_bn3(deconv2d(h2,
                                                 [self.batch_size, s_h4, s_w4, self.gf_dim*2],
                                                         name='g_h3')))
                h3 = conv_cond_concat(h3, yb)
                #[b,32,32,64*2]

                print "gh3",h3.get_shape().as_list()

                h4 = tf.nn.relu(self.g_bn4(deconv2d(h3,
                                  [self.batch_size, s_hout, s_wout, self.c_dim], name='g_h4')))
                print "gh4",h4.get_shape().as_list()

                return tf.nn.sigmoid(h4)
                #h4 = conv_cond_concat(h4, yb)
                #[b,64,64,64]


                #I replace tanh with sigmoid to save myself headaches --Larry
                #return tf.nn.sigmoid(
                #    deconv2d(h4, [self.batch_size, s_hout, s_wout, self.c_dim], name='g_hout'))


                
                
            
            
    def generator_taehoon(self, z, label,reuse=True):
        self.gf_dim=64 #output dim 64
        self.gfc_dim =1024 #num fully connected weights
        self.output_height=self.input_shape[1]
        self.output_width = self.input_shape[2]
        self.c_dim = self.input_shape[3]#num channels
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            if not self.label_dim:

                print "Generator is not set to use labels"
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4, s_h8, s_h16 = \
                                          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
                s_w2, s_w4, s_w8, s_w16 = \
                                          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)
                
                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
                
                self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))
                
                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))
                
                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))
                
                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))
                
                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                
                return tf.nn.tanh(h4)
            else:
                print "Generator is set to use labels"
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)
                
                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])
                z = tf.concat_v2([z, label], 1)
                
                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat_v2([h0, label], 1)
                
                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                
                h1 = conv_cond_concat(h1, yb)
                
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                   [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)
                #Does a tanh belong here?
                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))



    def generator_shekkizh(self,z,labels,reuse=False): ##genD
        """Of all the network configurations, this one appears to be the best balanced"""
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
                is_phase_train = True

            is_phase_train=True
            gen_dim=int(16)
            batch_size,image_size,_,num_channels = [int(d) for d in self.input_shape]

            #print batch_size,image_size,num_channels,gen_dim,self.z_dim
            
            
            z_dim = self.z_dim
            W_0 = utils.weight_variable([z_dim,int(64 * gen_dim / 2 * image_size / 16 * image_size / 16)],                               name="g_weights0")
            b_0 = utils.bias_variable([64 * gen_dim/ 2 * image_size / 16 * image_size / 16],
                                      name="g_bias0")
            z_0 = tf.matmul(z, W_0) + b_0
            h_0 = tf.reshape(z_0, [-1,image_size / 16, image_size / 16, 64 * gen_dim / 2])
            h_bn0 = utils.batch_norm(h_0, 64 * gen_dim / 2, is_phase_train,scope="g_bnorm0")
            h_relu0 = tf.nn.relu(h_bn0, name='g_relu0')
            
            W_2 = utils.weight_variable([5, 5, 64 * gen_dim / 4, 64 *
                                         gen_dim / 2],
                                        name="g_weights2")
            b_2 = utils.bias_variable([64 *gen_dim / 4], name="g_bias2")
            deconv_shape = tf.pack([tf.shape(h_relu0)[0],image_size / 8, image_size /
                                    8, 64 * gen_dim / 4])
            h_conv_t2 = utils.conv2d_transpose_strided(h_relu0,W_2, b_2,
                                                       output_shape=deconv_shape)
            h_bn2 = utils.batch_norm(h_conv_t2, 64* gen_dim / 4, is_phase_train,
                                     scope="g_bnorm2")
            h_relu2 = tf.nn.relu(h_bn2, name='g_relu2')
            
            
            W_3 = utils.weight_variable([5, 5, 64 * gen_dim / 8, 64 *
                                         gen_dim / 4],
                                        name="g_weights3")
            b_3 = utils.bias_variable([64 * gen_dim / 8], name="g_bias3")
            
            deconv_shape = tf.pack([tf.shape(h_relu2)[0],  image_size / 4, image_size /
                                    4, 64 * gen_dim / 8])
            h_conv_t3 = utils.conv2d_transpose_strided(h_relu2, W_3, b_3,
                                                       output_shape=deconv_shape)
            h_bn3 =utils.batch_norm(h_conv_t3, 64 * gen_dim / 8, is_phase_train,
                                    scope="g_bnorm3")
            h_relu3 = tf.nn.relu(h_bn3, name='g_relu3')
            #utils.add_activation_summary(h_relu3)
            
            W_4 = utils.weight_variable([5, 5, 64 * gen_dim / 16, 64 *
                                         gen_dim / 8],
                                        name="g_weights4")
            b_4 = utils.bias_variable([64 * gen_dim / 16], name="g_bias4")
            deconv_shape =tf.pack([tf.shape(h_relu3)[0],  image_size / 2, image_size /
                                   2, 64 * gen_dim / 16])
            h_conv_t4 = utils.conv2d_transpose_strided(h_relu3, W_4, b_4,output_shape=deconv_shape)
            h_bn4 = utils.batch_norm(h_conv_t4, 64 * gen_dim / 16, is_phase_train, scope="g_bnorm4")
            h_relu4 =tf.nn.relu(h_bn4, name='g_relu4')
            #utils.add_activation_summary(h_relu4)
            
            W_5 = utils.weight_variable([5, 5, num_channels, 64 *
                                         gen_dim / 16],
                                        name="g_weights5")
            b_5 =  utils.bias_variable([num_channels],
                                       name="g_bias5")
            deconv_shape =tf.pack([tf.shape(h_relu4)[0], image_size, image_size,
                                   num_channels])
            h_conv_t5 = utils.conv2d_transpose_strided(h_relu4,W_5, b_5,
                                                       output_shape=deconv_shape)
            generated_image = tf.nn.tanh(h_conv_t5,
                                         name='generated_image')
            
            return generated_image
        
        
        
    def discriminator_shekkizh(self,input_images,labels,reuse=False):
        ##disD
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            is_phase_train=True
            gen_dim=16
            batch_size,img_size,_,num_channels = self.input_shape
            
            
            W_conv0 = utils.weight_variable([5, 5, num_channels, 64 *1], name="d_weights0")
            b_conv0 = utils.bias_variable([64 * 1], name="d_bias0")
            h_conv0 = utils.conv2d_strided(input_images, W_conv0, b_conv0)
            h_bn0   = h_conv0
            # utils.batch_norm(h_conv0, 64 * 1,  is_phase_train, scope="d_bnorm0")
            h_relu0 = utils.leaky_relu(h_bn0, 0.2, name="d_relu0")
            #utils.add_activation_summary(h_relu0)
            
            W_conv1 = utils.weight_variable([5, 5, 64 * 1, 64 * 2],
                                            name="d_weights1")
            b_conv1 = utils.bias_variable([64 * 2],name="d_bias1")
            h_conv1 = utils.conv2d_strided(h_relu0,W_conv1, b_conv1)
            h_bn1 = utils.batch_norm(h_conv1, 64 * 2, is_phase_train, scope="d_bnorm1")
            h_relu1 = utils.leaky_relu(h_bn1, 0.2, name="d_relu1")
            #utils.add_activation_summary(h_relu1)
            
            W_conv2 = utils.weight_variable([5, 5, 64 * 2, 64 * 4], name="d_weights2")
            b_conv2 = utils.bias_variable([64 * 4],name="d_bias2")
            h_conv2 = utils.conv2d_strided(h_relu1,W_conv2, b_conv2)
            h_bn2 = utils.batch_norm(h_conv2, 64 * 4, is_phase_train, scope="d_bnorm2")
            h_relu2 = utils.leaky_relu(h_bn2, 0.2, name="d_relu2")
            #utils.add_activation_summary(h_relu2)
            
            W_conv3 = utils.weight_variable([5, 5, 64 * 4, 64 * 8],
                                            name="d_weights3")
            b_conv3 = utils.bias_variable([64 * 8],name="d_bias3")
            h_conv3 = utils.conv2d_strided(h_relu2,W_conv3, b_conv3)
            h_bn3 = utils.batch_norm(h_conv3, 64 * 8, is_phase_train, scope="d_bnorm3")
            h_relu3 = utils.leaky_relu(h_bn3, 0.2, name="d_relu3")
            #utils.add_activation_summary(h_relu3)
            
            shape = h_relu3.get_shape().as_list()
            h_3 = tf.reshape(h_relu3, [batch_size, (img_size //16)*(img_size // 16)*shape[3]])
            W_4 = utils.weight_variable([h_3.get_shape().as_list()[1], 1],name="W_4")
            b_4 = utils.bias_variable([1], name="d_bias4")
            h_4 = tf.matmul(h_3, W_4) + b_4
                
            return tf.nn.sigmoid(h_4), h_4
            


    

            
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_name = checkpoint_dir+os.sep+'checkpoint'    
        self.saver.save(self.sess,
                        checkpoint_name,
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" Retrieving checkpoints from", checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print ("Successfully loaded checkpoint from",checkpoint_dir)
            return True
        else:
            print ("Failed to load checkpoint",checkpoint_dir)
            return False



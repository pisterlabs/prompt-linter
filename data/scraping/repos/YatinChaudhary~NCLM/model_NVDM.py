"""NVDM Tensorflow implementation by Yishu Miao"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import os
import model.utils as utils
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.metrics.pairwise as pw
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import model.data_lstm as data

seed = 42
tf_op_seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

#learning_rate = 5e-5
#batch_size = 64
#n_hidden = 256
n_topic = 150
n_sample = 10
#non_linearity = tf.nn.tanh
non_linearity = tf.nn.sigmoid
GSM = True
concat_topic_emb_and_prop = False


class NVDM(object):
	""" Neural Variational Document Model -- BOW VAE.
	"""
	def __init__(self, params, prior_embeddings=None, initializer_nvdm=None, topic_coherence_embeddings=None):

		self.vocab_size = params.TM_vocab_length
		self.n_hidden = params.hidden_size_TM
		self.n_topic = n_topic
		self.n_sample = n_sample
		self.non_linearity = non_linearity
		self.learning_rate = params.learning_rate
		self.batch_size = params.batch_size

		self.x = tf.placeholder(tf.float32, [None, self.vocab_size], name='x')
		self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
		if params.use_sent_topic_rep:
			self.x_sent = tf.placeholder(tf.float32, [None, None, self.vocab_size], name='x_sent')

		if params.use_topic_embedding:
			self.x_doc_mask = tf.placeholder(tf.float32, [None, self.vocab_size], name='x_doc_mask')

		#self.input_batch_size = tf.placeholder(tf.int32, (), name='input_batch_size')
		self.input_batch_size = tf.shape(self.x)[0]
		if params.use_sent_topic_rep:
			self.input_batch_size_sent = tf.shape(self.x_sent)[0]
			self.input_batch_len_sent = tf.shape(self.x_sent)[1]
			self.batch_size_sent = self.input_batch_size_sent * self.input_batch_len_sent

		# encoder
		with tf.variable_scope('TM_encoder', reuse=tf.AUTO_REUSE): 
			self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
			#self.enc_vec = utils.mlp(self.x, [self.n_hidden, self.n_hidden], self.non_linearity)
			self.mean = utils.nvdm_linear(self.enc_vec,
										self.n_topic,
										scope='mean',
										#matrix_initializer=initializer_nvdm[1][0],
										matrix_initializer=None,
										#bias_initializer=initializer_nvdm[1][1])
										bias_initializer=None)
			self.logsigm = utils.nvdm_linear(self.enc_vec, 
									self.n_topic, 
									bias_start_zero=True,
									matrix_start_zero=True,
									scope='logsigm',
									#matrix_initializer=initializer_nvdm[2][0],
									matrix_initializer=None,
									#bias_initializer=initializer_nvdm[2][1])
									bias_initializer=None)
			self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
			#self.kld = self.mask*self.kld  # mask paddings
			self.kld = tf.multiply(self.mask, self.kld, name='kld')  # mask paddings

			if params.use_sent_topic_rep:
				self.x_sent_reshape = tf.reshape(self.x_sent, [-1, self.vocab_size])
				self.enc_vec_sent = utils.mlp(self.x_sent_reshape, [self.n_hidden], self.non_linearity)
				self.mean_sent = utils.nvdm_linear(self.enc_vec_sent, self.n_topic, scope='mean')
				self.logsigm_sent = utils.nvdm_linear(self.enc_vec_sent, 
										self.n_topic, 
										bias_start_zero=True,
										matrix_start_zero=True,
										scope='logsigm')

			if params.prior_emb_for_topics or params.topic_coherence_reg:
				W_prior = tf.get_variable(
					'embeddings_TM_prior',
					dtype=tf.float32,
					initializer=prior_embeddings,
					trainable=False
				)
		
		with tf.variable_scope('TM_decoder', reuse=tf.AUTO_REUSE):
			if self.n_sample == 1:
				eps = tf.random_normal((self.input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
				#doc_vec = tf.mul(tf.exp(self.logsigm), eps) + self.mean
				self.doc_vec = tf.add(tf.multiply(tf.exp(self.logsigm), eps), self.mean, name='doc_hidden')
				if GSM:
					self.doc_vec = tf.nn.softmax(self.doc_vec, axis=1)
				self.last_h = self.doc_vec
				logits_projected, self.decoding_matrix = utils.nvdm_linear(self.doc_vec, 
																		self.vocab_size, 
																		scope='projection', 
																		get_matrix=True,
																		#matrix_initializer=initializer_nvdm[3][0],
																		matrix_initializer=None,
																		#bias_initializer=initializer_nvdm[3][1])
																		bias_initializer=None)
				logits = tf.nn.log_softmax(logits_projected)
				self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)
			else:
				#eps = tf.random_normal((self.n_sample*self.batch_size, self.n_topic), mean=0.0, stddev=1.0)
				eps = tf.random_normal((self.n_sample*self.input_batch_size, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
				eps_list = tf.split(eps, self.n_sample, 0)
				recons_loss_list = []
				doc_vec_list = []
				for i in range(self.n_sample):
					if i > 0: tf.get_variable_scope().reuse_variables()
					curr_eps = eps_list[i]
					doc_vec = tf.add(tf.multiply(tf.exp(self.logsigm), curr_eps), self.mean)
					if GSM:
						doc_vec = tf.nn.softmax(doc_vec, axis=1)
					doc_vec_list.append(doc_vec)
					logits, self.decoding_matrix = utils.nvdm_linear(doc_vec, 
																	self.vocab_size, 
																	scope='projection',
																	get_matrix=True,
																	matrix_initializer=None,
																	bias_initializer=None)
					logits = tf.nn.log_softmax(logits)
					recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
				self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample
				self.doc_vec = tf.add_n(doc_vec_list) / self.n_sample
				self.last_h = self.doc_vec

			# TOPIC EMBEDDING CODE

			if params.use_topic_embedding:
				topics_masked = tf.multiply(tf.expand_dims(self.x_doc_mask, axis=1), tf.expand_dims(self.decoding_matrix, axis=0), name='topics_masked')
				self.top_k = tf.nn.top_k(topics_masked, k=params.use_k_topic_words, sorted=False)
				if params.prior_emb_for_topics:
					self.top_k_embeddings = tf.nn.embedding_lookup(W_prior, self.top_k.indices)
					if concat_topic_emb_and_prop:
						self.topic_emb_size = prior_embeddings.shape[1] + self.n_topic
					else:
						self.topic_emb_size = prior_embeddings.shape[1]
				else:
					self.top_k_embeddings = tf.nn.embedding_lookup(tf.transpose(self.decoding_matrix), self.top_k.indices)
					if concat_topic_emb_and_prop:
						self.topic_emb_size = self.n_topic * 2
					else:
						self.topic_emb_size = self.n_topic
				self.topic_embeddings = tf.reduce_mean(self.top_k_embeddings, axis=2, name='topic_embeddings')

				if params.use_k_topics > 0:
					# Masking document topic proportion vector
					top_k_h_values, top_k_h_indices = tf.nn.top_k(self.last_h, k=params.use_k_topics, sorted=False, name='top_k_h')
					row_numbers = tf.tile(tf.expand_dims(tf.range(0, self.input_batch_size), 1), [1, params.use_k_topics], name='row_numbers')
					full_indices = tf.concat([tf.expand_dims(row_numbers, -1), tf.expand_dims(top_k_h_indices, -1)], axis=2)
					full_indices = tf.reshape(full_indices, [-1, 2], name='full_indices')
					last_h_softmax = tf.scatter_nd(
						full_indices, 
						tf.reshape(tf.nn.softmax(top_k_h_values, axis=1), [-1]), 
						#tf.ones([self.input_batch_size * params.use_k_topics], dtype=tf.float32), 
						[self.input_batch_size, self.n_topic], 
						name='last_h_softmax'
					)
				else:
					last_h_softmax = tf.nn.softmax(self.last_h, axis=1, name='last_h_softmax')	
				
				self.last_h_topic_emb = tf.squeeze(tf.matmul(tf.expand_dims(last_h_softmax, axis=1), self.topic_embeddings), axis=1, name='last_h_topic_emb')
				if concat_topic_emb_and_prop:
					self.last_h_topic_emb = tf.concat([self.last_h_topic_emb, self.last_h], axis=1, name='last_h_topic_emb_concat')

			# Code segment for Sentence-level topical discourse

			if params.use_sent_topic_rep:
				if self.n_sample == 1:
					eps_sent = tf.random_normal((self.batch_size_sent, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
					self.last_h_sent = tf.add(tf.multiply(tf.exp(self.logsigm_sent), eps_sent), self.mean_sent, name='sent_hidden')
				else:
					eps_sent = tf.random_normal((self.n_sample*self.batch_size_sent, self.n_topic), mean=0.0, stddev=1.0, seed=seed)
					eps_sent_list = tf.split(eps_sent, self.n_sample, 0)
					recons_loss_list = []
					sent_vec_list = []
					for i in range(self.n_sample):
						if i > 0: tf.get_variable_scope().reuse_variables()
						curr_eps = eps_sent_list[i]
						sent_vec = tf.add(tf.multiply(tf.exp(self.logsigm_sent), curr_eps), self.mean_sent)
						if GSM:
							sent_vec = tf.nn.softmax(sent_vec, axis=1)
						sent_vec_list.append(sent_vec)
					self.last_h_sent = tf.add_n(sent_vec_list) / self.n_sample
				self.last_h_sent = tf.reshape(self.last_h_sent, [self.input_batch_size_sent, self.input_batch_len_sent, self.n_topic])

				if params.use_topic_embedding:
					if params.use_k_topics > 0:
						# Masking sentence topic proportion vector
						top_k_h_sent_values, top_k_h_sent_indices = tf.nn.top_k(self.last_h_sent, k=params.use_k_topics, sorted=False, name='top_k_h_sent')
						row_numbers_sent = tf.tile(tf.expand_dims(tf.range(0, self.batch_size_sent), 1), [1, params.use_k_topics], name='row_numbers_sent')
						full_indices_sent = tf.concat([tf.expand_dims(row_numbers_sent, -1), tf.expand_dims(top_k_h_sent_indices, -1)], axis=2)
						full_indices_sent = tf.reshape(full_indices_sent, [-1, 2], name='full_indices_sent')
						last_h_softmax_sent = tf.scatter_nd(
							full_indices_sent, 
							tf.reshape(tf.nn.softmax(top_k_h_sent_values, axis=1), [-1]), 
							[self.batch_size_sent, self.n_topic], 
							name='last_h_softmax_sent')
					else:
						last_h_softmax_sent = tf.nn.softmax(self.last_h_sent, axis=2, name='last_h_softmax_sent')
					
					self.last_h_topic_emb_sent = tf.matmul(last_h_softmax_sent, self.topic_embeddings, name='last_h_topic_emb_sent')
					if concat_topic_emb_and_prop:
						self.last_h_topic_emb_sent = tf.concat([self.last_h_topic_emb_sent, self.last_h_sent], axis=2, name='last_h_topic_emb_sent_concat')

		#self.objective_TM = self.recons_loss + self.kld
		#self.objective_TM = tf.add(self.recons_loss, self.kld, name='TM_loss_unnormed')
		self.final_loss = tf.add(self.recons_loss, self.kld, name='TM_loss_unnormed')
		self.objective_TM = tf.reduce_mean(self.final_loss)

		if params.TM_uniqueness_loss:
			## TCNLM topic uniqueness loss
			normed_topic_matrix = self.decoding_matrix / tf.reduce_sum(self.decoding_matrix, axis=1, keepdims=True)
			l2_normalized_topic_matrix = tf.nn.l2_normalize(normed_topic_matrix, axis=1)
			cosine_similarity = tf.matmul(
										l2_normalized_topic_matrix,
										l2_normalized_topic_matrix,
										transpose_a=False,
										transpose_b=True
										)
			cosine_distance = tf.subtract(1.0, cosine_similarity)
			mean_cosine_distance = tf.reduce_mean(cosine_distance)
			variance = tf.reduce_mean(tf.square(tf.subtract(cosine_distance, mean_cosine_distance)))
			#uniqueness_loss = mean_cosine_distance - variance
			uniqueness_loss = - mean_cosine_distance + variance
			self.objective_TM += params.alpha_uniqueness * uniqueness_loss
			#self.objective_TM += 0.01 * uniqueness_loss

		if params.topic_coherence_reg:
			#E_normalized = W_prior / tf.reduce_sum(W_prior, axis=1, keepdims=True)
			E_normalized = tf.nn.l2_normalize(W_prior, axis=1, name='E_normalized')
			#W_normalized = self.decoding_matrix / tf.reduce_sum(self.decoding_matrix, axis=1, keepdims=True)
			W_normalized = tf.nn.l2_normalize(self.decoding_matrix, axis=1, name='W_normalized')
			topic_vectors = tf.transpose(tf.matmul(W_normalized, E_normalized), [1,0], name='topic_vectors')
			#topic_vectors_normalized = topic_vectors / tf.reduce_sum(topic_vectors, axis=1, name='topic_vectors_normalized')
			topic_vectors_normalized = tf.nn.l2_normalize(topic_vectors, axis=0, name='topic_vectors_normalized')
			cos_sim_matrix = tf.transpose(tf.matmul(E_normalized, topic_vectors_normalized), [1,0], name='cos_sim_matrix')
			coherence_loss = - tf.reduce_sum(tf.multiply(cos_sim_matrix, W_normalized), name="coherence_loss")
			self.objective_TM += params.beta_coherence * coherence_loss
		

		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		#fullvars = tf.trainable_variables()

		#enc_vars = utils.variable_parser(fullvars, 'TM_encoder')
		enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder')
		#dec_vars = utils.variable_parser(fullvars, 'TM_decoder')
		dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')

		enc_grads = tf.gradients(self.objective_TM, enc_vars)
		dec_grads = tf.gradients(self.objective_TM, dec_vars)

		self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
		self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))


	## Pretraining of NVDM-TM

	def pretrain(self, dataset, params, session,
				training_epochs=20, alternate_epochs=10):
				#training_epochs=1000, alternate_epochs=10):

		log_dir = os.path.join(params.model, 'logs_nvdm_pretrain')
		model_dir_ir_nvdm = os.path.join(params.model, 'model_ir_nvdm_pretrain')
		model_dir_ppl_nvdm = os.path.join(params.model, 'model_ppl_nvdm_pretrain')

		if not os.path.isdir(log_dir):
			os.mkdir(log_dir)
		if not os.path.isdir(model_dir_ir_nvdm):
			os.mkdir(model_dir_ir_nvdm)
		if not os.path.isdir(model_dir_ppl_nvdm):
			os.mkdir(model_dir_ppl_nvdm)

		train_url = os.path.join(params.dataset, 'training_nvdm_docs_non_replicated.csv')
		dev_url = os.path.join(params.dataset, 'validation_nvdm_docs_non_replicated.csv')
		test_url = os.path.join(params.dataset, 'test_nvdm_docs_non_replicated.csv')

		train_set, train_count = utils.data_set(train_url)
		test_set, test_count = utils.data_set(test_url)
		dev_set, dev_count = utils.data_set(dev_url)

		dev_batches = utils.create_batches(len(dev_set), self.batch_size, shuffle=False)
		#dev_batches = utils.create_batches(len(dev_set), 512, shuffle=False)
		test_batches = utils.create_batches(len(test_set), self.batch_size, shuffle=False)
		#test_batches = utils.create_batches(len(test_set), 512, shuffle=False)
		
		training_labels = np.array(
			[[y] for y, _ in dataset.rows('training_nvdm_docs_non_replicated', num_epochs=1)]
		)
		validation_labels = np.array(
			[[y] for y, _ in dataset.rows('validation_nvdm_docs_non_replicated', num_epochs=1)]
		)
		test_labels = np.array(
			[[y] for y, _ in dataset.rows('test_nvdm_docs_non_replicated', num_epochs=1)]
		)

		patience = params.pretrain_patience
		patience_count = 0
		best_dev_ppl = np.inf
		best_test_ppl = np.inf
		best_val_nvdm_IR = -1.0
		best_test_nvdm_IR = -1.0

		enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder')
		dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')
		pretrain_saver = tf.train.Saver(enc_vars + dec_vars)

		ppl_model = False
		ir_model = False
		
		for epoch in range(training_epochs):
			epoch_counter = epoch + 1
			train_batches = utils.create_batches(len(train_set), self.batch_size, shuffle=True)
			#train_batches = utils.create_batches(len(train_set), 512, shuffle=True)
			
			#-------------------------------
			# train
			for switch in range(0, 2):
				if switch == 0:
					optim = self.optim_dec
					print_mode = 'updating decoder'
				else:
					optim = self.optim_enc
					print_mode = 'updating encoder'
				for i in range(alternate_epochs):
					print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
						train_batches, 
						train_set, 
						train_count, 
						params, 
						session,
						optimizer=optim
					)

					print('| Epoch train: {:d} |'.format(epoch_counter), 
						print_mode, '{:d}'.format(i),
						'| Corpus Perplexity: {:.5f}'.format(print_ppx),  # perplexity for all docs
						'| Per doc Perplexity: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
						'| KLD: {:.5}'.format(print_kld))
			
			if epoch_counter >= 1 and epoch_counter % params.nvdm_validation_ppl_freq == 0:
				ppl_model = True
				
				print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
					dev_batches, 
					dev_set, 
					dev_count, 
					params, 
					session
				)

				if print_ppx_perdoc < best_dev_ppl:
					best_dev_ppl = print_ppx_perdoc
					print("Saving best model.")
					pretrain_saver.save(session, model_dir_ppl_nvdm + '/model_ppl_nvdm_pretrain', global_step=1)
					patience_count = 0
				else:
					patience_count += 1

				print('| Epoch dev: {:d} |'.format(epoch_counter), 
					'| Corpus Perplexity: {:.9f} |'.format(print_ppx),
					'| Per doc Perplexity: {:.5f} |'.format(print_ppx_perdoc),
					'| KLD: {:.5} |'.format(print_kld),
					'| Best dev PPL: {:.5} |'.format(best_dev_ppl))
				
				with open(log_dir + "/logs_ppl_nvdm_pretrain.txt", "a") as f:
					f.write('| Epoch Val: {:d} || Val Corpus PPL: {:.9f} || Val Per doc PPL: {:.5f} || Best Val PPL: {:.5} || KLD Val: {:.5} |\n'.format(epoch+1, print_ppx, print_ppx_perdoc, best_dev_ppl, print_kld))

			if patience_count > patience:
				print("Early stopping.")
				break

		if ppl_model:
			print("Restoring best model.")
			pretrain_saver.restore(session, tf.train.latest_checkpoint(model_dir_ppl_nvdm))

			print("Calculating Val PPL.")
			print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
				dev_batches, 
				dev_set, 
				dev_count, 
				params, 
				session
			)

			print('| Val Corpus Perplexity: {:.9f}'.format(print_ppx),
					'| Val Per doc Perplexity: {:.5f}'.format(print_ppx_perdoc),
					'| Val KLD: {:.5}'.format(print_kld))
			
			with open(log_dir + "/logs_ppl_nvdm_pretrain.txt", "a") as f:
				f.write('\n\nVal Corpus PPL: {:.9f} || Val Per doc PPL: {:.5f} || KLD Val: {:.5} |\n'.format(print_ppx, print_ppx_perdoc, print_kld))
			
			print("Calculating Test PPL.")
			print_ppx, print_ppx_perdoc, print_kld = self.run_epoch(
				test_batches, 
				test_set, 
				test_count, 
				params, 
				session
			)

			print('| Test Corpus Perplexity: {:.9f}'.format(print_ppx),
					'| Test Per doc Perplexity: {:.5f}'.format(print_ppx_perdoc),
					'| Test KLD: {:.5}'.format(print_kld))

			with open(log_dir + "/logs_ppl_nvdm_pretrain.txt", "a") as f:
				f.write('\n\nTest Corpus PPL: {:.9f} || Test Per doc PPL: {:.5f} || KLD Test: {:.5} |\n'.format(print_ppx, print_ppx_perdoc, print_kld))

			# Topics with W matrix

			top_n_topic_words = 20
			w_h_top_words_indices = []
			W_topics = session.run(self.decoding_matrix)
			topics_list_W = []

			for h_num in range(np.array(W_topics).shape[0]):
				w_h_top_words_indices.append(np.argsort(W_topics[h_num, :])[::-1][:top_n_topic_words])

			with open(params.docnadeVocab, 'r') as f:
				vocab_docnade = [w.strip() for w in f.readlines()]
			#vocab_docnade = TM_vocab

			with open(os.path.join(log_dir, "TM_topics_pretrain.txt"), "w") as f:
				for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
					w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
					
					topics_list_W.append(w_h_top_words)
					
					print('h_num: %s' % h_num)
					print('w_h_top_words_indx: %s' % w_h_top_words_indx)
					print('w_h_top_words:%s' % w_h_top_words)
					print('----------------------------------------------------------------------')

					f.write('h_num: %s\n' % h_num)
					f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
					f.write('w_h_top_words:%s\n' % w_h_top_words)
					f.write('----------------------------------------------------------------------\n')

			# Compute Topic Coherence with internal corpus
			
			topic_file = os.path.join(log_dir, "TM_topics_pretrain.txt")
			ref_corpus = params.dataset + '_corpus'
			coherence_file_path = os.path.join(log_dir, "window_size_document")
			if not os.path.exists(coherence_file_path):
				os.makedirs(coherence_file_path)
			coherence_file = os.path.join(coherence_file_path, "topics-oc_bnc_internal.txt")
			wordcount_file = os.path.join(coherence_file_path, "wc-oc_bnc_internal.txt")

			os.system('python ./topic_coherence_code_python3/ComputeTopicCoherence.py ' + topic_file + ' ' + ref_corpus + ' ' + wordcount_file + ' ' + coherence_file)

			# Compute Topic Coherence with external corpus

			ref_corpus = "/home/ubuntu/TM_LM_code/topic_coherence_code/wiki_corpus"
			coherence_file = os.path.join(coherence_file_path, "topics-oc_bnc_external.txt")
			wordcount_file = os.path.join(coherence_file_path, "wc-oc_bnc_external.txt")

			os.system('python ./topic_coherence_code_python3/ComputeTopicCoherence.py ' + topic_file + ' ' + ref_corpus + ' ' + wordcount_file + ' ' + coherence_file)


	def hidden_vectors(self, data, params, session):
		vecs = []
		for y, x, count, mask in data:

			feed_dict = {
				self.x.name: x,
				self.mask.name: mask#,
			}
			
			vecs.extend(
				session.run([self.last_h], feed_dict=feed_dict)[0]
			)
		
		return np.array(vecs)

	def run_epoch(self, input_batches, input_set, input_count, params, session, optimizer=None):
		loss_sum = 0.0
		ppx_sum = 0.0
		kld_sum = 0.0
		word_count = 0
		doc_count = 0
		for idx_batch in input_batches:
			data_batch, count_batch, mask = utils.fetch_data(
			input_set, input_count, idx_batch, self.vocab_size)
			
			input_feed = {self.x.name: data_batch, self.mask.name: mask}
			
			if not optimizer is None:
				_, (loss, kld) = session.run((optimizer, 
											[self.final_loss, self.kld]),
											input_feed)
			else:
				loss, kld = session.run([self.final_loss, self.kld],
											input_feed)
			
			loss_sum += np.sum(loss)
			kld_sum += np.sum(kld) / np.sum(mask) 
			word_count += np.sum(count_batch)
			# to avoid nan error
			count_batch = np.add(count_batch, 1e-12)
			# per document loss
			ppx_sum += np.sum(np.divide(loss, count_batch)) 
			doc_count += np.sum(mask)
		print_ppx = np.exp(loss_sum / word_count)
		print_ppx_perdoc = np.exp(ppx_sum / doc_count)
		print_kld = kld_sum/len(input_batches)

		return print_ppx, print_ppx_perdoc, print_kld
import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data_lstm as data
import model.model_NVDM as m_NVDM
import model.model_LSTM as m_LSTM
import pickle
import keras.preprocessing.sequence as pp
import datetime
import sys
from math import *
from nltk.corpus import wordnet
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import time
import model.utils as u
from model.softmax import AdaptiveSoftmax
from model.softmax import FullSoftmax
#from fastText import load_model
from fasttext import load_model
from gensim.models import KeyedVectors

# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

#dir(tf.contrib)

def loadGloveModel(vocab, gloveFile=None, params=None):
	hidden_size = 300
	if gloveFile is None:
		if hidden_size == 50:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
		elif hidden_size == 100:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
		elif hidden_size == 200:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
		elif hidden_size == 300:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.300d.txt")
		else:
			print('Invalid dimension [%d] for Glove pretrained embedding matrix!!' %params.hidden_size)
			exit()

	print("Loading Glove Model")
	f = open(gloveFile, 'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	print("Done.", len(model), " words loaded!")

	missing_words = 0
	embedding_matrix = np.zeros((len(vocab), hidden_size), dtype=np.float32)
	for i, word in enumerate(vocab):
		if str(word).lower() in model.keys():
			if len(model[str(word).lower()]) == 0:
				embedding_matrix[i, :] = np.zeros((hidden_size), dtype=np.float32)
				missing_words += 1
			else:
				embedding_matrix[i, :] = np.array(model[str(word).lower()], dtype=np.float32)
		else:
			embedding_matrix[i, :] = np.zeros((hidden_size), dtype=np.float32)
	
	#embedding_matrix = tf.convert_to_tensor(embedding_matrix)
	print("Total missing words:%d out of %d" %(missing_words, len(vocab)))

	#return model
	return embedding_matrix

def init_embedding(model, idxvocab):
	word_emb = []
	for vi, v in enumerate(idxvocab):
		if v in model:
			word_emb.append(model[v])
		else:
			word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
	return np.array(word_emb, dtype=np.float32)

def loadWord2VecModel(vocab, word2vec_file=None, params=None):
	if word2vec_file is None:
		#word2vec_file = os.path.join(home_dir, "resources/pretrained_embeddings/GoogleNews-vectors-negative300.bin")
		word2vec_file = "./resources/pretrained_embeddings/GoogleNews-vectors-negative300.bin"
	model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
	embedding_matrix = init_embedding(model, vocab)
	return embedding_matrix

def loadFastTextModel(vocab, fasttext_file=None, params=None):
	if fasttext_file is None:
		#fasttext_file = os.path.join(home_dir, "resources/pretrained_embeddings/wiki.en.bin")
		fasttext_file = "./resources/pretrained_embeddings/wiki.en.bin"
	model = load_model(fasttext_file)
	prior_vecs = []
	for word in vocab:
		prior_vecs.append(model.get_word_vector(word.strip()))
	embedding_matrix = np.array(prior_vecs, dtype=np.float32)
	return embedding_matrix


def train(model, dataset, args):

	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=args.num_cores,
		intra_op_parallelism_threads=args.num_cores,
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session:

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		if args.pretrain_LM:
			print("Pretraining of LSTM-LM started.")
			model.pretrain(dataset, args, session)
			print("Pretraining of LSTM-LM finished.")
		else:
			print("No pretraining for LSTM-LM.")
		
		if args.pretrain_TM:
			print("Pretraining of TM started.")
			model.topic_model.pretrain(dataset, args, session)
			print("Pretraining of TM finished.")
		else:
			print("No pretraining for TM.")
		
		if args.combined_training:
			print("Combined training started.")
			train_NVDM_LM(model, dataset, args, session)
			print("Combined training finished.")
		else:
			print("No combined training.")

	print("Program finished.")


def train_NVDM_LM(model, dataset, params, session):
	log_dir = os.path.join(params.model, 'logs')
	model_dir_ir_TM = os.path.join(params.model, 'model_ir_TM')
	model_dir_ir_LM = os.path.join(params.model, 'model_ir_LM')
	model_dir_ir_comb = os.path.join(params.model, 'model_ir_comb')
	model_dir_ppl_comb = os.path.join(params.model, 'model_ppl_comb')
	model_dir_ppl_TM = os.path.join(params.model, 'model_ppl_TM')
	model_dir_ppl_LM = os.path.join(params.model, 'model_ppl_LM')

	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	if not os.path.isdir(model_dir_ir_TM):
		os.mkdir(model_dir_ir_TM)
	if not os.path.isdir(model_dir_ir_LM):
		os.mkdir(model_dir_ir_LM)
	if not os.path.isdir(model_dir_ir_comb):
		os.mkdir(model_dir_ir_comb)
	if not os.path.isdir(model_dir_ppl_comb):
		os.mkdir(model_dir_ppl_comb)
	if not os.path.isdir(model_dir_ppl_TM):
		os.mkdir(model_dir_ppl_TM)
	if not os.path.isdir(model_dir_ppl_LM):
		os.mkdir(model_dir_ppl_LM)
	
	avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
	tf.summary.scalar('loss', avg_loss)

	validation = tf.placeholder(tf.float32, [], 'validation_ph')
	validation_accuracy = tf.placeholder(tf.float32, [], 'validation_acc')
	tf.summary.scalar('validation', validation)
	tf.summary.scalar('validation_accuracy', validation_accuracy)

	summary_writer = tf.summary.FileWriter(log_dir, session.graph)
	summaries = tf.summary.merge_all()

	# TM Trainable variables
	TM_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_encoder') + \
							tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TM_decoder')
	
	# LM Trainable variables
	rnn_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RNN')
	if params.combination_type == 'concat':
		if params.common_space:
			rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_RNN')
		else:
			rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_combined_RNN')
	elif (params.combination_type == 'sum') or (params.combination_type == 'gating'):
		rnn_softmax_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_pretrain_RNN')
	LM_trainable_variables = rnn_trainable_variables + rnn_softmax_trainable_variables
	
	saver_TM = tf.train.Saver(TM_trainable_variables)
	saver_LM = tf.train.Saver(LM_trainable_variables)
	saver_comb = tf.train.Saver(TM_trainable_variables + LM_trainable_variables)
	
	if params.pretrain_LM:
		saver_LM.save(session, model_dir_ppl_LM + '/model_ppl_LM', global_step=1)

	if params.pretrain_TM:
		saver_TM.save(session, model_dir_ppl_TM + '/model_ppl_TM', global_step=1)

	losses = []

	training_data_filename_TM_for_LM = 'training_nvdm_docs_minus_sents'
	validation_data_filename_TM_for_LM = 'validation_nvdm_docs_minus_sents'
	test_data_filename_TM_for_LM = 'test_nvdm_docs_minus_sents'

	training_data_filename_TM = 'training_nvdm_docs_non_replicated'
	validation_data_filename_TM = 'validation_nvdm_docs_non_replicated'
	test_data_filename_TM = 'test_nvdm_docs_non_replicated'

	training_data_filename_LM_sents = 'training_lstm_sents'
	validation_data_filename_LM_sents = 'validation_lstm_sents'
	test_data_filename_LM_sents = 'test_lstm_sents'

	training_data_filename_LM_docs = 'training_lstm_docs'
	validation_data_filename_LM_docs = 'validation_lstm_docs'
	test_data_filename_LM_docs = 'test_lstm_docs'

	train_url = os.path.join(params.dataset, training_data_filename_TM + '.csv')
	dev_url = os.path.join(params.dataset, validation_data_filename_TM + '.csv')
	test_url = os.path.join(params.dataset, test_data_filename_TM + '.csv')

	train_set, train_count = u.data_set(train_url)
	test_set, test_count = u.data_set(test_url)
	dev_set, dev_count = u.data_set(dev_url)

	dev_batches = u.create_batches(len(dev_set), params.batch_size, shuffle=False)
	test_batches = u.create_batches(len(test_set), params.batch_size, shuffle=False)

	# This currently streams from disk. You set num_epochs=1 and
	# wrap this call with something like itertools.cycle to keep
	# this data in memory.
	training_data_TM = dataset.batches_nvdm_LM(training_data_filename_TM_for_LM, params.batch_size, params.TM_vocab_length, multilabel=params.multi_label)
	training_data_LM = dataset.batches_split(training_data_filename_LM_sents, params.batch_size, shuffle=False, multilabel=params.multi_label)

	best_val_nvdm_IR = -1.0
	best_val_lstm_IR = -1.0
	best_val_comb_IR = -1.0
	best_val_nll = np.inf
	best_val_nvdm_ppl = np.inf
	best_val_lstm_ppl = np.inf
	best_train_nll = np.inf

	training_labels = np.array(
		[[y] for y, _ in dataset.rows(training_data_filename_TM, num_epochs=1)]
	)
	validation_labels = np.array(
		[[y] for y, _ in dataset.rows(validation_data_filename_TM, num_epochs=1)]
	)
	test_labels = np.array(
		[[y] for y, _ in dataset.rows(test_data_filename_TM, num_epochs=1)]
	)
	
	training_labels_LM = np.array(
		[[y] for y, _ in dataset.rows(training_data_filename_LM_sents, num_epochs=1)]
	)

	ir_model = False
	ppl_model = False

	with open(params.dataset + "/vocab_lstm.vocab", 'r') as f:
		LM_vocab = [w.strip() for w in f.readlines()]

	with open(params.dataset + "/vocab_nvdm.vocab", 'r') as f:
		TM_vocab = [w.strip() for w in f.readlines()]

	if params.use_char_embeddings:
		with open(params.rnnVocab, 'r') as f:
			vocab_rnn_word = [w.strip() for w in f.readlines()]

		with open(params.rnnCharVocab, 'r') as f:
			vocab_rnn_char = [w.strip() for w in f.readlines()]
	

	TM_train_freq = (float(len(training_labels_LM)) // params.batch_size) + 1

	epoch = -1
	for step in range(params.num_steps + 1):
		
		if step % TM_train_freq == 0:
			epoch += 1
			train_batches = u.create_batches(len(train_set), model.topic_model.batch_size, shuffle=True)
			
			#-------------------------------
			# train
			for switch in range(0, 2):
				if switch == 0:
					optim = model.topic_model.optim_dec
					print_mode = 'updating decoder'
				else:
					optim = model.topic_model.optim_enc
					print_mode = 'updating encoder'
				
				for i in range(1):

					print_ppx, print_ppx_perdoc, print_kld = model.topic_model.run_epoch(
						train_batches, 
						train_set, 
						train_count, 
						params, 
						session,
						optimizer=optim
					)
					
					print('| Epoch train: {:d} |'.format(epoch+1), 
						print_mode, '{:d}'.format(i),
						'| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
						'| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
						'| KLD: {:.5}'.format(print_kld))
		
		y, x, count, mask = next(training_data_TM)
		y_rnn, x_rnn, rnn_seq_lengths, split_indices, x_rnn_original, seq_lengths_original = next(training_data_LM)
		if params.use_bilm:
			x_rnn_input = x_rnn
			x_rnn_output = x_rnn[:, 1:-1]
		else:
			x_rnn_input = x_rnn[:, :-1]
			x_rnn_output = x_rnn[:, 1:]
			rnn_seq_lengths -= 1

		if params.use_topic_embedding:
			x_doc_mask = m_LSTM.get_doc_topic_mask_nvdm(TM_vocab, LM_vocab, x, x_rnn_original)
			x_doc_mask = x_doc_mask[split_indices]
		
		x = x[split_indices]
		mask = mask[split_indices]
		y = [y[index] for index in split_indices]
		count = [count[index] for index in split_indices]

		train_feed_dict = {
			model.topic_model.x.name: x,
			model.topic_model.mask.name: mask,
			model.topic_model.input_batch_size: x.shape[0]
		}

		if params.use_topic_embedding:
			train_feed_dict[model.topic_model.x_doc_mask] = x_doc_mask

		if params.use_sent_topic_rep:
			x_rnn_new = m_LSTM.get_sent_topic_reps_nvdm(TM_vocab, LM_vocab, x_rnn, rnn_seq_lengths)
			train_feed_dict[model.topic_model.x_sent] = x_rnn_new

		train_feed_dict[model.x_rnn_input] = x_rnn_input
		train_feed_dict[model.x_rnn_output] = x_rnn_output
		train_feed_dict[model.y_rnn] = y_rnn
		train_feed_dict[model.rnn_seq_lengths] = rnn_seq_lengths
		train_feed_dict[model.lstm_dropout_keep_prob] = params.lstm_dropout_keep_prob
		train_feed_dict[model.tm_dropout_keep_prob] = params.tm_dropout_keep_prob

		if params.use_char_embeddings:
			x_rnn_char_input, rnn_char_seq_lengths = get_char_indices(x_rnn_input, vocab_rnn_word, vocab_rnn_char)
			train_feed_dict[model.x_rnn_char_input] = x_rnn_char_input
			train_feed_dict[model.rnn_char_seq_lengths] = rnn_char_seq_lengths
		
		if params.supervised:
			sys.exit()
		else:
			_, total_loss = session.run([model.opt_comb, model.total_loss], feed_dict=train_feed_dict)
		
		losses.append(total_loss)
		
		if (step % params.log_every == 0):
			print('{}: {:.6f}'.format(step, total_loss))
		
		if step >= 1 and step % params.lstm_validation_ppl_freq == 0:
			ppl_model = True

			total_val_nll, total_val_lstm_ppl = model.run_epoch_comb_nvdm(
				dataset.batches_split(validation_data_filename_LM_sents, params.batch_size, num_epochs=1, shuffle=False, multilabel=params.multi_label),
				dataset.batches_nvdm_LM(validation_data_filename_TM_for_LM, params.batch_size, params.TM_vocab_length, num_epochs=1, multilabel=params.multi_label),
				TM_vocab,
				LM_vocab,
				params,
				session
			)

			_, total_val_nvdm_ppl, _ = model.topic_model.run_epoch(
				dev_batches, 
				dev_set, 
				dev_count, 
				params, 
				session
			)
			
			if total_val_nvdm_ppl < best_val_nvdm_ppl:
				best_val_nvdm_ppl = total_val_nvdm_ppl
				print('saving: {}'.format(model_dir_ppl_TM))
				TM_saver_path = saver_TM.save(session, model_dir_ppl_TM + '/model_ppl_TM', global_step=1)
			
			if total_val_lstm_ppl < best_val_lstm_ppl:
				best_val_lstm_ppl = total_val_lstm_ppl
				print('saving: {}'.format(model_dir_ppl_LM))
				saver_comb.save(session, model_dir_ppl_LM + '/model_ppl_LM', global_step=1)
				patience_count = 0
			else:
				patience_count += 1
			
			# Early stopping
			if total_val_nll < best_val_nll:
				best_val_nll = total_val_nll

			print('NVDM val PPL: {:.3f},	LSTM val PPL: {:.3f}	(Best NVDM val PPL: {:.3f},	Best LSTM val PPL: {:.3f},	Best val NLL: {:.3f})'.format(
				total_val_nvdm_ppl,
				total_val_lstm_ppl,
				best_val_nvdm_ppl,
				best_val_lstm_ppl,
				best_val_nll
			))

			# logging information
			with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
				f.write("Step: %i,   NVDM val PPL: %s,   LSTM val PPL: %s,	(Best NVDM val PPL: %s,	Best LSTM val PPL: %s,	Best val NLL: %s)\n" %
						(step, total_val_nvdm_ppl, total_val_lstm_ppl, best_val_nvdm_ppl, best_val_lstm_ppl, best_val_nll))

			# Early stopping
			if patience_count > params.patience:
				print("Early stopping criterion satisfied.")
				break

	if ppl_model:
		saver_comb.restore(session, tf.train.latest_checkpoint(model_dir_ppl_LM))

		total_val_nll, total_val_lstm_ppl = model.run_epoch_comb_nvdm(
			dataset.batches_split(validation_data_filename_LM_sents, params.batch_size, num_epochs=1, shuffle=False, multilabel=params.multi_label),
			dataset.batches_nvdm_LM(validation_data_filename_TM_for_LM, params.batch_size, params.TM_vocab_length, num_epochs=1, multilabel=params.multi_label),
			TM_vocab,
			LM_vocab,
			params,
			session
		)

		_, total_val_nvdm_ppl, _ = model.topic_model.run_epoch(
			dev_batches, 
			dev_set, 
			dev_count, 
			params, 
			session
		)

		print('NVDM val PPL: {:.3f},	LSTM val PPL: {:.3f}'.format(total_val_nvdm_ppl, total_val_lstm_ppl))

		# logging information
		with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
			f.write("NVDM val PPL: %s,   LSTM val PPL: %s\n" % (total_val_nvdm_ppl, total_val_lstm_ppl))

		total_test_nll, total_test_lstm_ppl = model.run_epoch_comb_nvdm(
			dataset.batches_split(test_data_filename_LM_sents, params.batch_size, num_epochs=1, shuffle=False, multilabel=params.multi_label),
			dataset.batches_nvdm_LM(test_data_filename_TM_for_LM, params.batch_size, params.TM_vocab_length, num_epochs=1, multilabel=params.multi_label),
			TM_vocab,
			LM_vocab,
			params,
			session
		)

		_, total_test_nvdm_ppl, _ = model.topic_model.run_epoch(
			test_batches, 
			test_set, 
			test_count, 
			params, 
			session
		)

		print('NVDM test PPL: {:.3f},	LSTM test PPL: {:.3f}'.format(total_test_nvdm_ppl, total_test_lstm_ppl))

		# logging information
		with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
			f.write("NVDM test PPL: %s,   LSTM test PPL: %s\n" % (total_test_nvdm_ppl, total_test_lstm_ppl))

		# Topics with W matrix

		top_n_topic_words = 20
		w_h_top_words_indices = []
		W_topics = session.run(model.topic_model.decoding_matrix)
		topics_list_W = []

		for h_num in range(np.array(W_topics).shape[0]):
			w_h_top_words_indices.append(np.argsort(W_topics[h_num, :])[::-1][:top_n_topic_words])

		with open(params.docnadeVocab, 'r') as f:
			vocab_docnade = [w.strip() for w in f.readlines()]
		#vocab_docnade = TM_vocab

		with open(os.path.join(log_dir, "TM_topics.txt"), "w") as f:
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
			
		topic_file = os.path.join(log_dir, "TM_topics.txt")
		ref_corpus = params.dataset + '_corpus'
		coherence_file_path = os.path.join(log_dir, "topic_coherence")
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


def get_vectors_from_matrix(matrix, batches):
	# matrix: embedding matrix of shape = [vocab_size X embedding_size]
	vecs = []
	for _, x, seq_length in batches:
		temp_vec = np.zeros((matrix.shape[1]), dtype=np.float32)
		indices = x[0, :seq_length[0]]
		for index in indices:
			temp_vec += matrix[index, :]
		vecs.append(temp_vec)
	return np.array(vecs)

def softmax(X, theta = 1.0, axis = None):
	"""
	Compute the softmax of each element along an axis of X.

	Parameters
	----------
	X: ND-Array. Probably should be floats. 
	theta (optional): float parameter, used as a multiplier
		prior to exponentiation. Default = 1.0
	axis (optional): axis to compute values along. Default is the 
		first non-singleton axis.

	Returns an array the same size as X. The result will sum to 1
	along the specified axis.
	"""

	# make X at least 2d
	y = np.atleast_2d(X)

	# find axis
	if axis is None:
		axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

	# multiply y against the theta parameter, 
	y = y * float(theta)

	# subtract the max for numerical stability
	y = y - np.expand_dims(np.max(y, axis = axis), axis)
	
	# exponentiate y
	y = np.exp(y)

	# take the sum along the specified axis
	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

	# finally: divide elementwise
	p = y / ax_sum

	# flatten if X was 1D
	if len(X.shape) == 1: p = p.flatten()

	return p


def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 3)


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
	args.supervised = str2bool(args.supervised)
	args.use_TM_for_ir = str2bool(args.use_TM_for_ir)
	args.use_lstm_for_ir = str2bool(args.use_lstm_for_ir)
	args.use_combination_for_ir = str2bool(args.use_combination_for_ir)
	args.initialize_docnade = str2bool(args.initialize_docnade)
	args.initialize_nvdm = str2bool(args.initialize_nvdm)
	args.initialize_rnn = str2bool(args.initialize_rnn)
	args.update_docnade_w = str2bool(args.update_docnade_w)
	args.update_rnn_w = str2bool(args.update_rnn_w)
	args.include_lstm_loss = str2bool(args.include_lstm_loss)
	args.common_space = str2bool(args.common_space)
	args.deep = str2bool(args.deep)
	args.multi_label = str2bool(args.multi_label)
	args.reload = str2bool(args.reload)
	args.reload_train = str2bool(args.reload_train)
	args.reload_docnade_embeddings = str2bool(args.reload_docnade_embeddings)
	args.pretrain_LM = str2bool(args.pretrain_LM)
	args.pretrain_TM = str2bool(args.pretrain_TM)
	args.combined_training = str2bool(args.combined_training)
	args.use_char_embeddings = str2bool(args.use_char_embeddings)
	args.use_bilm = str2bool(args.use_bilm)
	args.use_crf = str2bool(args.use_crf)
	args.TM_uniqueness_loss = str2bool(args.TM_uniqueness_loss)
	args.use_topic_embedding = str2bool(args.use_topic_embedding)
	args.use_MOR = str2bool(args.use_MOR)
	args.use_sent_topic_rep = str2bool(args.use_sent_topic_rep)
	args.prior_emb_for_topics = str2bool(args.prior_emb_for_topics)
	args.topic_coherence_reg = str2bool(args.topic_coherence_reg)

	dataset = data.Dataset(args.dataset)
	args.rnnVocab = args.dataset + "/vocab_lstm.vocab"
	args.docnadeVocab = args.dataset + "/vocab_nvdm.vocab"
	args.rnnCharVocab = args.dataset + "/char_vocab_lstm.vocab"

	with open(args.rnnVocab, 'r') as f:
		vocab_rnn = [w.strip() for w in f.readlines()]

	with open(args.docnadeVocab, 'r') as f:
		vocab_docnade = [w.strip() for w in f.readlines()]

	LM_vocab_length = len(vocab_rnn)
	TM_vocab_length = len(vocab_docnade)
	
	now = datetime.datetime.now()

	if args.supervised:
		args.model += "_sup"

	if args.initialize_rnn:
		args.model += "_embprior"

	args.model += "_" + str(args.TM_type) + "_" + str(args.TM_lambda)

	if args.use_bilm:
		args.model += "_biLM"
	else:
		args.model += "_uniLM"
	
	args.model += "_" +str(args.softmax_type)

	#if args.pretrain_LM:
	#	args.model += "_pretr_LM"

	#if args.pretrain_TM:
	#	args.model += "_pretr_TM"

	if args.combined_training:
		args.model += "_comb_tr_" + str(args.docnade_loss_weight) + "_" + str(args.lstm_loss_weight)

	if args.lstm_dropout_keep_prob:
		args.model += "_lm_drop_" + str(args.lstm_dropout_keep_prob)

	if args.tm_dropout_keep_prob:
		args.model += "_tm_drop_" + str(args.tm_dropout_keep_prob)

	if args.use_char_embeddings:
		args.model += "_char_emb"
	
	if args.combination_type == 'concat':
		args.model += '_concat'
	elif args.combination_type == 'sum':
		args.model += "_sum"
	elif args.combination_type == 'gating':
		args.model += "_gating"
	else:
		print("Wrong value for args.combination_type: ", args.combination_type)
		sys.exit()

	if args.common_space:
		args.model += "_proj_" + str(args.concat_proj_activation)

	if args.initialize_docnade:
		args.model += "_init_TM"

	if args.initialize_nvdm:
		args.model += "_init_TM"

	if args.initialize_rnn:
		args.model += "_init_LM"

	if args.use_topic_embedding:
		args.model += "_topic_emb"
	else:
		args.model += "_topic_prop"

	if args.use_sent_topic_rep:
		args.model += "_doc_sent_rep"
	else:
		args.model += "_doc_rep"

	if args.TM_uniqueness_loss:
		args.model += "_TM_uniq_" + str(args.alpha_uniqueness)

	if args.topic_coherence_reg:
		args.model += "_TM_coh_" + str(args.beta_coherence)

	args.model += "_topic_words_" + str(args.use_k_topic_words)
	args.model += "_topics_" + str(args.use_k_topics)
	if args.prior_emb_for_topics:
		args.model += "_E_for_topics"

	args.model +=  "_" + str(args.activation) + "_hid_DNE_" + str(args.hidden_size_TM) + "_hid_RNN_" + str(args.hidden_size_LM) \
					+ "_voc_DNE_" + str(len(vocab_docnade)) + "_voc_RNN_" + str(len(vocab_rnn)) \
					+ "_lr_" + str(args.learning_rate) \
					+ "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)

	if not os.path.isdir(args.model):
		os.mkdir(args.model)

	with open(os.path.join(args.model, 'params.json'), 'w') as f:
		f.write(json.dumps(vars(args)))
	
	rnn_embedding_matrix = None
	
	if args.initialize_rnn:
		rnn_embedding_matrix = loadWord2VecModel(vocab_rnn, params=args)
		#rnn_embedding_matrix = np.load(args.dataset + "/word2vec_embeddings_LM_init.npy")
		print("RNN initialized.")
	
	docnade_initializer_list = []
	if args.initialize_docnade:
		print("Error: Invalid value for args.initialize_docnade == %s" % args.initialize_docnade)
		sys.exit()
	else:
		docnade_initializer_list = [None, None, None, None]
		print("DocNADE not initialized.")

	nvdm_dataset_name = args.dataset.strip().split("/")[-1] + "_"
	
	nvdm_initializer_list = []
	if args.initialize_nvdm:
		print("Error: Invalid value for args.initialize_nvdm == %s" % args.initialize_nvdm)
		sys.exit()
	else:
		nvdm_initializer_list = [[[None, None]], [None, None], [None, None], [None, None]]
		print("NVDM not initialized.")

	prior_embeddings_TM = None
	if args.prior_emb_for_topics or args.topic_coherence_reg:
		prior_embeddings_TM = loadFastTextModel(vocab_docnade, params=args)
		#prior_embeddings_TM = np.load(args.dataset + "/fasttext_embeddings_topics.npy")

	import pdb; pdb.set_trace()

	if args.TM_type == "nvdm":
		model_TM = m_NVDM.NVDM(args, prior_embeddings=prior_embeddings_TM, initializer_nvdm=nvdm_initializer_list)
	else:
		print("Error: Invalid value for args.TM_type == %s" % args.TM_type)
		sys.exit()
	
	model_LM = m_LSTM.BiLstmCrf(args, topic_model=model_TM, 
								TM_loss_weight=args.docnade_loss_weight,
								LM_loss_weight=args.lstm_loss_weight,
								W_initializer=rnn_embedding_matrix)
	
	train(model_LM, dataset, args)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True,
						help='path to model output directory')
	parser.add_argument('--dataset', type=str, required=True,
						help='path to the input dataset')
	parser.add_argument('--TM-vocab-length', type=int, default=2000,
						help='DocNADE vocab size')
	parser.add_argument('--LM-vocab-length', type=int, default=2000,
						help='RNN vocab size')
	parser.add_argument('--rnn-char-vocab-length', type=int, default=2000,
						help='RNN char vocab size')
	parser.add_argument('--hidden-size-TM', type=int, default=50,
						help='size of the hidden layer of DocNADE')
	parser.add_argument('--hidden-size-LM', type=int, default=600,
						help='size of the hidden layer of LSTM-LM')
	parser.add_argument('--input-size-LM', type=int, default=300,
						help='size of the input embeddings of LSTM-LM')
	parser.add_argument('--hidden-size-LM-char', type=int, default=25,
						help='size of the character embeddings of LSTM-LM')
	parser.add_argument('--activation', type=str, default='tanh',
						help='which activation to use: sigmoid|tanh')
	parser.add_argument('--learning-rate', type=float, default=0.0004,
						help='initial learning rate')
	parser.add_argument('--num-steps', type=int, default=50000,
						help='the number of steps for train')
	parser.add_argument('--pretrain-epochs', type=int, default=50000,
						help='the number of steps for pretraining')
	parser.add_argument('--batch-size', type=int, default=64,
						help='the batch size')
	parser.add_argument('--num-samples', type=int, default=None,
						help='softmax samples (default: full softmax)')
	parser.add_argument('--num-cores', type=int, default=2,
						help='the number of CPU cores to use')
	parser.add_argument('--log-every', type=int, default=10,
						help='print loss after this many steps')
	parser.add_argument('--num-classes', type=int, default=-1,
						help='number of classes')
	parser.add_argument('--supervised', type=str, default="False",
						help='whether to use supervised model or not')
	parser.add_argument('--use-TM-for-ir', type=str, default="True",
						help='whether to use only docnade hidden vectors for ir')
	parser.add_argument('--use-lstm-for-ir', type=str, default="True",
						help='whether to use only lstm hidden vectors for ir')
	parser.add_argument('--use-combination-for-ir', type=str, default="True",
						help='whether to use docnade + lstm hidden vectors for ir')
	parser.add_argument('--combination-type', type=str, default="concat",
						help='how to combine docnade and lstm hidden vectors for ir')
	parser.add_argument('--initialize-docnade', type=str, default="False",
						help='whether to initialize embedding matrix of docnade')
	parser.add_argument('--initialize-nvdm', type=str, default="False",
						help='whether to initialize embedding matrix of nvdm')
	parser.add_argument('--initialize-rnn', type=str, default="False",
						help='whether to initialize embedding matrix of rnn')
	parser.add_argument('--update-docnade-w', type=str, default="False",
						help='whether to update docnade embedding matrix')
	parser.add_argument('--update-rnn-w', type=str, default="False",
						help='whether to update rnn embedding matrix')
	parser.add_argument('--rnnVocab', type=str, default="False",
						help='path to vocabulary file used by RNN')
	parser.add_argument('--docnadeVocab', type=str, default="False",
						help='path to vocabulary file used by DocNADE')
	parser.add_argument('--test-ppl-freq', type=int, default=100,
						help='print and log test PPL after this many steps')
	parser.add_argument('--test-ir-freq', type=int, default=100,
						help='print and log test IR after this many steps')
	parser.add_argument('--docnade-validation-ppl-freq', type=int, default=100,
						help='print and log DocNADE validation PPL after this many steps')
	parser.add_argument('--nvdm-validation-ppl-freq', type=int, default=1,
						help='print and log NVDM validation PPL after this many steps')
	parser.add_argument('--lstm-validation-ppl-freq', type=int, default=100,
						help='print and log LSTM validation PPL after this many steps')
	parser.add_argument('--docnade-validation-ir-freq', type=int, default=100,
						help='print and log DocNADE validation IR after this many steps')
	parser.add_argument('--nvdm-validation-ir-freq', type=int, default=1,
						help='print and log NVDM validation IR after this many steps')
	parser.add_argument('--lstm-validation-ir-freq', type=int, default=100,
						help='print and log LSTM validation IR after this many steps')
	parser.add_argument('--validation-bs', type=int, default=64,
						help='the validation batch size')
	parser.add_argument('--test-bs', type=int, default=64,
						help='the test batch size')
	parser.add_argument('--patience', type=int, default=500,
						help='patience for early stopping')
	parser.add_argument('--include-lstm-loss', type=str, default="False",
						help='whether to include language modeling (RNN) loss into total loss')
	parser.add_argument('--common-space', type=str, default="False",
						help='whether to project hidden vectors to a common space or not')
	parser.add_argument('--deep-hidden-sizes', nargs='+', type=int,
						help='sizes of the hidden layers')
	parser.add_argument('--deep', type=str, default="False",
						help='whether to maked model deep or not')
	parser.add_argument('--multi-label', type=str, default="False",
						help='whether dataset is multi-label or not')
	parser.add_argument('--reload', type=str, default="False",
						help='whether to reload model and evaluate or not')
	parser.add_argument('--reload-train', type=str, default="False",
						help='whether to reload model and train or not')
	parser.add_argument('--reload-model-dir', type=str, default="",
						help='path to reload model directory')
	parser.add_argument('--trainfile', type=str, default="",
						help='path to training text file')
	parser.add_argument('--valfile', type=str, default="",
						help='path to validation text file')
	parser.add_argument('--testfile', type=str, default="",
						help='path to test text file')
	parser.add_argument('--docnade-loss-weight', type=float, default=1.0,
						help='weight for contribution of docnade loss in total loss')
	parser.add_argument('--lstm-loss-weight', type=float, default=1.0,
						help='weight for contribution of lstm loss in total loss')
	parser.add_argument('--lambda-hidden-lstm', type=float, default=1.0,
						help='weight for contribution of lstm loss in total loss')
	parser.add_argument('--reload-docnade-embeddings', type=str, default="False",
						help='whether to reload docnade embeddings from a pretrained model')
	parser.add_argument('--pretrain-LM', type=str, default="False",
						help='whether to pretrain LSTM-LM')
	parser.add_argument('--pretrain-TM', type=str, default="False",
						help='whether to pretrain DocNADE Topic Model')
	parser.add_argument('--combined-training', type=str, default="False",
						help='whether to run Combined Topic + Language model')
	parser.add_argument('--use-char-embeddings', type=str, default="False",
						help='whether to use character embeddings in LSTM-LM')
	parser.add_argument('--use-crf', type=str, default="False",
						help='whether to CRF in LSTM-LM')
	parser.add_argument('--lstm-dropout-keep-prob', type=float, default=1.0,
						help='whether to use dropout in LSTM-LM')
	parser.add_argument('--tm-dropout-keep-prob', type=float, default=1.0,
						help='whether to use dropout in TM')
	parser.add_argument('--use-bilm', type=str, default="False",
						help='whether to use bidirectional LSTM-LM')
	parser.add_argument('--TM-lambda', type=float, default=1.0,
						help='lambda for combining hidden units of TM and LSTM-LM')
	parser.add_argument('--TM-type', type=str, default="docnade",
						help='which topic model to be used')
	parser.add_argument('--softmax-type', type=str, default="full",
						help='which softmax model to be used')
	parser.add_argument('--alpha-uniqueness', type=float, default=0.0,
						help='weight of uniqness loss term')
	parser.add_argument('--TM-uniqueness-loss', type=str, default="False",
						help='if to include uniqueness loss term in training objective')
	parser.add_argument('--use-topic-embedding', type=str, default="False",
						help='if to include uniqueness loss term in training objective')
	parser.add_argument('--use-MOR', type=str, default="False",
						help='if to include separate word embedding matrix for each topic')
	parser.add_argument('--use-sent-topic-rep', type=str, default="False",
						help='if to use sentence - word representation in LM')
	parser.add_argument('--pretrain-patience', type=int, default=500,
						help='patience for early stopping pretraining')
	parser.add_argument('--use-k-topic-words', type=int, default=10,
						help='take top k words in each topic')
	parser.add_argument('--use-k-topics', type=int, default=5,
						help='take top k topics for each document/sentence')
	parser.add_argument('--prior-emb-for-topics', type=str, default="False",
						help='if to prior word embeddings for computing topic embeddings')
	parser.add_argument('--concat-proj-activation', type=str, default="linear",
						help='activation for projected vector after concatenation')
	parser.add_argument('--topic-coherence-reg', type=str, default="False",
						help='whether to use topic coherence regularization or not')
	parser.add_argument('--beta-coherence', type=float, default=0.0,
						help='weight for topic coherence regularization')
	
	return parser.parse_args()


if __name__ == '__main__':
	main(parse_args())

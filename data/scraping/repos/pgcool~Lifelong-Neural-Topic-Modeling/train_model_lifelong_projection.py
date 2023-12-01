import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model_supervised_projection as m
import model.evaluate as eval
import datetime
import json
import sys
import pickle
import time

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

dir(tf.contrib)


def loadGloveModel(gloveFile=None, params=None):
	if gloveFile is None:
		if params.hidden_size == 50:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
		elif params.hidden_size == 100:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
		elif params.hidden_size == 200:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
		elif params.hidden_size == 300:
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
	return model

def train(model, dataset, dataset_old, params, x_old_loss_values):
	log_dir = os.path.join(params.model, 'logs')
	model_dir_ir = os.path.join(params.model, 'model_ir')
	model_dir_ppl = os.path.join(params.model, 'model_ppl')
	model_dir_supervised = os.path.join(params.model, 'model_supervised')

	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params.num_cores,
		intra_op_parallelism_threads=params.num_cores,
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session:
		avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
		tf.summary.scalar('loss', avg_loss)

		validation = tf.placeholder(tf.float32, [], 'validation_ph')
		validation_accuracy = tf.placeholder(tf.float32, [], 'validation_acc')
		tf.summary.scalar('validation', validation)
		tf.summary.scalar('validation_accuracy', validation_accuracy)

		summary_writer = tf.summary.FileWriter(log_dir, session.graph)
		summaries = tf.summary.merge_all()
		saver = tf.train.Saver(tf.global_variables())

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		losses = []

		# This currently streams from disk. You set num_epochs=1 and
		# wrap this call with something like itertools.cycle to keep
		# this data in memory.
		# shuffle: the order of words in the sentence for DocNADE
		if params.bidirectional:
			pass
		else:
			training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)

			training_data_old_list = []
			training_doc_ids_old_list = []
			for i, dataset_temp in enumerate(dataset_old):
				training_data_old_list.append(dataset_temp.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.source_multi_label[i]))
				training_doc_ids_old_list.append(dataset_temp.batches('training_document_indices', params.batch_size, shuffle=True, multilabel=params.source_multi_label[i]))

		best_val_IR = 0.0
		best_val_nll = np.inf
		best_val_ppl = np.inf
		best_val_disc_accuracy = 0.0

		best_test_IR = 0.0
		best_test_nll = np.inf
		best_test_ppl = np.inf
		best_test_disc_accuracy = 0.0
		
		#if params.bidirectional or params.initialize_docnade:
		#	patience = 30
		#else:
		#	patience = params.patience
		
		patience = params.patience

		patience_count = 0
		patience_count_ir = 0
		best_train_nll = np.inf

		training_labels = np.array(
			[[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
		)
		validation_labels = np.array(
			[[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
		)
		test_labels = np.array(
			[[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
		)

		sal_docs_taken_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
		sal_docs_total_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
		
		for step in range(params.num_steps + 1):
			this_loss = -1.

			if params.bidirectional:
				pass
			else:
				y, x, seq_lengths = next(training_data)

				x_old_list_temp = []
				x_old_doc_ids_list = []
				seq_lengths_old_list = []
				for (training_data_old, training_doc_ids_old) in zip(training_data_old_list, training_doc_ids_old_list):
					y_old, x_old, seq_lengths_old = next(training_data_old)
					y_old_doc_ids, x_old_doc_ids, seq_lengths_old_doc_ids = next(training_doc_ids_old)
					y_old_doc_ids = np.array(y_old_doc_ids, dtype=np.float32)
					x_old_doc_ids = np.squeeze(x_old_doc_ids)

					x_old_list_temp.append(x_old)
					x_old_doc_ids_list.append(x_old_doc_ids)
					seq_lengths_old_list.append(seq_lengths_old)

				if len(x_old_list_temp) > 1:
					max_doc_len = 0
					for x_old_temp in x_old_list_temp:
						if x_old_temp.shape[1] > max_doc_len:
							max_doc_len = x_old_temp.shape[1]

					#print("Max doc length: ", max_doc_len)

					x_old_list = []
					for x_old_temp in x_old_list_temp:
						if x_old_temp.shape[1] != max_doc_len:
							pad_len = max_doc_len - x_old_temp.shape[1]
							x_old_list.append(np.pad(x_old_temp, ((0,0), (0,pad_len)), 'constant', constant_values=(0,0)))
						else:
							x_old_list.append(x_old_temp)
					x_old_list = np.array(x_old_list)
				else:
					x_old_list = x_old_list_temp
				
				#import pdb; pdb.set_trace()

				if x_old_loss_values is None:
					x_old_loss_input = np.sum(y_old_doc_ids)
				else:
					x_old_loss_input = np.mean(x_old_loss_values[x_old_doc_ids])
			
				if params.supervised:
					print("Error: params.supervised == ", params.supervised)
					sys.exit()
				else:
					if params.sal_loss and (params.sal_gamma == "manual"):
						_, loss, loss_unnormed, sal_gamma_mask_list = session.run([model.opt, model.loss_normed, model.loss_unnormed, model.sal_gamma_mask_list], feed_dict={
							model.x: x,
							model.y: y,
							model.x_old: x_old_list,
							model.x_old_doc_ids: x_old_doc_ids_list,
							model.seq_lengths: seq_lengths,
							model.seq_lengths_old: seq_lengths_old_list,
							model.x_old_loss: x_old_loss_input
						})
						this_loss = loss
						losses.append(this_loss)
						
						for i, sal_gamma_mask in enumerate(sal_gamma_mask_list):
							sal_docs_taken_list[i] += np.sum(sal_gamma_mask)
							sal_docs_total_list[i] += len(sal_gamma_mask)
					else:
						_, loss, loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict={
							model.x: x,
							model.y: y,
							model.x_old: x_old_list,
							model.x_old_doc_ids: x_old_doc_ids_list,
							model.seq_lengths: seq_lengths,
							model.seq_lengths_old: seq_lengths_old_list,
							model.x_old_loss: x_old_loss_input
						})
						this_loss = loss
						losses.append(this_loss)

			if (step % params.log_every == 0):
				print('{}: {:.6f}'.format(step, this_loss))

			if step and (step % params.validation_ppl_freq) == 0:
				if params.sal_loss:
					if params.sal_gamma == "manual":
						doc_str = ""
						for (taken, total) in zip(sal_docs_taken_list, sal_docs_total_list):
							doc_str += "(" + str(taken) + "/" + str(total) + ")"
						
						print("SAL docs: %s" % doc_str)
						with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
							f.write("SAL docs: %s\n" % (doc_str))

				sal_docs_taken_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
				sal_docs_total_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)

				this_val_nll = []
				this_val_loss_normed = []
				# val_loss_unnormed is NLL
				this_val_nll_bw = []
				this_val_loss_normed_bw = []

				this_val_disc_accuracy = []

				if params.bidirectional:
					pass
				else:
					for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
					#for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=False, multilabel=params.multi_label):
						if params.supervised:
							pass
						else:
							val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
								model.x: val_x,
								model.y: val_y,
								model.seq_lengths: val_seq_lengths
							})
						this_val_nll.append(val_loss_unnormed)
						this_val_loss_normed.append(val_loss_normed)
				
				if params.bidirectional:
					pass
				else:
					total_val_nll = np.mean(this_val_nll)
					total_val_ppl = np.exp(np.mean(this_val_loss_normed))

				if total_val_ppl < best_val_ppl:
					best_val_ppl = total_val_ppl
					print('saving: {}'.format(model_dir_ppl))
					saver.save(session, model_dir_ppl + '/model_ppl', global_step=1)

				# Early stopping
				if total_val_nll < best_val_nll:
					best_val_nll = total_val_nll
					patience_count = 0
				else:
					patience_count += 1

				

				print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f}'.format(
					total_val_ppl,
					best_val_ppl or 0.0,
					best_val_nll
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val PPL: %s,	 best val PPL: %s,	best val loss: %s\n" % 
							(step, total_val_ppl, best_val_ppl, best_val_nll))

				if patience_count > patience:
					print("Early stopping criterion satisfied.")
					break
			
			if step and (step % params.validation_ir_freq) == 0:
				#import pdb; pdb.set_trace()
				if params.sal_loss:
					if params.sal_gamma == "manual":
						doc_str = ""
						for (taken, total) in zip(sal_docs_taken_list, sal_docs_total_list):
							doc_str += "(" + str(taken) + "/" + str(total) + ")"
						
						print("SAL docs: %s" % doc_str)
						with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
							f.write("SAL docs: %s\n" % (doc_str))

				sal_docs_taken_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)
				sal_docs_total_list = np.zeros((len(params.sal_gamma_init)), dtype=np.int32)

				if params.supervised:
					pass
				else:
					if params.bidirectional:
						pass
					else:
						validation_vectors = m.vectors(
							model,
							dataset.batches(
								'validation_docnade',
								params.validation_bs,
								num_epochs=1,
								shuffle=True,
								multilabel=params.multi_label
							),
							session
						)

						training_vectors = m.vectors(
							model,
							dataset.batches(
								'training_docnade',
								params.validation_bs,
								num_epochs=1,
								shuffle=True,
								multilabel=params.multi_label
							),
							session
						)

					val = eval.evaluate(
						training_vectors,
						validation_vectors,
						training_labels,
						validation_labels,
						recall=[0.02],
						num_classes=params.num_classes,
						multi_label=params.multi_label
					)[0]

					if val > best_val_IR:
						best_val_IR = val
						print('saving: {}'.format(model_dir_ir))
						saver.save(session, model_dir_ir + '/model_ir', global_step=1)
						patience_count_ir = 0
					else:
						patience_count_ir += 1
					
					print('This val IR: {:.3f} (best val IR: {:.3f})'.format(
						val,
						best_val_IR or 0.0
					))

					# logging information
					with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
						f.write("Step: %i,	val IR: %s,	best val IR: %s\n" % 
								(step, val, best_val_IR))
				
				if patience_count_ir > patience:
					print("Early stopping criterion satisfied.")
					break
			
			if step and (step % params.test_ppl_freq) == 0:
				this_test_nll = []
				this_test_loss_normed = []
				this_test_nll_bw = []
				this_test_loss_normed_bw = []
				this_test_disc_accuracy = []

				if params.bidirectional:
					pass
				else:
					#for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
					for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=False, multilabel=params.multi_label):
						if params.supervised:
							pass
						else:
							test_loss_normed, test_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
								model.x: test_x,
								model.y: test_y,
								model.seq_lengths: test_seq_lengths
							})
						this_test_nll.append(test_loss_unnormed)
						this_test_loss_normed.append(test_loss_normed)

				if params.bidirectional:
					pass
				else:
					total_test_nll = np.mean(this_test_nll)
					total_test_ppl = np.exp(np.mean(this_test_loss_normed))

				if total_test_ppl < best_test_ppl:
					best_test_ppl = total_test_ppl

				if total_test_nll < best_test_nll:
					best_test_nll = total_test_nll

				print('This test PPL: {:.3f} (best test PPL: {:.3f},  best test loss: {:.3f})'.format(
					total_test_ppl,
					best_test_ppl or 0.0,
					best_test_nll
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	test PPL: %s,	best test PPL: %s,	best test loss: %s\n" % 
							(step, total_test_ppl, best_test_ppl, best_test_nll))

			
			if step >= 1 and (step % params.test_ir_freq) == 0:
				if params.supervised:
					pass
				else:
					if params.bidirectional:
						pass
					else:
						test_vectors = m.vectors(
							model,
							dataset.batches(
								'test_docnade',
								params.test_bs,
								num_epochs=1,
								shuffle=True,
								multilabel=params.multi_label
							),
							session
						)

						training_vectors = m.vectors(
							model,
							dataset.batches(
								'training_docnade',
								params.test_bs,
								num_epochs=1,
								shuffle=True,
								multilabel=params.multi_label
							),
							session
						)

					test = eval.evaluate(
						training_vectors,
						test_vectors,
						training_labels,
						test_labels,
						recall=[0.02],
						num_classes=params.num_classes,
						multi_label=params.multi_label
					)[0]

					if test > best_test_IR:
						best_test_IR = test
					
					print('This test IR: {:.3f} (best test IR: {:.3f})'.format(
						test,
						best_test_IR or 0.0
					))

					# logging information
					with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
						f.write("Step: %i,	test IR: %s,	best test IR: %s\n" % 
							(step, test, best_test_IR))


from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def compute_coherence(texts, list_of_topics, top_n_word_in_each_topic_list, reload_model_dir):

	dictionary = Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	print('corpus len:%s' %len(corpus))
	print('dictionary:%s' %dictionary)
	# https://github.com/earthquakesan/palmetto-py
	# compute_topic_coherence: PMI and other coherence types
	# from palmettopy.palmetto import Palmetto
	# palmetto = Palmetto()

	# coherence_types = ["ca", "cp", "cv", "npmi", "uci", "umass"] # for palmetto library
	coherence_types = ["c_v"]#, 'u_mass', 'c_v', 'c_uci', 'c_npmi'] # ["c_v"] # 'u_mass', 'c_v', 'c_uci', 'c_npmi',
	avg_coh_scores_dict = {}

	best_coh_type_value_topci_indx = {}
	for top_n in top_n_word_in_each_topic_list:
		avg_coh_scores_dict[top_n]= []
		best_coh_type_value_topci_indx[top_n] = [0,  0, []] # score, topic_indx, topics words


	h_num = 0
	with open(reload_model_dir, "w") as f:
		for topic_words_all in list_of_topics:
			h_num += 1
			for top_n in top_n_word_in_each_topic_list:
				topic_words = [topic_words_all[:top_n]]
				for coh_type in coherence_types:
					try:
						print('top_n: %s Topic Num: %s \nTopic Words: %s' % (top_n, h_num, topic_words))
						f.write('top_n: %s Topic Num: %s \nTopic Words: %s\n' % (top_n, h_num, topic_words))
						# print('topic_words_top_10_abs[%s]:%s' % (h_num, topic_words_top_10_abs[h_num]))
						# PMI = palmetto.get_coherence(topic_words_top_10[h_num], coherence_type=coh_type)
						PMI = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence=coh_type, processes=2).get_coherence()

						avg_coh_scores_dict[top_n].append(PMI)

						if PMI > best_coh_type_value_topci_indx[top_n][0]:
							best_coh_type_value_topci_indx[top_n] = [PMI, top_n, topic_words]

						print('Coh_type:%s  Topic Num:%s COH score:%s' % (coh_type, h_num, PMI))
						f.write('Coh_type:%s  Topic Num:%s COH score:%s\n' % (coh_type, h_num, PMI))

						'''
						output_topics_coh_filename_fp.write(str('h_num:') + str(h_num) + ' ' + str('PMI_') +
															str(coh_type) + ' ' + str('COH:') + str(PMI) + ' ' + str('topicsWords:'))

						for word in topic_words_top_10[h_num]:
							output_topics_coh_filename_fp.write(str(word) + ' ')

						output_topics_coh_filename_fp.write('\n')

						output_topics_coh_filename_fp.write(
							str('--------------------------------------------------------------') + '\n')
						'''


						print('--------------------------------------------------------------')
					except:
						continue
				print('========================================================================================================')

		for top_n in top_n_word_in_each_topic_list:
			print('top scores for top_%s:%s' %(top_n, best_coh_type_value_topci_indx[top_n]))
			print('-------------------------------------------------------------------')
			f.write('top scores for top_%s:%s\n' %(top_n, best_coh_type_value_topci_indx[top_n]))
			f.write('-------------------------------------------------------------------\n')

		for top_n in top_n_word_in_each_topic_list:
			print('Avg COH for top_%s topic words: %s' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
			print('-------------------------------------------------------------------')
			f.write('Avg COH for top_%s topic words: %s\n' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
			f.write('-------------------------------------------------------------------\n')


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

from math import *
from nltk.corpus import wordnet
def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 3)


def reload_evaluation_ppl(params, suffix=""):
	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params['num_cores'],
		intra_op_parallelism_threads=params['num_cores'],
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session_ppl:

		dataset = data.Dataset(params['dataset'])
		log_dir = os.path.join(params['model'], 'logs')
			
		saver_ppl = tf.train.import_meta_graph("model/" + params['reload_model_dir'] + "model_ppl/model_ppl-1.meta")
		saver_ppl.restore(session_ppl, tf.train.latest_checkpoint("model/" + params['reload_model_dir'] + "model_ppl/"))

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
		loss_normed = graph.get_tensor_by_name("loss_normed_x:0")
		loss_unnormed = graph.get_tensor_by_name("loss_unnormed_x:0")

		# TODO: Validation PPL

		this_val_nll = []
		this_val_loss_normed = []
		# val_loss_unnormed is NLL
		this_val_nll_bw = []
		this_val_loss_normed_bw = []

		this_val_disc_accuracy = []

		if params['bidirectional']:
			pass
		else:
			for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
			#for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
				if params['supervised']:
					pass
				else:
					val_loss_normed, val_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict={
						x: val_x,
						y: val_y,
						seq_lengths: val_seq_lengths
					})
				this_val_nll.append(val_loss_unnormed)
				this_val_loss_normed.append(val_loss_normed)
		
		if params['bidirectional']:
			total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
			total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
		else:
			total_val_nll = np.mean(this_val_nll)
			total_val_ppl = np.exp(np.mean(this_val_loss_normed))

		print('Val PPL: {:.3f},	Val loss: {:.3f}\n'.format(
			total_val_ppl,
			total_val_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "w") as f:
			f.write("Val PPL: %s,	Val loss: %s" % 
					(total_val_ppl, total_val_nll))
		
		# TODO: Test PPL

		this_test_nll = []
		this_test_loss_normed = []
		this_test_nll_bw = []
		this_test_loss_normed_bw = []
		this_test_disc_accuracy = []

		if params['bidirectional']:
			pass
		else:
			#for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
			for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
				if params['supervised']:
					pass
				else:
					test_loss_normed, test_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict={
						x: test_x,
						y: test_y,
						seq_lengths: test_seq_lengths
					})
				this_test_nll.append(test_loss_unnormed)
				this_test_loss_normed.append(test_loss_normed)

		if params['bidirectional']:
			total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
			total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
		else:
			total_test_nll = np.mean(this_test_nll)
			total_test_ppl = np.exp(np.mean(this_test_loss_normed))

		print('Test PPL: {:.3f},	Test loss: {:.3f}\n'.format(
			total_test_ppl,
			total_test_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "a") as f:
			f.write("\n\nTest PPL: %s,	Test loss: %s" % 
					(total_test_ppl, total_test_nll))

		W_target = session_ppl.run("embedding:0")
		bias_W_target = session_ppl.run("bias:0")
		U_target = session_ppl.run("U:0")
		bias_U_target = session_ppl.run("b:0")

		#import pdb; pdb.set_trace()

		source_data_W_projection_list = []
		source_data_U_projection_list = []
		if params['ll_loss'] and params['projection']:
			for i, source_data in enumerate(params['reload_source_data_list']):
				source_data_W_projection_list.append(session_ppl.run("ll_projection_W_" + str(i) + ":0"))
				source_data_U_projection_list.append(session_ppl.run("ll_projection_U_" + str(i) + ":0"))
		
		return W_target, bias_W_target, U_target, bias_U_target, source_data_W_projection_list, source_data_U_projection_list


def reload_evaluation_topics(W_target, U_target, params, suffix=""):

	log_dir = os.path.join(params['model'], 'logs')

	# Topics with W matrix

	top_n_topic_words = 20
	w_h_top_words_indices = []
	W_topics = W_target
	topics_list_W = []

	for h_num in range(np.array(W_topics).shape[1]):
		w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

	with open(params['docnadeVocab'], 'r') as f:
		vocab_docnade = [w.strip() for w in f.readlines()]

	assert(len(vocab_docnade) == W_target.shape[0])

	with open(os.path.join(log_dir, "topics_ppl_W_" + suffix + ".txt"), "w") as f:
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
			
	
	# Topics with V matrix

	top_n_topic_words = 20
	w_h_top_words_indices = []
	W_topics = U_target.T
	topics_list_V = []

	for h_num in range(np.array(W_topics).shape[1]):
		w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

	with open(params['docnadeVocab'], 'r') as f:
		vocab_docnade = [w.strip() for w in f.readlines()]

	with open(os.path.join(log_dir, "topics_ppl_V_" + suffix + ".txt"), "w") as f:
		for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
			w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
			topics_list_V.append(w_h_top_words)
			print('h_num: %s' % h_num)
			print('w_h_top_words_indx: %s' % w_h_top_words_indx)
			print('w_h_top_words:%s' % w_h_top_words)
			print('----------------------------------------------------------------------')

			f.write('h_num: %s\n' % h_num)
			f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
			f.write('w_h_top_words:%s\n' % w_h_top_words)
			f.write('----------------------------------------------------------------------\n')

	
	# TOPIC COHERENCE

	top_n_word_in_each_topic_list = [5, 10, 15, 20]

	text_filenames = [
		params['trainfile'],
		params['valfile'],
		params['testfile']
	]

	# read original text documents as list of words
	texts = []

	for file in text_filenames:
		print('filename:%s', file)
		for line in open(file, 'r').readlines():
			document = str(line).strip()
			texts.append(document.split())

	compute_coherence(texts, topics_list_W, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_W_" + suffix + ".txt"))
	#compute_coherence(texts, topics_list_V, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_V_" + suffix + ".txt"))

def reload_evaluation_ppl_source(model_ppl, dataset, params, suffix="", model_one_shot=None):
	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params['num_cores'],
		intra_op_parallelism_threads=params['num_cores'],
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session_ppl_source:

		log_dir = os.path.join(params['model'], 'logs')

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		# TODO: Validation PPL

		this_val_nll = []
		this_val_loss_normed = []
		# val_loss_unnormed is NLL
		this_val_nll_bw = []
		this_val_loss_normed_bw = []

		this_val_disc_accuracy = []

		if params['bidirectional']:
			pass
		else:
			for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
			#for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', batch_size=1, num_epochs=1, shuffle=False, multilabel=params['multi_label']):
				if params['supervised']:
					pass
				else:
					val_loss_normed, val_loss_unnormed = session_ppl_source.run([model_ppl.loss_normed, model_ppl.loss_unnormed], feed_dict={
						model_ppl.x: val_x,
						model_ppl.y: val_y,
						model_ppl.seq_lengths: val_seq_lengths
					})
				this_val_nll.append(val_loss_unnormed)
				this_val_loss_normed.append(val_loss_normed)
		
		if params['bidirectional']:
			total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
			total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
		else:
			total_val_nll = np.mean(this_val_nll)
			total_val_ppl = np.exp(np.mean(this_val_loss_normed))

		print('Val PPL: {:.3f},	Val loss: {:.3f}\n'.format(
			total_val_ppl,
			total_val_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "w") as f:
			f.write("Val PPL: %s,	Val loss: %s" % 
					(total_val_ppl, total_val_nll))
		

		# TODO: Test PPL

		this_test_nll = []
		this_test_loss_normed = []
		this_test_nll_bw = []
		this_test_loss_normed_bw = []
		this_test_disc_accuracy = []

		if params['bidirectional']:
			pass
		else:
			#for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
			for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=False, multilabel=params['multi_label']):
				if params['supervised']:
					pass
				else:
					test_loss_normed, test_loss_unnormed = session_ppl_source.run([model_ppl.loss_normed, model_ppl.loss_unnormed], feed_dict={
						model_ppl.x: test_x,
						model_ppl.y: test_y,
						model_ppl.seq_lengths: test_seq_lengths
					})
				this_test_nll.append(test_loss_unnormed)
				this_test_loss_normed.append(test_loss_normed)

		if params['bidirectional']:
			total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
			total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
		else:
			total_test_nll = np.mean(this_test_nll)
			total_test_ppl = np.exp(np.mean(this_test_loss_normed))

		print('Test PPL: {:.3f},	Test loss: {:.3f}\n'.format(
			total_test_ppl,
			total_test_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "a") as f:
			f.write("\n\nTest PPL: %s,	Test loss: %s" % 
					(total_test_ppl, total_test_nll))


def reload_evaluation_ir(params, training_vectors, validation_vectors, test_vectors, 
						training_labels, validation_labels, test_labels, 
						ir_ratio_list, W_matrix, suffix=""):
	log_dir = os.path.join(params['model'], 'logs')

	#ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
	ir_ratio_list = [0.02]

	val_ir_list = eval.evaluate(
		training_vectors,
		validation_vectors,
		training_labels,
		validation_labels,
		recall=ir_ratio_list,
		num_classes=params['num_classes'],
		multi_label=params['multi_label']
	)

	# logging information
	with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "w") as f:
		f.write("\n\nFractions list: %s" % (ir_ratio_list))
		f.write("\nVal IR: %s" % (val_ir_list))

	test_ir_list = eval.evaluate(
		training_vectors,
		test_vectors,
		training_labels,
		test_labels,
		recall=ir_ratio_list,
		num_classes=params['num_classes'],
		multi_label=params['multi_label']
	)

	# logging information
	with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "a") as f:
		f.write("\n\nFractions list: %s" % (ir_ratio_list))
		f.write("\nTest IR: %s" % (test_ir_list))
		

def reload_evaluation_ir_source(model_ir, dataset, params, ir_ratio_list, suffix):
	with tf.Session() as session_ir:
		log_dir = os.path.join(params['model'], 'logs')

		#ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
		ir_ratio_list = [0.02]

		training_labels = np.array(
			[[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
		)
		validation_labels = np.array(
			[[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
		)
		test_labels = np.array(
			[[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
		)

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()
		
		if params['bidirectional']:
			pass
		else:
			print("Getting DocNADE document vector representation.")
			training_vectors = m.vectors(
				model_ir,
				dataset.batches(
					'training_docnade',
					params['validation_bs'],
					num_epochs=1,
					shuffle=True,
					multilabel=params['multi_label']
				),
				session_ir
			)

			validation_vectors = m.vectors(
				model_ir,
				dataset.batches(
					'validation_docnade',
					params['validation_bs'],
					num_epochs=1,
					shuffle=True,
					multilabel=params['multi_label']
				),
				session_ir
			)

			test_vectors = m.vectors(
				model_ir,
				dataset.batches(
					'test_docnade',
					params['test_bs'],
					num_epochs=1,
					shuffle=True,
					multilabel=params['multi_label']
				),
				session_ir
			)
		
		# Validation IR

		#import pdb; pdb.set_trace()

		val_list = eval.evaluate(
			training_vectors,
			validation_vectors,
			training_labels,
			validation_labels,
			recall=ir_ratio_list,
			num_classes=params['num_classes'],
			multi_label=params['multi_label']
		)
		
		print('Val IR: ', val_list)

		# logging information
		with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "w") as f:
			f.write("\n\nFractions list: %s" % (ir_ratio_list))
			f.write("\nVal IR: %s" % (val_list))
		
		
		# Test IR
		
		test_list = eval.evaluate(
			training_vectors,
			test_vectors,
			training_labels,
			test_labels,
			recall=ir_ratio_list,
			num_classes=params['num_classes'],
			multi_label=params['multi_label']
		)
		
		print('Test IR: ', test_list)

		# logging information
		with open(os.path.join(log_dir, "reload_info_ir_" + suffix + ".txt"), "a") as f:
			f.write("\n\nFractions list: %s" % (ir_ratio_list))
			f.write("\n\nTest IR: %s" % (test_list))


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def get_prior_matrix(prior_embedding_path, prior_vocab, docnade_vocab, hidden_size):
	prior_embedding_matrix = np.load(prior_embedding_path)
	
	W_old_indices = []
	W_new_indices = []
	W_old_matrix = []
	prior_matrix = np.zeros((len(docnade_vocab), hidden_size), dtype=np.float32)
	for i, word in enumerate(docnade_vocab):
		try:
			index = prior_vocab.index(word)
		except ValueError:
			continue
		prior_matrix[i, :] = prior_embedding_matrix[index, :]
		W_old_matrix.append(prior_embedding_matrix[index, :])
		W_old_indices.append(index)
		W_new_indices.append(i)
	
	return prior_matrix, np.array(W_old_matrix, dtype=np.float32), W_old_indices, W_new_indices

def main(args):
	args.reload = str2bool(args.reload)
	args.supervised = str2bool(args.supervised)
	args.initialize_docnade = str2bool(args.initialize_docnade)
	args.bidirectional = str2bool(args.bidirectional)
	args.projection = str2bool(args.projection)
	args.deep = str2bool(args.deep)
	args.multi_label = str2bool(args.multi_label)
	args.shuffle_reload = str2bool(args.shuffle_reload)

	args.sal_loss = str2bool(args.sal_loss)
	args.ll_loss = str2bool(args.ll_loss)
	args.pretraining_target = str2bool(args.pretraining_target)
	args.bias_sharing = str2bool(args.bias_sharing)
	args.use_embeddings_prior = str2bool(args.use_embeddings_prior)
	args.source_multi_label = [str2bool(value) for value in args.reload_source_multi_label]

	if args.reload:
		with open("model/" + args.reload_model_dir + "params.json") as f:
			params = json.load(f)

		params['trainfile'] = args.trainfile
		params['valfile'] = args.valfile
		params['testfile'] = args.testfile

		source_datasets_path = "/home/ubuntu/DocNADE_Lifelong_Learning/datasets"
		source_trainfiles = [source_datasets_path + "/20NS/20_news_groups_MVG_train.txt",
							source_datasets_path + "/R21578/train.txt",
							source_datasets_path + "/TMN/TMN_train.txt",
							source_datasets_path + "/AGnews/train.txt"]
		source_valfiles = [source_datasets_path + "/20NS/20_news_groups_MVG_val.txt",
							source_datasets_path + "/R21578/val.txt",
							source_datasets_path + "/TMN/TMN_val.txt",
							source_datasets_path + "/AGnews/val.txt"]
		source_testfiles = [source_datasets_path + "/20NS/20_news_groups_MVG_test.txt",
							source_datasets_path + "/R21578/test.txt",
							source_datasets_path + "/TMN/TMN_test.txt",
							source_datasets_path + "/AGnews/test.txt"]
		source_vocabs = [source_datasets_path + "/20NS/vocab_docnade.vocab",
						source_datasets_path + "/R21578/vocab_docnade.vocab",
						source_datasets_path + "/TMN/vocab_docnade.vocab",
						source_datasets_path + "/AGnews/vocab_docnade.vocab"]


		params['reload_model_dir'] = args.reload_model_dir

		reload_ir = False
		if os.path.isdir("model/" + args.reload_model_dir + "/model_ir"):
			reload_ir = True

		reload_ppl = False
		if os.path.isdir("model/" + args.reload_model_dir + "/model_ppl"):
			reload_ppl = True

		if reload_ppl:
			W_target, bias_W_target, U_target, bias_U_target, \
			source_data_W_projection_list, source_data_U_projection_list = reload_evaluation_ppl(params, suffix="target")
			reload_evaluation_topics(W_target, U_target, params, suffix="target")

			## SOURCE DATA
			params['reload_source_data_list'] = args.reload_source_data_list
			params['bias_W_old_path_list'] = args.bias_W_old_path_list
			params['bias_U_old_path_list'] = args.bias_U_old_path_list

			if (params['use_embeddings_prior']) and (not params['ll_loss']) and (not params['sal_loss']):
				params['projection'] = False
				params['ll_loss'] = True
			
			if params['ll_loss']:
				source_multi_label = [str2bool(value) for value in args.reload_source_multi_label]
				source_num_classes = args.reload_source_num_classes

				with open(params['docnadeVocab'], "r") as f:
					target_data_vocab = [line.strip().lower() for line in f.readlines()]
				
				for i, source_data in enumerate(params['reload_source_data_list']):
					with open(source_data + "/vocab_docnade.vocab", "r") as f:
						source_data_vocab = [line.strip().lower() for line in f.readlines()]

					params['multi_label'] = source_multi_label[i]
					params['num_classes'] = source_num_classes[i]

					if params['projection']:
						source_data_W_projection = source_data_W_projection_list[i]

					source_data_W_original = np.load(params['W_old_path_list'][i])
					source_data_bias_W_original = np.load(params['bias_W_old_path_list'][i])

					source_data_W_merged = np.zeros_like(source_data_W_original, dtype=np.float32)
					for j, word in enumerate(source_data_vocab):
						try:
							index = target_data_vocab.index(word)
						except ValueError:
							source_data_W_merged[j, :] = source_data_W_original[j, :]
							continue
						
						if params['projection']:
							source_data_W_merged[j, :] = np.dot(W_target[index, :], source_data_W_projection)
						else:
							source_data_W_merged[j, :] = W_target[index, :]
						
					if params['projection']:
						source_data_U_projection = source_data_U_projection_list[i]

					source_data_U_original = np.load(params['U_old_path_list'][i])
					source_data_bias_U_original = np.load(params['bias_U_old_path_list'][i])

					source_data_U_merged = np.zeros_like(source_data_U_original, dtype=np.float32)
					source_data_bias_U_merged = np.zeros_like(source_data_bias_U_original, dtype=np.float32)
					for j, word in enumerate(source_data_vocab):
						try:
							index = target_data_vocab.index(word)
						except ValueError:
							source_data_U_merged[:, j] = source_data_U_original[:, j]
							source_data_bias_U_merged[j] = source_data_bias_U_original[j]
							continue
						source_data_bias_U_merged[j] = bias_U_target[index]

						if params['projection']:
							source_data_U_merged[:, j] = np.dot(source_data_U_projection, U_target[:, index])
						else:
							source_data_U_merged[:, j] = U_target[:, index]
						
				
					x_source = tf.placeholder(tf.int32, shape=(None, None), name='x_source')
					if params['multi_label']:
						y_source = tf.placeholder(tf.string, shape=(None), name='y_source')
					else:
						y_source = tf.placeholder(tf.int32, shape=(None), name='y_source')
					seq_lengths_source = tf.placeholder(tf.int32, shape=(None), name='seq_lengths_source')

					model_ppl_source = m.DocNADE_TL_reload(x_source, y_source, seq_lengths_source, params, 
														W_initializer=None, W_reload=source_data_W_merged, 
														W_embeddings_reload=[], W_prior_proj_reload=None, 
														bias_reload=source_data_bias_W_original, bias_bw_reload=None, 
														V_reload=source_data_U_merged, b_reload=source_data_bias_U_original, 
														b_bw_reload=None, W_list_reload=[], bias_list_reload=[], 
														lambda_embeddings_reload=[])

					print("DocNADE PPL source created")
					
					model_ppl_source_one_shot = None
					source_dataset = data.Dataset(source_data)
					reload_evaluation_ppl_source(model_ppl_source, source_dataset, params, suffix="source_" + str(i), model_one_shot=model_ppl_source_one_shot)

					params['trainfile'] = source_trainfiles[i]
					params['valfile'] = source_valfiles[i]
					params['testfile'] = source_testfiles[i]
					params['docnadeVocab'] = source_vocabs[i]
					#reload_evaluation_topics(source_data_W_merged, source_data_U_merged, params, suffix="source_" + str(i))

		# Reloading and evaluating on Information Retrieval
		if reload_ir:
			sess_ir = tf.Session()	
			
			saver_ir = tf.train.import_meta_graph("model/" + args.reload_model_dir + "model_ir/model_ir-1.meta")
			saver_ir.restore(sess_ir, tf.train.latest_checkpoint("model/" + args.reload_model_dir + "model_ir/"))

			graph = tf.get_default_graph()

			x = graph.get_tensor_by_name("x:0")
			seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
			last_hidden = graph.get_tensor_by_name("last_hidden:0")

			## TARGET DATA
			dataset = data.Dataset(params['dataset'])

			training_labels = np.array(
				[[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
			)
			validation_labels = np.array(
				[[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
			)
			test_labels = np.array(
				[[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
			)

			hidden_vectors_val = []
			for va_y, va_x, va_seq_lengths in dataset.batches('validation_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
				hidden_vec = sess_ir.run([last_hidden], feed_dict={
					x: va_x,
					seq_lengths: va_seq_lengths
				})
				hidden_vectors_val.append(hidden_vec[0])
			hidden_vectors_val = np.squeeze(np.array(hidden_vectors_val, dtype=np.float32))

			hidden_vectors_tr = []
			for tr_y, tr_x, tr_seq_lengths in dataset.batches('training_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
				hidden_vec = sess_ir.run([last_hidden], feed_dict={
					x: tr_x,
					seq_lengths: tr_seq_lengths
				})
				hidden_vectors_tr.append(hidden_vec[0])
			hidden_vectors_tr = np.squeeze(np.array(hidden_vectors_tr, dtype=np.float32))

			hidden_vectors_test = []
			for te_y, te_x, te_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
				hidden_vec = sess_ir.run([last_hidden], feed_dict={
					x: te_x,
					seq_lengths: te_seq_lengths
				})
				hidden_vectors_test.append(hidden_vec[0])
			hidden_vectors_test = np.squeeze(np.array(hidden_vectors_test, dtype=np.float32))

			W_target = sess_ir.run("embedding:0")
			bias_W_target = sess_ir.run("bias:0")
			U_target = sess_ir.run("U:0")
			bias_U_target = sess_ir.run("b:0")

			ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

			reload_evaluation_ir(params, hidden_vectors_tr, hidden_vectors_val, hidden_vectors_test, 
								training_labels, validation_labels, test_labels, 
								ir_ratio_list, W_target, suffix="target")
			
			## SOURCE DATA
			params['reload_source_data_list'] = args.reload_source_data_list
			params['bias_W_old_path_list'] = args.bias_W_old_path_list
			params['bias_U_old_path_list'] = args.bias_U_old_path_list

			if (params['use_embeddings_prior']) and (not params['ll_loss']) and (not params['sal_loss']):
				params['projection'] = False
				params['ll_loss'] = True
			
			if params['ll_loss']:
				source_multi_label = [str2bool(value) for value in args.reload_source_multi_label]
				source_num_classes = args.reload_source_num_classes

				with open(params['docnadeVocab'], "r") as f:
					target_data_vocab = [line.strip().lower() for line in f.readlines()]
				
				for i, source_data in enumerate(params['reload_source_data_list']):
					with open(source_data + "/vocab_docnade.vocab", "r") as f:
						source_data_vocab = [line.strip().lower() for line in f.readlines()]

					params['multi_label'] = source_multi_label[i]
					params['num_classes'] = source_num_classes[i]

					if params['projection']:
						source_data_W_projection = sess_ir.run("ll_projection_W_" + str(i) + ":0")

					source_data_W_original = np.load(params['W_old_path_list'][i])
					source_data_bias_W_original = np.load(params['bias_W_old_path_list'][i])

					source_data_W_merged = np.zeros_like(source_data_W_original, dtype=np.float32)
					for j, word in enumerate(source_data_vocab):
						try:
							index = target_data_vocab.index(word)
						except ValueError:
							source_data_W_merged[j, :] = source_data_W_original[j, :]
							continue

						if params['projection']:
							source_data_W_merged[j, :] = np.dot(W_target[index, :], source_data_W_projection)
						else:
							source_data_W_merged[j, :] = W_target[index, :]

					if params['projection']:
						source_data_U_projection = sess_ir.run("ll_projection_U_" + str(i) + ":0")

					source_data_U_original = np.load(params['U_old_path_list'][i])
					source_data_bias_U_original = np.load(params['bias_U_old_path_list'][i])

					source_data_U_merged = np.zeros_like(source_data_U_original, dtype=np.float32)
					source_data_bias_U_merged = np.zeros_like(source_data_bias_U_original, dtype=np.float32)
					for j, word in enumerate(source_data_vocab):
						try:
							index = target_data_vocab.index(word)
						except ValueError:
							source_data_U_merged[:, j] = source_data_U_original[:, j]
							source_data_bias_U_merged[j] = source_data_bias_U_original[j]
							continue
						source_data_bias_U_merged[j] = bias_U_target[index]

						if params['projection']:
							source_data_U_merged[:, j] = np.dot(source_data_U_projection, U_target[:, index])
						else:
							source_data_U_merged[:, j] = U_target[:, index]
			
					x_source = tf.placeholder(tf.int32, shape=(None, None), name='x_source')
					if params['multi_label']:
						y_source = tf.placeholder(tf.string, shape=(None), name='y_source')
					else:
						y_source = tf.placeholder(tf.int32, shape=(None), name='y_source')
					seq_lengths_source = tf.placeholder(tf.int32, shape=(None), name='seq_lengths_source')

					#params['use_embeddings_prior'] = False

					model_ir_source = m.DocNADE_TL_reload(x_source, y_source, seq_lengths_source, params, 
														W_initializer=None, W_reload=source_data_W_merged, 
														W_embeddings_reload=[], W_prior_proj_reload=None, 
														bias_reload=source_data_bias_W_original, bias_bw_reload=None, 
														V_reload=source_data_U_merged, b_reload=source_data_bias_U_original, 
														b_bw_reload=None, W_list_reload=[], bias_list_reload=[], 
														lambda_embeddings_reload=[])

					print("DocNADE IR source created")

					ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

					source_dataset = data.Dataset(source_data)
					reload_evaluation_ir_source(model_ir_source, source_dataset, params, ir_ratio_list, suffix="source_" + str(i))
	
			
	else:
		x = tf.placeholder(tf.int32, shape=(None, None), name='x')
		x_bw = tf.placeholder(tf.int32, shape=(None, None), name='x_bw')
		if args.multi_label:
			y = tf.placeholder(tf.string, shape=(None), name='y')
		else:
			y = tf.placeholder(tf.int32, shape=(None), name='y')
		seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')

		
		x_old = tf.placeholder(tf.int32, shape=(len(args.sal_threshold), None, None), name='x_old')
		x_old_doc_ids = tf.placeholder(tf.int32, shape=(len(args.sal_threshold), None), name='x_old_doc_ids')
		seq_lengths_old = tf.placeholder(tf.int32, shape=(len(args.sal_threshold), None), name='seq_lengths_old')
		x_old_loss = tf.placeholder(tf.float32, shape=(), name='x_old_loss')

		now = datetime.datetime.now()

		if args.bidirectional:
			args.model += "_iDocNADE"
		else:
			args.model += "_DocNADE"

		if args.supervised:
			args.model += "_supervised"

		if args.use_embeddings_prior:
			args.model += "_emb_lambda_" + str(args.lambda_embeddings) + "_" + "_".join([str(lamb) for lamb in args.lambda_embeddings_list])

		if args.W_pretrained_path or args.U_pretrained_path:
			args.model += "_pretr_reload_"

		if args.pretraining_target:
			args.model += "_pretr_targ_" + str(args.pretraining_epochs)

		if args.bias_sharing:
			args.model += "_bias_sharing_"
		
		args.model +=  "_act_" + str(args.activation) + "_hid_" + str(args.hidden_size) \
						+ "_vocab_" + str(args.vocab_size) + "_lr_" + str(args.learning_rate) \

		if args.sal_loss:
			if args.sal_gamma == "automatic":
				#args.model += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + str(args.sal_gamma_init)
				args.model += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + "_".join([str(val) for val in args.sal_gamma_init])
			else:
				args.model += "_SAL_loss_" + str(args.sal_loss) + "_" + str(args.sal_gamma) + "_" + "_".join([str(val) for val in args.sal_threshold]) + "_" + "_".join([str(val) for val in args.sal_gamma_init])

		if args.ll_loss:
			#args.model += "_LL_loss_" + str(args.ll_loss) + "_" + str(args.ll_lambda) + "_" + str(args.ll_lambda_init)
			args.model += "_LL_loss_" + str(args.ll_loss) + "_" + str(args.ll_lambda) + "_".join([str(lamb) for lamb in args.ll_lambda_init])

		if args.projection:
			args.model += "_projection"
		
		args.model += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
		
		if not os.path.isdir(args.model):
			os.mkdir(args.model)

		docnade_vocab = args.docnadeVocab
		with open(docnade_vocab, 'r') as f:
			vocab_docnade = [w.strip() for w in f.readlines()]

		with open(os.path.join(args.model, 'params.json'), 'w') as f:
			f.write(json.dumps(vars(args)))

		dataset = data.Dataset(args.dataset)
		#dataset_old = data.Dataset(args.dataset_old)
		dataset_old_list = []

		for old_dataset in args.dataset_old:
			dataset_old_list.append(data.Dataset(old_dataset))

		if args.initialize_docnade:
			glove_embeddings = loadGloveModel(params=args)

		docnade_embedding_matrix = None
		if args.initialize_docnade:
			missing_words = 0
			docnade_embedding_matrix = np.zeros((len(vocab_docnade), args.hidden_size), dtype=np.float32)
			for i, word in enumerate(vocab_docnade):
				if str(word).lower() in glove_embeddings.keys():
					if len(glove_embeddings[str(word).lower()]) == 0:
						docnade_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
						missing_words += 1
					else:
						docnade_embedding_matrix[i, :] = np.array(glove_embeddings[str(word).lower()], dtype=np.float32)
				else:
					docnade_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
					missing_words += 1

			docnade_embedding_matrix = tf.convert_to_tensor(docnade_embedding_matrix)
			print("Total missing words:%d out of %d" %(missing_words, len(vocab_docnade)))

		W_pretrained_matrix = None
		if args.W_pretrained_path:
			W_pretrained_matrix = np.load(args.W_pretrained_path)
			print("pretrained W loaded.")

		U_pretrained_matrix = None
		if args.U_pretrained_path:
			U_pretrained_matrix = np.load(args.U_pretrained_path)
			print("pretrained U loaded.")

		W_old_indices_list = []
		W_new_indices_list = []
		W_old_matrices_list = []
		W_embeddings_matrices_list = []
		if args.use_embeddings_prior or args.ll_loss:
			for i, W_old_path in enumerate(args.W_old_path_list):
				with open(args.W_old_vocab_path_list[i], "r") as f:
					temp_vocab = [str(word).lower().strip() for word in f.readlines()]

				prior_matrix, W_old_matrix, W_old_indices, W_new_indices = get_prior_matrix(W_old_path, temp_vocab, vocab_docnade, args.hidden_size)
				W_embeddings_matrices_list.append(prior_matrix)
				W_old_matrices_list.append(W_old_matrix)
				W_old_indices_list.append(W_old_indices)
				W_new_indices_list.append(W_new_indices)
			print("Loaded W_embeddings_matrices_list and W_embeddings_indices_list.")

			args.lambda_embeddings_list = np.array(args.lambda_embeddings_list, dtype=np.float32)
			
		
		U_old_matrices_list = []
		if args.ll_loss:
			for i, (U_old_path, W_old_indices) in enumerate(zip(args.U_old_path_list, W_old_indices_list)):
				prior_matrix = np.load(U_old_path)
				prior_matrix = np.take(prior_matrix, W_old_indices, axis=1)
				U_old_matrices_list.append(prior_matrix)
			print("Loaded U_old_list.")

			args.ll_lambda_init = np.array(args.ll_lambda_init, dtype=np.float32)
		
		
		x_old_loss_values = None
		args.sal_threshold_list = []
		if args.sal_gamma == "manual":
			for sal_threshold in args.sal_threshold:
				args.sal_threshold_list.append(np.ones((args.batch_size), dtype=np.float32) * sal_threshold)


		if args.bidirectional:
			print("Error: args.bidirectional == ", args.bidirectional)
			sys.exit()
		else:
			model = m.DocNADE_TL(x, y, x_old, x_old_doc_ids, seq_lengths, seq_lengths_old, args,  x_old_loss, \
								W_old_list=W_old_matrices_list, U_old_list=U_old_matrices_list, \
								W_embeddings_matrices_list=W_embeddings_matrices_list, W_old_indices_list=W_old_indices_list, \
								lambda_embeddings_list=args.lambda_embeddings_list, W_new_indices_list=W_new_indices_list, \
								W_pretrained=W_pretrained_matrix, U_pretrained=U_pretrained_matrix)
			print("DocNADE created")
		
		train(model, dataset, dataset_old_list, args, x_old_loss_values)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True,
						help='path to model output directory')
	parser.add_argument('--dataset', type=str, required=True,
						help='path to the input dataset')
	parser.add_argument('--vocab-size', type=int, default=2000,
						help='the vocab size')
	parser.add_argument('--hidden-size', type=int, default=50,
						help='size of the hidden layer')
	parser.add_argument('--activation', type=str, default='tanh',
						help='which activation to use: sigmoid|tanh')
	parser.add_argument('--learning-rate', type=float, default=0.0004,
						help='initial learning rate')
	parser.add_argument('--num-steps', type=int, default=50000,
						help='the number of steps to train for')
	parser.add_argument('--batch-size', type=int, default=64,
						help='the batch size')
	parser.add_argument('--num-samples', type=int, default=None,
						help='softmax samples (default: full softmax)')
	parser.add_argument('--num-cores', type=int, default=2,
						help='the number of CPU cores to use')
	parser.add_argument('--log-every', type=int, default=10,
						help='print loss after this many steps')
	parser.add_argument('--validation-ppl-freq', type=int, default=500,
						help='print loss after this many steps')

	parser.add_argument('--num-classes', type=int, default=-1,
						help='number of classes')
	parser.add_argument('--supervised', type=str, default="False",
						help='whether to use supervised model or not')
	#parser.add_argument('--hidden-sizes', nargs='+', type=int,
	#					help='sizes of the hidden layers')

	parser.add_argument('--initialize-docnade', type=str, default="False",
						help='whether to embedding matrix of docnade')
	parser.add_argument('--docnadeVocab', type=str, default="False",
						help='path to vocabulary file used by DocNADE')
	parser.add_argument('--test-ppl-freq', type=int, default=100,
						help='print and log test PPL after this many steps')
	parser.add_argument('--test-ir-freq', type=int, default=100,
						help='print and log test IR after this many steps')
	parser.add_argument('--patience', type=int, default=10,
						help='print and log test IR after this many steps')
	parser.add_argument('--validation-bs', type=int, default=64,
						help='the batch size for validation evaluation')
	parser.add_argument('--test-bs', type=int, default=64,
						help='the batch size for test evaluation')
	parser.add_argument('--validation-ir-freq', type=int, default=500,
						help='print loss after this many steps')
	parser.add_argument('--bidirectional', type=str, default="False",
						help='whether to use bidirectional DocNADE model or not')
	parser.add_argument('--combination-type', type=str, default="concat",
						help='combination type for bidirectional docnade')
	parser.add_argument('--generative-loss-weight', type=float, default=0.0,
						help='weight for generative loss in total loss')
	parser.add_argument('--projection', type=str, default="False",
						help='whether to project prior embeddings or not')
	parser.add_argument('--reload', type=str, default="False",
						help='whether to reload model or not')
	parser.add_argument('--reload-model-dir', type=str,
						help='path for model to be reloaded')
	parser.add_argument('--model-type', type=str,
						help='type of model to be reloaded')
	parser.add_argument('--deep-hidden-sizes', nargs='+', type=int,
						help='sizes of the hidden layers')
	parser.add_argument('--deep', type=str, default="False",
						help='whether to maked model deep or not')
	parser.add_argument('--multi-label', type=str, default="False",
						help='whether dataset is multi-label or not')
	parser.add_argument('--shuffle-reload', type=str, default="True",
						help='whether dataset is shuffled or not')
	parser.add_argument('--trainfile', type=str, required=True,
						help='path to train text file')
	parser.add_argument('--valfile', type=str, required=True,
						help='path to validation text file')
	parser.add_argument('--testfile', type=str, required=True,
						help='path to test text file')
	#parser.add_argument('--lambda-embeddings', type=float, default=0.0,
	#					help='combination weight for prior embeddings into docnade')
	parser.add_argument('--W-pretrained-path', type=str, default="",
						help='path for pretrained W matrix')
	parser.add_argument('--U-pretrained-path', type=str, default="",
						help='path for pretrained U matrix')

	parser.add_argument('--sal-loss', type=str, default="False",
						help='whether to include SAL loss')
	parser.add_argument('--sal-gamma', type=str, default="automatic",
						help='"automatic" or "manual"')
	parser.add_argument('--sal-gamma-init', type=float, nargs='+', default=[],
						help='initialization value for SAL gamma variable')
	parser.add_argument('--ll-loss', type=str, default="False",
						help='whether to include LL loss')
	parser.add_argument('--ll-lambda', type=str, default="automatic",
						help='"automatic" or "manual"')
	parser.add_argument('--ll-lambda-init', type=float, nargs='+', default=[],
						help='"automatic" or "manual"')
	parser.add_argument('--dataset-old', type=str, nargs='+', required=True,
						help='path to the old datasets')
	parser.add_argument('--pretraining-target', type=str, default="False",
						help='whether to do pretraining on target data or not')
	parser.add_argument('--pretraining-epochs', type=int, default=50,
						help='number of epochs for pretraining')
	parser.add_argument('--bias-sharing', type=str, default="True",
						help='whether to share encoding and decoding bias with old dataset or not')
	parser.add_argument('--sal-threshold', type=float, nargs='+', default=[],
						help='theshold on NLL for old dataset')
	parser.add_argument('--W-old-path-list', type=str, nargs='+', default=[],
						help='path to the W matrices of source datasets')
	parser.add_argument('--U-old-path-list', type=str, nargs='+', default=[],
						help='path to the U matrices of source datasets')
	parser.add_argument('--W-old-vocab-path-list', type=str, nargs='+', default=[],
						help='path to the vocab of source datasets')
	parser.add_argument('--use-embeddings-prior', type=str, default="False",
						help='whether to embedings as prior or not')
	parser.add_argument('--lambda-embeddings', type=str, default="",
						help='make embeddings lambda trainable or not')
	parser.add_argument('--lambda-embeddings-list', type=float, nargs='+', default=[],
						help='list of lambda for every embedding prior')

	parser.add_argument('--reload-source-data-list', type=str, nargs='+', default=[],
						help='list of source datasets')
	parser.add_argument('--bias-W-old-path-list', type=str, nargs='+', default=[],
						help='path to the bias of W matrices of source datasets')
	parser.add_argument('--bias-U-old-path-list', type=str, nargs='+', default=[],
						help='path to the bias of U matrices of source datasets')
	parser.add_argument('--reload-source-multi-label', type=str, nargs='+', default=[], required=True,
						help='whether source datasets are multi-label or not')
	parser.add_argument('--reload-source-num-classes', type=int, nargs='+', default=[], required=True,
						help='number of classes in source datasets')

	return parser.parse_args()


if __name__ == '__main__':
	main(parse_args())

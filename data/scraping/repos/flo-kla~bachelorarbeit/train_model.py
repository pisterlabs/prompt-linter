import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as eval
import datetime
import json
import sys
import pickle
from tqdm import tqdm 
# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

dir(tf.contrib)


def loadGloveModel(gloveFile=None, params=None):
    '''
    This function loads GloVe embeddings as per hidden size of DocNADE or iDocNADE.
    '''
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


def train(model, dataset, params):
    '''
    This function runs training of DocNADE/iDocNADE
    based on the given parameters.
    Also logs the training and validation PPL and IR scores in 
    log directory.
    For information about various training parameters see ReadME.md file
    '''
    log_dir = os.path.join(params.model, 'logs')
    model_dir_ir = os.path.join(params.model, 'model_ir')
    model_dir_ppl = os.path.join(params.model, 'model_ppl')

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

        # Shuffle: the order of words in the sentence for DocNADE to avoid overfitting
        if params.bidirectional:
            training_data = dataset.batches_bidirectional('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)
        else:
            training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)

        # Global variables for best training, validation and test scores
        best_val_IR = 0.0
        best_val_nll = np.inf
        best_val_ppl = np.inf
        best_val_disc_accuracy = 0.0

        best_test_IR = 0.0
        best_test_nll = np.inf
        best_test_ppl = np.inf
        best_test_disc_accuracy = 0.0
        
        # Patience is set differently for DocNADE and iDocNADE
        #if params.bidirectional or params.initialize_docnade:
        #    patience = 30
        #else:
        #    patience = params.patience
        patience = params.patience
        
        patience_count = 0
        #patience_count_ir = 0
        best_train_nll = np.inf

        # Loading labels for Information Retrieval (IR)
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
        )
        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
        )
        test_labels = np.array(
            [[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
        )

        # Start of training
        print("Training started.\n")

        for step in range(params.num_steps + 1):
            this_loss = -1.

            if params.bidirectional:
                # Getting data batch by batch
                y, x, x_bw, seq_lengths = next(training_data)

                _, loss_normed, loss_unnormed, loss_normed_bw, loss_unnormed_bw = session.run([model.opt, model.loss_normed, model.loss_unnormed,
                                                                            model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                    model.x: x,
                    model.x_bw: x_bw,
                    model.y: y,
                    model.seq_lengths: seq_lengths
                })
                this_loss = 0.5 * (loss_unnormed + loss_unnormed_bw)
                losses.append(this_loss)
            else:
                # Getting data batch by batch
                y, x, seq_lengths = next(training_data)
                
                _, loss, loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict={
                    model.x: x,
                    model.y: y,
                    model.seq_lengths: seq_lengths
                })
                this_loss = loss
                losses.append(this_loss)

            # Printing training loss
            if (step % params.log_every == 0):
                print('{}: {:.6f}'.format(step, this_loss))


            # Calculating PPL for validation set as per "params.validation_ppl_freq" parameter
            if step and (step % params.validation_ppl_freq) == 0:
                # val_loss_unnormed is for Negative Log Likelihood (NLL)
                # val_loss_normed is for Perplexity (PPL)

                this_val_nll = []
                this_val_loss_normed = []
                this_val_nll_bw = []
                this_val_loss_normed_bw = []

                if params.bidirectional:
                    # Getting iDocNADE validation data as per validation batch size parameter "params.validation_bs"
                    for val_y, val_x, val_x_bw, val_seq_lengths in tqdm(dataset.batches_bidirectional('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label)):
                        val_loss_normed, val_loss_unnormed, \
                        val_loss_normed_bw, val_loss_unnormed_bw = session.run([model.loss_normed, model.loss_unnormed, 
                                                                                model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                            model.x: val_x,
                            model.x_bw: val_x_bw,
                            model.y: val_y,
                            model.seq_lengths: val_seq_lengths
                        })
                        this_val_nll.append(val_loss_unnormed)
                        this_val_loss_normed.append(val_loss_normed)
                        this_val_nll_bw.append(val_loss_unnormed_bw)
                        this_val_loss_normed_bw.append(val_loss_normed_bw)
                else:
                    # Getting DocNADE validation data as per validation batch size parameter "params.validation_bs"
                    for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
                            model.x: val_x,
                            model.y: val_y,
                            model.seq_lengths: val_seq_lengths
                        })
                        this_val_nll.append(val_loss_unnormed)
                        this_val_loss_normed.append(val_loss_normed)
                
                # Calculating PPL and NLL on full validation set
                if params.bidirectional:
                    print("Calculating stuff")
                    total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
                    total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
                    print("Calculating stuff finished")
                else:
                    total_val_nll = np.mean(this_val_nll)
                    total_val_ppl = np.exp(np.mean(this_val_loss_normed))

                # Only saving latest best model
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

                # Print validation PPL and IR statistics
                print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f})'.format(
                    total_val_ppl,
                    best_val_ppl or 0.0,
                    best_val_nll
                ))

                # logging information
                with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                    f.write("Step: %i,    val PPL: %s,     best val PPL: %s,    best val loss: %s\n" % 
                            (step, total_val_ppl, best_val_ppl, best_val_nll))

                if patience_count > patience:
                    print("Early stopping criterion satisfied.")
                    break
            
            # Calculating IR for validation set as per "params.validation_ir_freq" parameter
            if step >= 1 and step % params.validation_ir_freq == 0:
                this_val_nll = []
                this_val_loss_normed = []
                this_val_nll_bw = []
                this_val_loss_normed_bw = []

                if params.bidirectional:
                    # Getting iDocNADE validation data as per validation batch size parameter "params.validation_bs"
                    for val_y, val_x, val_x_bw, val_seq_lengths in tqdm(dataset.batches_bidirectional('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label)):
                        _, val_loss_normed, val_loss_unnormed, \
                        val_loss_normed_bw, val_loss_unnormed_bw = session.run([model.opt, model.loss_normed, model.loss_unnormed, 
                                                                                model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                            model.x: val_x,
                            model.x_bw: val_x_bw,
                            model.y: val_y,
                            model.seq_lengths: val_seq_lengths
                        })
                        this_val_nll.append(val_loss_unnormed)
                        this_val_loss_normed.append(val_loss_normed)
                        this_val_nll_bw.append(val_loss_unnormed_bw)
                        this_val_loss_normed_bw.append(val_loss_normed_bw)
                else:
                    # Getting DocNADE validation data as per validation batch size parameter "params.validation_bs"
                    for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        _, val_loss_normed, val_loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict={
                            model.x: val_x,
                            model.y: val_y,
                            model.seq_lengths: val_seq_lengths
                        })
                        this_val_nll.append(val_loss_unnormed)
                        this_val_loss_normed.append(val_loss_normed)
                
                # Calculating PPL and NLL on full validation set
                if params.bidirectional:
                    total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
                    total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
                else:
                    total_val_nll = np.mean(this_val_nll)
                    total_val_ppl = np.exp(np.mean(this_val_loss_normed))

                # Only saving latest best model
                if total_val_ppl < best_val_ppl:
                    best_val_ppl = total_val_ppl

                # Early stopping
                if total_val_nll < best_val_nll:
                    best_val_nll = total_val_nll
                    patience_count = 0
                else:
                    patience_count += 1

                # Print validation PPL and IR statistics
                print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f})'.format(
                    total_val_ppl,
                    best_val_ppl or 0.0,
                    best_val_nll
                ))

                # Getting hidden vectors using iDocNADE/DocNADE model for all documents in validation set
                if params.bidirectional:
                    validation_vectors = m.vectors_bidirectional(
                        model,
                        dataset.batches_bidirectional(
                            'validation_docnade',
                            params.validation_bs,
                            num_epochs=1,
                            shuffle=True,
                            multilabel=params.multi_label
                        ),
                        session,
                        params.combination_type
                    )

                    training_vectors = m.vectors_bidirectional(
                        model,
                        dataset.batches_bidirectional(
                            'training_docnade',
                            params.validation_bs,
                            num_epochs=1,
                            shuffle=True,
                            multilabel=params.multi_label
                        ),
                        session,
                        params.combination_type
                    )
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

                # Calculating validation IR using "evaluate.py" in "model" directory
                val = eval.evaluate(
                    training_vectors,
                    validation_vectors,
                    training_labels,
                    validation_labels,
                    recall=[0.02],
                    num_classes=params.num_classes,
                    multi_label=params.multi_label
                )[0]

                # Saving best IR model
                if val > best_val_IR:
                    best_val_IR = val
                    print('saving: {}'.format(model_dir_ir))
                    saver.save(session, model_dir_ir + '/model_ir', global_step=1)
                #    patience_count = 0
                #else:
                #    patience_count += 1
                
                print('This val IR: {:.3f} (best val IR: {:.3f})'.format(
                    val,
                    best_val_IR or 0.0
                ))

                # logging information
                with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                    f.write("Step: %i,    val IR: %s,    best val IR: %s\n" % 
                            (step, val, best_val_IR))

                # Saving summaries
                if params.bidirectional:
                    summary, = session.run([summaries], feed_dict={
                        model.x: x,
                        model.x_bw: x_bw,
                        model.y: y,
                        model.seq_lengths: seq_lengths,
                        validation: val,
                        validation_accuracy: 0.0,
                        avg_loss: np.average(losses)
                    })
                else:
                    summary, = session.run([summaries], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths,
                        validation: val,
                        validation_accuracy: 0.0,
                        avg_loss: np.average(losses)
                    })
                summary_writer.add_summary(summary, step)
                summary_writer.flush()
                losses = []

                if patience_count > patience:
                    print("Early stopping criterion satisfied.")
                    break
            
            # Calculating PPL for test set as per "params.test_ppl_freq" parameter
            if step and (step % params.test_ppl_freq) == 0:
                # test_loss_unnormed is for Negative Log Likelihood (NLL)
                # test_loss_normed is for Perplexity (PPL)

                this_test_nll = []
                this_test_loss_normed = []
                this_test_nll_bw = []
                this_test_loss_normed_bw = []

                if params.bidirectional:
                    # Getting iDocNADE test set as per test batch size parameter "params.test_bs"
                    for test_y, test_x, test_x_bw, test_seq_lengths in dataset.batches_bidirectional('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        test_loss_normed, test_loss_unnormed, \
                        test_loss_normed_bw, test_loss_unnormed_bw = session.run([model.loss_normed, model.loss_unnormed, 
                                                                                    model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                            model.x: test_x,
                            model.x_bw: test_x_bw,
                            model.y: test_y,
                            model.seq_lengths: test_seq_lengths
                        })
                        this_test_nll.append(test_loss_unnormed)
                        this_test_loss_normed.append(test_loss_normed)
                        this_test_nll_bw.append(test_loss_unnormed_bw)
                        this_test_loss_normed_bw.append(test_loss_normed_bw)
                else:
                    # Getting DocNADE test set as per test batch size parameter "params.test_bs"
                    for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        test_loss_normed, test_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
                            model.x: test_x,
                            model.y: test_y,
                            model.seq_lengths: test_seq_lengths
                        })
                        this_test_nll.append(test_loss_unnormed)
                        this_test_loss_normed.append(test_loss_normed)

                # Calculating PPl and NLL on full test set
                if params.bidirectional:
                    total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
                    total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
                else:
                    total_test_nll = np.mean(this_test_nll)
                    total_test_ppl = np.exp(np.mean(this_test_loss_normed))

                # Saving best test set values
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
                    f.write("Step: %i,    test PPL: %s,    best test PPL: %s,    best test loss: %s\n" % 
                            (step, total_test_ppl, best_test_ppl, best_test_nll))

            # Calculating IR for test set as per "params.test_ir_freq" parameter
            if step >= 1 and (step % params.test_ir_freq) == 0:

                if params.bidirectional:
                    # Getting iDocNADE test set as per test batch size parameter "params.test_bs"
                    for test_y, test_x, test_x_bw, test_seq_lengths in dataset.batches_bidirectional('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        _, test_loss_normed, test_loss_unnormed, \
                        test_loss_normed_bw, test_loss_unnormed_bw = session.run([model.opt, model.loss_normed, model.loss_unnormed, 
                                                                                    model.loss_normed_bw, model.loss_unnormed_bw], feed_dict={
                            model.x: test_x,
                            model.x_bw: test_x_bw,
                            model.y: test_y,
                            model.seq_lengths: test_seq_lengths
                        })
                else:
                    # Getting DocNADE test set as per test batch size parameter "params.test_bs"
                    for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                        _, test_loss_normed, test_loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict={
                            model.x: test_x,
                            model.y: test_y,
                            model.seq_lengths: test_seq_lengths
                        })

                # Getting hidden vectors using iDocNADE/DocNADE model for all documents in test set
                if params.bidirectional:
                    test_vectors = m.vectors_bidirectional(
                        model,
                        dataset.batches_bidirectional(
                            'test_docnade',
                            params.test_bs,
                            num_epochs=1,
                            shuffle=True,
                            multilabel=params.multi_label
                        ),
                        session,
                        params.combination_type
                    )

                    training_vectors = m.vectors_bidirectional(
                        model,
                        dataset.batches_bidirectional(
                            'training_docnade',
                            params.test_bs,
                            num_epochs=1,
                            shuffle=True,
                            multilabel=params.multi_label
                        ),
                        session,
                        params.combination_type
                    )
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

                # Calculating test IR using "evaluate.py" in "model" directory
                test = eval.evaluate(
                    training_vectors,
                    test_vectors,
                    training_labels,
                    test_labels,
                    recall=[0.02],
                    num_classes=params.num_classes,
                    multi_label=params.multi_label
                )[0]

                # Saving and printing best test IR value
                if test > best_test_IR:
                    best_test_IR = test
                
                print('This test IR: {:.3f} (best test IR: {:.3f})'.format(
                    test,
                    best_test_IR or 0.0
                ))
        
        print("Training finished.")



from math import *
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def compute_coherence(texts, list_of_topics, top_n_word_in_each_topic_list, reload_model_dir):
    '''
    Function to compute Topic Coherence based on different types available.

    list_of_topics:                 list of list of topic words
    top_n_word_in_each_topic_list:  list of number of words to count from beginning
                                    to calculate topic coherence
    reload_model_dir:               model directory created when running the experiments
    '''

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
    '''
    Function to get document representation vectors using embedding
    matrix of iDocNADE/DocNADE

    matrix: embedding matrix of shape = [vocab_size X embedding_size]
    batches: instance data.batches/data.batches_bidirectional function
    '''

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
    '''
    Function to calculate L2 norm of a vector

    x: a vector or a list
    '''
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y,i):
    '''
    Function to calculate cosine similarity between x and y.

    x: a vector or a list
    y: a vector or a list
    '''
    if (i%10000)==0:
        print("Iteration: "+str(i))
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


def reload_evaluation_ir(model_ir, dataset, params):
    '''
    Main function which does evaluation on reloaded model.

    model_ir:  iDocNADE/DocNADE model instance with weights/biases from best IR model
    datset:    an instance of data object
    params:    parameters saved during training of the reloaded model
    '''
    print("doing reload eval now")
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params['num_cores'],
        intra_op_parallelism_threads=params['num_cores'],
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        log_dir = os.path.join(params['model'], 'logs')

        # List of ratios on which IR has to be calculated
        ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

        # List of "reciprocal of regularization strength" to be used 
        # in classification using LogiticRegression module of sklearn
        c_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]

        # Loading labels for IR and Classification
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
        
        # Calculating hidden vector representation for training, validation and test documents
        if params['bidirectional']:
            training_vectors = m.vectors_bidirectional(
                model_ir,
                dataset.batches_bidirectional(
                    'training_docnade',
                    params['validation_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session,
                params['combination_type']
            )

            validation_vectors = m.vectors_bidirectional(
                model_ir,
                dataset.batches_bidirectional(
                    'validation_docnade',
                    params['validation_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session,
                params['combination_type']
            )

            test_vectors = m.vectors_bidirectional(
                model_ir,
                dataset.batches_bidirectional(
                    'test_docnade',
                    params['test_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session,
                params['combination_type']
            )
        else:
            training_vectors = m.vectors(
                model_ir,
                dataset.batches(
                    'training_docnade',
                    params['validation_bs'],
                    num_epochs=1,
                    shuffle=True,
                    multilabel=params['multi_label']
                ),
                session
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
                session
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
                session
            )
        
        
        print("calculated hidden vec representations done")
        # Calculating IR for validation set using "evaluate.py" in "model" directory
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
        with open(os.path.join(log_dir, "reload_info_ir.txt"), "w") as f:
            f.write("\n\nFractions list: %s" % (ir_ratio_list))
            f.write("\nVal IR: %s" % (val_list))
        

        print("starting to calc test IR now")
        # Calculating IR for test set using "evaluate.py" in "model" directory
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
        with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
            f.write("\n\nFractions list: %s" % (ir_ratio_list))
            f.write("\n\nTest IR: %s" % (test_list))
        
        print("logging done")
        # Document classification using LogisticRegression from scikit-learn
        test_acc = []
        test_f1 = []
        val_acc = []
        val_f1 = []

        test_acc_W = []
        test_f1_W = []
        val_acc_W = []
        val_f1_W = []

        # Loading train, validation and test set labels for classification
        y_train = np.array(
            [y for y, _ in dataset.rows('training_docnade', num_epochs=1)]
        )
        y_val = np.array(
            [y for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
        )
        y_test = np.array(
            [y for y, _ in dataset.rows('test_docnade', num_epochs=1)]
        )
        print("Hallo")
        # Getting document representation using embedding matrix of DocNADE/iDocNADE
        training_vectors_W = get_vectors_from_matrix(model_ir.W, dataset.batches('training_docnade', 1, num_epochs=1, shuffle=False, multilabel=params['multi_label']))
        validation_vectors_W = get_vectors_from_matrix(model_ir.W, dataset.batches('validation_docnade', 1, num_epochs=1, shuffle=False, multilabel=params['multi_label']))
        test_vectors_W = get_vectors_from_matrix(model_ir.W, dataset.batches('test_docnade', 1, num_epochs=1, shuffle=False, multilabel=params['multi_label']))
        print("Hallo2222")
        if not params['multi_label']:
            train_data = (training_vectors, np.array(y_train, dtype=np.int32))
            validation_data = (validation_vectors, np.array(y_val, dtype=np.int32))
            test_data = (test_vectors, np.array(y_test, dtype=np.int32))

            # Performing classification
            test_acc, test_f1, val_acc, val_f1 = eval.perform_classification(train_data, validation_data, test_data, c_list)
            #test_acc, test_f1 = eval.perform_classification_test(train_data, test_data, c_list)

            # logging information
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation accuracy with h vector IR: %s" % (val_acc))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))

            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation F1 score with h vector IR: %s" % (val_f1))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))

            print("more logging done")
            train_data_W = (training_vectors_W, np.array(y_train, dtype=np.int32))
            validation_data_W = (validation_vectors_W, np.array(y_val, dtype=np.int32))
            test_data_W = (test_vectors_W, np.array(y_test, dtype=np.int32))

            # Performing classification
            test_acc_W, test_f1_W, val_acc_W, val_f1_W = eval.perform_classification(train_data_W, validation_data_W, test_data_W, c_list)
            #test_acc_W, test_f1_W = eval.perform_classification_test(train_data_W, test_data_W, c_list)

            # logging information
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation accuracy with W matrix IR: %s" % (val_acc_W))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest accuracy with W matrix IR: %s" % (test_acc_W))

            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation F1 score with W matrix IR: %s" % (val_f1_W))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest F1 score with W matrix IR: %s" % (test_f1_W))
            
        else:
            total_labels = []

            # Creating one-hot labels for multi label dataset
            y_train_new = [label.strip().split(':') for label in y_train]
            y_val_new = [label.strip().split(':') for label in y_val]
            y_test_new = [label.strip().split(':') for label in y_test]

            total_labels.extend(y_train_new)

            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            mlb.fit(total_labels)
            y_train_one_hot = mlb.transform(y_train_new)
            y_val_one_hot = mlb.transform(y_val_new)
            y_test_one_hot = mlb.transform(y_test_new)

            train_data = (training_vectors, y_train_one_hot)
            validation_data = (validation_vectors, y_val_one_hot)
            test_data = (test_vectors, y_test_one_hot)

            # Performing multi label classification
            test_acc, test_f1, val_acc, val_f1 = eval.perform_classification_multi(train_data, validation_data, test_data, c_list)
            #test_acc, test_f1 = eval.perform_classification_test_multi(train_data, test_data, c_list)

            # logging information
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation accuracy with h vector IR: %s" % (val_acc))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))

            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation F1 score with h vector IR: %s" % (val_f1))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))

            train_data_W = (training_vectors_W, y_train_one_hot)
            validation_data_W = (validation_vectors_W, y_val_one_hot)
            test_data_W = (test_vectors_W, y_test_one_hot)

            # Performing multi label classification
            test_acc_W, test_f1_W, val_acc_W, val_f1_W = eval.perform_classification_multi(train_data_W, validation_data_W, test_data_W, c_list)
            #test_acc_W, test_f1_W = eval.perform_classification_test_multi(train_data_W, test_data_W, c_list)

            # logging information
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation accuracy with W matrix IR: %s" % (val_acc_W))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest accuracy with W matrix IR: %s" % (test_acc_W))

            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nValidation F1 score with W matrix IR: %s" % (val_f1_W))
            
            with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
                f.write("\n\nTest F1 score with W matrix IR: %s" % (test_f1_W))
        
        

def reload_evaluation_ppl(model_ppl, dataset, params):
    '''
    Main function which does evaluation on reloaded model.

    model_ppl: iDocNADE/DocNADE model instance with weights/biases from best PPL model
    datset:    an instance of data object
    params:    parameters saved during training of the reloaded model
    '''
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params['num_cores'],
        intra_op_parallelism_threads=params['num_cores'],
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        log_dir = os.path.join(params['model'], 'logs')

        # List of ratios on which IR has to be calculated
        ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

        # List of "reciprocal of regularization strength" to be used 
        # in classification using LogiticRegression module of sklearn
        c_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]

        # Loading labels for IR and Classification
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
        

        # Calculating PPL for validation set

        # val_loss_unnormed is for Negative Log Likelihood (NLL)
        # val_loss_normed is for Perplexity (PPL)

        this_val_nll = []
        this_val_loss_normed = []
        this_val_nll_bw = []
        this_val_loss_normed_bw = []

        if params['bidirectional']:
            # Getting iDocNADE validation set
            for val_y, val_x, val_x_bw, val_seq_lengths in dataset.batches_bidirectional('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                val_loss_normed, val_loss_unnormed, \
                val_loss_normed_bw, val_loss_unnormed_bw = session.run([model_ppl.loss_normed, model_ppl.loss_unnormed, 
                                                                        model_ppl.loss_normed_bw, model_ppl.loss_unnormed_bw], feed_dict={
                    model_ppl.x: val_x,
                    model_ppl.x_bw: val_x_bw,
                    model_ppl.y: val_y,
                    model_ppl.seq_lengths: val_seq_lengths
                })
                this_val_nll.append(val_loss_unnormed)
                this_val_loss_normed.append(val_loss_normed)
                this_val_nll_bw.append(val_loss_unnormed_bw)
                this_val_loss_normed_bw.append(val_loss_normed_bw)
        else:
            # Getting DocNADE validation set
            for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                val_loss_normed, val_loss_unnormed = session.run([model_ppl.loss_normed, model_ppl.loss_unnormed], feed_dict={
                    model_ppl.x: val_x,
                    model_ppl.y: val_y,
                    model_ppl.seq_lengths: val_seq_lengths
                })
                this_val_nll.append(val_loss_unnormed)
                this_val_loss_normed.append(val_loss_normed)
        
        # Calculating PPl and NLL on validation set
        if params['bidirectional']:
            total_val_nll = 0.5 * (np.mean(this_val_nll) + np.mean(this_val_nll_bw))
            total_val_ppl = 0.5 * (np.exp(np.mean(this_val_loss_normed)) + np.exp(np.mean(this_val_loss_normed_bw)))
        else:
            total_val_nll = np.mean(this_val_nll)
            total_val_ppl = np.exp(np.mean(this_val_loss_normed))

        print('Val PPL: {:.3f},    Val loss: {:.3f}\n'.format(
            total_val_ppl,
            total_val_nll
        ))

        # logging information
        with open(os.path.join(log_dir, "reload_info_ppl.txt"), "w") as f:
            f.write("Val PPL: %s,    Val loss: %s" % 
                    (total_val_ppl, total_val_nll))
        
        #here
        # Calculating PPL for test set

        # test_loss_unnormed is for Negative Log Likelihood (NLL)
        # test_loss_normed is for Perplexity (PPL)

        this_test_nll = []
        this_test_loss_normed = []
        this_test_nll_bw = []
        this_test_loss_normed_bw = []

        if params['bidirectional']:
            # Getting iDocNADE test set
            for test_y, test_x, test_x_bw, test_seq_lengths in dataset.batches_bidirectional('test_docnade', params['test_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                test_loss_normed, test_loss_unnormed, \
                test_loss_normed_bw, test_loss_unnormed_bw = session.run([model_ppl.loss_normed, model_ppl.loss_unnormed, 
                                                                        model_ppl.loss_normed_bw, model_ppl.loss_unnormed_bw], feed_dict={
                    model_ppl.x: test_x,
                    model_ppl.x_bw: test_x_bw,
                    model_ppl.y: test_y,
                    model_ppl.seq_lengths: test_seq_lengths
                })
                this_test_nll.append(test_loss_unnormed)
                this_test_loss_normed.append(test_loss_normed)
                this_test_nll_bw.append(test_loss_unnormed_bw)
                this_test_loss_normed_bw.append(test_loss_normed_bw)
        else:
            # Getting DocNADE test set
            for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=True, multilabel=params['multi_label']):
                test_loss_normed, test_loss_unnormed = session.run([model_ppl.loss_normed, model_ppl.loss_unnormed], feed_dict={
                    model_ppl.x: test_x,
                    model_ppl.y: test_y,
                    model_ppl.seq_lengths: test_seq_lengths
                })
                this_test_nll.append(test_loss_unnormed)
                this_test_loss_normed.append(test_loss_normed)

        # Calculating PPl and NLL on full test set
        if params['bidirectional']:
            total_test_nll = 0.5 * (np.mean(this_test_nll) + np.mean(this_test_nll_bw))
            total_test_ppl = 0.5 * (np.exp(np.mean(this_test_loss_normed)) + np.exp(np.mean(this_test_loss_normed_bw)))
        else:
            total_test_nll = np.mean(this_test_nll)
            total_test_ppl = np.exp(np.mean(this_test_loss_normed))

        print('Test PPL: {:.3f},    Test loss: {:.3f}\n'.format(
            total_test_ppl,
            total_test_nll
        ))

        # logging information
        with open(os.path.join(log_dir, "reload_info_ppl.txt"), "a") as f:
            f.write("\n\nTest PPL: %s,    Test loss: %s" % 
                    (total_test_ppl, total_test_nll))
        
        
        
        # Topics calculation with W matrix
        top_n_topic_words = 20
        w_h_top_words_indices = []
        W_topics = model_ppl.W
        topics_list_W = []

        for h_num in range(np.array(W_topics).shape[1]):
            w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

        with open(params['docnadeVocab'], 'r') as f:
            vocab_docnade = [w.strip() for w in f.readlines()]

        with open(os.path.join(log_dir, "topics_ppl_W.txt"), "w") as f:
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

        # Topics calculation with V matrix
        top_n_topic_words = 20
        w_h_top_words_indices = []
        W_topics = model_ppl.V.T
        topics_list_V = []

        for h_num in range(np.array(W_topics).shape[1]):
            w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

        with open(params['docnadeVocab'], 'r') as f:
            vocab_docnade = [w.strip() for w in f.readlines()]

        with open(os.path.join(log_dir, "topics_ppl_V.txt"), "w") as f:
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
        
        # Calculating topic coherence of the topics calculated above
        top_n_word_in_each_topic_list = [10, 20]

        text_filenames = [
            params['trainfile'],
            params['valfile'],
            params['testfile']
        ]

        # read original text documents as list of words
        texts = []

        for file in text_filenames:
            print('filename:%s', file)
            for line in open(file.rstrip('\r'), 'r').readlines():
                document = str(line).split('\t')[1]
                document = document.encode(encoding="utf-8",errors="ignore").decode('utf-8',errors='ignore')
                texts.append(document.split())

        compute_coherence(texts, topics_list_W, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_W.txt"))
        compute_coherence(texts, topics_list_V, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_V.txt"))

        print("Done with computing W and V coherence")
        """
        # Calculating nearest neighbors using DocNADE/iDocNADE W embedding matrix
        W = model_ppl.W
        vocab = vocab_docnade
        sim_mat = dict()
        debug_i=0
        print("W-Matrix: "+str(W.shape[0]))
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                sim_mat[str(vocab[i]) + "_" + str(vocab[j])] = np.around(cosine_similarity(W[i][:], W[j][:],debug_i), decimals=3)
                debug_i+=1
                
        sorted_sim_mat = sim_mat

        print('shape sorted_sim_mat:', len(sorted_sim_mat))

        top_5_words_for_each_word = dict()
        for key in sorted_sim_mat.keys():
            value = sorted_sim_mat[key]
            target_word = str(key).split('_')[0]

            if not target_word in top_5_words_for_each_word.keys():
                top_5_words_for_each_word[target_word] = []

            second_word = str(key).split('_')[1]

            if target_word != second_word:
                top_5_words_for_each_word[target_word].append((second_word, value))

        top_n_similar_words_for_each_word = {}
        top_n = 20

        with open(os.path.join(log_dir, "nearest_neighbors_ppl_W.txt"), "w") as f:
            for target_word in top_5_words_for_each_word.keys():

                if not len(target_word) > 2: continue

                sample_word_tupls = top_5_words_for_each_word[target_word]
                sorted_by_sim = sorted(sample_word_tupls, key=lambda tup: tup[1])[::-1][:top_n * top_n]
                
                sorted_by_sim = [(x, np.around(y, decimals=2)) for (x, y)
                                in sorted_by_sim if wordnet.synsets(x)
                                and len(x) > 2][:top_n]  # check if the word is in english dict
                
                top_n_similar_words_for_each_word[target_word] = sorted_by_sim
                top_words = [word for word, sim in sorted_by_sim]
                print('target word:%s  Top_words:%s' % (target_word, top_words))
                print('======================================================================================================')

                f.write('target word:%s  Top_words:%s\n' % (target_word, top_words))
                f.write('======================================================================================================\n')

            print('len top_5_words_for_each_word:%s' % len(top_5_words_for_each_word))
            f.write('len top_5_words_for_each_word:%s' % len(top_5_words_for_each_word))


        # Calculating nearest neighbors using DocNADE/iDocNADE V matrix
        W = model_ppl.V.T
        vocab = vocab_docnade
        sim_mat = dict()
        debug_j=0
        print("V-Matrix: "+str(W.shape[0]))
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                sim_mat[str(vocab[i]) + "_" + str(vocab[j])] = np.around(cosine_similarity(W[i][:], W[j][:],debug_j), decimals=3)
                debug_j+=1
                
        sorted_sim_mat = sim_mat

        print('shape sorted_sim_mat:', len(sorted_sim_mat))

        top_5_words_for_each_word = dict()
        for key in sorted_sim_mat.keys():
            value = sorted_sim_mat[key]
            target_word = str(key).split('_')[0]

            if not target_word in top_5_words_for_each_word.keys():
                top_5_words_for_each_word[target_word] = []

            second_word = str(key).split('_')[1]

            if target_word != second_word:
                top_5_words_for_each_word[target_word].append((second_word, value))

        top_n_similar_words_for_each_word = {}
        top_n = 20

        with open(os.path.join(log_dir, "nearest_neighbors_ppl_V.txt"), "w") as f:
            for target_word in top_5_words_for_each_word.keys():

                if not len(target_word) > 2: continue

                sample_word_tupls = top_5_words_for_each_word[target_word]
                sorted_by_sim = sorted(sample_word_tupls, key=lambda tup: tup[1])[::-1][:top_n * top_n]
                
                sorted_by_sim = [(x, np.around(y, decimals=2)) for (x, y)
                                in sorted_by_sim if wordnet.synsets(x)
                                and len(x) > 2][:top_n]  # check if the word is in english dict
                
                top_n_similar_words_for_each_word[target_word] = sorted_by_sim
                top_words = [word for word, sim in sorted_by_sim]
                print('target word:%s  Top_words:%s' % (target_word, top_words))
                print('======================================================================================================')

                f.write('target word:%s  Top_words:%s\n' % (target_word, top_words))
                f.write('======================================================================================================\n')

            print('len top_5_words_for_each_word:%s' % len(top_5_words_for_each_word))
            f.write('len top_5_words_for_each_word:%s' % len(top_5_words_for_each_word))
            """

def str2bool(v):
    '''
    Function to convert string parameters to boolean
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    # Converting string parameters parsed from configuration file to boolean type
    args.reload = str2bool(args.reload)
    args.initialize_docnade = str2bool(args.initialize_docnade)
    args.bidirectional = str2bool(args.bidirectional)
    args.projection = str2bool(args.projection)
    args.deep = str2bool(args.deep)
    args.multi_label = str2bool(args.multi_label)

    # Creating placeholders for tensorflow tensors
    x = tf.placeholder(tf.int32, shape=(None, None), name='x')
    x_bw = tf.placeholder(tf.int32, shape=(None, None), name='x_bw')
    if args.multi_label:
        y = tf.placeholder(tf.string, shape=(None), name='y')
    else:
        y = tf.placeholder(tf.int32, shape=(None), name='y')
    seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')

    
    if args.reload:
        # Model reload
        with open("model/" + args.reload_model_dir + "params.json") as f:
            params = json.load(f)

        if not 'multi_label' in params.keys():
            params['multi_label'] = False

        params['trainfile'] = args.trainfile
        params['valfile'] = args.valfile
        params['testfile'] = args.testfile

        params['reload_model_dir'] = args.reload_model_dir

        reload_ir = False
        if os.path.isdir("model/" + args.reload_model_dir + "/model_ir"):
            reload_ir = True

        reload_ppl = False
        if os.path.isdir("model/" + args.reload_model_dir + "/model_ppl"):
            reload_ppl = True

        if reload_ppl:
            sess_ppl = tf.Session()

            saver_ppl = tf.train.import_meta_graph("model/" + args.reload_model_dir + "/model_ppl/model_ppl-1.meta")
            saver_ppl.restore(sess_ppl, tf.train.latest_checkpoint("model/" + args.reload_model_dir + "/model_ppl/"))

            # Loading best saved IR/PPL weights and biases
            if params['projection']:
                embedding_prior_projection_ppl = sess_ppl.run("embedding_prior_projection:0")
            else:
                embedding_prior_projection_ppl = None

            if params['initialize_docnade']:
                embedding_prior_ppl = sess_ppl.run("embedding_prior:0")
            else:
                embedding_prior_ppl = None

            W_list_ppl = []
            bias_list_ppl = []
            bias_bw_list_ppl = []

            if params['bidirectional']:
                
                if params['deep']:
                    for index, size in enumerate(params['deep_hidden_sizes']):
                        embedding_name_ppl = "embedding_" + str(index) + ":0"
                        embedding_temp_ppl = sess_ppl.run(embedding_name_ppl)
                        W_list_ppl.append(embedding_temp_ppl)

                        bias_name_ppl = "bias_" + str(index) + ":0"
                        bias_temp_ppl = sess_ppl.run(bias_name_ppl)
                        bias_list_ppl.append(bias_temp_ppl)

                        bias_bw_name_ppl = "bias_bw_" + str(index) + ":0"
                        bias_bw_temp_ppl = sess_ppl.run(bias_bw_name_ppl)
                        bias_bw_list_ppl.append(bias_bw_temp_ppl)
                
                embedding_ppl = sess_ppl.run("embedding:0")
                bias_ppl = sess_ppl.run("bias:0")
                bias_bw_ppl = sess_ppl.run("bias_bw:0")
                V_ppl = sess_ppl.run("softmax/w:0")
                bias_V_ppl = sess_ppl.run("b:0")
                bias_bw_V_ppl = sess_ppl.run("b_bw:0")

                # Creating iDocNADE model based on best saved PPL model
                model_ppl = m.iDocNADE_reload(x, x_bw, y, seq_lengths, params, W_initializer=embedding_prior_ppl, W_reload=embedding_ppl, 
                                        W_prior_reload=embedding_prior_ppl, W_prior_proj_reload=embedding_prior_projection_ppl, 
                                        bias_reload=bias_ppl, bias_bw_reload=bias_bw_ppl, V_reload=V_ppl, b_reload=bias_V_ppl, b_bw_reload=bias_bw_V_ppl,
                                        W_list_reload=W_list_ppl, bias_list_reload=bias_list_ppl, bias_bw_list_reload=bias_bw_list_ppl,
                                        lambda_embeddings=params['lambda_embeddings'])
                
                print("iDocNADE PPL created")
                
            else:
                
                if params['deep']:
                    for index, size in enumerate(params['deep_hidden_sizes']):
                        embedding_name_ppl = "embedding_" + str(index) + ":0"
                        embedding_temp_ppl = sess_ppl.run(embedding_name_ppl)
                        W_list_ppl.append(embedding_temp_ppl)

                        bias_name_ppl = "bias_" + str(index) + ":0"
                        bias_temp_ppl = sess_ppl.run(bias_name_ppl)
                        bias_list_ppl.append(bias_temp_ppl)
                
                embedding_ppl = sess_ppl.run("embedding:0")
                bias_ppl = sess_ppl.run("bias:0")
                V_ppl = sess_ppl.run("softmax/w:0")
                bias_V_ppl = sess_ppl.run("b:0")

                # Creating DocNADE model based on best saved PPL model
                model_ppl = m.DocNADE_reload(x, y, seq_lengths, params, W_initializer=embedding_prior_ppl, W_reload=embedding_ppl, 
                                        W_prior_reload=embedding_prior_ppl, W_prior_proj_reload=embedding_prior_projection_ppl, 
                                        bias_reload=bias_ppl, bias_bw_reload=None, V_reload=V_ppl, b_reload=bias_V_ppl, b_bw_reload=None,
                                        W_list_reload=W_list_ppl, bias_list_reload=bias_list_ppl, lambda_embeddings=params['lambda_embeddings'])
                print("DocNADE PPL created")
            
            dataset = data.Dataset(params['dataset'])
            reload_evaluation_ppl(model_ppl, dataset, params)

        if reload_ir:
            sess_ir = tf.Session()
            
            saver_ir = tf.train.import_meta_graph("model/" + args.reload_model_dir + "/model_ir/model_ir-1.meta")
            saver_ir.restore(sess_ir, tf.train.latest_checkpoint("model/" + args.reload_model_dir + "/model_ir/"))

            # Loading best saved IR/PPL weights and biases
            if params['projection']:
                embedding_prior_projection_ir = sess_ir.run("embedding_prior_projection:0")
            else:
                embedding_prior_projection_ir = None

            if params['initialize_docnade']:
                embedding_prior_ir = sess_ir.run("embedding_prior:0")
            else:
                embedding_prior_ir = None

            W_list_ir = []
            bias_list_ir = []
            bias_bw_list_ir = []

            if params['bidirectional']:
                
                if params['deep']:
                    for index, size in enumerate(params['deep_hidden_sizes']):
                        embedding_name_ir = "embedding_" + str(index) + ":0"
                        embedding_temp_ir = sess_ir.run(embedding_name_ir)
                        W_list_ir.append(embedding_temp_ir)

                        bias_name_ir = "bias_" + str(index) + ":0"
                        bias_temp_ir = sess_ir.run(bias_name_ir)
                        bias_list_ir.append(bias_temp_ir)

                        bias_bw_name_ir = "bias_bw_" + str(index) + ":0"
                        bias_bw_temp_ir = sess_ir.run(bias_bw_name_ir)
                        bias_bw_list_ir.append(bias_bw_temp_ir)
                
                embedding_ir = sess_ir.run("embedding:0")
                bias_ir = sess_ir.run("bias:0")
                bias_bw_ir = sess_ir.run("bias_bw:0")
                V_ir = sess_ir.run("softmax/w:0")
                bias_V_ir = sess_ir.run("b:0")
                bias_bw_V_ir = sess_ir.run("b_bw:0")

                # Creating iDocNADE model based on best saved IR model
                model_ir = m.iDocNADE_reload(x, x_bw, y, seq_lengths, params, W_initializer=embedding_prior_ir, W_reload=embedding_ir, 
                                        W_prior_reload=embedding_prior_ir, W_prior_proj_reload=embedding_prior_projection_ir, 
                                        bias_reload=bias_ir, bias_bw_reload=bias_bw_ir, V_reload=V_ir, b_reload=bias_V_ir, b_bw_reload=bias_bw_V_ir,
                                        W_list_reload=W_list_ir, bias_list_reload=bias_list_ir, bias_bw_list_reload=bias_bw_list_ir,
                                        lambda_embeddings=params['lambda_embeddings'])

                print("iDocNADE IR created")
                
            else:
                
                if params['deep']:
                    for index, size in enumerate(params['deep_hidden_sizes']):
                        embedding_name_ir = "embedding_" + str(index) + ":0"
                        embedding_temp_ir = sess_ir.run(embedding_name_ir)
                        W_list_ir.append(embedding_temp_ir)

                        bias_name_ir = "bias_" + str(index) + ":0"
                        bias_temp_ir = sess_ir.run(bias_name_ir)
                        bias_list_ir.append(bias_temp_ir)
                
                embedding_ir = sess_ir.run("embedding:0")
                bias_ir = sess_ir.run("bias:0")
                V_ir = sess_ir.run("softmax/w:0")
                bias_V_ir = sess_ir.run("b:0")

                # Creating DocNADE model based on best saved IR model
                model_ir = m.DocNADE_reload(x, y, seq_lengths, params, W_initializer=embedding_prior_ir, W_reload=embedding_ir, 
                                        W_prior_reload=embedding_prior_ir, W_prior_proj_reload=embedding_prior_projection_ir, 
                                        bias_reload=bias_ir, bias_bw_reload=None, V_reload=V_ir, b_reload=bias_V_ir, b_bw_reload=None,
                                        W_list_reload=W_list_ir, bias_list_reload=bias_list_ir, lambda_embeddings=params['lambda_embeddings'])
                print("DocNADE IR created")

            dataset = data.Dataset(params['dataset'])
            reload_evaluation_ir(model_ir, dataset, params)
    else:
        # Model training
        now = datetime.datetime.now()

        if args.bidirectional:
            args.model += "_iDocNADE"
        else:
            args.model += "_DocNADE"

        if args.initialize_docnade:
            args.model += "_embprior"

        if args.pretrained_embeddings_path:
            args.model += "_pretrained"
        
        args.model +=  "_act_" + str(args.activation) + "_hidden_" + str(args.hidden_size) + "_vocab_" + str(args.vocab_size) \
                        + "_lr_" + str(args.learning_rate) + "_proj_" + str(args.projection) + "_deep_" + str(args.deep) \
                        + "_lambda_" + str(args.lambda_embeddings) + "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
        
        if not os.path.isdir(args.model):
            os.mkdir(args.model)

        docnade_vocab = args.docnadeVocab

        #if args.bidirectional or args.initialize_docnade:
        #    args.patience = 30

        with open(os.path.join(args.model, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))

        dataset = data.Dataset(args.dataset)

        if args.initialize_docnade:
            glove_embeddings = loadGloveModel(params=args)

        with open(docnade_vocab, 'r') as f:
            vocab_docnade = [w.strip() for w in f.readlines()]

        # Creating embedding prior matrix using glove embeddings
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

        # Load pretrained DocNADE embeddings for iDocNADE setting
        docnade_pretrained_matrix = None
        if args.pretrained_embeddings_path:
            with open(args.pretrained_embeddings_path, "rb") as f:
                docnade_pretrained_matrix = pickle.load(f)
            print("pretrained embeddings loaded.")
        
        if args.bidirectional:
            model = m.iDocNADE(x, x_bw, y, seq_lengths, args, W_initializer=docnade_embedding_matrix, lambda_embeddings=args.lambda_embeddings, W_pretrained=docnade_pretrained_matrix)
            print("iDocNADE created")
        else:
            model = m.DocNADE(x, y, seq_lengths, args, W_initializer=docnade_embedding_matrix, lambda_embeddings=args.lambda_embeddings, W_pretrained=docnade_pretrained_matrix)
            print("DocNADE created")
        
        # Train function
        train(model, dataset, args)


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
                        help='which activation to use: sigmoid|tanh|relu')
    parser.add_argument('--learning-rate', type=float, default=0.0004,
                        help='learning rate')
    parser.add_argument('--num-steps', type=int, default=50000,
                        help='the number of steps to train for')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training data')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='softmax samples (default: full softmax)')
    parser.add_argument('--num-cores', type=int, default=8,
                        help='the number of CPU cores to use')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print training loss after this many steps')
    parser.add_argument('--validation-ppl-freq', type=int, default=500,
                        help='print validation PPL and NLL after this many steps')
    parser.add_argument('--validation-ir-freq', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--num-classes', type=int, default=-1,
                        help='number of classes')
    parser.add_argument('--initialize-docnade', type=str, default="False",
                        help='whether to include glove embedding prior or not')
    parser.add_argument('--docnadeVocab', type=str, default="False",
                        help='path to vocabulary file used by DocNADE')
    parser.add_argument('--test-ppl-freq', type=int, default=100,
                        help='print and log test PPL and NLL after this many steps')
    parser.add_argument('--test-ir-freq', type=int, default=100,
                        help='print and log test IR after this many steps')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping criterion')
    parser.add_argument('--validation-bs', type=int, default=64,
                        help='the batch size for validation evaluation')
    parser.add_argument('--test-bs', type=int, default=64,
                        help='the batch size for test evaluation')
    parser.add_argument('--bidirectional', type=str, default="False",
                        help='whether to use iDocNADE model or not')
    parser.add_argument('--combination-type', type=str, default="concat",
                        help='combination type for iDocNADE forward and backward hidden document representation')
    parser.add_argument('--projection', type=str, default="False",
                        help='whether to project prior embeddings or not')
    parser.add_argument('--reload', type=str, default="False",
                        help='whether to reload model or not')
    parser.add_argument('--reload-model-dir', type=str, default="",
                        help='path for model to be reloaded')
    parser.add_argument('--deep-hidden-sizes', nargs='+', type=int,
                        help='sizes of the deep hidden layers for deepDocNADE')
    parser.add_argument('--deep', type=str, default="False",
                        help='whether to maked model deep (deepDocNADE) or not (docNADE/iDocNADE)')
    parser.add_argument('--multi-label', type=str, default="False",
                        help='whether dataset is multi-label or not')
    parser.add_argument('--trainfile', type=str, default="",
                        help='path to train text file')
    parser.add_argument('--valfile', type=str, default="",
                        help='path to validation text file')
    parser.add_argument('--testfile', type=str, default="",
                        help='path to test text file')
    parser.add_argument('--lambda-embeddings', type=float, default=0.0,
                        help='combination weight for prior embeddings into docnade')
    parser.add_argument('--pretrained-embeddings-path', type=str, default="",
                        help='path for pretrained embeddings')


    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data_lstm as data
import model.model_supervised_lstm as m
import model.evaluate as eval
import pickle
import keras.preprocessing.sequence as pp
import datetime
#import sys

from math import *
from nltk.corpus import wordnet

from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import time

# sys.setdefaultencoding() does not exist, here!
#reload(sys)  
#sys.setdefaultencoding('UTF8')

os.environ['CUDA_VISIBLE_DEVICES'] = ''

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

#dir(tf.contrib)

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


def train(model, dataset, mapping_dict, params, max_seq_length):
    log_dir = os.path.join(params.model, 'logs')
    model_dir_ir_docnade = os.path.join(params.model, 'model_ir_docnade')
    model_dir_ir_lstm = os.path.join(params.model, 'model_ir_lstm')
    model_dir_ir_comb = os.path.join(params.model, 'model_ir_comb')
    model_dir_ppl = os.path.join(params.model, 'model_ppl')
    model_dir_ppl_docnade = os.path.join(params.model, 'model_ppl_docnade')
    model_dir_supervised = os.path.join(params.model, 'model_supervised')

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(model_dir_ir_docnade):
        os.mkdir(model_dir_ir_docnade)
    if not os.path.isdir(model_dir_ir_lstm):
        os.mkdir(model_dir_ir_lstm)
    if not os.path.isdir(model_dir_ir_comb):
        os.mkdir(model_dir_ir_comb)
    if not os.path.isdir(model_dir_ppl):
        os.mkdir(model_dir_ppl)
    if not os.path.isdir(model_dir_ppl_docnade):
        os.mkdir(model_dir_ppl_docnade)
    if not os.path.isdir(model_dir_supervised):
        os.mkdir(model_dir_supervised)

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
        training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)
        #validation_data = dataset.batches('validation_docnade', 50, num_epochs=1, shuffle=True)
        #test_data = dataset.batches('test_docnade', 50, num_epochs=1, shuffle=True)

        training_data_rnn = dataset.batches('training_lstm', params.batch_size, shuffle=False, max_len=max_seq_length, multilabel=params.multi_label)
        #validation_data_rnn = dataset.batches('validation_lstm', 50, num_epochs=1, shuffle=False, max_len=max_seq_length)
        #test_data_rnn = dataset.batches('test_lstm', 50, num_epochs=1, shuffle=False, max_len=max_seq_length)

        best_val_IR = 0.0
        best_val_lstm_IR = 0.0
        best_val_combined_IR = 0.0
        best_val_nll = np.inf
        best_val_ppl = np.inf
        best_val_docnade_ppl = np.inf
        best_val_lstm_ppl = np.inf
        best_val_disc_accuracy = 0.0

        best_test_IR = 0.0
        best_test_combined_IR = 0.0
        best_test_nll = np.inf
        best_test_ppl = np.inf
        best_test_docnade_ppl = np.inf
        best_test_lstm_ppl = np.inf
        best_test_disc_accuracy = 0.0

        if params.initialize_docnade:
            patience = 30
        else:
            patience = params.patience
        
        patience_count = 0
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
        time_1 = time.time()
        for step in range(params.num_steps + 1):
            this_loss = -1.
            y, x, seq_lengths = next(training_data)
            y_rnn, x_rnn, rnn_seq_lengths = next(training_data_rnn)
            #transition_matrix = get_transition_matrix(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict)
            transition_matrix = get_transition_indices(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_length)
            
            if params.supervised:
                #_, loss_normed, loss_unnormed, loss_normed_docnade, loss_normed_lstm, disc_loss, disc_accuracy = session.run([model.opt, model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.loss_normed_lstm, model.disc_loss, model.disc_accuracy], feed_dict={
                _, loss_unnormed, disc_loss, disc_accuracy = session.run([model.opt, model.loss_unnormed, model.disc_loss, model.disc_accuracy], feed_dict={
                    model.x: x,
                    model.y: y,
                    model.seq_lengths: seq_lengths,
                    model.x_rnn: x_rnn,
                    model.rnn_seq_lengths: rnn_seq_lengths,
                    model.rnn_transition_matrix: transition_matrix,
                    model.docnade_loss_weight: params.docnade_loss_weight,
                    model.lstm_loss_weight: params.lstm_loss_weight
                    #model.lambda_hidden_lstm: params.lambda_hidden_lstm
                })
            else:
                _, loss_unnormed = session.run([model.opt, model.loss_unnormed], feed_dict={
                    model.x: x,
                    model.y: y,
                    model.seq_lengths: seq_lengths,
                    model.x_rnn: x_rnn,
                    model.rnn_seq_lengths: rnn_seq_lengths,
                    model.rnn_transition_matrix: transition_matrix,
                    model.docnade_loss_weight: params.docnade_loss_weight,
                    model.lstm_loss_weight: params.lstm_loss_weight
                    #model.lambda_hidden_lstm: params.lambda_hidden_lstm
                })

            
            if params.supervised:
                losses.append(loss_unnormed + disc_loss)
            else:
                losses.append(loss_unnormed)

            if (step % params.log_every == 0):
                print('{}: {:.6f}'.format(step, loss_unnormed))
                
            if params.supervised:
                print("accuracy: ", disc_accuracy)

            if step >= 1 and step % params.validation_ppl_freq == 0:
                time_2 = time.time()
                print("Time elapsed: %s" % (time_2 - time_1))
                
                this_val_nll = []
                this_val_loss_normed = []
                this_val_docnade_loss_normed = []
                this_val_lstm_loss_normed = []

                this_val_disc_accuracy = []

                validation_data_rnn = dataset.batches('validation_lstm', params.validation_bs, num_epochs=1, shuffle=False, max_len=max_seq_length, multilabel=params.multi_label)
                
                for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                    val_y_rnn, val_x_rnn, rnn_val_seq_lengths = next(validation_data_rnn)
                    #val_transition_matrix = get_transition_matrix(val_x, val_x_rnn, val_seq_lengths, rnn_val_seq_lengths, mapping_dict)
                    val_transition_matrix = get_transition_indices(val_x, val_x_rnn, val_seq_lengths, rnn_val_seq_lengths, mapping_dict, max_seq_length)
                
                    if params.supervised:
                        val_loss_normed, val_loss_unnormed, val_loss_normed_docnade, val_loss_normed_lstm, val_disc_loss, val_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.loss_normed_lstm, model.disc_loss, model.disc_accuracy], feed_dict={
                        #val_loss_normed, val_loss_unnormed, val_loss_normed_docnade, val_loss_normed_lstm, val_disc_loss, val_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.lstm.loss_normed, model.disc_loss, model.disc_accuracy], feed_dict={
                            model.x: val_x,
                            model.y: val_y,
                            model.seq_lengths: val_seq_lengths,
                            model.x_rnn: val_x_rnn,
                            model.rnn_seq_lengths: rnn_val_seq_lengths,
                            model.rnn_transition_matrix: val_transition_matrix,
                            model.docnade_loss_weight: params.docnade_loss_weight,
                            model.lstm_loss_weight: params.lstm_loss_weight
                            #model.lambda_hidden_lstm: params.lambda_hidden_lstm
                        })
                        this_val_disc_accuracy.append(val_disc_accuracy)
                    else:
                        val_loss_normed, val_loss_unnormed, val_loss_normed_docnade, val_loss_normed_lstm = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.loss_normed_lstm], feed_dict={
                        #val_loss_normed, val_loss_unnormed, val_loss_normed_docnade, val_loss_normed_lstm = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.lstm.loss_normed], feed_dict={
                            model.x: val_x,
                            model.y: val_y,
                            model.seq_lengths: val_seq_lengths,
                            model.x_rnn: val_x_rnn,
                            model.rnn_seq_lengths: rnn_val_seq_lengths,
                            model.rnn_transition_matrix: val_transition_matrix,
                            model.docnade_loss_weight: params.docnade_loss_weight,
                            model.lstm_loss_weight: params.lstm_loss_weight
                            #model.lambda_hidden_lstm: params.lambda_hidden_lstm
                        })

                    this_val_nll.append(val_loss_unnormed)
                    this_val_loss_normed.append(val_loss_normed)
                    this_val_docnade_loss_normed.append(val_loss_normed_docnade)
                    this_val_lstm_loss_normed.append(val_loss_normed_lstm)

                total_val_nll = np.mean(this_val_nll)
                total_val_ppl = np.exp(np.mean(this_val_loss_normed))
                total_val_docnade_ppl = np.exp(np.mean(this_val_docnade_loss_normed))
                total_val_lstm_ppl = np.exp(np.mean(this_val_lstm_loss_normed))

                if total_val_ppl < best_val_ppl:
                    best_val_ppl = total_val_ppl
                    print('saving: {}'.format(model_dir_ppl))
                    saver.save(session, model_dir_ppl + '/model_ppl', global_step=1)

                if total_val_docnade_ppl < best_val_docnade_ppl:
                    best_val_docnade_ppl = total_val_docnade_ppl
                    print('saving: {}'.format(model_dir_ppl_docnade))
                    saver.save(session, model_dir_ppl_docnade + '/model_ppl_docnade', global_step=1)
                
                # Early stopping
                if total_val_nll < best_val_nll:
                    best_val_nll = total_val_nll
                    patience_count = 0
                else:
                    patience_count += 1

                if total_val_docnade_ppl < best_val_docnade_ppl:
                    best_val_docnade_ppl = total_val_docnade_ppl

                if total_val_lstm_ppl < best_val_lstm_ppl:
                    best_val_lstm_ppl = total_val_lstm_ppl

                print('Combined val PPL: {:.3f},    DocNADE val PPL: {:.3f},    LSTM val PPL: {:.3f}    (Best Combined val PPL: {:.3f},    Best DocNADE val PPL: {:.3f},    Best LSTM val PPL: {:.3f},    Best Combined val loss: {:.3f})'.format(
                    total_val_ppl,
                    total_val_docnade_ppl,
                    total_val_lstm_ppl,
                    best_val_ppl or 0.0,
                    best_val_docnade_ppl,
                    best_val_lstm_ppl,
                    best_val_nll
                ))

                # logging information
                with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                    f.write("Step: %i,   Combined val PPL: %s,   DocNADE val PPL: %s,   LSTM val PPL: %s,    Best Combined val PPL: %s,    Best DocNADE val PPL: %s,    Best LSTM val PPL: %s,    Best val loss: %s\n" %
                            (step, total_val_ppl, total_val_docnade_ppl, total_val_lstm_ppl, best_val_ppl, best_val_docnade_ppl, best_val_lstm_ppl, best_val_nll))

                # Early stopping
                if patience_count > patience:
                    print("Early stopping criterion satisfied.")
                    break
            
            if step >= 1 and step % params.validation_ir_freq == 0:
                if params.supervised:
                    total_val_disc_accuracy = np.mean(this_val_disc_accuracy)

                    if total_val_disc_accuracy > best_val_disc_accuracy:
                        best_val_disc_accuracy = total_val_disc_accuracy
                        print('saving: {}'.format(model_dir_supervised))
                        saver.save(session, model_dir_supervised + '/model_supervised', global_step=1)

                    print('This val accuracy: {:.3f} (best val accuracy: {:.3f})'.format(
                        total_val_disc_accuracy,
                        best_val_disc_accuracy or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    This val accuracy: %s,    best val accuracy: %s\n" % 
                                (step, total_val_disc_accuracy, best_val_disc_accuracy))

                    summary, = session.run([summaries], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths,
                        model.x_rnn: x_rnn,
                        model.rnn_seq_lengths: rnn_seq_lengths,
                        model.rnn_transition_matrix: transition_matrix,
                        model.docnade_loss_weight: params.docnade_loss_weight,
                        model.lstm_loss_weight: params.lstm_loss_weight,
                        #model.lambda_hidden_lstm: params.lambda_hidden_lstm,
                        validation: 0.0,
                        validation_accuracy: total_val_disc_accuracy,
                        avg_loss: np.average(losses)
                    })
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                    losses = []
                else:
                    val_docnade_ir = -1.
                    val_comb_ir = -1.
                    val_lstm_ir = -1.

                    if params.use_docnade_for_ir:
                        # Only docnade IR calculation
                        validation_vectors_docnade = m.vectors_docnade(
                            model,
                            dataset.batches(
                                'validation_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'validation_lstm',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        training_vectors_docnade = m.vectors_docnade(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'training_lstm',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        val_docnade_ir = eval.evaluate(
                            training_vectors_docnade,
                            validation_vectors_docnade,
                            training_labels,
                            validation_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                    if params.use_combination_for_ir:
                        # DocNADE + LSTM combined IR calculation
                        validation_vectors_comb = m.vectors_comb(
                            model,
                            dataset.batches(
                                'validation_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'validation_lstm',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            params.combination_type,
                            session,
                            max_seq_length,
                            params.docnade_loss_weight,
                            params.lstm_loss_weight
                        )

                        training_vectors_comb = m.vectors_comb(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.batch_size,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'training_lstm',
                                params.batch_size,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            params.combination_type,
                            session,
                            max_seq_length,
                            params.docnade_loss_weight,
                            params.lstm_loss_weight
                        )

                        val_comb_ir = eval.evaluate(
                            training_vectors_comb,
                            validation_vectors_comb,
                            training_labels,
                            validation_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                    if params.use_lstm_for_ir:
                        # Only LSTM IR calculation
                        validation_vectors_lstm = m.vectors_lstm(
                            model,
                            dataset.batches(
                                'validation_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'validation_lstm',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        training_vectors_lstm = m.vectors_lstm(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'training_lstm',
                                params.validation_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        val_lstm_ir = eval.evaluate(
                            training_vectors_lstm,
                            validation_vectors_lstm,
                            training_labels,
                            validation_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                    if val_docnade_ir > best_val_IR:
                        best_val_IR = val_docnade_ir
                        print('saving: {}'.format(model_dir_ir_docnade))
                        saver.save(session, model_dir_ir_docnade + '/model_ir_docnade', global_step=1)

                    if val_lstm_ir > best_val_lstm_IR:
                        best_val_lstm_IR = val_lstm_ir
                        print('saving: {}'.format(model_dir_ir_lstm))
                        saver.save(session, model_dir_ir_lstm + '/model_ir_lstm', global_step=1)

                    if val_comb_ir > best_val_combined_IR:
                        best_val_combined_IR = val_comb_ir
                        print('saving: {}'.format(model_dir_ir_comb))
                        saver.save(session, model_dir_ir_comb + '/model_ir_comb', global_step=1)
                    
                    print('DocNADE IR: {:.3f},  Combined IR: {:.3f},  LSTM IR: {:.3f}  (best val IR: {:.3f}) (best val comb IR: {:.3f})'.format(
                        val_docnade_ir,
                        val_comb_ir,
                        val_lstm_ir,
                        best_val_IR or 0.0,
                        best_val_combined_IR or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    DocNADE IR: %s,    LSTM IR: %s,    Combined IR: %s,    best val IR: %s, best val Combined IR: %s\n" %
                                (step, val_docnade_ir, val_lstm_ir, val_comb_ir, best_val_IR, best_val_combined_IR))

                    summary, = session.run([summaries], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.seq_lengths: seq_lengths,
                        model.x_rnn: x_rnn,
                        model.rnn_seq_lengths: rnn_seq_lengths,
                        model.rnn_transition_matrix: transition_matrix,
                        model.docnade_loss_weight: params.docnade_loss_weight,
                        model.lstm_loss_weight: params.lstm_loss_weight,
                        #model.lambda_hidden_lstm: params.lambda_hidden_lstm,
                        validation: val_comb_ir,
                        validation_accuracy: 0.0,
                        avg_loss: np.average(losses)
                    })
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                    losses = []

            
            if step >= 1 and step % params.test_ppl_freq == 0:
                this_test_nll = []
                this_test_loss_normed = []
                this_test_docnade_loss_normed = []
                this_test_lstm_loss_normed = []
                this_test_disc_accuracy = []

                test_data_rnn = dataset.batches('test_lstm', params.test_bs, num_epochs=1, shuffle=False, max_len=max_seq_length, multilabel=params.multi_label)
                
                for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params.test_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
                    test_y_rnn, test_x_rnn, rnn_test_seq_lengths = next(test_data_rnn)
                    #val_transition_matrix = get_transition_matrix(val_x, val_x_rnn, val_seq_lengths, rnn_val_seq_lengths, mapping_dict)
                    test_transition_matrix = get_transition_indices(test_x, test_x_rnn, test_seq_lengths, rnn_test_seq_lengths, mapping_dict, max_seq_length)
                
                    if params.supervised:
                        test_loss_normed, test_loss_unnormed, test_loss_normed_docnade, test_loss_normed_lstm, test_disc_loss, test_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.loss_normed_lstm, model.disc_loss, model.disc_accuracy], feed_dict={
                        #test_loss_normed, test_loss_unnormed, test_loss_normed_docnade, test_loss_normed_lstm, test_disc_loss, test_disc_accuracy = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.lstm.loss_normed, model.disc_loss, model.disc_accuracy], feed_dict={
                            model.x: test_x,
                            model.y: test_y,
                            model.seq_lengths: test_seq_lengths,
                            model.x_rnn: test_x_rnn,
                            model.rnn_seq_lengths: rnn_test_seq_lengths,
                            model.rnn_transition_matrix: test_transition_matrix,
                            model.docnade_loss_weight: params.docnade_loss_weight,
                            model.lstm_loss_weight: params.lstm_loss_weight
                            #model.lambda_hidden_lstm: params.lambda_hidden_lstm
                        })
                        this_test_disc_accuracy.append(test_disc_accuracy)
                    else:
                        test_loss_normed, test_loss_unnormed, test_loss_normed_docnade, test_loss_normed_lstm = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.loss_normed_lstm], feed_dict={
                        #test_loss_normed, test_loss_unnormed, test_loss_normed_docnade, test_loss_normed_lstm = session.run([model.loss_normed, model.loss_unnormed, model.loss_normed_docnade, model.lstm.loss_normed], feed_dict={
                            model.x: test_x,
                            model.y: test_y,
                            model.seq_lengths: test_seq_lengths,
                            model.x_rnn: test_x_rnn,
                            model.rnn_seq_lengths: rnn_test_seq_lengths,
                            model.rnn_transition_matrix: test_transition_matrix,
                            model.docnade_loss_weight: params.docnade_loss_weight,
                            model.lstm_loss_weight: params.lstm_loss_weight
                            #model.lambda_hidden_lstm: params.lambda_hidden_lstm
                        })

                    this_test_nll.append(test_loss_unnormed)
                    this_test_loss_normed.append(test_loss_normed)
                    this_test_docnade_loss_normed.append(test_loss_normed_docnade)
                    this_test_lstm_loss_normed.append(test_loss_normed_lstm)

                total_test_nll = np.mean(this_test_nll)
                total_test_ppl = np.exp(np.mean(this_test_loss_normed))
                total_test_docnade_ppl = np.exp(np.mean(this_test_docnade_loss_normed))
                total_test_lstm_ppl = np.exp(np.mean(this_test_lstm_loss_normed))

                if total_test_ppl < best_test_ppl:
                    best_test_ppl = total_test_ppl

                if total_test_nll < best_test_nll:
                    best_test_nll = total_test_nll

                if total_test_docnade_ppl < best_test_docnade_ppl:
                    best_test_docnade_ppl = total_test_docnade_ppl
                
                if total_test_lstm_ppl < best_test_lstm_ppl:
                    best_test_lstm_ppl = total_test_lstm_ppl

                print('Combined test PPL: {:.3f},    DocNADE test PPL: {:.3f},    LSTM test PPL: {:.3f}    (Best Combined test PPL: {:.3f},    Best DocNADE test PPL: {:.3f},    Best LSTM test PPL: {:.3f},    Best Combined test loss: {:.3f})'.format(
                    total_test_ppl,
                    total_test_docnade_ppl,
                    total_test_lstm_ppl,
                    best_test_ppl or 0.0,
                    best_test_docnade_ppl,
                    best_test_lstm_ppl,
                    best_test_nll
                ))

                # logging information
                with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                    f.write("Step: %i,   Combined test PPL: %s,   DocNADE test PPL: %s,   LSTM test PPL: %s,    Best Combined test PPL: %s,     Best DocNADE test PPL: %s,    Best LSTM test PPL: %s,    Best test loss: %s\n" %
                            (step, total_test_ppl, total_test_docnade_ppl, total_test_lstm_ppl, best_test_ppl, best_test_docnade_ppl, best_test_lstm_ppl, best_test_nll))

            if step >= 1 and step % params.test_ir_freq == 0:
                if params.supervised:
                    total_test_disc_accuracy = np.mean(this_test_disc_accuracy)

                    if total_test_disc_accuracy > best_test_disc_accuracy:
                        best_test_disc_accuracy = total_test_disc_accuracy

                    print('This test accuracy: {:.3f} (best test accuracy: {:.3f})'.format(
                        total_test_disc_accuracy,
                        best_test_disc_accuracy or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    This test accuracy: %s,    best test accuracy: %s\n" % 
                                (step, total_test_disc_accuracy, best_test_disc_accuracy))
                else:
                    test_docnade_ir = -1.
                    test_comb_ir = -1.
                    test_lstm_ir = -1.
                    
                    if params.use_docnade_for_ir:
                        # Only docnade IR calculation
                        test_vectors_docnade = m.vectors_docnade(
                            model,
                            dataset.batches(
                                'test_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'test_lstm',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        training_vectors_docnade = m.vectors_docnade(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'training_lstm',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        test_docnade_ir = eval.evaluate(
                            training_vectors_docnade,
                            test_vectors_docnade,
                            training_labels,
                            test_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                    if params.use_combination_for_ir:
                        # DocNADE + LSTM combined IR calculation
                        test_vectors_comb = m.vectors_comb(
                            model,
                            dataset.batches(
                                'test_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'test_lstm',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            params.combination_type,
                            session,
                            max_seq_length,
                            params.docnade_loss_weight,
                            params.lstm_loss_weight
                        )

                        training_vectors_comb = m.vectors_comb(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'training_lstm',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            params.combination_type,
                            session,
                            max_seq_length,
                            params.docnade_loss_weight,
                            params.lstm_loss_weight
                        )

                        test_comb_ir = eval.evaluate(
                            training_vectors_comb,
                            test_vectors_comb,
                            training_labels,
                            test_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                    if params.use_lstm_for_ir:
                        # Only LSTM IR calculation
                        test_vectors_lstm = m.vectors_lstm(
                            model,
                            dataset.batches(
                                'test_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'test_lstm',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        training_vectors_lstm = m.vectors_lstm(
                            model,
                            dataset.batches(
                                'training_docnade',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=True,
                                multilabel=params.multi_label
                            ),
                            dataset.batches(
                                'training_lstm',
                                params.test_bs,
                                num_epochs=1,
                                shuffle=False,
                                max_len=max_seq_length,
                                multilabel=params.multi_label
                            ),
                            mapping_dict,
                            session,
                            max_seq_length
                        )

                        test_lstm_ir = eval.evaluate(
                            training_vectors_lstm,
                            test_vectors_lstm,
                            training_labels,
                            test_labels,
                            recall=[0.02],
                            num_classes=params.num_classes,
                            multi_label=params.multi_label
                        )[0]

                    if test_docnade_ir > best_test_IR:
                        best_test_IR = test_docnade_ir

                    if test_comb_ir > best_test_combined_IR:
                        best_test_combined_IR = test_comb_ir
                    
                    print('DocNADE Test IR: {:.3f},  Combined Test IR: {:.3f},  LSTM Test IR: {:.3f}  (best Test IR: {:.3f}) (best Test comb IR: {:.3f})'.format(
                        test_docnade_ir,
                        test_comb_ir,
                        test_lstm_ir,
                        best_test_IR or 0.0,
                        best_test_combined_IR or 0.0
                    ))

                    # logging information
                    with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
                        f.write("Step: %i,    DocNADE Test IR: %s,    LSTM Test IR: %s,    Combined Test IR: %s,    best Test IR: %s, best Test Combined IR: %s" %
                                (step, test_docnade_ir, test_lstm_ir, test_comb_ir, best_test_IR, best_test_combined_IR))



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


def get_transition_matrix(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths, mapping_dict):
    assert (docnade_indices.shape[0] == lstm_indices.shape[0])

    batch_size, d_seq_len = docnade_indices.shape
    l_seq_len = lstm_indices.shape[1]

    #transition_matrix = np.zeros((batch_size, d_seq_len, l_seq_len), dtype=np.float32)
    transition_matrices = []
    for d_indices, l_indices, d_seq, l_seq in zip(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths):
        mapped_d_indices = np.zeros((d_seq, 1))
        mapped_d_indices[:,0] = np.array([int(mapping_dict[int(d_index)]) for d_index in d_indices[:d_seq]])
        temp_d = np.tile(mapped_d_indices, (1, l_seq))

        mapped_l_indices = np.zeros((1, l_seq))
        mapped_l_indices[0, :] = l_indices[:l_seq]
        temp_l = np.tile(mapped_l_indices, (d_seq, 1))

        transition_matrix = np.zeros((d_seq_len, l_seq_len), dtype=np.float32)
        transition_matrix[:d_seq, :l_seq] = np.array((temp_d == temp_l), dtype=np.float32)
        transition_matrices.append(transition_matrix)
    transition_matrices_stacked = np.stack(transition_matrices)

    return transition_matrices_stacked


def get_transition_indices(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_len):
    assert (docnade_indices.shape[0] == lstm_indices.shape[0])

    batch_size, d_seq_len = docnade_indices.shape
    l_seq_len = lstm_indices.shape[1]

    transition_indices = []
    max_length = 0
    for d_indices, l_indices, d_seq, l_seq in zip(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths):
        #mapped_l_indices = np.zeros((1, l_seq))
        mapped_l_indices = np.array(l_indices[:l_seq])

        d_transition_indices = []
        for d_index in d_indices:
            indices = np.where(mapped_l_indices == mapping_dict[d_index])[0].tolist()
            if indices == []:
                d_transition_indices.append([0])
            else:
                d_transition_indices.append(indices)
        try:
            d_transition_indices = pp.pad_sequences(d_transition_indices, dtype='int32', padding='post', value=max_seq_len)
        except:
            import pdb; pdb.set_trace()
        transition_indices.append(d_transition_indices)
        if d_transition_indices.shape[1] > max_length:
            max_length = d_transition_indices.shape[1]

    transition_index_matrix = np.ones((batch_size, d_seq_len, max_length), dtype=np.int32) * max_seq_len
    for i, mat in enumerate(transition_indices):
        transition_index_matrix[i, :, :mat.shape[1]] = mat

    #transition_indices = pp.pad_sequences(transition_indices, dtype='int32', padding='post', value=max_seq_len)
    #transition_index_matrix = np.array(transition_index_matrix)
    return transition_index_matrix

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def reload_evaluation(dataset, params, max_seq_len, mapping_dict):
    # to do
	return 
	
	
def main(args):
    max_seq_len = 0
    # max_seq_len = 27649 # 20NS data
    # max_seq_len = 917 # R8 data
    # max_seq_len = 938 # DBpediatopics data
    # max_seq_len = 182 # AGnews data
    # max_seq_len = 50 # TMN data
    # max_seq_len = 72 # Subjectivity data

    args.supervised = str2bool(args.supervised)
    args.use_docnade_for_ir = str2bool(args.use_docnade_for_ir)
    args.use_lstm_for_ir = str2bool(args.use_lstm_for_ir)
    args.use_combination_for_ir = str2bool(args.use_combination_for_ir)
    args.initialize_docnade = str2bool(args.initialize_docnade)
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

    dataset = data.Dataset(args.dataset)

    # Finding max sequence length in LSTM dataset
    training_batches = dataset.batches('training_lstm', 1000, num_epochs=1, shuffle=False, multilabel=args.multi_label)
    for y, x, seq_lengths in training_batches:
        print("max length: ", x.shape[1])
        if x.shape[1] > max_seq_len:
            max_seq_len = x.shape[1]
    
    training_batches = dataset.batches('validation_lstm', 50, num_epochs=1, shuffle=False, multilabel=args.multi_label)
    for y, x, seq_lengths in training_batches:
        print("max length: ", x.shape[1])
        if x.shape[1] > max_seq_len:
            max_seq_len = x.shape[1]
    
    training_batches = dataset.batches('test_lstm', 1000, num_epochs=1, shuffle=False, multilabel=args.multi_label)
    for y, x, seq_lengths in training_batches:
        print("max length: ", x.shape[1])
        if x.shape[1] > max_seq_len:
            max_seq_len = x.shape[1]

    print("LSTM max sequence length: %s\n\n" % max_seq_len)

    if args.reload:
        # to do 
        return 
    else:
        now = datetime.datetime.now()

        if args.supervised:
            args.model += "_supervised"

        if args.initialize_rnn:
            args.model += "_embprior"

        args.model +=  "_" + str(args.activation) + "_hid_" + str(args.hidden_size) + "_voc_" + str(args.vocab_size) \
                        + "_lr_" + str(args.learning_rate) + "_comb_type_" + str(args.combination_type) + "_l_update_" \
                        + str(args.update_rnn_w) + "_deep_" + str(args.deep) + "_d_weight_" + str(args.docnade_loss_weight) \
                        + "_l_weight_" + str(args.lstm_loss_weight) + "_lambda_" + str(args.lambda_hidden_lstm) \
                        + "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)

        if not os.path.isdir(args.model):
            os.mkdir(args.model)

        with open(os.path.join(args.model, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))


        if args.initialize_rnn:
            glove_embeddings = loadGloveModel(params=args)

        rnn_vocab = args.rnnVocab
        docnade_vocab = args.docnadeVocab

        with open(rnn_vocab, 'r') as f:
            vocab_lstm = [w.strip() for w in f.readlines()]
        
        rnn_embedding_matrix = None
        
        if args.initialize_rnn:
            missing_words = 0
            rnn_embedding_matrix = np.zeros((len(vocab_lstm), args.hidden_size), dtype=np.float32)
            for i, word in enumerate(vocab_lstm):
                if str(word).lower() in glove_embeddings.keys():
                    if len(glove_embeddings[str(word).lower()]) == 0:
                        rnn_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
                        missing_words += 1
                    else:
                        rnn_embedding_matrix[i, :] = np.array(glove_embeddings[str(word).lower()], dtype=np.float32)
                else:
                    rnn_embedding_matrix[i, :] = np.zeros((args.hidden_size), dtype=np.float32)
            
            rnn_embedding_matrix = tf.convert_to_tensor(rnn_embedding_matrix)
            print("Total missing words:%d out of %d" %(missing_words, len(vocab_lstm)))
            print("RNN initialize.")

        with open(docnade_vocab, 'r') as f:
            vocab_docnade = [w.strip() for w in f.readlines()]
        
        
        docnade_embedding_matrix = None
        
        if args.initialize_docnade:
            f = open(args.docnade_embeddings_path, "rb")
            docnade_embedding_matrix = pickle.load(f)
            print("docnade embedding loaded.")

        mapping_dict = pickle.load(open(args.mapping_dict, "rb"))
        rnn_vocab_length = len(vocab_lstm)

        x = tf.placeholder(tf.int32, shape=(None, None), name='x')
        if args.multi_label:
            y = tf.placeholder(tf.string, shape=(None), name='y')
        else:
            y = tf.placeholder(tf.int32, shape=(None), name='y')
        seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')
        x_rnn = tf.placeholder(tf.int32, shape=(None, None), name='x_rnn')
        rnn_seq_lengths = tf.placeholder(tf.int32, shape=(None), name='rnn_seq_lengths')
        transition_matrix = tf.placeholder(tf.int32, shape=(None, None, None), name='transition_matrix')

        docnade_loss_weight = tf.placeholder(tf.float32, name='docnade_loss_weight')
        lstm_loss_weight = tf.placeholder(tf.float32, name='lstm_loss_weight')

        model = m.DocNADE(x, y, seq_lengths, args, x_rnn, rnn_seq_lengths, transition_matrix,
                        rnn_vocab_length, max_seq_len, docnade_loss_weight, lstm_loss_weight, args.lambda_hidden_lstm,
                        W_initializer_docnade=docnade_embedding_matrix, W_initializer_rnn=rnn_embedding_matrix)
        train(model, dataset, mapping_dict, args, max_seq_len)


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
    parser.add_argument('--num-classes', type=int, default=-1,
                        help='number of classes')
    parser.add_argument('--supervised', type=str, default="False",
                        help='whether to use supervised model or not')
    parser.add_argument('--mapping-dict', type=str, required=True,
                        help='path to the mapping dictionary from docnade to rnn-lstm dataset')
    parser.add_argument('--use-docnade-for-ir', type=str, default="True",
                        help='whether to use only docnade hidden vectors for ir')
    parser.add_argument('--use-lstm-for-ir', type=str, default="True",
                        help='whether to use only lstm hidden vectors for ir')
    parser.add_argument('--use-combination-for-ir', type=str, default="True",
                        help='whether to use docnade + lstm hidden vectors for ir')
    parser.add_argument('--combination-type', type=str, default="concat",
                        help='how to combine docnade and lstm hidden vectors for ir')
    parser.add_argument('--initialize-docnade', type=str, default="False",
                        help='whether to embedding matrix of docnade')
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
    parser.add_argument('--validation-ppl-freq', type=int, default=100,
                        help='print and log validation PPL after this many steps')
    parser.add_argument('--validation-ir-freq', type=int, default=100,
                        help='print and log validation IR after this many steps')
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
    parser.add_argument('--docnade-embeddings-path', type=str, default="",
                        help='pretrained docnade embeddings path')


    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

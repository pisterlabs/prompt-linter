"""
Modifications copyright (C) 2020 Michael Strobl
"""

import tensorflow as tf
import numpy as np
import sys

from tensorflow.core.protobuf import saver_pb2

from entity_linker.evaluation import evaluate_inference, evaluate_types
from entity_linker.models.base import Model
from entity_linker.models.figer_model.context_encoder import ContextEncoderModel
from entity_linker.models.figer_model.coherence_model import CoherenceModel
from entity_linker.models.figer_model.labeling_model import LabelingModel
from entity_linker.models.figer_model.entity_posterior import EntityPosterior


np.set_printoptions(precision=5)


class ELModel(Model):
    """Unsupervised Clustering using Discrete-State VAE"""

    def __init__(self, sess, reader, max_steps, pretrain_max_steps,
                 word_embed_dim, context_encoded_dim,
                 context_encoder_lstmsize, context_encoder_num_layers,
                 coherence_numlayers, jointff_numlayers,
                 learning_rate, dropout_keep_prob, reg_constant,
                 checkpoint_dir, optimizer, strict=False,
                 pretrain_word_embed=True, typing=True, el=True,
                 coherence=False, textcontext=False,
                 WDLength=100, Fsize=5, entyping=True):
        self.optimizer = optimizer
        self.sess = sess
        self.reader = reader  # Reader class
        self.strict = strict
        self.pretrain_word_embed = pretrain_word_embed
        assert self.pretrain_word_embed, "Only use pretrained word embeddings"
        self.typing = typing    # Bool - Perform typing
        self.el = el    # Bool - Perform Entity-Linking
        self.coherence = coherence
        self.textcontext = textcontext
        if not (self.coherence or self.textcontext):
            print("Both textcontext and coherence cannot be False")
            sys.exit(0)
        self.WDLength = WDLength
        self.Fsize = Fsize
        self.entyping = entyping


        self.max_steps = max_steps  # Max num of steps of training to run
        self.pretrain_max_steps = pretrain_max_steps
        self.batch_size = reader.batch_size
        self.reg_constant = reg_constant
        self.dropout_keep_prob = dropout_keep_prob
        self.lr = learning_rate

        # Num of clusters = Number of entities in dataset.
        self.num_labels = self.reader.num_labels
        self.num_words = self.reader.num_words
        self.num_knwn_entities = self.reader.num_knwn_entities
        self.num_cand_entities = self.reader.num_cands

        # Size of word embeddings
        if not self.pretrain_word_embed:
            self.word_embed_dim = word_embed_dim
        else:
            self.word_embed_dim = 300

        # Context encoders
        self.context_encoded_dim = context_encoded_dim
        self.context_encoder_lstmsize = context_encoder_lstmsize
        self.context_encoder_num_layers = context_encoder_num_layers
        # Coherence Encoder
        self.coherence_numlayers = coherence_numlayers
        # Joint FeedForward
        self.jointff_numlayers = jointff_numlayers

        self.checkpoint_dir = checkpoint_dir

        self.embeddings_scope = "embeddings"
        self.word_embed_var_name = "word_embeddings"
        self.encoder_model_scope = "context_encoder"
        self.coherence_model_scope = "coherence_encoder"
        self.wikidesc_model_scope = "wikidesc_encoder"
        self.joint_context_scope = "joint_context"
        self.label_model_scope = "labeling_model"
        self.labeling_loss_scope = "labeling_loss"
        self.entity_posterior_scope = "en_posterior_model"
        self.posterior_loss_scope = "en_posterior_loss"
        self.wikidesc_loss_scope = "wikidesc_loss"
        self.optim_scope = "labeling_optimization"

        self._attrs=[
          "textcontext", "coherence", "typing",
          "pretrain_word_embed", "word_embed_dim", "num_words", "num_labels",
          "num_knwn_entities", "context_encoded_dim", "context_encoder_lstmsize",
          "context_encoder_num_layers", "coherence_numlayers",
          "reg_constant", "strict", "lr", "optimizer"]


        #GPU Allocations
        self.device_placements = {
          'cpu': '/cpu:0',
          'gpu': '/cpu:0'
        }

        with tf.compat.v1.variable_scope("figer_model") as scope:
            self.learning_rate = tf.Variable(self.lr, name='learning_rate',
                                             trainable=False)
            self.global_step = tf.Variable(0, name='global_step', trainable=False,
                                           dtype=tf.int32)
            self.increment_global_step_op = self.global_step.assign_add(1)

            self.build_placeholders()

            # Encoder Models : Name LSTM, Text FF and Links FF networks
            with tf.compat.v1.variable_scope(self.encoder_model_scope) as scope:
                if self.pretrain_word_embed == False:
                    self.left_context_embeddings = tf.nn.embedding_lookup(
                      self.word_embeddings, self.left_batch, name="left_embeddings")
                    self.right_context_embeddings = tf.nn.embedding_lookup(
                      self.word_embeddings, self.right_batch, name="right_embeddings")


                if self.textcontext:
                    self.context_encoder_model = ContextEncoderModel(
                      num_layers=self.context_encoder_num_layers,
                      batch_size=self.batch_size,
                      lstm_size=self.context_encoder_lstmsize,
                      left_embed_batch=self.left_context_embeddings,
                      left_lengths=self.left_lengths,
                      right_embed_batch=self.right_context_embeddings,
                      right_lengths=self.right_lengths,
                      context_encoded_dim=self.context_encoded_dim,
                      scope_name=self.encoder_model_scope,
                      device=self.device_placements['gpu'],
                      dropout_keep_prob=self.dropout_keep_prob)

                if self.coherence:
                    self.coherence_model = CoherenceModel(
                      num_layers=self.coherence_numlayers,
                      batch_size=self.batch_size,
                      input_size=self.reader.num_cohstr,
                      coherence_indices=self.coherence_indices,
                      coherence_values=self.coherence_values,
                      coherence_matshape=self.coherence_matshape,
                      context_encoded_dim=self.context_encoded_dim,
                      scope_name=self.coherence_model_scope,
                      device=self.device_placements['gpu'],
                      dropout_keep_prob=self.dropout_keep_prob)

                if self.coherence and self.textcontext:
                    # [B, 2*context_encoded_dim]
                    joint_context_encoded = tf.concat(
                      [self.context_encoder_model.context_encoded,
                          self.coherence_model.coherence_encoded], 1,
                      name='joint_context_encoded')

                    context_vec_size = 2*self.context_encoded_dim

                    ### WITH FF AFTER CONCAT  ##########
                    trans_weights = tf.compat.v1.get_variable(
                      name="joint_trans_weights",
                      shape=[context_vec_size, context_vec_size],
                      initializer=tf.random_normal_initializer(
                        mean=0.0,
                        stddev=1.0/(100.0)))

                    # [B, context_encoded_dim]
                    joint_context_encoded = tf.matmul(joint_context_encoded, trans_weights)
                    self.joint_context_encoded = tf.nn.relu(joint_context_encoded)
                    ####################################

                elif self.textcontext:
                    self.joint_context_encoded = self.context_encoder_model.context_encoded
                    context_vec_size = self.context_encoded_dim
                elif self.coherence:
                    self.joint_context_encoded = self.coherence_model.coherence_encoded
                    context_vec_size = self.context_encoded_dim
                else:
                    print("ERROR:Atleast one of local or "
                          "document context needed.")
                    sys.exit(0)

                self.posterior_model = EntityPosterior(
                  batch_size=self.batch_size,
                  num_knwn_entities=self.num_knwn_entities,
                  context_encoded_dim=context_vec_size,
                  context_encoded=self.joint_context_encoded,
                  entity_ids=self.sampled_entity_ids,
                  scope_name=self.entity_posterior_scope,
                  device_embeds=self.device_placements['gpu'],
                  device_gpu=self.device_placements['gpu'])

                self.labeling_model = LabelingModel(
                  batch_size=self.batch_size,
                  num_labels=self.num_labels,
                  context_encoded_dim=context_vec_size,
                  true_entity_embeddings=self.posterior_model.trueentity_embeddings,
                  word_embed_dim=self.word_embed_dim,
                  context_encoded=self.joint_context_encoded,
                  mention_embed=None,
                  scope_name=self.label_model_scope,
                  device=self.device_placements['gpu'])

    ################ end Initialize  #############################################


    def build_placeholders(self):
        # Left Context
        self.left_batch = tf.compat.v1.placeholder(
          tf.int32, [self.batch_size, None], name="left_batch")
        self.left_context_embeddings = tf.compat.v1.placeholder(
          tf.float32, [self.batch_size, None, self.word_embed_dim], name="left_embeddings")
        self.left_lengths = tf.compat.v1.placeholder(
          tf.int32, [self.batch_size], name="left_lengths")

        # Right Context
        self.right_batch = tf.compat.v1.placeholder(
          tf.int32, [self.batch_size, None], name="right_batch")
        self.right_context_embeddings = tf.compat.v1.placeholder(
          tf.float32, [self.batch_size, None, self.word_embed_dim], name="right_embeddings")
        self.right_lengths = tf.compat.v1.placeholder(
          tf.int32, [self.batch_size], name="right_lengths")

        # Mention Embedding
        self.mention_embed = tf.compat.v1.placeholder(
          tf.float32, [self.batch_size, self.word_embed_dim], name="mentions_embed")

        # Wiki Description Batch
        self.wikidesc_batch = tf.compat.v1.placeholder(
          tf.float32, [self.batch_size, self.WDLength, self.word_embed_dim],
          name="wikidesc_batch")

        # Labels
        self.labels_batch = tf.compat.v1.placeholder(
          tf.float32, [self.batch_size, self.num_labels], name="true_labels")

        # Candidates, Priors and True Entities Ids
        self.sampled_entity_ids = tf.compat.v1.placeholder(
          tf.int32, [self.batch_size, self.num_cand_entities], name="sampled_candidate_entities")

        self.entity_priors = tf.compat.v1.placeholder(
          tf.float32, [self.batch_size, self.num_cand_entities], name="entitiy_priors")

        self.true_entity_ids = tf.compat.v1.placeholder(
          tf.int32, [self.batch_size], name="true_entities_in_sampled")

        # Coherence
        self.coherence_indices = tf.compat.v1.placeholder(
          tf.int64, [None, 2], name="coherence_indices")

        self.coherence_values = tf.compat.v1.placeholder(
          tf.float32, [None], name="coherence_values")

        self.coherence_matshape = tf.compat.v1.placeholder(
          tf.int64, [2], name="coherence_matshape")

        #END-Placeholders

        if self.pretrain_word_embed == False:
            with tf.compat.v1.variable_scope(self.embeddings_scope) as s:
                with tf.device(self.device_placements['cpu']) as d:
                    self.word_embeddings = tf.compat.v1.get_variable(
                      name=self.word_embed_var_name,
                      shape=[self.num_words, self.word_embed_dim],
                      initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=(1.0/100.0)))


    # #####################      TEST     ##################################
    def load_ckpt_model(self, ckptpath=None):
        saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.all_variables())
        # (Try) Load all pretraining model variables
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptpath)

        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        tf.compat.v1.get_default_graph().finalize()

    def loadModel(self, ckptpath=None):
        #saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.all_variables(),write_version = saver_pb2.SaverDef.V1)

        saver = tf.compat.v1.train.Saver()

        # (Try) Load all pretraining model variables
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptpath)





        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        tf.compat.v1.get_default_graph().finalize()





    def doInference(self):
        r = self.inference_run()
        return r

    def inference(self, ckptpath=None):
        saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.all_variables())
        # (Try) Load all pretraining model variables
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptpath)

        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        tf.compat.v1.get_default_graph().finalize()

        r = self.inference_run()
        return r

    def inference_run(self):
        assert self.reader.typeOfReader == "inference"
        # assert self.reader.batch_size == 1
        self.reader.reset_test()
        numInstances = 0

        # For types: List contains numpy matrices with row_size = BatchSize
        predLabelScoresnumpymat_list = []
        # For EL : Lists contain one list per mention
        widIdxs_list = []       # Candidate WID IDXs (First is true)
        condProbs_list = []     # Crosswikis conditional priors
        contextProbs_list = []  # Predicted Entity prob using context

        while self.reader.epochs < 1 and self.reader.men_idx < len(self.reader.mentions):
            result = self.reader.next_test_batch()



            if result != None:
                (left_batch, left_lengths,
                         right_batch, right_lengths,
                         coherence_batch,
                         wid_idxs_batch, wid_cprobs_batch) = result


                # Candidates for entity linking
                # feed_dict = {self.sampled_entity_ids: wid_idxs_batch,
                #              self.entity_priors: wid_cprobs_batch}
                feed_dict = {self.sampled_entity_ids: wid_idxs_batch}
                # Required Context
                if self.textcontext:
                    if not self.pretrain_word_embed:
                        context_dict = {
                          self.left_batch: left_batch,
                          self.right_batch: right_batch,
                          self.left_lengths: left_lengths,
                          self.right_lengths: right_lengths}
                    else:
                        context_dict = {
                          self.left_context_embeddings: left_batch,
                          self.right_context_embeddings: right_batch,
                          self.left_lengths: left_lengths,
                          self.right_lengths: right_lengths}
                    feed_dict.update(context_dict)
                if self.coherence:
                    coherence_dict = {self.coherence_indices: coherence_batch[0],
                                      self.coherence_values: coherence_batch[1],
                                      self.coherence_matshape: coherence_batch[2]}
                    feed_dict.update(coherence_dict)

                fetch_tensors = [self.labeling_model.label_probs,
                                 self.posterior_model.entity_posteriors]

                fetches = self.sess.run(fetch_tensors, feed_dict=feed_dict)

                [label_sigms, context_probs] = fetches

                predLabelScoresnumpymat_list.append(label_sigms)
                condProbs_list.extend(wid_cprobs_batch)
                widIdxs_list.extend(wid_idxs_batch)
                contextProbs_list.extend(context_probs.tolist())
                numInstances += self.reader.batch_size


                #print('processed a batch')

        # print("Num of instances {}".format(numInstances))
        # print("Starting Type and EL Evaluations ... ")
        # pred_TypeSetsList: [B, Types], For each mention, list of pred types
        pred_TypeSetsList = evaluate_types.evaluate(
            predLabelScoresnumpymat_list,
            self.reader.idx2label)

        # evWTs:For each mention: Contains a list of [WTs, WIDs, Probs]
        # Each element above has (MaxPrior, MaxContext, MaxJoint)
        # sortedContextWTs: Titles sorted in decreasing context prob
        (jointProbs_list,
         evWTs,
         sortedContextWTs) = evaluate_inference.evaluateEL(
            condProbs_list, widIdxs_list, contextProbs_list,
            self.reader.idx2knwid, self.reader.wid2WikiTitle,
            verbose=False)

        return (predLabelScoresnumpymat_list,
                widIdxs_list, condProbs_list, contextProbs_list,
                jointProbs_list, evWTs, pred_TypeSetsList)

    def softmax(self, scores):
        expc = np.exp(scores)
        sumc = np.sum(expc)
        softmax_out = expc/sumc
        return softmax_out

#!/usr/bin/env python

# Standard
import os
import sys
import random
import datetime

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

# PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

# Utils
from topic_labeling import TopicLabeling
from vae_avitm_paper import VaeAvitmModel
from sentiment_classifier import SentimentClassifier

from utils import write_file
from utils import AnnotatedList, normalize_list
from utils import orthogonal_reg_loss, adv_cross_entropy
from utils import weights_init_xavier, weights_init_kaiming, weights_init_normal, weights_init_sparse

# Topic Class
from topic_class import Topic

# Gensim
from gensim.models.coherencemodel import CoherenceModel

# JSON
import json

# T-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 


# DIATOM Class
class AdversarialVaeModel(nn.Module):
    def __init__(self, exp, input_dimensionality, num_of_classes, vae_hidd_size, num_of_OpinionTopics, num_of_PlotTopics, 
                 encoder_layers=1, generator_layers=4, beta_s=1.0, beta_a=1.0, encoder_dropout=False,  dropout_prob=0.0, 
                 generator_shortcut=False, generator_transform=None, interaction="dot_prod", plug_Plots=False, device="cpu"):

        super(AdversarialVaeModel, self).__init__()

        # Args includes all the meta information about the experiment
        self.exp  = exp
        self.args = exp.args

        self.beta_a = beta_a
        self.beta_s = beta_s
        
        self.input_dimensionality = input_dimensionality
        self.num_of_OpinionTopics = num_of_OpinionTopics
        self.num_of_PlotTopics    = num_of_PlotTopics

        # Prior mean and variance
        self.priors = dict()
        self.priors["prior_mean_Plot"]        = torch.Tensor(1, self.num_of_PlotTopics).fill_(0).to(device)
        self.priors["prior_variance_Plot"]    = 0.995
        self.priors["prior_var_Plot"]         = torch.Tensor(1, self.num_of_PlotTopics).fill_(self.priors["prior_variance_Plot"]).to(device)
        self.priors["prior_logvar_Plot"]      = self.priors["prior_var_Plot"].log()

        self.priors["prior_mean_Opinion"]     = torch.Tensor(1, self.num_of_OpinionTopics).fill_(0).to(device)
        self.priors["prior_variance_Opinion"] = 0.995
        self.priors["prior_var_Opinion"]      = torch.Tensor(1, self.num_of_OpinionTopics).fill_(self.priors["prior_variance_Opinion"]).to(device)
        self.priors["prior_logvar_Opinion"]   = self.priors["prior_var_Opinion"].to(device).log()
                      
        # Flags
        self.interaction    = interaction
        self.plug_Plots     = plug_Plots
        self.topicType      = "both"
        self.wordEmb        = None
        self.alsoAspectLoss = True
        self.alsoSentLoss   = True

        # Training Device
        self.device         = device

        # - Inint VAE components -
        self.aspect_vae_model = VaeAvitmModel(input_dimensionality, d_e=vae_hidd_size, d_t=num_of_PlotTopics, 
                                                encoder_layers=encoder_layers, generator_layers=generator_layers, without_decoder=True, 
                                                encoder_dropout=True, dropout_rate=dropout_prob, sparsity=self.args.de_sparsity, generator_shortcut=False, 
                                                generator_transform='softmax', device=device).to(device)

        self.sent_vae_model   = VaeAvitmModel(input_dimensionality, d_e=vae_hidd_size, d_t=num_of_OpinionTopics, 
                                                encoder_layers=encoder_layers, generator_layers=generator_layers, without_decoder=True, 
                                                encoder_dropout=True, dropout_rate=dropout_prob, sparsity=self.args.de_sparsity, generator_shortcut=False, 
                                                generator_transform='softmax', device=device).to(device)

        self.plot_vae_model   = VaeAvitmModel(input_dimensionality, d_e=vae_hidd_size, d_t=num_of_PlotTopics, 
                                                encoder_layers=encoder_layers, generator_layers=generator_layers, without_decoder=False, 
                                                encoder_dropout=True, dropout_rate=dropout_prob, sparsity=self.args.de_sparsity, generator_shortcut=False, 
                                                generator_transform='softmax', device=device).to(device)

        # - Sentiment classifier -
        self.num_of_classes    = num_of_classes
        self.sent_class_model  = SentimentClassifier(input_dimensionality, 
                                                     num_of_OpinionTopics, 
                                                     num_of_classes, 
                                                     hid_size=self.args.sent_classi_hid_size,
                                                     device=device).to(device)

        # - Plot discriminator/classifier -
        # It is not an actual sentiment classifier, just reusing the same class.
        self.plot_discri_model = SentimentClassifier(input_dimensionality, 
                                                     num_of_PlotTopics, 
                                                     num_of_classes=2, 
                                                     hid_size=self.args.plot_classi_hid_size,
                                                     device=device).to(device)

        # - Linear projection for possible asymmetric number of topics -
        if self.num_of_PlotTopics != self.num_of_OpinionTopics:
            self.plotScaling = nn.Linear(self.num_of_PlotTopics, self.num_of_OpinionTopics)

        # Dropout
        self.r_drop = nn.Dropout(dropout_prob)

        # - Decoder matrix -
        if self.interaction == "dot_prod":
            self.de = nn.Linear(self.num_of_PlotTopics*self.num_of_OpinionTopics, self.input_dimensionality)
        elif self.interaction == "concat":
            self.de = nn.Linear(self.num_of_PlotTopics + num_of_OpinionTopics, self.input_dimensionality)
        elif self.interaction == "onlySent":
            self.de = nn.Linear(self.num_of_OpinionTopics, self.input_dimensionality)
        elif self.interaction == "onlyNeutral":
            self.de = nn.Linear(self.num_of_PlotTopics, self.input_dimensionality)

        # Batch Norm.
        self.de_bn = nn.BatchNorm1d(self.input_dimensionality)

        # Orthogonal Reg.
        self.ortho_regul_flag = True
        

        # --- INIT ---
        # Decoder initialization
        weights_init_sparse(self.de, sparsity=self.args.de_sparsity)
        if self.num_of_PlotTopics != self.num_of_OpinionTopics:
            weights_init_xavier(self.plotScaling)
    
        
    def decoder(self, r):
        p_x_given_h = F.softmax(self.de_bn(self.de(r)), dim=1)
        return p_x_given_h
                
                
    def forward(self, x, x_plots=None, perplexity=False):
        # --- Split reviews and plots ---
        if self.plug_Plots and not perplexity:
            x_plots = x[:, self.input_dimensionality:]
            x       = x[:, :self.input_dimensionality]

        # --- Encoders ---
        mean_a, logvar_a, var_a, z_a  = self.aspect_vae_model(x)
        mean_s, logvar_s, var_s, z_s  = self.sent_vae_model(x)

        # Plot Encoder
        if self.plug_Plots and not perplexity:
            mean_p, logvar_p, var_p, z_p, p_x_given_h_plots = self.plot_vae_model(x_plots)
            conc_z_a_z_p    = torch.cat((z_a, z_p), 0)
            y_p_pred        = self.plot_discri_model(conc_z_a_z_p[torch.randperm(conc_z_a_z_p.size()[0])])
        else:
            y_p_pred = mean_p = logvar_p = var_p = z_p = p_x_given_h_plots = None

        # --- Interaction ---
        interaction_vec = self.z_interaction(z_a, z_s)

        # --- Decoder ---
        p_x_given_h     = self.decoder(interaction_vec)

        # --- Adversarial prediction ---
        y_s_pred = self.sent_class_model(z_s)
        if self.num_of_PlotTopics != self.num_of_OpinionTopics:
            y_a_pred = self.sent_class_model(self.plotScaling(z_a))
        else:
            y_a_pred = self.sent_class_model(z_a)

        # # -- Orthogonal regularization --
        if self.ortho_regul_flag:
            decoder_weights = self.de.weight.data.transpose(0,1).to(self.device, non_blocking=True)
            orth_loss       = orthogonal_reg_loss(self.device, decoder_weights)
        else:
            orth_loss = 0.0

        return [z_a, z_s, p_x_given_h, interaction_vec, mean_a, logvar_a, var_a, mean_s, logvar_s, var_s, \
                y_a_pred, y_s_pred, y_p_pred, mean_p, logvar_p, var_p, z_p, p_x_given_h_plots, orth_loss]
    
        
    def save_params(self, filename):
        torch.save(self.state_dict(), filename)
    

    def load_params(self, filename):
        self.load_state_dict(torch.load(filename))


    def z_interaction(self, z_a, z_s):
        interaction_vec = None

        if self.interaction == "dot_prod":
            interaction_vec = torch.bmm(z_a.unsqueeze(2), z_s.unsqueeze(2).transpose(1,2))
            batch_size      = interaction_vec.size()[0]
            interaction_vec = interaction_vec.view(batch_size, -1)

        # --- Interaction through concatination ---
        # interaction_vec: (batch_size, 2*#topics)
        elif self.interaction == "concat":
            interaction_vec  = torch.cat((z_a, z_s), 1) 

        # -- Interaction without interaction :) ---
        elif self.interaction == "onlySent":
            interaction_vec = z_s

        # -- Interaction without interaction :) ---
        elif self.interaction == "onlyNeutral":
            interaction_vec = z_a

        return interaction_vec



    ###################
    # FREEZE PARAMETERS
    ###################
    def freeze_sent_discriminators(self, freeze):
        # Freeze or defrost discriminators parameters
        if freeze:
            print("Sentiment discriminator parameters have been frozen.")
            self.sent_class_model.freeze_parameters(freeze)
        else:
            print("Sentiment discriminator parameters have been DE-frozen")
            self.sent_class_model.freeze_parameters(freeze)

    def freeze_plot_vae_and_discriminators(self, freeze):
        for m in [self.plot_vae_model, self.plot_discri_model]:
            for param in m.parameters():
                if freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            if freeze:
                m.frozen = True
            else:
                m.frozen = False

    def freeze_aspect_sent_VAE_encoders(self, freeze):
        for m in [self.aspect_vae_model, self.sent_vae_model]:
            for param in m.parameters():
                if freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            if freeze:
                m.frozen = True
            else:
                m.frozen = False

    def freeze_VAEdecoder(self, freeze):
        for param in self.de.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        if freeze:
            self.de.frozen = True
        else:
            self.de.frozen = False

    def remove_ortoghonalization_regularizer(self, remove):
        if remove:
            self.ortho_regul_flag = False
        else:
            self.ortho_regul_flag = True



    ###############
    # LOSS
    ###############
    def compute_KLD(self, posterior_mean, posterior_logvar, posterior_var, prior_mean, 
                    prior_var, prior_logvar, num_of_topics):
        # see Appendix B from paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        prior_mean      = prior_mean.expand_as(posterior_mean)
        prior_var       = prior_var.expand_as(posterior_mean)
        prior_logvar    = prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        KLD             = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - num_of_topics )
        return KLD
    

    def loss_function_bagOfWords(self, posterior_mean_a, posterior_logvar_a, posterior_var_a, 
                                       posterior_mean_s, posterior_logvar_s, posterior_var_s, 
                                       p_x_given_h, DocTerm_batch, avg_loss=True):

        KLD_a = self.compute_KLD(posterior_mean_a, posterior_logvar_a, posterior_var_a, 
                                 self.priors["prior_mean_Plot"], self.priors["prior_var_Plot"], 
                                 self.priors["prior_logvar_Plot"], self.num_of_PlotTopics)

        KLD_s = self.compute_KLD(posterior_mean_s, posterior_logvar_s, posterior_var_s, 
                                 self.priors["prior_mean_Opinion"], self.priors["prior_var_Opinion"], 
                                 self.priors["prior_logvar_Opinion"], self.num_of_OpinionTopics)

        nll_term = -(DocTerm_batch * (p_x_given_h+1e-10).log()).sum(1)

        loss     = self.beta_a*KLD_a + self.beta_s*KLD_s  + nll_term

        if avg_loss:
            loss = loss.mean()

        return (loss, nll_term)



    def overall_loss_func(self, reconstr_loss, y_adv_pred, y_sent_pred, y_sent_labels, y_p_pred, orth_loss, 
                          recontr_loss_plot=0.0, perplexity=False, test=False):
        alpha   = 1.0 / (self.args.vocab_size * 0.5)
        beta    = 0.0
        gamma   = 0.0
        delta   = 1.0 / (self.args.vocab_size * 1.3)
        epsilon = 0.0
        zeta    = 0.0       

        adv_sent_loss     = 0.0 
        sent_loss         = 0.0    
        adv_plot_loss     = 0.0

        if not perplexity:

            if self.alsoSentLoss:
                sent_loss     = F.cross_entropy(y_sent_pred.to(self.device), y_sent_labels.to(self.device)) 
                gamma         = 1.0 

            if self.alsoAspectLoss:
                uniform_dist  = torch.Tensor(len(y_adv_pred), self.num_of_classes).fill_((1./self.num_of_classes)).to(self.device)
                # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
                # https://github.com/alinlab/Confident_classifier/blob/master/src/run_joint_confidence.py
                adv_sent_loss = F.kl_div(F.log_softmax(y_adv_pred), uniform_dist, reduction='sum')*self.num_of_classes
                beta          =  2.0 

            
            if self.plug_Plots and self.plot_vae_model.frozen != True:
                adv_plot_loss = adv_cross_entropy(y_p_pred.to(self.device))
                epsilon       = 1.0  
                zeta          = 1.0

            overall_loss = alpha*reconstr_loss.mean() + beta*adv_sent_loss + gamma*sent_loss + delta*orth_loss 
                           + epsilon*adv_plot_loss + zeta*recontr_loss_plot

        else:
            overall_loss = reconstr_loss


        list_of_loss = [overall_loss, alpha*reconstr_loss.mean(), beta*adv_sent_loss, gamma*sent_loss, \
                        delta*orth_loss, epsilon*adv_plot_loss, zeta*recontr_loss_plot]

        if test:
            list_of_loss.append(y_sent_pred)            

        return list_of_loss


    def loss(self, list_of_computed_params, DocTerm_batch, labels_batch, avg_loss=True, perplexity=False, test=False):
        # Unzipping list values
        z_a, z_s, p_x_given_h, interaction_vec, mean_a, logvar_a, var_a, mean_s, logvar_s, var_s, y_a_pred, y_s_pred, y_p_pred, mean_p, logvar_p, var_p, z_p, p_x_given_h_plots, orth_loss = list_of_computed_params

        if self.plug_Plots and not perplexity:
            DocTerm_batch                        =  DocTerm_batch[:, :self.input_dimensionality]
            params_from_vae                      =  [mean_p, logvar_p, var_p, z_p, p_x_given_h_plots]
            recontr_loss_plot, nll_term_p, KLD_p =  self.plot_vae_model.loss(params_from_vae, DocTerm_batch)
        else:
            recontr_loss_plot = 0.0


        # p_x_given_h: (batch_size, input_dimensionality)
        reconstr_loss, nll_term  = self.loss_function_bagOfWords(mean_a, logvar_a, var_a, mean_s, logvar_s, var_s, p_x_given_h, 
                                                                 DocTerm_batch, avg_loss=avg_loss)

        list_of_loss             = self.overall_loss_func(reconstr_loss, y_a_pred, y_s_pred, labels_batch, 
                                                         y_p_pred, orth_loss, recontr_loss_plot=recontr_loss_plot, perplexity=perplexity, test=test)
        
        return list_of_loss



    ###############
    # PRINT TOPICS
    ###############
    def t_uniqueness(self, dataset, annotated_sents_dataset, topicType='All'):
        list_of_topics  = self.get_list_of_topics(dataset, annotated_sents_dataset=annotated_sents_dataset)
        num_of_topWords = len(list_of_topics[0].words_in_topic)
        word_counter    = 0.0

        # Choose topics of interest
        list_of_filtered_topics = []
        if topicType == 'All':
            list_of_filtered_topics = list_of_topics
            word_counter            = Counter([w for t in list_of_filtered_topics for w in t.words_in_topic])

        elif topicType == 'Neutral':
            list_of_filtered_topics = [t for t in list_of_topics if t.topic_type == "Plot"]
            word_counter            = Counter([w for t in list_of_filtered_topics for w in t.words_in_topic])
            if len(list_of_filtered_topics) == 0:
                print("ATT: Zero neutral topics.")

        elif topicType == 'Opinion':
            list_of_filtered_topics = [t for t in list_of_topics if t.topic_type == "Opinion"]
            word_counter            = Counter([w for t in list_of_filtered_topics for w in t.words_in_topic])
            if len(list_of_filtered_topics) == 0:
                print("ATT: Zero opinion topics.")
        else:
            print("ERR: TopicType for uniqueness not recognized.")
            sys.exit(0)

        # -- Final TU is in between 1/#Topics and #Topics --
        # (Topic Modeling with Wasserstein Autoencoders, Nan et al. 2019)
        tu = 0.0
        for t in list_of_filtered_topics:
            t_tu = 0.0
            for w in t.words_in_topic:
                t_tu += (1/word_counter[w])
            tu += (t_tu / num_of_topWords)

        return tu




    ###############
    # PRINT TOPICS
    ###############
    def print_ordered_topics(self, ch_ordered_topic_list, ordered_ch, ch_order=None):
        if ch_order is None:
            for t,ch in zip(ch_ordered_topic_list, ordered_ch):
                print( ("{:<105} | {:>}").format("  ".join(t), ch) )
        else:
            for t,ch,t_ID in zip(ch_ordered_topic_list, ordered_ch, ch_order):
                print( ("{:<105} | {:<3} | {:>}").format("  ".join(t), t_ID, ch) )

        print("Avg: ", np.mean(ordered_ch))
        print("Std: ", np.std(ordered_ch))


    def sort_topics_by_coherence(self, ch_model, topic_list):
        topic_ch_list = ch_model.get_coherence_per_topic()

        # Ordering topics by coherence. The [::-1] is to get the reversed order.
        ch_order              = list(np.argsort(topic_ch_list).tolist())
        ch_order              = ch_order[::-1]
        ch_ordered_topic_list = [topic_list[i] for i in ch_order]
        ordered_ch            = [topic_ch_list[i] for i in ch_order]

        return ch_ordered_topic_list, ordered_ch, ch_order


    def print_topics(self, dataset, gensim_dataset, id2token, visualized_words, annotated_sents_dataset=None):
        list_of_topics        = self.get_list_of_topics(dataset, annotated_sents_dataset=annotated_sents_dataset)
        list_of_OpinionTopics = [t.words_in_topic for t in list_of_topics if t.topic_type=="Opinion"]
        list_of_NeutralTopics = [t.words_in_topic for t in list_of_topics if t.topic_type=="Plot"]

        avg_tCh_mass_list = []
        avg_tCh_cv_list   = []
        for i, highest_topic_list  in enumerate([list_of_NeutralTopics, list_of_OpinionTopics]):
            if i == 0:
                print("\nNeutral Topics:")
            else:
                print("\nSentiment Topics")
       
            # define other remaining metrics available
            ch_umass       = CoherenceModel(topics=highest_topic_list, corpus=gensim_dataset.corpus_vect_gensim, 
                                            dictionary=gensim_dataset.dictionary_gens, coherence='u_mass')
            
            ch_cv          = CoherenceModel(topics=highest_topic_list, dictionary=gensim_dataset.dictionary_gens, 
                                            texts=gensim_dataset.tokenized_corpus, coherence="c_v" )

            # --- U_MASS ---
            # print("\nU_MASS:")
            ch_ordered_topic_list, ordered_ch, ch_order = self.sort_topics_by_coherence(ch_umass, highest_topic_list)
            avg_tCh_mass_list.append(np.mean(ordered_ch)) 

            # --- C_V ---
            print("\nTOPICS - C_V:")
            ch_ordered_topic_list, ordered_ch, ch_order = self.sort_topics_by_coherence(ch_cv, highest_topic_list)
            self.print_ordered_topics(ch_ordered_topic_list, ordered_ch, ch_order)
            avg_tCh_cv_list.append(np.mean(ordered_ch))

        return (avg_tCh_mass_list, avg_tCh_cv_list)


    def getTopicWordDistr(self):
        return self.de.weight.data.cpu().numpy()


    def print_TopicSentences(self, exp, dataset, epoch, export=False, annotated_sents_dataset=None):

        TopicWordDistr = self.getTopicWordDistr()

        # - Configure TopicLabeling object -
        tl = TopicLabeling(self.exp, self.num_of_OpinionTopics+self.num_of_PlotTopics, 
                           self.aspect_vae_model, self.sent_vae_model, self.interaction)

        # - Get topic sentences with and without applying SVD -
        topic_sentences            = self.get_topic_sentences(dataset, 
                                                             TopicLabelingObj=tl,
                                                             TopicWordDistr=TopicWordDistr, 
                                                             rmpc_svd=False, 
                                                             annotated_sents_dataset=annotated_sents_dataset)

        topic_sentences_svdremoved = self.get_topic_sentences(dataset, 
                                                             TopicLabelingObj=tl,
                                                             TopicWordDistr=TopicWordDistr,
                                                             rmpc_svd=True, 
                                                             annotated_sents_dataset=annotated_sents_dataset)

        # --- Get topics ----
        list_of_topics = self.get_list_of_topics(dataset, 
                                                 TopicWordDistr, 
                                                 topic_sentences_svdremoved,
                                                 annotated_sents_dataset)

        # --- Export JSON files ---
        if export:
            self.export_topics_sents_2Json(exp, epoch, topic_sentences, topic_sentences_svdremoved, list_of_topics)

        # --- Write topic details with sentence labels to file ---
        str_with_topic_details = ""
        
        for t in list_of_topics:
            str_with_topic_details += t.print_topic_details()

        if annotated_sents_dataset is not None:
            write_file(exp.exp_folder_path+"annotatedTopicSentences_E_"+str(epoch)+".txt",  str_with_topic_details)
        else:
            write_file(exp.exp_folder_path+"datasetTopicSentences_E_"+str(epoch)+".txt",  str_with_topic_details)


        # --- Compute topic-type statistics ---
        if annotated_sents_dataset is not None:
            self.compute_topic_type_coherence_statistics(dataset, annotated_sents_dataset, list_of_topics)



    def get_topic_sentences(self, dataset, TopicLabelingObj=None, TopicWordDistr=None, rmpc_svd=True, annotated_sents_dataset=None):

        if TopicWordDistr is None:
            TopicWordDistr = self.getTopicWordDistr()

        # - Configure TopicLabeling object -
        if TopicLabelingObj is None:
            tl = TopicLabeling(self.exp, self.num_of_OpinionTopics+self.num_of_PlotTopics, self.aspect_vae_model, 
                               self.sent_vae_model, self.interaction)
        else:
            tl = TopicLabelingObj

        # # - Get topic sentences with and without applying SVD -
        topic_sentences  = tl.getTopicsSentences(dataset, 
                                                 TopicWordDistr, 
                                                 rmpc_svd=rmpc_svd, 
                                                 annotated_sents_dataset=annotated_sents_dataset)

        return topic_sentences



    def get_list_of_topics(self, dataset, TopicWordDistr=None, topic_sentences=None, annotated_sents_dataset=None): 

        if TopicWordDistr is None:
            TopicWordDistr = self.getTopicWordDistr()

        if topic_sentences is None:
            topic_sentences = self.get_topic_sentences(dataset=dataset, 
                                                       TopicWordDistr=TopicWordDistr,
                                                       rmpc_svd=True, 
                                                       annotated_sents_dataset=annotated_sents_dataset)

        # --- Get topics ----
        if self.interaction == "onlySent":
            weights_a       = TopicWordDistr[:, :self.num_of_OpinionTopics]
            weights_s       = TopicWordDistr[:, self.num_of_OpinionTopics:]
        else:
            weights_a       = TopicWordDistr[:, :self.num_of_PlotTopics]
            weights_s       = TopicWordDistr[:, self.num_of_PlotTopics:]


        list_of_topics = []

        # Duplicate For-loop for code readability
        if self.interaction != "onlySent":
            for j in range(self.num_of_PlotTopics):
                # "order" are the indexes of the potential ordered sequence
                order = list( np.argsort(weights_a[:, j])[::-1].tolist())
                list_of_words = [dataset["id2token"][z] for z in order[:10]]
                list_of_topics.append(Topic(list_of_words, 
                                            topic_type         = "Plot", 
                                            list_of_weights    = weights_a[order[:10], j],
                                            list_of_sentences  = topic_sentences[j], 
                                            # list_sentence_scores = topic_sentences_scores           
                                            )
                                    )         
                if annotated_sents_dataset is not None:
                    list_of_topics[-1].label2text = ("Positive", "Negative", "Plot", "None")              
                                         
        if self.interaction != "onlyNeutral":
            for j in range(self.num_of_OpinionTopics):
                # "order" are the indexes of the potential ordered sequence
                order = list( np.argsort(weights_s[:, j])[::-1].tolist())
                list_of_words = [dataset["id2token"][z] for z in order[:10]]
                list_of_topics.append(Topic(list_of_words, 
                                            topic_type         = "Opinion", 
                                            list_of_weights    = weights_s[order[:10], j],
                                            list_of_sentences  = topic_sentences[self.num_of_PlotTopics+j], 
                                            # list_sentence_scores = topic_sentences_scores                                         
                                            )
                                    )
                if annotated_sents_dataset is not None:
                    list_of_topics[-1].label2text = ("Positive", "Negative", "Plot", "None")  

        return list_of_topics


    def compute_topic_type_coherence_statistics(self, dataset, annotated_sents_dataset, list_of_topics=None, expectation_indipendent=True):
        if list_of_topics is None:
            list_of_topics = self.get_list_of_topics(dataset, annotated_sents_dataset=annotated_sents_dataset)

        # --- Compute topic type-coherence statistics ---
        actual_opinion_topics = 0
        actual_plot_topics    = 0
        actual_opinion_topic_rate = -1
        actual_plot_topic_rate = -1

        if expectation_indipendent:
            for t in list_of_topics:
                if t.most_common_label is None:
                    t.compute_topic_type_coherence()

                if t.label2text[t.most_common_label] in ["Positive", "Negative"]:
                    actual_opinion_topics += 1
                elif t.label2text[t.most_common_label] in ["Plot", "None"]:
                    actual_plot_topics    += 1

            actual_opinion_topic_rate = actual_opinion_topics / (self.num_of_OpinionTopics +self.num_of_PlotTopics)
            actual_plot_topic_rate    = actual_plot_topics / (self.num_of_OpinionTopics +self.num_of_PlotTopics)

        else:
            for t in list_of_topics:
                if t.most_common_label is None:
                    t.compute_topic_type_coherence()

                if t.topic_type == "Opinion":
                    if t.label2text[t.most_common_label] in ["Positive", "Negative"]:
                        actual_opinion_topics += 1
                elif t.topic_type == "Plot":
                    if t.label2text[t.most_common_label] in ["Plot", "None"]:
                        actual_plot_topics += 1

            actual_opinion_topic_rate = actual_opinion_topics / self.num_of_OpinionTopics
            actual_plot_topic_rate    = actual_plot_topics    / self.num_of_PlotTopics

        return actual_opinion_topic_rate, actual_plot_topic_rate

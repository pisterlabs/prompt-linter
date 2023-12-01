from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, bert_embeddings_from_list
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
import os
import numpy as np
import pickle
import torch
from contextualized_topic_models.datasets.dataset import CTMDatasetTopReg
from contextualized_topic_models.utils.data_preparation import TextHandler, labels_from_file

handler = TextHandler("contextualized_topic_models/data/wiki/wiki_train_en_prep.txt")
handler.prepare()

train_bert = bert_embeddings_from_file('contextualized_topic_models/data/wiki/wiki_train_en_unprep.txt', \
       '../sentence-transformers/sentence_transformers/output/training_wiki_topics_4_xlm-roberta-base-2020-10-24_13-38-14/')
labels, nb_labels = labels_from_file('contextualized_topic_models/data/wiki/topic_full_misc.json')
training_dataset = CTMDatasetTopReg(handler.bow, train_bert, handler.idx2token, labels)

num_topics = 100
# the nb_labels argument turns this from a CTM into a TCCTM
ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=60, hidden_sizes=(100,),
          inference_type="contextual", n_components=num_topics, num_data_loader_workers=0, nb_labels=nb_labels)
ctm.fit(training_dataset)
ctm.save("models/wiki/wiki_xlmr_en_topics_4_topreg")

from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, bert_embeddings_from_list
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
import os
import numpy as np
import pickle
import torch
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import TextHandler

handler = TextHandler("contextualized_topic_models/data/wiki/wiki_train_en_prep.txt")
handler.prepare()

#train_bert = bert_embeddings_from_file('contextualized_topic_models/data/wiki/wiki_train_en_unprep.txt', 'distiluse-base-multilingual-cased')
# train_bert = bert_embeddings_from_file('contextualized_topic_models/data/wiki/wiki_train_en_unprep.txt', 'xlm-r-100langs-bert-base-nli-mean-tokens')
train_bert = bert_embeddings_from_file('contextualized_topic_models/data/wiki/wiki_train_en_unprep.txt', \
        '../sentence-transformers/sentence_transformers/output/training_nli_wiki-xlmr-2020-12-15_00-20-18')
training_dataset = CTMDataset(handler.bow, train_bert, handler.idx2token)

num_topics = 100
#ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=60, hidden_sizes=(100,),
#          inference_type="contextual", n_components=num_topics, num_data_loader_workers=0)
# ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=60, hidden_sizes=(100,),
#          inference_type="contextual", n_components=num_topics, num_data_loader_workers=0)
ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=60, hidden_sizes=(100,),
          inference_type="contextual", n_components=num_topics, num_data_loader_workers=0)
ctm.fit(training_dataset)
ctm.save("models/wiki/wiki_xlmr_en_nli_ct")

# filehandler = open("iqos_en.ctm", 'wb')
# torch.save(ctm, "wiki_en_xlmr_topicsiqos_1.ctm", pickle_protocol=4)
# with open("contextualized_topic_models/data/wiki/wiki_train_en_prep_sub.txt", "r") as en:
#     texts = [doc.split() for doc in en.read().splitlines()]

# npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
# print(npmi.score())

import os
import pandas as pd
import torch
import string
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from models.multilingual_contrast import MultilingualContrastiveTM
from utils.data_preparation import MultilingualTopicModelDataPreparation
from utils.preprocessing import WhiteSpacePreprocessingMultilingual

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='MultilingualContrast', type=str)
argparser.add_argument('--data_path', default='data/', type=str)
argparser.add_argument('--save_path', default='trained_models/', type=str)
argparser.add_argument('--train_data', default='wikiarticles.csv', type=str)
argparser.add_argument('--num_topics', default=100, type=int)
argparser.add_argument('--num_epochs', default=100, type=int)
argparser.add_argument('--langs', default='en,de', type=str)
argparser.add_argument('--sbert_model', default='paraphrase-multilingual-mpnet-base-v2', type=str)
argparser.add_argument('--text_enc_dim', default=512, type=int)
argparser.add_argument('--batch_size', default=32, type=int)
argparser.add_argument('--max_seq_length', default=200, type=int)
argparser.add_argument('--kl_weight', default=0.01, type=int, help='weight for the KLD loss')
argparser.add_argument('--cl_weight', default=50, type=int, help='weight for the contrastive loss')
args = argparser.parse_args()

print("\n" + "-"*5, "Train Contrastive PLTM", "-"*5)
print("model_name:", args.model_name)
print("data_path:", args.data_path)
print("save_path:", args.save_path)
print("train_data:", args.train_data)
print("num_topics:", args.num_topics)
print("num_epochs:", args.num_epochs)
print("langs:", args.langs)
print("sbert_model:", args.sbert_model)
print("text_enc_dim:", args.text_enc_dim)
print("batch_size:", args.batch_size)
print("max_seq_length:", args.max_seq_length)
print("kl_weight:", args.kl_weight)
print("cl_weight:", args.cl_weight)
print("-"*50 + "\n")

# stopwords lang dict
lang_dict = {'en': 'english',
             'de': 'german',
             'fi': 'finnish'}

# ----- load dataset -----
df = pd.read_csv(os.path.join(args.data_path, args.train_data))
# print("df:", df.shape)
languages = args.langs.lower().split(',')
languages = [l.strip() for l in languages]
print('languages:', languages)

documents = [list(df[lang+'_text']) for lang in languages]


# ----- preprocess documents -----
lang_stopwords = [lang_dict[l] for l in languages]
preproc_pipeline = WhiteSpacePreprocessingMultilingual(documents=documents,
                                                       stopwords_languages=lang_stopwords,
                                                       max_len=args.max_seq_length)
preprocessed_docs, raw_docs, vocab = preproc_pipeline.preprocess()
for l in range(len(languages)):
    print("-"*5, "lang", l, ":", languages[l].upper(), "-"*5)
    print('preprocessed_docs:', len(preprocessed_docs[l]))
    print('raw_docs:', len(raw_docs[l]))
    print('vocab:', len(vocab[l]))

#  ----- encode documents -----
qt = MultilingualTopicModelDataPreparation(args.sbert_model, vocabularies=vocab)

training_dataset = qt.fit(text_for_contextual=raw_docs, text_for_bow=preprocessed_docs)


# ----- initialize model -----
loss_weights = {"KL": args.kl_weight,
                "CL": args.cl_weight}

contrast_model = MultilingualContrastiveTM(bow_size=qt.vocab_sizes[0],
                                           contextual_size=args.text_enc_dim,
                                           n_components=args.num_topics,
                                           num_epochs=args.num_epochs,
                                           languages=languages,
                                           batch_size=args.batch_size,
                                           loss_weights=loss_weights)

# ----- topic inference -----
contrast_model.fit(training_dataset)

# ----- save model -----
save_filepath = os.path.join(args.save_path, args.model_name + '_K' + str(args.num_topics)
                             + '_epochs' + str(args.num_epochs)
                             + '_' + args.sbert_model)
contrast_model.save(save_filepath)
print("Done! Saved model as", save_filepath)


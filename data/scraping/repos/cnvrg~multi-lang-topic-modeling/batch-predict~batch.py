# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import spacy
import gensim.corpora as corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import translators as ts
import joblib
from random import randrange
import os
import argparse
import tqdm
import numpy as np
from gensim.models import CoherenceModel
import langid

parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--input_file",
    action="store",
    dest="input_file",
    default="/data/topic_data/sample_topic_data.csv",
    required=True,
    help="""location of input file""",
)
parser.add_argument(
    "--model_file",
    action="store",
    dest="model_file",
    default="lda.sav",
    required=True,
    help="""saved model file""",
)
parser.add_argument(
    "--topic_word_cnt",
    action="store",
    dest="topic_word_cnt",
    default="10",
    required=True,
    help="""number of words describing each topic""",
)
parser.add_argument(
    "--dictionary_path",
    action="store",
    dest="dictionary_path",
    default="/input/lda/corp_dict_file.sav",
    required=True,
    help="""corpora dictionary file""",
)
parser.add_argument(
    "--model_results_path",
    action="store",
    dest="model_results_path",
    default="/input/lda/model_results.csv",
    required=True,
    help="""path of lda model prepared with same data""",
)

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
args = parser.parse_args()
input_file = pd.read_csv(args.input_file)
topic_cnt = int(pd.read_csv(args.model_results_path)['Topics_Count'][0])
model_file = args.model_file
topic_word_cnt = int(args.topic_word_cnt)
topic_word_cnt_file = cnvrg_workdir+'/topic_word_cnt.csv'
pd.DataFrame([topic_word_cnt],columns=['Topic_Word_Cnt']).to_csv(topic_word_cnt_file)
lda_model = joblib.load(model_file)
corp_dict = joblib.load(args.dictionary_path)

cnt = 0
if os.path.exists("/input/train/lda_model.sav"):
    print('Do Nothing')
else:
    print('Stand Alone Batch')
    for i in input_file.text:
        lang_cde = langid.classify(i)[0]
        if lang_cde != 'en':
            translated_val = ts.google(i, from_language=lang_cde, to_language='en')
            input_file.at[cnt,'trans_val'] = translated_val
            input_file.at[cnt,'title'] = 'https://'+lang_cde+'.w'
        else:
            input_file.at[cnt,'trans_val'] = i
            input_file.at[cnt,'title'] = 'https://'+lang_cde+'.w'
        cnt = cnt+1
    input_file = input_file.drop(['text'], axis=1)
    input_file.rename(columns={'trans_val': 'text','title': 'title'}, inplace=True)
    input_file.to_csv('/cnvrg/input_temp.csv')

df = input_file
df['text'] = df['text'].apply(str)
# Remove punctuation
df['text_processed'] = df['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
df['text_processed'] = df['text_processed'].map(lambda x: x.lower())

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data = df.text_processed.values.tolist()
data_words = list(sent_to_words(data))
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
                                'NOUN', 'ADJ', 'VERB', 'ADV'])
texts = data_lemmatized
corpus = [corp_dict.doc2bow(text) for text in texts]

docnames = ["Doc" + str(i) for i in range(len(df.text))]
col_res = []
for i in range(topic_cnt):
    col_res.append(f'Topic-{i}')
    col_res.append(f'Topic-{i}-prob')

final_df = pd.DataFrame(columns=col_res, index=docnames)

cnt_row = 0
for j in range(len(corpus)):
    cnt_col = 0
    cnt_pred_col = 1
    cnt_pointer = 0
    for index, score in lda_model[corpus[j]]:
        if index == cnt_pointer:
            final_df.at['Doc'+str(cnt_row), final_df.columns[cnt_col]
                        ] = lda_model.print_topic(index, topic_word_cnt)
            final_df.at['Doc'+str(cnt_row),
                        final_df.columns[cnt_pred_col]] = score
            cnt_col = cnt_col+2
            cnt_pred_col = cnt_col+1
        elif index > cnt_pointer:
            diff = index-cnt_pointer
            final_df.at['Doc'+str(cnt_row), final_df.columns[cnt_col+2*(diff)]
                        ] = lda_model.print_topic(index, topic_word_cnt)
            final_df.at['Doc'+str(cnt_row),
                        final_df.columns[cnt_pred_col+2*(diff)]] = score
            cnt_col = cnt_col+2
            cnt_pred_col = cnt_col+1
        cnt_pointer = cnt_pointer+1
    cnt_row = cnt_row+1

prob_cols = [col for col in final_df.columns if 'prob' in col]
final_df[prob_cols] = final_df[prob_cols].fillna(0)
for k in range(final_df.shape[1]):
    for m in range(final_df.shape[0]):
        if (pd.isna(final_df.iloc[[m], [k]].values[0][0]) == True):
            final_df.iloc[[m], [k]] = final_df.iloc[:,[k]].drop_duplicates().values[0][0]

final_df['Document_Number'] = final_df.index


df['Document_Number'] = ['Doc'+str(x) for x in list(df.index)]
#df.to_csv('/cnvrg/df_temp.csv')
final_df = final_df.merge(df[['title', 'Document_Number']], on='Document_Number')
final_df['Lang_Code'] = final_df['title'].str[8:10]

#final_df.to_csv('/cnvrg/final_temp.csv')
lang_codes = list(final_df['Lang_Code'].unique())

if len(lang_codes) > 1 != 'en':
    english_topics = []
    topics_number = []
    for idx, topic in lda_model.print_topics(-1,topic_word_cnt):
        english_topics.append(topic)
        topics_number.append(idx)

    translation_frame = pd.DataFrame(english_topics, columns=['original_vals'])
    if len(lang_codes) == 1:
        lang_codes.remove('en')
    else:
        lang_codes.remove('en')

    topic_iteration = 0
    for lang_cde in lang_codes:
        topics_list = []
        for topic_iteration in range(len(english_topics)):
            ac = english_topics[topic_iteration].split()
            translated_val = [ts.google(z, from_language='en', to_language=lang_cde) for z in [y.group() for y in [re.search('"(.*)"', x) for x in list(filter(('+').__ne__, ac))]]]
            joined_string = [y+'×'+z for y, z in zip([x[0:5] for x in list(filter(('+').__ne__, ac))], translated_val)]
            translated_string = " + ".join([y for y in joined_string])
            topics_list.append(translated_string.replace('"', ''))
        translation_frame[lang_cde] = topics_list
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    consolidated_df = final_df
    col_res_1 = []
    for i in range(topic_cnt):
        col_res_1.append(f'Topic_{i}')
    lang_df = pd.DataFrame(columns=col_res_1)
    j = 0

    for lang_code in lang_codes:
        for i in range(topic_cnt):
            lang_df.at[j, 'Topic_'+str(i)] = translation_frame[lang_code][i]
        lang_df.at[j, "Lang_Code"] = lang_code
        j = j+1
    consolidated_df = consolidated_df.merge(lang_df, how='left', left_on=[
                                            'Lang_Code'], right_on=['Lang_Code'])

    prob_cols = [x for x in list(consolidated_df.columns) if x.endswith('prob')]
    old_cols = [x for x in list(consolidated_df.columns) if '-' in x and 'prob' not in x]
    new_cols = [x for x in list(consolidated_df.columns) if '_' in x and len(x)<9]
    everything_else = ['Lang_Code', 'Document_Number', 'title']
    final_order = []
    final_order.extend(everything_else)
    for i in range(len(old_cols)):
        final_order.append(new_cols[i])
        final_order.append(prob_cols[i])
    final_order.extend(old_cols)
    nu_to_remove = len(old_cols)
    del final_order[-nu_to_remove:]
    topic_output = consolidated_df[final_order]

    old_sub_frame = consolidated_df[consolidated_df['Lang_Code'] == 'en'][old_cols]
    new_sub_frame = topic_output[topic_output['Lang_Code'] != 'en'][new_cols]
    old_sub_frame.columns = old_sub_frame.columns.str.replace("-", "_")
    combined_sub_frame = pd.concat([old_sub_frame,new_sub_frame],axis=0)
    topic_output[new_cols] = combined_sub_frame[new_cols]
    topic_output.columns.str.replace("_","-")
    final_file_path = cnvrg_workdir+'/topic_model_output.csv'
    topic_output = topic_output.drop('title', axis=1)
    topic_output.to_csv(final_file_path)
else:
    final_df = final_df.drop('title', axis=1)
    alternate_cols = list(final_df.columns)
    alternate_cols = [alternate_cols[-1]] + alternate_cols[:-1]
    alternate_cols = [alternate_cols[0]]+[alternate_cols[-1]] + alternate_cols[1:len(alternate_cols)-1]
    topic_output = final_df[alternate_cols]
    final_file_path = cnvrg_workdir+'/topic_model_output.csv'
    #topic_output = final_df.drop('title', axis=1)
    topic_output.to_csv(final_file_path)

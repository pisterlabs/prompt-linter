'''
Toolkits for analysis
'''
import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

import pickle
import os
import json

from pos import CMUTweetTagger as tagger


def doc_stats(df):
    '''Provide basic document stats analysis:
        * Number of Document (num_doc) in the datasets
        * Average (aver_doc) & Median (median_doc) number of word per document;
    '''
    doc_stat = dict()
    doc_stat['num_doc'] = len(df)
    docs = [[word for word in doc.split() if len(word) > 0] for doc in df.text]
    doc_stat['aver_doc'] = sum([len(doc) for doc in docs])/doc_stat['num_doc']
    doc_stat['median_doc'] = np.median([len(doc) for doc in docs])
    return doc_stat


def label_stats(df):
    '''Provide label stats
    '''
    label_stat = dict()
    value_counts = df['label'].value_counts()
    vc_sum = sum(value_counts)
    for key in df['label'].unique():
        label_stat[key] = {'count':0, 'percentage':0.0}
        label_stat[key]['count'] = value_counts[key]
        label_stat[key]['percentage'] = round(value_counts[key]/vc_sum, 3)
    return pd.DataFrame(label_stat)


def merge_data(dpath1, dpath2):
    with open('all_data.tsv', 'w') as wfile:
        with open(dpath1) as dfile:
            wfile.write(dfile.readline()) # column names
            for line in dfile:
                wfile.write(line)
        with open(dpath2) as dfile:
            dfile.readline()
            for line in dfile:
                wfile.write(line)
    return


def encode_ethnicity(ethnicity):
    if ethnicity == 'white':
        return '0'
    elif ethnicity == 'black':
        return '1'
    elif ethnicity == 'hispanic':
        return '1'
    elif ethnicity == 'asian':
        return '1'
#    elif ethnicity == 'black':
#        return '1'
#    elif ethnicity == 'hispanic':
#        return '2'
#    elif ethnicity == 'asian':
#        return '3'
    
    return 'x'


def encode_age(age):
    if age == 'x':
        return age
    age = float(age)
    
    if age > 30:
        return '1'
    else:
        return '0'
    

def encode_gender(gender):
    if gender == 'x':
        return 'x'
    if gender == 'male':
        return '0'
    else:
        return '1'

    
def encode_country(country):
    if country == 'x':
        return 'x'
    elif country == 'United States':
        return '1'
    else:
        return '0'

    
def encode_region(state):
    
    '''US only, 4 regions'''
    south = {
        'Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina',
        'South Carolina', 'Virginia', 'West Virginia', 'Alabama', 
        'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana',
        'Oklahoma', 'Texas'}
    northeast = {
        'Connecticut', 'Maine', 'New Hampshire', 'Rhode Island', 
        'Vermont', 'New Jersey', 'New York', 'Pennsylvania', 
        'Massachusetts', 'District of Columbia'}
    midwest = {
        'Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 
        'Iowa', 'Kansas', 'Minnesota', 'Montana', 'Nebraska', 
        'North Dakota', 'South Dakota'}
    west = {
        'Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada',
        'New Mexico', 'Utah', 'Wyoming', 'Alaska', 'California', 
        'Hawaii', 'Oregon', 'Washington'}

    if state in south:
        return '0'
    elif state in west:
        return '1'
    elif state in midwest:
        return '2'
    elif state in northeast:
        return '3'
    else:
        return 'x'


def encode_utable(dpath='utable.tsv'):
    with open(dpath) as dfile:
        with open('utable_encoded.tsv', 'w') as wfile:
            wfile.write(dfile.readline())
            for line in dfile:
                line = line.strip().split('\t')
                line[1] = encode_gender(line[1])
                line[2] = encode_age(line[2])
                if line[5] == 'United States':
                    line[4] = encode_region(line[4])
                line[5] = encode_country(line[5])
                if line[6] != 'x':
                    if line[6] == 'white':
                        line[6] = '0'
                    else:
                        line[6] = '1'
                wfile.write('\t'.join(line)+'\n')


def encode_label(label):
    if label in ['neither', 'normal']:
        return '0'
    else:
        return '1'


def encode_data(dpath):
    ''' Encode the demographic attributes
    '''
    new_cols = 'tid\tuid\ttext\tdate\tgender\tage'+\
        '\tregion\tcountry\tethnicity\tlabel\n'
    wfile = open('all_data_encoded.tsv', 'w')
    wfile.write(new_cols)
    
    with open(dpath) as dfile:
        dfile.readline() # skip column names
        for line in dfile:
            
            line = line.strip().split('\t')
            try:
                line[4] = encode_gender(line[4].strip())
                line[5] = encode_age(line[5].strip())
                line[7] = encode_region(line[7].strip())
                line[8] = encode_country(line[8].strip())
                line[9] = encode_ethnicity(line[9].strip())
                line[10] = encode_label(line[10].strip())
            except:
                print(line)
                continue
            
            line = line[:6] + line[7:]
            wfile.write('\t'.join(line)+'\n')
    wfile.flush()
    wfile.close()


def reformat_data(datap, att_idx):
    print('Reformat Data......')
    data = []
    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            # filter out the attributes are all null
            if line[4] == 'x' and line[5] == 'x' and \
                line[6] == 'x' and line[7] == 'x' and line[8] == 'x':
                continue

            line[-1] = int(line[-1]) # label
            
            for att in att_idx:
                if line[att_idx[att][0]] == 'x':
                    continue
                # str to int attributes
                line[att_idx[att][0]] = int(line[att_idx[att][0]])

                # this step is to merge attributes into binary values
                if att_idx[att][1]:
                    for item in att_idx[att][1]:
                        if line[att_idx[att][0]] in att_idx[att][1][item]:
                            line[att_idx[att][0]] = item
                            break
            data.append(line)

    return data


def train_topics(datap):
    # build or load topic model
    dictp = './topic/twitter.dict'
    tmodelp = './topic/twitter.model'
    text_idx = 2 # the column idx of tweets in tsv file
    topic_ns = [5, 10, 15, 20] # number of topics, to choose the best topic k

    # load corpus
    corpus = []
    with open(datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            corpus.append(line[text_idx].split())

    print('Build Dictionary for topic model...')
    if os.path.exists(dictp):
        dictionary = Dictionary.load(dictp)
    else:
        dictionary = Dictionary(corpus)
        dictionary.save(dictp)

    print('Training Topic Models......')
    if os.path.exists(tmodelp):
        best_m = LdaModel.load(tmodelp)
    else:
        # document to indices
        doc_matrix = [dictionary.doc2bow(doc) for doc in corpus]

        # find the best number of topics    
        best_s = -10
        best_m = None
        for idx in range(len(topic_ns)):
            print('Trying topic number: ', topic_ns[idx])
            ldamodel = LdaMulticore(doc_matrix,
                    id2word=dictionary, num_topics=topic_ns[idx],
                    passes=1000, alpha='symmetric', eta=None)

            cm = CoherenceModel(
                model=ldamodel, corpus=doc_matrix, 
                coherence='c_npmi', texts=corpus,
            )
            if cm.get_coherence() > best_s:
                best_s = cm.get_coherence()
                best_m = ldamodel
                best_m.save(tmodelp)

            print(
                'Topic number ' + str(topic_ns[idx]) + \
                ', get coherence: ' + str(cm.get_coherence())
            )

        corpus = None
        del corpus # release memory
    return dictionary, best_m


def get_pos(datap):
    '''Obtain part of speech tags from the Tweebo parser'''
    text_idx = 2 # the column idx of tweets in tsv file
    tweets = []
    tagged_datap = './pos/data_tags.tsv'
    tagged_data = []

    if not os.path.exists(tagged_datap):
        with open(datap) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')
                tweets.append(line[text_idx])

        # tagging
        print('Tagging tweets...')
        tags = []
        last = 0
        for idx in range(10000, len(tweets)+10000, 10000):
            tags.extend(tagger.runtagger_parse(tweets[last:idx]))
            last = idx

        # write a new data file
        print('Creating tagged data...')
        with open(tagged_datap, 'w') as wfile:
            with open(datap) as dfile:
                wfile.write(dfile.readline())
                for idx, line in enumerate(dfile):
                    line = line.strip().split('\t')
                    # filter out the attributes are all null
                    # if line[4] == 'x' and line[5] == 'x' and \
                        # line[6] == 'x' and line[7] == 'x' and line[8] == 'x':
                        # continue

                    line[text_idx] = json.dumps(tags[idx])
                    wfile.write('\t'.join(line)+'\n')

        tags = None
        del tags

    with open(tagged_datap) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            tagged_data.append(line)
        
    return tagged_data


if __name__ == '__main__':
    datap = './all_data_encoded.tsv'
    #word_overlap(datap)
    #get_pos(datap)
    #topic_distance('./all_data_encoded.tsv')
    #pos_distance(datap)
    #predictability(datap)
    #encode_utable(dpath='utable.tsv')

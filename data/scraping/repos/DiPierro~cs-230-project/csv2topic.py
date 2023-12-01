"""
csv2topic.py

The following script drives the TopicVec model executed in topicvecDir.py,
running a series of experiments using different hyperparameters. It is adapted
from the csv2topic.py source code for "Generative Topic Embedding: a Continuous 
Representation of Documents" (ACL 2016) by Shaohua Li, Tat-Seng Chua, Jun Zhu 
and Chunyan Miao. The full implementation is here: https://github.com/askerlee/topicvec
"""
import sys
import pdb
import os
import csv
from collections import OrderedDict
import itertools

from topicvecDir import topicvecDir
from utils import * 

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import gensim.downloader
from gensim.models import Phrases

from numpy import geomspace

def main():
    config = get_base_config()
    glove_vectors = load_glove_embeddings()
    docwords, file_rownames = read_data(config)
    # Set test = True to run in test mode.
    run_experiments(config, docwords, file_rownames, glove_vectors)

def get_base_config():
    """Initializes dictionary to sensible defaults."""
    custom_stopwords = \
        "alameda burlingame cupertino hayward hercules mountain view mtc oakland jose leandro mateo santa clara stockton sunnyvale city council agenda minutes committee commission milpitas meeting authorities city exhibit report attachment recommendation district ordinance code supervisors councilmember supervisor manager seconding authorizing approving"
        
    # Edit this dict with desired hyperparameters to run in test mode
    config = dict(
        csv_filenames = ["/Users/amydipierro/GitHub/cs230/nlp/data/processed/train/train_200.csv"], 
        csv_results = 'experiments/num_topics/train.csv',
        short_name = None,
        unigramFilename = "data/raw/top1grams-wiki.txt",
        word_vec_file = "data/raw/25000-180000-500-BLK-8.0.vec",
        K = 100,
        N0 = 500,
        max_l = 5,
        init_l = 1,
        max_grad_norm = 0,
        # cap the sum of Em when updating topic embeddings
        # to avoid too big gradients
        grad_scale_Em_base = 2500,
        topW = 50,
        topTopicMassFracPrintThres = 0.1,
        alpha0 = 0.1,
        alpha1 = 0.1,
        iniDelta = 0.1,
        MAX_EM_ITERS = 100,
        topicDiff_tolerance = 2e-3,
        printTopics_iterNum = 1,
        zero_topic0 = True,
        useDrdtApprox = False,
        customStopwords = custom_stopwords,
        remove_stop = True,
        normalize_vecs = False,
        # shift all embeddings in a document, so that their average is 0
        rebase_vecs = True,
        rebase_norm_thres = 0.2,
        evalKmeans = False,
        verbose = 1,
        seed = 0,
        logfilename = None,
        test = False # Change to true to run in test mode
    )
    return config

def load_glove_embeddings():
    """Loads pre-trained GloVe embeddings"""
    return gensim.downloader.load('glove-wiki-gigaword-50')

def read_data(config):
    """Reads data from provided csv file of with processed doc text"""
    data = config['csv_filenames'][0]
    docwords = []
    file_rownames = []

    with open(data, 'r') as f:
        csv_text = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
        csv.field_size_limit(sys.maxsize)
        row_num = 0
        for row in csv_text:
            tokens = []
            # Reformat the incoming text
            text = row[-1][2:-2].replace("'", "").split(",")
            for token in text:
                token = token.strip()
                if token != ' ':
                    tokens.append(token)
            # topicvecDir.py needs this nested list format
            # in order to run correctly.
            tokens = [tokens]
            file_rownames.append(data + str(row_num))
            docwords.append(tokens)
            row_num += 1
    
    # Add bigrams
    for outer_list in docwords:
        bigram = Phrases(outer_list)
        for i in range(len(outer_list)):
            for token in bigram[outer_list[i]]:
                if '_' in token:
                    outer_list[i].append(token)
    
    return docwords, file_rownames

def run_experiments(config, docwords, file_rownames, glove_vectors):
    """Configures experiments and runs them in a loop."""
    # Edit this input dictionary to change the experiments to run.
    if not config['test']:
        input_dict = OrderedDict([
            ('num_topics', [100]),
            ('alpha0', geomspace(start=0.05, stop=0.15, num=10)),
            ('alpha1', geomspace(start=0.05, stop=0.15, num=10)),
            ('delta', geomspace(start=0.05, stop=0.15, num=10))
        ]) 
        
        for param in input_dict:
            for i, val in enumerate(input_dict[param]):
                path = 'experiments/{}/train.csv'.format(param)
                config['csv_results'] = os.path.join(os.getcwd(), path)
                config['short_name'] = param
                config['logfilename'] = '{}_{}'.format(param, val)
                if param == 'num_topics':
                    config['K'] = val
                elif param == 'alpha0':
                    config['alpha0'] = val
                elif param == 'alpha1':
                    config['alpha1'] = val
                elif param == 'delta':
                    config['iniDelta'] = val
                topicvec = run_topicvec(config, docwords, file_rownames)
                results_dict = get_coherence(config, topicvec, docwords, glove_vectors)
                write_results(config, results_dict)
                eval_results(config, glove_vectors, param)
    else:
        path = 'experiments/num_topics/train.csv'
        config['csv_results'] = os.path.join(os.getcwd(), path)
        config['short_name'] = 'test'
        config['logfilename'] = 'test'
        topicvec = run_topicvec(config, docwords, file_rownames)
        results_dict = get_coherence(config, topicvec, docwords, glove_vectors)
        write_results(config, results_dict)
        eval_results(config, glove_vectors, param)

def run_topicvec(config, docwords, file_rownames):
    """Runs the TopicVec pipeline. Adapted from the original open source code."""
    topicvec = topicvecDir(**config)
    topicvec.setDocs( docwords, file_rownames )    
    best_last_Ts, Em, docs_Em, Pi = topicvec.inference()

    basename = os.path.basename(config['logfilename'])
    basetrunk = os.path.splitext(basename)[0]

    best_it, best_T, best_loglike = best_last_Ts[0]
    save_matrix_as_text( basetrunk + "-em%d-best.topic.vec" %best_it, "topic", best_T  )

    if best_last_Ts[1]:
        last_it, last_T, last_loglike = best_last_Ts[1]
        save_matrix_as_text( basetrunk + "-em%d-last.topic.vec" %last_it, "topic", last_T  )
    
    return topicvec

def get_coherence(config, topicvec, docwords, glove_vectors):
    """Calculate UMass and w2v (GloVe) coherence scores."""

    # Clean docwords
    docs = []
    for word_list in docwords:
        doc = word_list[0]
        docs.append(doc)
    
    # Clean topics
    byte_topics = topicvec.printTopWordsInTopics(topicvec.docs_theta, True)
    word_topics = []
    one_word = []
    for topic in byte_topics:
        new_topic = []
        for word in topic:
            if type(word) != str:
                word = word.decode()
            new_topic.append(word)
            one = [word]
            one_word.append(one)
        word_topics.append(new_topic)

    # Get dictionary
    vocab_dict = Dictionary(docs[1:])
    # Make sure words in topics are in the dictionary
    vocab_dict.add_documents(one_word) 
    
    # Get corpus
    corpus = [vocab_dict.doc2bow(doc) for doc in docs]
    
    # Calculate UMass coherence score
    # The closer to 0, the more coherent
    cm = CoherenceModel(topics=word_topics, corpus=corpus, dictionary=vocab_dict, coherence='u_mass')
    umass_coherence = cm.get_coherence() 

    # Calculate GloVe coherence score
    # Ranges between 0 and 1
    # The closer to 1, the better
    cm = CoherenceModel(topics=word_topics, corpus=corpus, dictionary=vocab_dict, coherence='c_w2v', keyed_vectors=glove_vectors)
    glove_coherence = cm.get_coherence()
    
    # Log coherence score and other metrics
    results_dict = OrderedDict([
        ('num_topics', config['K']),
        ('alpha0', config['alpha0']),
        ('alpha1', config['alpha1']),
        ('delta', config['iniDelta']),
        ('umass', umass_coherence),
        ('glove', glove_coherence)
    ])

    return results_dict

def write_results(config, results_dict):
    """Export results to a csv"""
    # Write out our results
    csv_path = config['csv_results']
    new_file = not os.path.exists(csv_path)
    
    with open(csv_path, 'a') as out_file:       
        dict_writer = csv.DictWriter(out_file, results_dict.keys())
        if new_file:
            dict_writer.writeheader()
        dict_writer.writerow(results_dict)

def eval_results(config, glove_vectors, param):
    """Evaluate the results on the dev set."""
    if not config['test']:
        config['csv_filenames'] = ["data/processed/dev/dev_200.csv"]
        config['csv_results'] = config['csv_results'].replace('train', 'dev')
    else:
        config['csv_filenames'] = ["data/processed/test/test_200.csv"]
        config['csv_results'] = config['csv_results'].replace('train', 'test')
    docwords, file_rownames = read_data(config)
    topicvec = run_topicvec(config, docwords, file_rownames)
    results_dict = get_coherence(config, topicvec, docwords, glove_vectors)
    write_results(config, results_dict)
    
if __name__ == '__main__':
    main()
  

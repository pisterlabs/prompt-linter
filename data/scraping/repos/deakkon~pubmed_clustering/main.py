import numpy as np
np.random.seed(42)

import sys
import argparse

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from utils.preprocess import preprocess_text
from collections import Counter
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

from utils.exploration import Exploration
from utils.transformers import TokenizePreprocessor, sentence_tokenize, vectorizer
import pandas as pd
import time

from sklearn.externals import joblib

preprocess_text = preprocess_text()
tokenizer = TokenizePreprocessor()
ex = Exploration()

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Add some integers.')
    parser.add_argument('-m',
                        '--mode',
                        choices=['train', 'test', 'external'],
                        help='Run the pipeline on the train, test or external dataset; \n "train" invokes training while test and external invoke inference on test or external dataset respectively.')

    parser.add_argument('-i',
                        '--in',
                        help='Path to input file. Use only if --mode == external.')

    parser.add_argument('-n',
                        '--name',
                        help='Name of the output file. In case of none, external_timestamp is used. Results are stored in report/results/')

    parser.add_argument(
                        '-k',
                        '--nr_tokens',
                        help='Number of most frequent tokens in individual clusters to show.',
                        default=10
            )

    parser.add_argument('-fi',
                        '--feature_input',
                        choices=[['title'],
                                 ['abstract'],
                                 ['title', 'abstract'],
                                 ['title', 'NE'],
                                 ['title', 'abstract', 'NE']],
                        default=['title', 'abstract', 'NE'],
                        help='Input fields. Defaults to ["title", "abstract", "NE"].')


    args = vars(parser.parse_args())

    if not args['mode']:
        raise ValueError("No mode parameter supplied. 'train' invokes training and inference on the ground truth data set. test and external invoke inference on test or external dataset respectively.")

    if args['mode'] == "external" and not args['in']:
        raise ValueError("For evalution of an external dataset, make sure a path for an input file is passed.")

    k = int(args['nr_tokens'])

    if args['mode'] == 'train':
        gold_data = preprocess_text.get_text(preprocess_text.gold_data_labeled.PMID.values.tolist(), 'gold_text',preprocess_text.gold_data_labeled.Label.values.tolist())
        labels_true = preprocess_text.gold_data_labeled.Label.values.tolist()

        ex.ground_truth_cluster_analysis(gold_data,
                                        labels_true,
                                        ['lsa','tfidf', "lda"],
                                        ['title'],
                                        ['abstract'],
                                        ['title', 'abstract'],
                                        ['title', 'NE'],
                                        ['title', 'abstract', 'NE']
                                         )

        ex.inference(gold_data, args['feature_input'], "ground_truth", k, labels=labels_true)

    if args['mode'] == 'test':

        test_data = preprocess_text.get_text(preprocess_text.test_data.PMID.values.tolist(), 'test_data')
        ex.inference(test_data, args['feature_input'], "test_set", k)

    if args['mode'] == 'external':

        if not args['name']:
            file_name = "external_{}".format(time.strftime("%Y%m%d_%H%M%S"))
        else:
            file_name = args['name']

        df = pd.read_csv(args['in'], sep='\t', header=None, names=['PMID'])

        data = preprocess_text.get_text(preprocess_text.test_data.PMID.values.tolist(), file_name)
        ex.inference(data, args['feature_input'], file_name, k)

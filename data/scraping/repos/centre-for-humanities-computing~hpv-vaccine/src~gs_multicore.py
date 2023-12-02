'''
Gensim's LdaMulticore.
Fit & save many models, find a good one.
'''
import os
import gc
import argparse
from itertools import chain
from time import time

import ndjson

from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis.gensim

from src.utility.general import make_folders


def lda_MULTICORE(texts,
                  n_topics_range, iterations, passes, workers,
                  out_dir, verbose=True, save_doc_top=True):
    '''Fit topic models and search for optimal hyperparameters.

    Dirtier, multicore version for faster running of HOPE stuff.


    Parameters
    ----------
    texts : list
        preprocessed corpus, where texts[0] is a document
        and texts[0][0] is a token.

    n_topics_range : range of int
        range of integers to use as the number of topics
        in interations of the topic model.

    iterations : int
        maximum number of iterations for each topic models

    passes : int
        maximum number of passes (start iterations again) for each topic models

    workers : int
        how many CPUs to initiate?

    out_dir : str
        path to a directory, where results will be saved (in a child directory).

    verbose : bool
        give comments about the progress?

    save_doc_top : bool
        save documet-topic matices from models?


    Exports
    -------
    out_dir/report_lines/*
        pickled dict with model information
        (n topics, model coherence, per-topic coherence, hyperparameters)
        
    out_dir/models/*
        gensim objects, where the model is saved.
        
    out_dir/plots/*
        pyLDAvis visualizations of the model
    '''
    # check how legit out_dir is
    make_folders(out_dir)

    # if a single model is to be fitted,
    # make sure it can be "iterated"
    if isinstance(n_topics_range, int):
        n_topics_range = [n_topics_range]

    # input texts to gensim format
    dictionary = corpora.Dictionary(texts)
    bows = [dictionary.doc2bow(tl) for tl in texts]

    # iterate
    report_list = []
    for n_top in chain(n_topics_range):

        if verbose:
            print("{} topics".format(n_top))

        start_time = time()

        # paths for saving
        ## it's not very elegant defining the paths here
        ## after there already is funciton make_folders
        filename = str(n_top) + "T_" + 'ASM'
        report_path = os.path.join(
            out_dir,
            'report_lines',
            filename + '.ndjson'
        )

        model_path = os.path.join(
            out_dir,
            'models',
            filename + '.model'
        )

        pyldavis_path = os.path.join(
            out_dir,
            'plots',
            filename + '_pyldavis.html'
        )

        doctop_path = os.path.join(
            out_dir,
            'doctop_mats',
            filename + '_mat.ndjson'
        )

        model = LdaMulticore(
            corpus=bows,
            num_topics=n_top,
            id2word=dictionary,
            workers=workers,
            chunksize=2000,
            passes=passes,
            batch=False,
            alpha='symmetric',
            eta=None,
            decay=0.5,
            offset=1.0,
            eval_every=10,
            iterations=iterations,
            gamma_threshold=0.001,
            random_state=None,
            minimum_probability=0.01,
            minimum_phi_value=0.01,
            per_word_topics=False,
        )

        # track time usage
        training_time = time() - start_time
        if verbose:
            print('    Time: {}'.format(training_time))

        # coherence
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            corpus=bows,
            coherence='c_v'
        )

        coh_score = coherence_model.get_coherence()
        coh_topics = coherence_model.get_coherence_per_topic()

        if verbose:
            print('    Coherence: {}'.format(coh_score.round(2)))

        # save priors
        alpha = model.alpha.tolist()
        eta = model.eta.tolist()

        # save report
        report = (n_top, alpha, eta, training_time, coh_score, coh_topics)
        report_list.append(report)
        with open(report_path, 'w') as f:
            ndjson.dump(report, f)

        # save model
        model.save(model_path)

        # produce a visualization
        # it is imperative that sort_topics should never be turned on!
        vis = pyLDAvis.gensim.prepare(
            model, bows, dictionary, sort_topics=False
        )

        pyLDAvis.save_html(vis, pyldavis_path)

        # save document-topic matrix
        if save_doc_top:
            # keep minimum_probability at 0 for a complete matrix
            doc_top = [model.get_document_topics(doc, minimum_probability=0)
                       for doc in model[bows]]

            # unnest (n topic, prob) tuples
            # float to convert from np.float32 which is not
            # JSON serializable
            doc_top_prob = [
                [float(prob) for i, prob in doc]
                for doc in doc_top
            ]

            # save the matrix as ndjson
            with open(doctop_path, 'w') as f:
                ndjson.dump(doc_top_prob, f)
        
        gc.collect()

    return None


def gridsearch(args):
    # load data
    print('[info] Importing data form {}'.format(args['data']))
    with open(args['data']) as f:
        texts = ndjson.load(f)
        texts = [doc['text'] for doc in texts]

    n_topics_range = range(5, 55, 5)
    
    lda_MULTICORE(
        texts=texts,
        n_topics_range=n_topics_range,
        iterations=2000,
        passes=2,
        workers=int(args['workers']),
        out_dir=args['batchname'],
        verbose=True,
        save_doc_top=True
    )

    return None


def arguments():
    '''
    python3 gs_multicore.py -d data/AAA.ndjson -bn models/200921_AAA_mc -w 4
    '''
    # init
    ap = argparse.ArgumentParser(description="Tmux friendly driver for LDA multicore grid search")
    # possible arguments
    ap.add_argument("-d", "--data", required=True, help="path to texts to process")
    ap.add_argument("-bn", "--batchname", required=True, help="folder to save models to")
    ap.add_argument("-w", "--workers", required=False, default=32, help="number of CPUs to initiate")
    # parse
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = arguments()
    gridsearch(args)

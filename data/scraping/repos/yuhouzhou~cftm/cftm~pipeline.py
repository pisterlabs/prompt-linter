import cftm_parser
import preprocessing as pp
from spacy.lang.de.stop_words import STOP_WORDS
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pickle
import pyLDAvis.gensim
import webbrowser
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import datetime
import yaml
import ntpath
import logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'pipeline',
        type=int, nargs=3,
        help=('Define pipeline: \n'
              '    Require 3 numbers. Each number switches on or off a part of the pipeline.\n'
              '    The first position is preprocessing,\n'
              '    the second position is modelling,\n'
              '    the third position is visualization.\n'
              '    Enter 1 to turn on, 0 to turn off the specific part of pipeline.\n'
              '    Example: 1 1 1 to run the full pipeline,\n'
              '    1 1 0 to run preprocessing and modelling but without visualization.\n'
              '    ')
    )

    parser.add_argument(
        '-p', '--path_file',
        type=str, nargs='?', const='./path.yaml', default='./path.yaml'
    )

    parser.add_argument(
        '-o', '--observation_n',
        type=int, nargs='?', const=-1, default=-1
    )

    parser.add_argument(
        '-m', '--agg_metadata',
        type=str, nargs='?', const='DATE', default='DATE'
    )

    parser.add_argument(
        '-l', '--agg_length',
        type=int, nargs='?', const=-1, default=-1
    )

    parser.add_argument(
        '-r', '--topic_range',
        type=int, nargs=3, default=[1, 10, 1]
    )

    parser.add_argument(
        '-c', '--chunksize',
        type=int, nargs='?', const=2000, default=2000
    )

    parser.add_argument(
        '-i', '--iterations',
        type=int, nargs='?', const=50, default=50
    )

    parser.add_argument(
        '-s', '--seed',
        type=int, nargs='?', const=1, default=1
    )

    args = parser.parse_args()

    parquet_path1, parquet_path2, archive_fore_path, data_path, model_path, pic_path, html_path, log_path = yaml.load(
        open(args.path_file)).values()

    dt = str(datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S'))
    archive_path = archive_fore_path + dt + '/'
    if not os.path.exists(archive_fore_path):
        os.mkdir(archive_fore_path)
    os.mkdir(archive_path)

    with open(archive_path + 'commandline_args.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

    preproc, modelling, visualization = args.pipeline
    if preproc:
        # Parsing
        df_pd = cftm_parser.parquet_transform(parquet_path1, parquet_path2, n=args.observation_n)

        # Pre-processing
        stopwords = list(STOP_WORDS)
        texts, dictionary, corpus = pp.preprocessor(df_pd, stopwords=stopwords, language='de', text='TEXT',
                                                    metadata=args.agg_metadata, min_len=args.agg_length)
        training_data = {"texts": texts, "dictionary": dictionary, "corpus": corpus}
        pickle.dump(training_data, open(data_path, 'wb'))
        pickle.dump(training_data, open(archive_path + ntpath.split(data_path)[1], 'wb'))
    elif modelling or visualization:
        try:
            training_data = pickle.load(open(data_path, 'rb'))
            texts, dictionary, corpus = training_data['texts'], training_data['dictionary'], training_data['corpus']
        except FileNotFoundError:
            print("> Training Data Not Found!")
            exit(1)

    if modelling:
        # Logger
        logging.basicConfig(filename=archive_path + ntpath.split(log_path)[1],
                            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # Model Generation
        lda_lst = []
        coherence_lst = []
        n_topic_min = args.topic_range[0]
        n_topics_max = args.topic_range[1]
        step = args.topic_range[2]
        print("> Topic modelling started at ", datetime.datetime.now())
        try:
            for i in tqdm(range(n_topic_min, n_topics_max + 1, step)):
                # Data Modelling
                # Gensim LDA model set update_every=1 by default, meaning it uses Online LDA;
                # The algorithm update the model after read every chunk-size number of documents.
                # If you set passes > 1, then
                # the algorithm will read through the whole corpus multiple times.
                # It is a similar parameter with 'epoch' in neural networks.

                MODE = 'multicore'
                if MODE == 'distributed':
                    lda = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary, distributed=False,
                                   update_every=1, chunksize=args.chunksize, passes=1, iterations=args.iterations,
                                   random_state=args.seed, eval_every=None)
                elif MODE == 'multicore':
                    lda = LdaMulticore(corpus=corpus, num_topics=i, id2word=dictionary, chunksize=args.chunksize, workers=3,
                                       passes=1, iterations=args.iterations, random_state=args.seed)

                lda_lst.append(lda)

                # Model Evaluation
                cm = CoherenceModel(model=lda, texts=texts, corpus=corpus, dictionary=dictionary,
                                    coherence='c_v')
                coherence = cm.get_coherence()
                coherence_lst.append(coherence)

                lda_pickle = {"model_lst": lda_lst, "coherence_lst": coherence_lst,
                              "n_topics_min": n_topic_min, "n_topics_max": n_topics_max, "step": step}
                pickle.dump(lda_pickle, open(model_path, 'wb'))
                pickle.dump(lda_pickle,
                            open(archive_path + ntpath.split(model_path)[1], 'wb'))
        # If the modelling time is too long, the program can be interrupted by keyboard.
        # All the generated content will be saved.
        # TODO:
        #  Does not work due to scipy BUG
        #  https://github.com/ContinuumIO/anaconda-issues/issues/905
        #  and
        #  https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
        except KeyboardInterrupt:
            if len(lda_lst) == len(coherence_lst):
                print("> The modelling processing is stopped, but generated Models and their coherence are saved.")
            else:
                print("> The modelling processing is stopped. Generated Models are saved, "
                      "but the coherence of the last model is failed to save.")

    elif visualization:
        try:
            lda_pickle = pickle.load(open(model_path, 'rb'))
            lda_lst, coherence_lst, n_topic_min, n_topics_max, step = lda_pickle.values()
        except FileNotFoundError:
            print("> Model Not Found!")
            exit(1)

    if visualization:
        print("> Plot Topic Coherence vs. Number of Topics started...")
        # Plot Topic Coherence
        index = int(np.argmax(coherence_lst))
        lda = lda_lst[index]
        plt.scatter(range(n_topic_min, n_topics_max + 1, step), coherence_lst, s=5)
        plt.scatter(n_topic_min + index * step, coherence_lst[index], color='r')
        plt.annotate(str(n_topic_min + index * step) + ', ' + str(coherence_lst[index]),
                     (n_topic_min + index * step, coherence_lst[index]))
        plt.title('Topic Coherence vs. Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('Topic Coherence (By $C_V$)')
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        plt.savefig(pic_path)
        plt.savefig(archive_path + ntpath.split(pic_path)[1])
        print("> Plot Topic Coherence vs. Number of Topics finished!")

        # Data Visualization
        # TODO: When use all the topics, it consumes much memory.
        print("> pyLDAvis started...")
        vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary=dictionary, mds='mmds')
        pyLDAvis.save_html(vis, html_path)
        pyLDAvis.save_html(vis, archive_path + ntpath.split(html_path)[1])
        webbrowser.open(os.path.abspath(html_path), new=2)
        print("> pyLDAvis finished!")

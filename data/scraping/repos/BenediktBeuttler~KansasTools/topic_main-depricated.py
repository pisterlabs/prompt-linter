from HelperFunctions import *
import stanza

import sys

import pickle
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

from pprint import pprint

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Main Resource:
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Other Resources
# https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
# http://dsspace.wzb.eu/pyug/topicmodeling2/slides.html
# https://de.dariah.eu/text-analysis-with-topic-models
# https://blog.codecentric.de/2017/01/topic-modeling-codecentric-blog-articles/
# http://inhaltsanalyse-mit-r.de/themenmodelle.html

if __name__ == '__main__':

    # General settings
    malletPath = "/Users/zweiss/Documents/mallet-2.0.8/bin/mallet"
    loadDataFromCommandLine = False  # change here to enable/disable hard coded settings
    be_verbose = True
    use_pickled_data = False
    save_as_pickle = True
    use_mallet_model = True
    nTopics = 10

    pos_list = ["NOUN"] #if use_only_nouns else ["NOUN", "VERB", "ADV", "ADJ"]
    lemmaPickleFile = "corpusLemmas-nouns.p" #if use_only_nouns_and_verbs else "corpusLemmas.p"
    fileNamePickleFile = "corpusFileNames-nouns.p"# if use_only_nouns_and_verbs else "corpusFileNames.p"
    bigramPickleFile = "corpusBigrams-nouns.p"# if use_only_nouns_and else "corpusBigrams.p"
    trigramPickleFile = "corpusTrigrams-nouns.p" #if use_only_nouns_and else "corpusTrigrams.p"
    modelListFile = "modelList-nouns.p" #if use_only_nouns_and else "modelList.p"
    modelCoherenceFile = "modelCoherence-nouns.p"# if use_only_nouns_and else "modelCoherence.p"

    modelTopicFile = "topic_classification-"

    # STEP 0: Obtain user settings
    if loadDataFromCommandLine:
        if len(sys.argv) < 5:
            print("Call:\n> python3 topic_main-depricated.py STR_INDIR STR_OUTDIR BOOL_USE_PICKLED_DATA SAVE_AS_PICKLE")
            sys.exit(0)
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
        use_pickled_data = sys.argv[3].lower() == "true"
        save_as_pickle = sys.argv[4].lower() == "true"
    else:
        if be_verbose:
            print("Using hard coded settings")
        in_dir = "/Users/zweiss/Documents/Forschung/Projekte/KANSAS/data/alpha-corpus2/"
        out_dir = "/Users/zweiss/Dropbox/current-research/KANSAS-TopicModeling/"
    os.makedirs(out_dir, exist_ok=True)

    # STEP 1: Get NLP annotation of all data files and extract content lemmata
    if use_pickled_data:
        if be_verbose:
            print("Loading pickled NLP")
        corpus_lemmas = pickle.load(open(lemmaPickleFile, "rb"))
        corpus_file_names = pickle.load(open(fileNamePickleFile, "rb"))
    else:
        if be_verbose:
            print("Starting NLP annotation")
        nlp = stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma')  # initialize pipeline
        all_corpus_files = rec_file_list(in_dir, file_ending="-clean.txt")  # get all files from input dir
        corpus_lemmas = []
        corpus_file_names = []
        nfiles = len(all_corpus_files)
        for i, c_file in enumerate(all_corpus_files):
            if be_verbose:
                print("... {}/{}".format(i, nfiles))
            c_text = load_text(c_file)
            # ignore empty files
            if len(c_text) == 0:
                continue
            doc = nlp(c_text)

            content_lemmas = extract_content_lemmas(annotated_document=doc,
                                                    set_content_pos=pos_list)
            corpus_lemmas.append(content_lemmas)
            corpus_file_names.append(c_file)
        if be_verbose:
            print("NLP annotation complete: {} annotations / {} file names".format(len(corpus_lemmas),
                                                                                   len(corpus_file_names)))
    # save documents
    if save_as_pickle:
        if be_verbose:
            print("Pickling corpus data")
        pickle.dump(corpus_lemmas, open(lemmaPickleFile, "wb"))
        pickle.dump(corpus_file_names, open(fileNamePickleFile, "wb"))

    # STEP 2: Calculate bigrams and trigrams
    if use_pickled_data:
        if be_verbose:
            print("Loading pickled ngrams")
        data_words_bigrams = pickle.load(open(bigramPickleFile, "rb"))
        data_words_trigrams = pickle.load(open(trigramPickleFile, "rb"))
    else:
        if be_verbose:
            print("Calculating ngrams")
        data_words_bigrams = calculate_ngram_model(txt_data=corpus_lemmas, min_count=5, threshold=100)
        data_words_trigrams = calculate_ngram_model(txt_data=data_words_bigrams, threshold=100)
    if save_as_pickle:
        if be_verbose:
            print("Pickling ngrams")
        pickle.dump(data_words_bigrams, open(bigramPickleFile, "wb"))
        pickle.dump(data_words_trigrams, open(trigramPickleFile, "wb"))

    # STEP 3: Create the Dictionary
    id2word = corpora.Dictionary(data_words_trigrams)
    texts = data_words_trigrams  # Create Corpus
    corpus = [id2word.doc2bow(text) for text in texts]  # Term Document Frequency

    # View
    if be_verbose:
        # For example, (0, 1) in id2word[:1] implies, word id 0 occurs once in the first document.
        # Likewise, word id 1 occurs twice and so on.
        # If you want to see what word a given id corresponds to, pass the id as a key to the dictionary.
        # Human readable format of corpus (term-frequency)
        print([[(id2word[cid], freq) for cid, freq in cp] for cp in corpus[:1]])

    # STEP 4: Find the right number of topics
    # Can take a long time to run.
    if use_pickled_data:
        model_list = pickle.load(open(modelListFile, "rb"))
        coherence_values = pickle.load(open(modelCoherenceFile, "rb"))
    else:
        model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=corpus_lemmas,
                                                                mallet_path=malletPath, start=2, limit=nTopics, step=2)
    if save_as_pickle:
        if be_verbose:
            print("Pickling topic models")
        pickle.dump(model_list, open(modelListFile, "wb"))
        pickle.dump(coherence_values, open(modelCoherenceFile, "wb"))

    # Show graph
    limit = nTopics;
    start = 2;
    step = 2;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    plt.savefig('ntopics.png')
    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    # either 10 (idx 4) or 22 (idx 10) or 34 (idx 16) topics!

    # MANUAL SELECTION of 3 topic numbers
    model_idx_to_be_compared = [1, 2, 3]

    # TODO this is not working yet

    # STEP 5 get model with each potential topic distrubition
    df_dominant_topic = []
    for i, current_number_topics in enumerate(model_idx_to_be_compared):
        if use_pickled_data:
            df_dominant_topic[i] = pickle.load(open(modelTopicFile[i]+".p", "rb"))
        else:
            # Save topics for highest selected number of topics
            df_dominant_topic[i] = format_topic_df(optimal_model=model_list[current_number_topics],
                                                   corpus=corpus,
                                                   texts=corpus_lemmas,
                                                   corpus_file_names=corpus_file_names)
            df_dominant_topic[i].to_csv(os.path.join(out_dir, modelTopicFile[i]+".csv"), sep="\t", index=False)

            if save_as_pickle:
                pickle.dump(df_dominant_topic[i], open(modelTopicFile[i]+".p", "wb"))


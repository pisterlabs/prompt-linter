import os
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pandas as pd
import numpy as np


def load_text(in_file):
    with open(in_file, 'r', encoding='utf-8') as instr:
        rval = instr.read()
    return rval.strip()


def rec_file_list(in_dir, file_ending=".txt"):
    rval = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if not f.endswith(file_ending) or f.startswith("."):
                continue
            rval.append(os.path.join(root, f))
    return rval


def calculate_ngram_model(txt_data, min_count=-1, threshold=100):
    # higher threshold fewer phrases
    if min_count != -1:
        ngram = gensim.models.Phrases(txt_data, min_count=min_count, threshold=threshold)
    else:
        ngram = gensim.models.Phrases(txt_data, threshold=threshold)

    # Faster way to get a sentence clubbed as a trigram/bigram
    ngram_mod = gensim.models.phrases.Phraser(ngram)

    rval = [ngram[doc] for doc in txt_data]
    return rval


def extract_content_lemmas(annotated_document, set_content_pos):
    return [word.lemma for sent in annotated_document.sentences for word in sent.words if word.pos in set_content_pos]


def compute_coherence_values(dictionary, corpus, texts, limit, mallet_path, start=2, step=3,
                             be_verbose=False):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    mallet_path : Path to mallet

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence model
    """
    coherence_models = []
    model_list = []
    for num_topics in range(start, limit, step):
        model, coherencemodel = compute_coherence_value(dictionary=dictionary, corpus=corpus, texts=texts,
                                                        num_topics=num_topics, mallet_path=mallet_path)

        if be_verbose:
            print("{}N.Topics={}: Coherence.Score={}{}".format("=" * 500, num_topics, get_aggregate_coherence(coherencemodel),
                                                               "=" * 500))
        model_list.append(model)
        coherence_models.append(coherencemodel)
    return model_list, coherence_models


def get_aggregate_coherence(c_coherence_model):
    return np.nanmean(c_coherence_model.get_coherence_per_topic())


def compute_coherence_value(dictionary, corpus, texts, num_topics, mallet_path):
    """
    Compute c_v coherence for a given number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    num_topics : Num of topics
    mallet_path : Path to mallet

    Returns:
    -------
    model : LDA topic model
    coherence_value : Coherence model corresponding to the LDA model
    """
    model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    return model, coherencemodel


def format_topics_sentences(ldamodel, corpus, texts, corpus_file_names, keep_all_topics=True,
                            verbose=False):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    ntotal = len(ldamodel[corpus])
    for i, row in enumerate(ldamodel[corpus]):
        if verbose:
            print("progress ... {}/{}".format(i, ntotal))
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if keep_all_topics or j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num),
                                                                  round(prop_topic,4),
                                                                  topic_keywords,
                                                                  corpus_file_names[i]]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'File_Name']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def format_topic_df(optimal_model, corpus, texts, corpus_file_names, keep_all_topics=True,
                            verbose=False):
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts,
                                                      keep_all_topics=keep_all_topics,
                                                      corpus_file_names=corpus_file_names, verbose=verbose)
    df_topic_sents_keywords = df_topic_sents_keywords.reset_index()
    #df_topic_sents_keywords.columns = ['Index', 'Document.Id', 'Dominant.Topic', 'Topic.Perc.Contrib', 'Topic.Keywords',
    #                                   'File.Name', 'Text']
    return df_topic_sents_keywords


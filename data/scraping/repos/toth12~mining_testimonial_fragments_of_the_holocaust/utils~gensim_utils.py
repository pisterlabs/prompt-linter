f"""Functions to use Gensim."""
from gensim.corpora import Dictionary
import pandas as pd
from gensim.models import Word2Vec
from gensim import utils
import string
from nltk.corpus import stopwords
import constants
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers import LdaMallet
import numpy as np
from gensim.models import CoherenceModel
import matplotlib as plt
plt.use('PS')


def initialize_gensim_dictionary(text):
    """Initialize a gensim dictionary model.

    Parameters
    ----------
    text : {list}
        List of tokens as bag of words.

    Returns
    -------
     Gensim dictionary model
    """
    dct = Dictionary(text)
    return dct


def add_documents_to_gensim_dictionary(gensim_dictionary_model, text):
    """Update an existing gensim dictionary model with new texts.

    Function does not return the model; it just updates it.

    Parameters
    ----------
    gensim_dictionary_model : {gensim dictionary model}
    text : {list}
        List of tokens as bag of words.
    """
    gensim_dictionary_model.add_documents(text)


def get_gensim_dictionary(gensim_dictionary_model):
    """Print the vocabulary in a gensim dictionary model.

    Parameters
    ----------
    gensim_dictionary_model : {gensim_dictionary_model}

    Returns
    -------
    [list]
        List of individual words in the model in the order of indices.
    """
    return list(gensim_dictionary_model.token2id.keys())


def get_df_in_dictionary(gensim_dictionary_model, as_pandas_df=False):
    """Get the document frequency of each word in a gensim dictionary model.

    Before applying this method, it is recommended to cut the vocabulary.

    Parameters
    ----------
    gensim_dictionary_model : {gensim_dictionary_model}
    as_pandas_df : {bool}, optional
        If true returns the document frequencies in a panda dataframe.

    Returns
    -------
    Panda Dataframe or Dictionary.
        Document frequency of each word in a model as panda dataframe or
        dictionary.
    """
    look_up = gensim_dictionary_model.dfs
    dictionary = get_gensim_dictionary(gensim_dictionary_model)
    result = {}
    for i, word in enumerate(dictionary):
        result[word] = look_up[i]

    if as_pandas_df:
        dfobj = pd.DataFrame(result.items())
        return dfobj
    else:
        return result


def load_gensim_dictionary_model(path_to_gensim_dictionary_model):
    """Load a gensim dictionary model.

        Parameters
    ----------
    path_to_gensim_dictionary_model : {str}
        Absolute path to a gensim dictionary model.

    Returns
    -------
    Gensim dictionary model.
    """
    dct = Dictionary.load(path_to_gensim_dictionary_model)
    return dct


def build_gensim_synset_model_from_sentences(sentences, window=5):
    r"""Construct a gensim synset model from sentences that are list of tokens.

    The default number of dimensions in the model is 100. \
    Words below count 5 are dropped. Words not beginning with \
    letters of the alphabet are trimmed.

    Parameters
    ----------
    sentences : {[type]}
        [description]
    window : {number}, optional
        [description] (the default is 5, which [default_description])

    Returns
    -------
    [type]
        [description]
    """
    model = Word2Vec(
        sentences, size=100, window=5, min_count=5, workers=4,
        trim_rule=trim_rule)
    return model


def trim_rule(word, count, min_count):
    r"""Define rules to trim vocabulary.

    It drops all words the first character of \
    which is not an element of alphabet when constructing a gensim \
    synset model. Can be used as argument of the building process.

    Parameters
    ----------
    word : {str}
    count : {int}
    min_count : {int}

    Returns
    -------
    gensim.utils.RULE_DISCARD
    """
    if ((word[0] not in string.ascii_uppercase + string.ascii_lowercase) or
            (word in set(stopwords.words('english')))):
        return utils.RULE_DISCARD


def initialize_gensim_synset_model_with_dictionary(dictionary, window):
    r"""Use a gensim dictionary model gensim the vocabulary of a synset model.

    Method iterates through the dictionary to extract vocab without trimming \
    or count. Vocabulary preprocessing is needed (minimum count is set to 1).\
    The synset model is initialized, hence window is to set here. Use the \
    model to train the synset model with sentences.


    Parameters
    ----------
    dictionary : {gensim dictionary}
    window : {int}
        Window used to train the synset model.

    Returns
    -------
    gensim synset model
        Synset model initialized with the vocab.
    """
    result = []
    for i, word in enumerate(dictionary):
        result.append([dictionary[word]])
    model = Word2Vec(min_count=1, window=int(window), sorted_vocab=1)
    model.build_vocab(result)
    return model


def find_similar_terms(term, path_to_model, n=10):
    r"""Find similar terms to a given term in a trained gensim model.

    If term is not present it returns an emty dictionary.

    Parameters
    ----------
    term : {str}
        The term the synonym of which is searched.
    path_to_model : {str}
        Absolute path to the model to be loaded.
    n : {int}, optional
        Number of closest terms to return(the default is 10)

    Returns
    -------
    list of tuples
        List of n closest terms with cosine similarity. Each term is a tuple.
    """
    model = Word2Vec.load(path_to_model)
    similar_terms = model.wv.most_similar(term, topn=n)
    return similar_terms


def build_gensim_phrase_model_from_sentences(sentences,
                                             threshold=2, min_count=5):
    r"""Build a gensim phrase model from a list of tokenized sentences.

    Sensitivity of the model can be adjust by the treshold value.

    Parameters
    ----------
    sentences : {list of lists or iterable as list of lists}
        List of sentences that are lists of tokens or an iterable with \
        this structure.
    min_count : {int}
        (the default is 5)
    threshold : {int}, optional
        Adjust how sensitive the model is (the default is 2)

    Returns

    genim phrase model
        Returns a trained phrase model
    """
    phrases = Phrases(sentences, min_count=5, threshold=threshold)
    return phrases


def identify_phrases(sentence, path_to_gensim_phrase_model):
    """Identify multiword expression by a trained phrase model.

    Parameters
    ----------
    sentence : {list}
        list with tokens as elements
    path_to_gensim_phrase_model : {str}
        Absolute path to the model.

    Returns
    -------
    list
        List with tokens as elements.
    """
    phrase_model = Phrases.load(path_to_gensim_phrase_model)
    phraser_model = Phraser(phrase_model)
    new_sentence = phraser_model[sentence]
    return new_sentence


def train_lda_topic_model_with_mallet(texts, path_mallet,
                                      terms_to_remove=[], num_topics=50,
                                      no_below=10, no_above=0.9,
                                      scoring=False, start=2, step=3):
    r"""Train an LDA topic model from texts as bag of words.

    Texts are to be preprocessed. Terms (for instance, stopwords) to be /
    excluded can be added. Phrasing is not supported. The function can be used
    also for scoring the results that different topic numbers would give.
    If scoring is true, it scores each number of topic between the value of
    start parameter and the top number of topics with steps that defines the
    incrementation. If set to 1, every topic number is scored. First, it builds
    a gensim dictionary from all texts, then extremes (documentum frequency
    not below and percentage not above) are removed and a bag of words model
    built as input to LDA. This bag of words model is simply the count of terms
    (not tf-idf).

    Parameters
    ----------
    texts : {list of list}
        Each text is a list of terms (bag of words).
    path_mallet : {str}
        Absolute path to Mallet model.
    terms_to_remove : {list}, optional
        [description] (the default is [])
    num_topics : {number}, optional
        Number of topics (the default is 50).
    scoring : {bool}, optional
        Whether the function is used to score different number of topics.
    start : {number}, optional
        The starting number of topics for scoring (the default is 2)
    step : {number}, optional
        Intervals when scoring (the default is 3).

    Returns
    -------
    list of int
        gensim LDA model and the Gensim trained corpus object as input.
    gensim LDA model and gensim corpus model
        if scoring false, gensim LDA model and the Gensim trained corpus object
        as input.

    """
    preprocessed_corpus = []
    print ('training of gensim corpus began')
    for i, text in enumerate(texts):
        if i == 0:
            # todo filter here
            text = text.split()

            # Additional filtering steps #
            """
            filtered_text = [word for word in text if (word[0] in
                    string.ascii_uppercase + string.ascii_lowercase)]

            filtered_text = [word for word in filtered_text if
                    (word not in set(stopwords.words('english')))]
            preprocessed_corpus.append(filtered_text)
            """

            dct = initialize_gensim_dictionary([text])
        else:
            text = text.split()
            # Additional filtering steps

            """
            filtered_text = [word for word in text if (word[0] in
                    string.ascii_uppercase + string.ascii_lowercase)]

            filtered_text = [word for word in filtered_text if
                    (word not in set(stopwords.words('english')))]
            preprocessed_corpus.append(filtered_text)
            """
            add_documents_to_gensim_dictionary(dct, [text])
    # todo:this is to be integrated to the building process

    if len(terms_to_remove) > 0:
        for term in terms_to_remove:
            dct.filter_tokens(bad_ids=[dct.token2id[term]])

    dct.filter_extremes(no_below=no_below, no_above=no_above)

    gensim_corpus = [dct.doc2bow(bag_of_word.split()) for bag_of_word in texts]
    print ('gensim corpus done')
    if scoring:

        coherence_values = []

        for n in range(start, num_topics, step):

            lda = LdaMallet(constants.PATH_TO_MALLET,
                            gensim_corpus, id2word=dct,
                            num_topics=n)
            coherencemodel = CoherenceModel(model=lda,
                                            texts=preprocessed_corpus,
                                            dictionary=dct, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return coherence_values

    else:
        lda = LdaMallet(constants.PATH_TO_MALLET, gensim_corpus,
                        id2word=dct, num_topics=num_topics)
        # Visualize LDA results, poor results obtained.
        # from gensim.models.wrappers import ldamallet
        # lda_model = ldamallet.malletmodel2ldamodel(lda)
        # vis = pyLDAvis.gensim.prepare(lda_model, gensim_corpus, dct)
        # pyLDAvis.save_html(vis , 'test.html')
        return {'model': lda, 'corpus': gensim_corpus}


def post_process_result_of_lda_topic_model(lda_model, gensim_corpus,
                                           document_collection,
                                           document_collection_filtered,
                                           n_closest=25):
    """Get the n_closest texts to each topic and the document topic matrix.

    Parameters
    ----------
    lda_model : {gensim mallet lda model}
        The result of LDA training.
    gensim_corpus : {gensim corpus}
        The original corpus used to train the model.
    document_collection : {list}
        The original results retrieved with BlackLab.
    document_collection_filtered : {list}
        Document collection used to feed the model after filtering.
    n_closest : {number}, optional
        The number of closest documents to a given topic (the default is 25)

    Returns
    -------
    dict
       Dictionary consisting of two elements. 1. 'topic_documents':
       list of dictionary, each topic ('topic_words') and the n_closest
       text ('texts') to that topic. This latter one is a list of
       dictionaries with keys 'matched_text' (input to the model)
       and 'matched_text_words' (the original unfiltered text and
       'testimony_id'. 2. 'document_topic_matrix': this is numpy.ndarray

    """
    # Prepare containers to store results
    # Container to keep the document topic matrix
    n_closest = - n_closest
    document_topic_matrix = []
    # Container to keep topics and the closest texts to each topic
    topic_closest_doc_with_topics_words = []
    # Container to keep topics
    all_topics = lda_model.show_topics(50)

    # Create an LDA corpus from the original gensim corpus
    lda_corpus = lda_model[gensim_corpus]

    # Iterate through the lda corpus and create the document topic matrix
    for i, documents in enumerate(lda_corpus):
        # Data returned is not proper numpy matrix
        document_topic_matrix.append(
            np.array([elements[1]for elements in documents]))

    # Create the proper numpy matrix
    document_topic_matrix = np.vstack(document_topic_matrix)

    # Find the closest texts to a given topic
    # Iterate through the transpose of the document topic matrix
    for i, element in enumerate(document_topic_matrix.T):
        # Identify the id of 15 closest texts of each topic
        closest = element.argsort(axis=0)[n_closest:][::-1]
        # Create a container to keep each text with the id above
        texts = []
        for element in closest:
            texts.append({'matched_text':
                         document_collection_filtered[element],
                         'matched_text_words':
                          document_collection[element]['match_word'],
                          'testimony_id': document_collection[element]
                          ['testimony_id']})

        # Append them to container
        topic_closest_doc_with_topics_words.append({'texts': texts,
                                                    'topic_words':
                                                    all_topics[i]})

    return {'topic_documents': topic_closest_doc_with_topics_words,
            'document_topic_matrix': document_topic_matrix}


def write_topics_texts_to_file(topics_texts,
                               path_to_file,
                               query_parameters=None):
    r"""Create a text file with each topic words.

    Each set of topic words is followed by those texts that are the closest \
    to that set of topic words.

    Parameters
    ----------
    topics_texts : {list}
       List of dictionaries that are topics ('topic_words')
       and texts ('texts') Each text is a dictionary of 'matched_text',
       'matched_text_words',\'testimony_id']).
    path_to_file : {str}
       Absolute path to a text where results are written.
    query_parameters : {str}, optional
       Query used to retrieve the texts in the data (the default is None)
    """
    output_text = ''
    if query_parameters:
        for element in query_parameters:
            output_text = output_text + \
                str(element[0]) + \
                ': ' + str(element[1]) + \
                ' \n '
    for i, element in enumerate(topics_texts):
        topic_words = ' '.join(
            [str(topic) for topic in element['topic_words']])
        output_text = output_text + str(i) + '. ' + topic_words + ':' + '\n\n'
        for f, document in enumerate(element['texts']):
            output_text = output_text + \
                str(f) + '. ' + document['testimony_id'] + '.\n'
            output_text = output_text + ' Original text:\n' + \
                document['matched_text_words'] + '\n\n'
            output_text = output_text + ' Input text:\n' \
                + document['matched_text'] + '\n\n'
        output_text = output_text + '-----------------------\n\n'
    f = open(path_to_file, 'w')
    f.write(output_text)
    f.close()


def visualize_topic_scoring(scores, limit, start, step, path_to_output_file):
    """Render topic scoring on a plit.

    Parameters
    ----------
    scores : {list}
        List of scores
    limit : {number}
    start : {number}
    step : {number}
    path_to_output_file : {str}
        Absolute path to file.
    """
    x = range(start, limit, step)
    plt.plot(x, scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(path_to_output_file)


if __name__ == '__main__':
    import sys
    sys.path.append("..")

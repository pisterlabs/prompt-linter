import collections
import gensim.corpora as corpora
import nltk
import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from sentence_transformers import SentenceTransformer, models
from typing import List, Dict, Tuple, Union, Literal

nltk.download('stopwords')

nltk.download('punkt')


def initialize_model(embedding_model: Union[SentenceTransformer, str] = "specter",
                     nr_topics: int = None,
                     n_gram_range: Union[Literal["unigram", "bigram"], Tuple[int, int]] = "unigram",
                     verbose="False"):
    """
        Conducts type checks on the input variables and converts certain parameters to the proper types dependent on their input.
        Not intended to be called by the user, but instead used internally by the fit function.


        Parameters
        ----------
        embedding_model
                Either a SentenceTransformer, or a string with values "specter". "aspire", or "scibert".
        nr_topics: int
                The desired number of topics, if not specified the results will be determined by HDBSCAN's reduction step.
        n_gram_range
                String indicating whether the user would like unigram or bigram or tuple of ints.
        verbose
                Boolean indicating verbose output.

        Returns
        -------
        embedding_model
                SentenceTransformer language model
        n_gram_range
                tuple indicating the level of n_gram
            """

    if not isinstance(nr_topics, int) and nr_topics is not None:
        raise TypeError("nr_topics must be an int")
    if not isinstance(verbose, bool) and nr_topics is not None:
        raise TypeError("verbose must be a bool")

    if isinstance(n_gram_range, str):
        if n_gram_range == "unigram":
            n_gram_range = (1, 1)
        elif n_gram_range == "bigram":
            n_gram_range = (2, 2)
        else:
            raise AttributeError("n_gram_type must equal \"unigram\" or \"bigram\" if it is a string")
    elif isinstance(n_gram_range, tuple):
        pass
    else:
        raise TypeError("n_gram_range must be an string or tuple")

    if isinstance(embedding_model, str):
        if embedding_model.lower() == "specter":
            embedding_model = SentenceTransformer('allenai-specter')

        elif embedding_model.lower() == "aspire":
            word_embedding_model = models.Transformer('allenai/aspire-sentence-embedder', max_seq_length=512)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
            embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        elif embedding_model.lower() == "scibert":
            word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=512)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
            embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        else:
            raise AttributeError("embedding model should either be a string specifying one of the 3 pre-loaded "
                                 "models, or the desired language model")
    embedding_model = embedding_model

    n_gram_range = n_gram_range

    return embedding_model, n_gram_range


def fit(data: List[str],
        embedding_model: Union[SentenceTransformer, str] = "specter",
        nr_topics: int = None,
        n_gram_range: Union[Literal["unigram", "bigram"], Tuple[int, int]] = "unigram",
        verbose: bool = False):
    """
        Fits a BERTopic model on the data. After fitting the topic assigned to each document is stored
        in the 'topics' attribute, the coherence and diversity measures are stored in the
        'coherence' and 'diversity' attributes respectively, and the amount of documents assigned to each topic
        are stored in the 'topic_sizes' attribute.


        Parameters
        ----------
        data
                List of the documents
        embedding_model
                Either a SentenceTransformer, or a string with values "specter". "aspire", or "scibert".
        nr_topics
                The desired number of topics, if not specified the results will be determined by HDBSCAN's reduction step.
        n_gram_range
                String indicating whether the user would like unigram or bigram, or tuple of ints.
        verbose
                Boolean indicating verbose output.

        Returns
        -------
        topics
                a list of integers representing the topic the corresponding document was assigned to
        probabilities
                a
        metrics
                a dictionary of the 2 metrics with keys "Coherence" and "Diversity"
        topic_sizes
                a dictionary with key: topic number and the value: the number of documents assigned the said topic
        topic_model
                the fitted BERTopic model
        topic_words
                a dictionary with key: the topic number and the value: a list of strings
        """
    opts = dict()

    if not isinstance(embedding_model, SentenceTransformer) and not isinstance(n_gram_range, tuple):
        embedding_model, opts['n_gram_range'] = initialize_model(embedding_model, nr_topics, n_gram_range, verbose)

    if not isinstance(data, list) or (isinstance(data, list) and not isinstance(data[0], str)):
        raise TypeError("Data must be a list of strings")

    topic_model = BERTopic(embedding_model=embedding_model,
                           nr_topics=nr_topics,
                           verbose=verbose,
                           **opts)
    topics, probabilities = topic_model.fit_transform(data)
    metrics = _calculate_metrics(data, topic_model, topics)
    topic_sizes = _calculate_topic_sizes(topics)
    topic_words = topic_model.topic_representations_

    return topics, probabilities, topic_sizes, topic_model, topic_words, metrics


def _calculate_topic_sizes(topics: List[int]):
    """
    counts the number of documents assigned to each topic and stores the topic sizes in the 'topic_sizes'
    attribute in the form of a dict.

    Parameters
    ----------
    topics
        List of topic assignments for each document

    Returns
    -------
    Sets the topic size parameter to a collection object containing the topics and their sizes.

    """

    counter = collections.Counter(topics)
    topic_sizes = dict(counter)
    return topic_sizes


def _calculate_metrics(texts: List[str], topic_model: BERTopic, topics: List[int]):
    """
    Calculates the Topic Coherence and Topic Diversity of the topic model.
    Parameters
    ----------
    texts
         The list of documents
    topic_model
        Fitted topic model
    topics
        list of topic assignments

    Returns
    -------
    dict
        A dict containing the metrics as keys, and their respective scores as values.
    """

    topic_words = {}
    topic_dict = topic_model.topic_representations_
    for k in topic_dict.keys():
        topic_words[k] = [x[0] for x in topic_dict[k]]
    word_list = list(topic_words.values())
    word_list.pop(0)

    topic_diversity = TopicDiversity(topk=10)

    coherence_score = _calculate_coherence(topic_model, texts, topics)

    output_tm = {"topics": word_list}
    diversity_score = topic_diversity.score(output_tm)

    return {"Coherence": coherence_score, "Diversity": diversity_score}


def _calculate_coherence(topic_model: BERTopic, docs: List[str], topics):
    """
    Internal method for calculating the coherence when the n_gram_range is set to something other than (1,1)
    Parameters
    ----------
    topic_model
        The BERTopic object
    docs
        The list of documents

    Returns
    -------
        The coherence score

    """

    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs,
                              "ID": range(len(docs)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                   for topic in range(len(set(topics)) - 1)]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     processes=1,
                                     coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence


def create_barcharts(topics, topic_model: BERTopic, path=""):
    """
    creates and saves the BERTopic barcharts

    Parameters
    ----------
    topics
            the list of topic assignments output from the fit function
    topic_model
            fitted BERTopic model output from the fit function
    path
        A string containing the desired path to save the barchart

    Returns
    -------
    Barcharts are saved in .html and .png format in the desired directory.

    """

    if not isinstance(path, str):
        raise TypeError("path must be a string")

    if path != "":
        path = path + 'barchart'

    # model_path = base_path + iter_version + beamline + "_" + model_use + "_" + n_gram_type + "/"
    fig = topic_model.visualize_barchart(top_n_topics=len(topics))
    fig_name = path + ".html"
    fig_name_png = path + ".png"
    if path != "":
        fig.write_html(fig_name)
        fig.write_image(fig_name_png)
    return fig

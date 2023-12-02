"""
Functional interface to the He et al Attention Based Aspect Extraction model

Module provides helper functions for pre-processing training data,
interacting with the HeEtAl2017Build class, and post-processing
outputs from the model
"""

from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
import gensim
from keras.optimizers import Adam

#from fntk.topic_modeling.metrics import coherence
from .he_et_al_2017 import HeEtAl2017Build

def word_to_vec(corpus, doc_sent_word=True, **kwargs):
    """
    Train word vectors with `gensim.models.Word2Vec`

    Parameters
    ----------
    corpus : list of lists
        Corpus is a either a list of documents, which is in turn a list of sentences,
        or just a list of sentences.  A sentence is a list of words/tokens
    doc_sent_word : boolean, optional
        Set to `True` if the corpus is a list of documents, `False` if corpus is a
        list of sentences
    kwargs : dict
        keyword args to be passed to gensim.Word2Vec

    Returns
    -------
    weights : np.array of shape (`vocab_size`, `word_embed_dim`)
    vocab : dict
        keys are the original words/tokens
        values are the token's index, as int
    """

    if doc_sent_word:
        class CorpusIterator(object):
            """Generator to serve up sentences to Word2Vec"""
            def __init__(self, corpus):
                self.corpus = corpus

            def __iter__(self):
                for doc in corpus:
                    for sentence in doc:
                        yield sentence
    else:
        class CorpusIterator(object):
            """Generator to serve up sentences to Word2Vec"""
            def __init__(self, corpus):
                self.corpus = corpus

            def __iter__(self):
                for doc in corpus:
                    yield doc

    corpus_gen = CorpusIterator(corpus)

    model = gensim.models.Word2Vec(corpus_gen, **kwargs)
    weights = model.wv.syn0

    vocab = {k: v.index for k, v in model.wv.vocab.items()}

    return weights, vocab

def replace_rare_words(corpus, common_word_freq, unknown_token='<UNK>'):
    """
    Replace rarely appearing words with sentinel unknown token

    Parameters
    ----------
    corpus : list of lists
        List of documents where each document is a list of words/tokens
    common_word_freq : int
        Minimum number of times word/token must appear in corpus to be included.
        Any word occurring less than this will be replaced with `unknown_token`
    unknown_token : str, optional
        String used to overwrite word/token that appear less than `common_word_freq`

    Returns
    -------
    corpus_modified : list of list
        A copy of the `corpus`, with rare words replaced with `unknown_token`
    """
    word_count = Counter()
    for doc in corpus:
        word_count.update(doc)

    corpus_modified = [
        [
            unknown_token if word_count[word] < common_word_freq else word
            for word in doc
        ]
        for doc in corpus
    ]

    return corpus_modified


def indexize_corpus(corpus, vocab_dict):
    """
    Replace words in sentences with their word vector indexes

    Parameters
    ----------
    corpus : list of lists
        Corpus is list of sentences, each sentence is a list of words/tokens
    vocab_dict : dict
        keyed by word, value is index

    Returns
    -------
    corpus : list of lists
        Copy of corpus with the words replaced with their index

    See Also
    --------
    reverse_indexize_corpus

    """
    corpus = [[vocab_dict[w] for w in doc] for doc in corpus]
    return corpus

def reverse_indexize_corpus(corpus, vocab_dict):
    """
    Replace word index in sentences with word/token

    Parameters
    ----------
    corpus : list of lists
        Corpus is list of sentences, each sentence is a list of word indexes
    vocab_dict : dict
        keyed by word, value is index

    Returns
    -------
    corpus : list of lists
        Copy of corpus with the word_index replaced with their word

    See Also
    --------
    indexize_corpus
    """
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}

    corpus = [[inv_vocab_dict[w] for w in doc] for doc in corpus]
    return corpus


# post-processing functions consumed by build functio
# Maybe these belong outside present module

def most_similar_words_for_topics(n_words_per_topic, word_vecs, topic_vecs, inv_vocab_dict):
    """
    For each topic, gives list of words with largest cosine similarities

    Parameters
    ----------
    n_words_per_topic : int
        number of words to include per topic
    word_vecs : np.array of shape (`vocab_size`, `word_embed_dim`)
    topic_vecs : np.array of shape (`n_topics`, `word_embed_dim`)
    inv_vocab_dict : dict
        Keyed by word index, value is word

    Returns
    -------
    topic_words : dict
        Keys of dict are the topic index
        Value is a length n_words_per_topic list of tuples, each tuple is
        a (word_index, word, cosine sim to topic)
    """

    def most_similar_words_to_topic(topic_sim, n_words_per_topic, inv_vocab_dict):
        """
        Extract closest words to a topic, given the topic similarities

        Parameters
        ----------
        topic_sim : list
            Each entry gives cosine similarity of a given word to the topic
        n_words_per_topic : int
            Number of entries to return
        inv_vocab_dict : dict
            Keyed by word index, value is word

        Returns
        -------
        list of 3-tuple
            List of length `n_words_per_topic`, corresponding to the `n_words_per_topic`
            closest words to the topic.   Each entry is (word_index, word, similarity)
        """
        sorted_by_sim = sorted([(i, v) for i, v in enumerate(topic_sim) if not np.isnan(v)], key=lambda x: -x[1])
        sorted_by_sim = sorted_by_sim[:n_words_per_topic]
        sorted_by_sim = [(idx, inv_vocab_dict[idx], sim) for (idx, sim) in sorted_by_sim]
        return sorted_by_sim

    def row_normalize(mat):
        """
        L2 normalizes rows of matrix

        Parameters
        ----------
        mat : np.array

        Returns
        -------
        np.array
            Array with same shape as `mat`
        """
        row_norms = np.sqrt(np.sum(np.square(mat), axis=1))
        return mat/row_norms[:, None]

    word_vecs_n = row_normalize(word_vecs)
    tv_n = row_normalize(topic_vecs)
    similarity = np.matmul(word_vecs_n, tv_n.transpose()).transpose()

    topic_words = {
        j: most_similar_words_to_topic(topic_sim, n_words_per_topic, inv_vocab_dict)
        for j, topic_sim in enumerate(similarity)}

    return topic_words


def build_he_et_al_2017(
        corpus, labeled_validation=None, word2vec_kwargs=None,
        vocab_dict=None, word_embed_dim=None, vocab_size=None, min_sent_len=1,
        max_sent_len=None, n_topics=None, n_neg_samples=1, optimizer=Adam(lr=0.001),
        pretrained_word_vecs=None, pretrained_topic_vecs=None, kmeans_kwargs=None,
        word_embed_kwargs=None, topic_kwargs=None, attention_kwargs=None,
        fit_kwargs=None, n_words_per_topic=10, train_test_split_kwargs=None,
        unknown_token='<UNK>', pad_token='<PAD>', seed=1357, **kwargs
    ):
    """
    Functional interface for the He et al 2017 Attention Based Aspect Extraction model

    Model described in He, Lee, Ng, Dahlmeier
    An Unsupervised Neural Attention Model for Aspect Extraction
    http://www.aclweb.org/anthology/P/P17/P17-1036.pdf

    This function serves as a functional interface to neuraltopics.HeEtAl2017Build.
    If you are looking for greater flexibility in terms
    of the inputs or outputs, you may want to work with the build class directly.

    Parameters
    ----------
    corpus : list of lists
        List of documents, each document is a list of words.  It is assumed
        pre-preocessing like removing stop words, etc has already occurred.
        But it is not assumed that the words have been turned into indexes
    labeled_validation : list of list of dicts, optional
        List of documents, each document is a dict with keys for 'words' and 'labels'.
        'words' is a list of the words/tokens in the sentences, 'labels' is the list of
        gold-standard topic labels.  It is assumed pre-preocessing like removing stop words,
        etc has already occurred, but it is not assumed that the words have been turned
        into indexes
    word2vec_kwargs : dict
        kwargs to be passed to `gensim.word2vec`.  Only needs to be provided if
        `pretrained_word_vecs` are not provided.   If providing `word2vec_kwargs`, then
        specify only one of `word_embed_dim` and `word2vec_kwargs['size']`
    unknown_token : str
        Rare occurring words will be replaced with unknown_token
    pad_token : str
        Token used to pad sentences to `max_sent_len`.
    vocab_dict : dict, optional
        Keyed by words/tokens, values are the associated indexes.  Must be provided if
        providing `pretrained_word_vecs`.  Otherwise, if constructing word vecs
        (ie, not providing `pretrained_word_vecs`) then `vocab_dict` will be
        auto-generated.
    train_test_split_kwargs : dict, optional
        kwargs to be passed to sklearn `train_test_split`
        If None, will use the full corpus as both the train and test data
    fit_kwargs : dict, optional
        kwargs passed to keras' fit method.
        Usually required to include `fit_kwargs['batch_size']` and `fit_kwargs['epochs]`
    n_words_per_topic : int
        Controls number of top words to display per topic.
    word_embed_dim : int, optional
        Dimension of the word embeddings.  Set to `None` if using
        `pretrained_word_vecs`, since value will be inferred in that case.
    vocab_size : int, optional
        Number of distinct words in the vocabulary.  Set to `None` if using
        `pretrained_word_vecs`, since value will be inferred in that case.
    max_sent_len : int, optional
        Maximum allowed Word length of sentences.  If `None` then allows all
        lengths.
    min_sent_len : int, optional
        Minimum allowed sentence length.  This is useful to filter out bad data,
        for example extremely short sentence fragments.
    n_topics : int, optional
        Number of topics to use.  Set to `None` if using `pretrained_topic_vecs`
        since value will be inferred in that case.
    pretrained_word_vecs : np.array of shape (`vocab_size`, `word_embed_dim`)
        Matrix describing word vectors.
        Currently, model requires `pretrained_word_vecs` to be provided.
    n_neg_samples : int, optional
        Number of negative samples to associate with each positive sample
    optimizer : keras.optimizer
        Optimizer to use in keras's compile and fit
    pretrained_topic_vecs : np.array of shape (`n_topics`, `word_embed_dim`)
        Matrix describing topic vectors.
        If `pretrained_topic_vecs` are not provided then topic vectors will
        be initialized via k-means clustering on the word vecs with
        `n_topics` clusters.
    kmeans_kwargs : dict, optional
        kwargs passed to kmeans.  Only used in the event that
        `pretrained_topic_vecs` are not provided.  Do not provide `n_clusters` to
        kmeans here, as that value will be inferred from `n_topics`.
    word_embed_kwargs : dict, optional
        kwargs passed to `keras.layers.Embedding` for the shared word embeddings.
        The following values will be overwritten or set as defaults
        'input_dim' will be set to `vocab_size`
        'output_dim' will be set to `word_embed_dim`
        'weights' will be set to `word_vecs`
        'trainable' will be set to 'False', unless explicitly overwritten here
    topic_kwargs : dict, optional
        kwargs passed to `keras.layers.Dense` for the topic embedding layer
        The following values will be overwritten or set as defaults
        'trainable' will be set to `True` unless explicitly overwritten here
        'units' will be set to `word_embed_dim`
        'use_bias' will be set to `False`
        'weights' will be set to transpose of the `topic_vecs`
        'kernel_regularizer' if not passed, it will be set to `ABAEL2TopicReg`
        'topic_l2_reg' takes a float (defaults to `1.0`) and will be passed
        to `ABAEL2TopicReg`, provided kernel regularizer is `ABAEL2TopicReg`
    attention_kwargs : dict, optional
        kwargs passed to the ABAEAttentionLayer shared_attention layer
    **kwargs :
        kwargs to be passed through to HeEtAl2017Build

    Returns
    -------

    """
    # not sure if this is enough.  My understanding is that this does not
    # necessary control random seeds inside of tensorflow....
    np.random.seed(seed)

    if train_test_split_kwargs:
        (train, test) = train_test_split(corpus, **train_test_split_kwargs)
    else:
        train = corpus
        test = corpus

    if pretrained_word_vecs is None:
        if vocab_dict is not None:
            raise ValueError("""
                Unexpected input `vocab_dict`.  Should only specify `vocab_dict` when
                providing `pretrained_word_vecs`
            """)

        if word2vec_kwargs is None:
            word2vec_kwargs = {}
        if word_embed_dim and (word2vec_kwargs.get('size', word_embed_dim) != word_embed_dim):
            raise ValueError("""
                The provided word_embed_dim and word2vec_kwargs['size'] are contradictory.
                You are better off just providing word_embed_dim, in which case the function
                will auto populate word2vecs_kwargs['size'].""")
        word2vec_kwargs.setdefault('size', word_embed_dim)
        word2vec_kwargs.setdefault('min_count', 5)
        word2vec_kwargs.setdefault('window', 10)
        word2vec_kwargs.setdefault('negative', 5)

        # TODO
        # I think this belongs elsewhere...
        common_word_freq = word2vec_kwargs['min_count']
        train = replace_rare_words(train, common_word_freq, unknown_token)
        test = replace_rare_words(test, common_word_freq, unknown_token)

        word_vecs, vocab_dict = word_to_vec(train, doc_sent_word=False, **word2vec_kwargs)
    else:
        if vocab_dict is None:
            raise ValueError("""
                Missing input `vocab_dict`.  `vocab_dict` needs to be provided
                when `pretrained_word_vecs` are provided
            """)
        word_vecs = pretrained_word_vecs

    vocab_size, word_embed_dim = word_vecs.shape


    train = [sent for sent in train if len(sent) >= min_sent_len]
    if test:
        test = [sent for sent in test if len(sent) >= min_sent_len]
    if max_sent_len is not None:
        train = [sent for sent in train if len(sent) <= max_sent_len]
        if test:
            test = [sent for sent in test if len(sent) <= max_sent_len]

    # Handles padding sentences to fixed length
    lengths = Counter(len(sent) for sent in train + test)
    max_sent_len = max(lengths)

    vocab_dict[pad_token] = -1
    print("Maximum sentence length {}".format(max_sent_len))
    print("Sentence Length Distribution: {}".format(lengths))

    def pad_sentence(sent, pad_token, pad_to_len):
        return sent + [pad_token] * (pad_to_len - len(sent))

    train = [pad_sentence(sent, pad_token, max_sent_len) for sent in train]
    if test:
        test = [pad_sentence(sent, pad_token, max_sent_len) for sent in test]

    # Create row of all zeros to handle the "PAD"
    word_vecs = np.append(np.zeros([1, word_vecs.shape[1]]), word_vecs, axis=0)

    # But what happens when it looks up index -1, if `unknown_token` not seen in training?
    vocab_dict.setdefault(unknown_token, len(vocab_dict))
    vocab_dict = {k: v + 1 for k, v in vocab_dict.items()}

    def constant_factory(value):
        return lambda: value
    vocab_dict = defaultdict(constant_factory(vocab_dict[unknown_token]), vocab_dict)

    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    inv_vocab_dict = defaultdict(constant_factory(unknown_token), inv_vocab_dict)

    train = indexize_corpus(train, vocab_dict)
    if test:
        test = indexize_corpus(test, vocab_dict)

    abae_model = HeEtAl2017Build(
        word_embed_dim=word_vecs.shape[1],
        vocab_size=word_vecs.shape[0],
        sent_len=max_sent_len,
        n_topics=n_topics,
        pretrained_word_vecs=word_vecs,
        n_neg_samples=n_neg_samples,
        optimizer=optimizer,
        pretrained_topic_vecs=pretrained_topic_vecs,
        kmeans_kwargs=kmeans_kwargs,
        word_embed_kwargs=word_embed_kwargs,
        topic_kwargs=topic_kwargs,
        attention_kwargs=attention_kwargs,
        **kwargs)

    abae_trained = abae_model.abae_fit(train, **fit_kwargs)

    if not test:
        test = train
    [sent_embed, topic_weights, reconstructed] = abae_trained.predict(test)

    topic_vecs = abae_model.topic_vecs
    attention = abae_model.attention

    # gives closest words for each topic
    topic_words = most_similar_words_for_topics(
        n_words_per_topic, word_vecs, topic_vecs, inv_vocab_dict)
    print("\nClosest words per topic:  (and cosine similarity)")
    for topic, words in topic_words.items():
        print("Topic {}:\n\t{}".format(topic, words))

#    # gives coherence score for each topic
#    # Note that 'irrelevant' topics can keep high 'coherence'.  Problem is
#    # that when topic is 'irrelevant', then there aren't many opportunities
#    # to find documents with one topic word but not another
#    coherences = {
#        topic: coherence(test, [x[0] for x in words])
#        for topic, words in topic_words.items()}
#    print("\nTopic coherences (range -infty to 0; closer to zero is more coherent)")
#    for topic, words in coherences.items():
#        print("Topic {}:\n\t{}".format(topic, words))

    # gives attention weights per word
    attended = abae_model.attention_weights(test)
    sentence_attention = [
        [(inv_vocab_dict[w], a) for (w, a) in zip(sent, attended[i])]
        for i, sent in enumerate(test)]

    print("\nFirst 20 test sentences and their attended words")
    for sent in sentence_attention[:20]:
        print([(w, "%0.4f" % a) for (w, a) in sent])

    # gives decomposition of topics per sentence
    sentence_topics = [
        sorted([x for x in enumerate(topic_weights[i])], key=lambda x: -x[1])
        for i, sent in enumerate(test)]

    print("\nFirst 20 test sentences and their topic distributions")
    for sent in sentence_topics[:20]:
        print([(idx, "%0.4f" % weight) for (idx, weight) in sent])

    test_summary = {'sentence_attention' : sentence_attention, 'sentence_topics': sentence_topics}


    # TODO:
    #     Abstract out the similarities between testing and validation
    validation_summary = {}
    if labeled_validation is not None:
        #labeled_validation = flatten_to_bag_of_sentences(labeled_validation)

        if max_sent_len is not None:
            labeled_validation = [
                sent for sent in labeled_validation if len(sent['words']) <= max_sent_len]
        labeled_validations = [
            sent for sent in labeled_validation if len(sent['words']) >= min_sent_len]

        labels = []
        val = []
        for sent in labeled_validation:
            labels.append(sent['topics'])
            val.append(sent['words'])

        #if pretrained_word_vecs is None:
        #    val = replace_rare_words(val, common_word_freq, unknown_token)
        val = indexize_corpus(val, vocab_dict)

        validation_summary['labels'] = labels
        validation_summary['sentences'] = [[inv_vocab_dict[w] for w in sent] for sent in val]

        [sent_embed_val, topic_weights_val, reconstructed_val] = abae_trained.predict(val)

        attended_val = abae_model.attention_weights(val)
        validation_summary['sentence_attention'] = [
            [(inv_vocab_dict[w], a) for (w, a) in zip(sent, attended_val[i])]
            for i, sent in enumerate(val)]

        validation_summary['sentence_topics'] = [
            sorted([x for x in enumerate(topic_weights_val[i])], key=lambda x: -x[1])
            for i, sent in enumerate(val)]

    return {
        'topic_words': topic_words,
        'attended': attended,
        'model': abae_model,
        'test': test_summary,
        'validation': validation_summary}

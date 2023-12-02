import gensim.corpora as corpora
import gensim.models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import keras
from keras.layers import Input, Dense
from keras.models import Model
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import pyLDAvis.gensim
import pyLDAvis
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(file):
    """Load file into Pandas dataframe"""
    return pd.read_csv(file)

def get_corpus(data, text_column):
    """Extract corpus series from dataframe.

    Args:
        data (dataframe): Pandas dataframe.
        text_column (str): Series name containing corpus.

    Returns:
        List of documents.
    """
    replace_dict = {"'": '',
                    '[': '',
                    ']': ''
                   }
    corpus = data[text_column].map(lambda x: ''.join(replace_dict[char] if char in replace_dict else char for char in x)
                                               .split(', ') if type(x) != list else x
                                  ).tolist()
    return corpus

class LDA:
    """Trains an LDA model.

    In addition to training an LDA model in Gensim, there are
    methods to trim the corpus vocabulary. Additionally, one
    can further extract topic probability vectors for each
    document in the corpus.

    Attributes:
        corpus (list): List of documents.
    """
    def __init__(self, corpus):
        """Inits class with corpus."""
        self.corpus = corpus
        self.pruned_vocab = None
        self.pruned_corpus = None
        self.formatted_corpus = None
        self.formatted_dict = None
        self.model = None

    def prune_vocab(self,
                    min_df=10,
                    vectorizer=None
                   ):
        """Removes words from corpus vocabulary under specified threshold.

        Args:
            min_df (int): threshold for word occurrence. Default is 10, i.e.,
              the word must appear in a minimum of 10 documents.
            vectorizer(obj): A count vectorizer. Default is sklearn's CountVectorizer.

        Returns:
            Self. Pruned_vocab attribute stores of words meeting inclusion criterion.
        """
        if not vectorizer:
            vectorizer = CountVectorizer(min_df = min_df)
        vectorizer.fit_transform([' '.join(word) for word in [doc for doc in self.corpus]])
        self.pruned_vocab = vectorizer.get_feature_names()
        return self

    def prune_corpus(self):
        """Filters corpus based on `.prune_vocab()` results.

        Returns:
            Self. Pruned_corpus attribute stores new corpus.
        """
        if not self.pruned_vocab:
            raise ValueError("Run 'prune_vocab()' before pruning corpus.")
        pruned_corpus = []
        for doc in self.corpus:
            pruned_corpus.append(
                [word for word in doc if word in self.pruned_vocab])
        self.pruned_corpus = pruned_corpus
        return self

    def format_corpus(self, lda_corpus = None,
                      prune_corpus = True,
                      min_df = 10
                     ):
        """Formats corpus for use in a Gensim LDA model.

        After checking parameters for if any pruning has been/
        should be done, the method creates a Gensim dictionary
        and coverts each document into a bag-of-words format.

        Args:
            lda_corpus (list): List of documents.
            prune_corpus (bool): Specifies whether a corpus has been
              pruned or if pruning should be done. Default is true.
            min_df (int): Frequency criterion for vocabulary inclusion
              passed into the `.prune_vocab()` method if pruning is done.
              Default is 10 occurrences.

        Returns:
            Self. The formatted_dict attribute stores the dictionary,
            and the formatted_corpus attribute stores the corpus.
        """
        if prune_corpus and not self.pruned_corpus:
            self.prune_vocab(min_df = min_df)
            self.prune_corpus()
            lda_corpus = self.pruned_corpus
        elif self.pruned_corpus:
            lda_corpus = self.pruned_corpus
        else:
            lda_corpus = self.corpus
        self.formatted_dict = corpora.Dictionary(lda_corpus)
        self.formatted_corpus = [
            self.formatted_dict.doc2bow(doc) for doc in lda_corpus]
        return self

    def train(self,
              mallet_path=None,
              corpus=None,
              id2word=None,
              num_topics=20,
              alpha='asymmetric', #https://papers.nips.cc/paper/3854-rethinking-lda-why-priors-matter.pdf
              eta='auto', #might as well let it learn!
              passes=20,
              iterations = 50,
              optimize_interval = 10, #going with their suggestion http://mallet.cs.umass.edu/topics.php
              workers = multiprocessing.cpu_count() - 1,
              random_state=611
              ):
        """Trains an LDA model.

        There are 2 different model options: Gensim's LDAMulticore and Gensim's Mallet wrapper.

        Args:
            mallet_path (str): Path to mallet model file.
            corpus (obj): Gensim formatted list of documents. See `format.corpus()`.
            id2word (obj): Gensim formatted dictionary. See `format.corpus()`.
            num_topics (int): Number of topics to extract.
            alpha, eta, passes, iterations, optimize interval, workers, random_state:
              Parameters specific to Gensim's LDA models. Please refer to
              https://radimrehurek.com/gensim/models/ldamodel.html.

        Returns:
            Trained Gensim LDA model.
        """
        if self.formatted_corpus and self.formatted_dict:
            corpus = self.formatted_corpus
            id2word = self.formatted_dict
        else:
            raise ValueError("Please run 'format_corpus()' before training LDA model.")
        if mallet_path:
            model = gensim.models.wrappers.LdaMallet(mallet_path = mallet_path,
                                                     corpus = corpus,
                                                     id2word = id2word,
                                                     num_topics = num_topics,
                                                     workers = workers,
                                                     optimize_interval = optimize_interval,
                                                     random_seed = 611
                                                    )
            model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
        else:
            model = gensim.models.LdaMulticore(corpus = corpus,
                                               id2word = id2word,
                                               num_topics = num_topics,
                                               alpha = alpha,
                                               eta = eta,
                                               passes = passes,
                                               iterations = iterations,
                                               random_state = 611
                                              )
        return model

    def get_vec_lda(self,
                    model,
                    corpus,
                    num_topics):
        """Gets topic probability vectors for each document.

        Args:
            model (obj): Trained Gensim LDA model.
            corpus (obj): Gensim formatted corpus.
            num_topics (int): Number of topics in trained model.

        Returns:
            Array with each element representing 1 document and
            each element's length equal to `num_topics`.
        """
        n_doc = len(corpus)
        vec_lda = np.zeros((n_doc, num_topics))
        for i in range(n_doc):
            for topic, prob in model.get_document_topics(corpus[i]):
                vec_lda[i, topic] = prob
        return vec_lda

class LdaEval:
    """This class contains several methods to evaluate an LDA model.

    Attributes:
        model (obj): Trained Gensim LDA model.
    """
    def __init__(self, model):
        """Inits class with model."""
        self.model = model
        self.score_df = None

    def coherence_score(self,
                        texts,
                        dictionary,
                        coherence = 'c_v'
                       ):
        """Returns model coherence score.

        Args:
            texts (obj): Gensim formatted corpus. Must be same corpus
              that was used to train original model.
            dictionary (obj): Gensim formatted dictionary. Must be
              same dictionary used to train original model.
            coherence (str): For all options, see
              https://radimrehurek.com/gensim/models/coherencemodel.html.
              Default is 'c_v'.

        Returns:
            A float value indicating the coherence score.
        """
        coherence_model = CoherenceModel(model = self.model,
                                         texts = texts,
                                         dictionary = dictionary,
                                         coherence = coherence
                                         )

        coherence_score = coherence_model.get_coherence()

        return coherence_score

    def lda_vis(self, corpus, dictionary):
        """Plots pyLDAvis object """
        pyLDAvis.enable_notebook()
        LDAvis_prepared = pyLDAvis.gensim.prepare(topic_model = self.model,
                                                  corpus = corpus,
                                                  dictionary = dictionary
                                                  )
        return LDAvis_prepared

class D2V:
    """Tags documents and fits a Gensim Doc2Vec model.

    Using a Pandas dataframe, this can be used to fit a
    Doc2Vec model to train custom document embeddings.
    Default parameters for the model are for a PV-DM model,
    and PV-DBOW embeddings can also be computed. Additionally,
    Doc2Vec parameters can be manually modified.

    Attributes:
        corpus (list): List of documents.
        lda_model (obj): Gensim LDA model.
        lda_vocab (obj): Gensim formatted dictionary.
    """

    def __init__(self, corpus, lda_model=None, lda_vocab=None):
        """Inits class with corpus and optional LDA model and dictionary."""
        self.corpus = corpus
        if lda_model:
            self.lda_model = lda_model
        if lda_vocab:
            self.lda_vocab = lda_vocab
        self.topic_tags = None
        self.docs_tagged = None

    def get_topic_tags(self):
        """Adds tags for document topic probabilities.

        If you want to get custom document embeddings for topics,
        this method scans each document's word for the highest topic
        probability for each word in each document. Then, all topics
        that appear are added to the document tags for future model
        fitting.

        Returns:
            Self. The topic_tags attribute stores the tags.
        """
        assert self.lda_model, "Need LDA model to retrieve topic tags."
        assert self.lda_vocab, "Need vocabulary list to retrieve topic tags."

        word_topic_dict = {}
        for topic in self.lda_model.show_topics(-1, len(self.lda_vocab), formatted=False):
            # topic[0] = topic
            # topic[1] = word, probability
            for wordprob in topic[1]:
                if wordprob[0] in word_topic_dict:  # check if word's in dict
                    # check if current dict probability < current prob
                    if word_topic_dict[wordprob[0]][1] < wordprob[1]:
                        word_topic_dict[wordprob[0]] = (
                            f'topic_{topic[0]}', wordprob[1])
                    else:
                        pass
                else:
                    word_topic_dict[wordprob[0]] = (
                        f'topic_{topic[0]}', wordprob[1])
        self.topic_tags = word_topic_dict
        return self

    def tag_docs(self, topic2vec=False):
        """Return a list of tagged documents formatted for Doc2Vec.

        Method iterates through the corpus, turning
        each into its own document with its own doc tag. If topic2vec,
        topic tags are also added to each document.

        topic2vec (bool): Default is false.

        Returns:
            Self. The docs_tagged attribute stores the tagged documents.
        """
        tagged_docs = []
        if topic2vec:
            for i in range(len(self.corpus)):
                tags = []
                for word in self.corpus[i]:
                    tags.append(self.topic_tags[word][0])

                tags.append(f'doc_{i}')
                tagged_docs.append(TaggedDocument(self.corpus[i], list(set(tags))))
        else:
            for idx, doc in enumerate(self.corpus):
                tagged_docs.append(TaggedDocument(doc, f'doc_{idx}'))
        self.docs_tagged = tagged_docs
        return self

    def model_train(self,
                    vector_size=400,
                    dm=0,
                    dbow_words=1,
                    min_count=10,
                    epochs=40,
                    workers=multiprocessing.cpu_count() - 1
                    ):
        """Trains a Doc2Vec model.

        This method is an aggregation of the several steps needed to train
        a Doc2Vec model. The model is first instantiated with a vocabulary
        based on the tagged documents, and the model is then trained on these
        data. For further documentation, see Gensim's official Doc2Vec docs
        (https://radimrehurek.com/gensim/models/doc2vec.html).

        Args:
            See https://radimrehurek.com/gensim/models/doc2vec.html.

        Returns:
            A trained Doc2Vec model.
        """
        assert self.docs_tagged, 'No TaggedDocuments found. Please run "tag_docs" method prior to "model_train".'

        model = Doc2Vec(vector_size=vector_size,
                        dm=dm,
                        dbow_words=dbow_words,
                        min_count=min_count,
                        epochs=epochs,
                        workers=workers
                        )

        model.build_vocab(self.docs_tagged)
        model.train(self.docs_tagged,
                    total_examples=len(self.corpus),
                    epochs=model.epochs
                    )

        return model

    def get_vec_d2v(self, model):
        """For all documents in the model, this gets each document's document vectors.

        Args:
            Model (obj): Trained Doc2Vec model.

        Returns:
            Array of dimensions (# docs)x(vector size).
        """
        doc_vectors = np.array([model.docvecs[f'doc_{i}'] for i in range(len(self.corpus))])
        return doc_vectors
class Bert:
    """This class vectorizes text using BERT sentence embeddings.

    Attributes:
        Corpus (list): List of documents.
        Transformer (obj): sentence-transformer object.
    """
    """Inits class with corpus and transformer."""
    def __init__(self, corpus, transformer):
        self.corpus = corpus
        self.transformer = transformer
        self.joined_corpus = None

    def join_docs(self):
        """Formats documents for transformation.

        Returns:
            Self. The joined_corpus attribute stores the formatted corpus.
        """
        sentence_transformer_docs = [' '.join(word) for word in [doc for doc in self.corpus]]
        self.joined_corpus = sentence_transformer_docs
        return self

    def transform_corpus(self):
        """Transforms corpus using BERT embeddings.

        Returns:
            An array of dimensions (# docs)*(embedding vector length).
        """
        assert self.joined_corpus, "Run '.join_docs()' before encoding documents."

        bert_docvecs = np.array(self.transformer.encode(self.joined_corpus))
        return bert_docvecs

class Autoencoder:
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer

    Taken from: https://github.com/Stveshawn/contextual_topic_identification/blob/master/model/Autoencoder.py
    """
    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(X_train, X_train,
                                        epochs=200,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(X_test, X_test), verbose=0)

class ConcatVectors:
    """This class concatenates document vectors.

    This class is adapted from https://github.com/Stveshawn/contextual_topic_identification/blob/master/model/model.py.
    The "15" in each concatenation was originally set by that repository's author as the
    relative LDA importance parameter, and it has been left unchanged.

    Attributes:
        lda_vecs (array): Array containing LDA vectors.
        d2v_vecs (array): Array containing Doc2Vec vectors.
        bert_vecs (array): Array containing BERT sentence-transformer vectors.
    """
    def __init__(self, lda_vecs, d2v_vecs, bert_vecs):
        """Inits class with lda_vecs, d2v_vecs, and bert_vecs"""
        self.lda_vecs = lda_vecs
        self.d2v_vecs = d2v_vecs
        self.bert_vecs = bert_vecs

    def transform_lda_d2v(self):
        """Concatenates lda and doc2vec vectors.

        Returns:
            Array of concatenated vectors.
        """
        concatted_vecs = np.c_[self.lda_vecs * 15, self.d2v_vecs]
        return concatted_vecs

    def transform_lda_bert(self):
        """Concatenates lda and BERT vectors.

        Returns:
            Array of concatenated vectors.
        """
        concatted_vecs = np.c_[self.lda_vecs * 15, self.bert_vecs]
        return concatted_vecs

def save(lda, lda_vecs, d2v, bert_vecs, lda_d2v_model, lda_d2v_vecs, lda_bert_model, lda_bert_vecs):
    """Saves lda vectors, doc2vec vectors, lda-bert vectors, and lda-doc2vec vectors."""
    lda.save('../models/lda/lda_model')
    with open('../models/lda/lda_vecs.pkl', 'wb') as f:
        pickle.dump(lda_vecs, f)

    d2v.save('../models/d2v/d2v_model')

    with open('../models/bert/bert_docvecs.pkl', 'wb') as f:
        pickle.dump(bert_vecs, f)

    lda_d2v_json = lda_d2v_model.to_json()
    with open("../models/lda_d2v/lda_d2v_autoencoder.json", "w") as f:
        f.write(lda_d2v_json)
    lda_d2v_model.save_weights("../models/lda_d2v/lda_d2v_model.h5")
    with open("../models/lda_d2v/lda_d2v_vectors.pkl", 'wb') as f:
        pickle.dump(lda_d2v_vecs, f)

    lda_bert_json = lda_bert_model.to_json()
    with open("../models/lda_bert/lda_bert_autoencoder.json", "w") as f:
        f.write(lda_bert_json)
    lda_bert_model.save_weights("../models/lda_bert/lda_bert_model.h5")
    with open("../models/lda_bert/lda_bert_vectors.pkl", 'wb') as f:
        pickle.dump(lda_bert_vecs, f)

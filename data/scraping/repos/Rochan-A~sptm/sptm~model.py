# -*- coding: utf-8 -*-

"""
    Adjust, train and optimize Model
"""

#######################################

import logging
import codecs
import multiprocessing
from shutil import copy2

import gensim.corpora as corpora
import gensim.models.wrappers as Wrappers
import gensim.utils as utils
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

from sptm.utils import force_unicode

__author__ = "Rochan Avlur Venkat"
__credits__ = ["Anupam Mediratta"]
__license__ = "MIT"
__version__ = "1.1.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

# Setup logging for gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                                        level=logging.INFO)

class Model:
    """Adjust, train and optimize LDA model

    This class is responsible for traning the Topic Model using Mallet's
    LDA which can be found [here](http://mallet.cs.umass.edu/topics.php)

    Attributes:
        mallet_path: Path to Mallet binary
        tokens: List of lists containing data index number and tokens
        id2word: Dictionary of the Corpus
        corpus: Term Document frequency
        alpha = Model alpha hyperparameter
        workers = Number of workers spawned while training the model
        prefix = prefix
        optimize_interval = Number of iterations after which to re-evaluate
            hyperparameters
        iterations = Number of iterations
        topic_threshold = topic threshold
        num_topics = Number of topics
        lda_model_mallet = Gensim Mallet LDA wrapper object
    """

    def __init__(self, mallet_path, tokens=None, input_path=None):
        """Inits Model with Mallet path, tokenized data or input path to open
        saved tokenized data.

        NOTE: If both input_path and tokens is given, tokens will always take
        higher preference

        Args:
            mallet_path: Location of Mallet binary
            input_path: Location of saved preprocessed tokens file
            tokens: tokens of preprocessed data

        Raises:
            IOError: Tokens file not found or not in specified format
            Exception: Not in specified structure
        """
        self.mallet_path = mallet_path
        self.tokens = []

        if (tokens is not None and input_path is None) or \
                            (tokens is not None and input_path is not None):
            # Use tokens list passed as an argument
            print('Using tokens passed as argument')
            try:
                for i, val in enumerate(tokens):
                    self.tokens.append(tokens[i][1:])
            except:
                raise Exception("Tokens list does not follow required " + \
                                                                "structure")

        elif tokens is None and input_path is not None:

            # Read the saved tokens file
            print('Opening tokens file')
            try:
                with codecs.open(input_path, 'r', encoding='utf8') as F:
                    for row in F:
                        token_in_row = row.split(",")
                        for i, val in enumerate(token_in_row):
                            token_in_row[i] = force_unicode(token_in_row[i])
                        self.tokens.append(token_in_row[1:])
            except IOError:
                raise IOError("File not found")
            except:
                raise Exception("Tokens list does not follow required " + \
                                                                "structure")
        elif tokens is None and input_path is None:
            print("Assuming load model from saved file, use Model.load()")
        else:
            print("Missing tokens data")

    def fit(self):
        """Generate the id2word dictionary and term document frequency of
        the given tokens

        NOTE: Should be called only after making sure that the tokens
        have been properly read

        Raises:
            Exception: self.tokens empty or not in required format
        """
        try:
                # Create Dictionary
                self.id2word = corpora.Dictionary(self.tokens)
                # Term Document Frequency
                self.corpus = \
                        [self.id2word.doc2bow(text) for text in self.tokens]
        except:
            raise Exception('tokens not compatible')

    def params(self, alpha=50, workers=multiprocessing.cpu_count(), \
                    prefix=None, optimize_interval=0, iterations=1000, \
                                        topic_threshold=0.0, num_topics=100):
        """Model parameters

        NOTE: These are the same parameters used while traning models
        for coherence computation. Call this function to re-initialize
        parameter values in that case

        Args:
            alpha: Alpha value (Dirichlet Hyperparameter)
            workers: Number of threads to spawn to parallel traning process
            prefix: prefix
            optimize_interval: Number of intervals after which to recompute
                hyperparameters
            iterations: Number of iterations
            topic_threshold: Topic threshold
            num_topics: Number of topics
        """
        self.alpha = alpha
        self.workers = workers
        self.prefix = prefix
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        self.topic_threshold = topic_threshold
        self.num_topics = num_topics

    def train(self):
        """Train LDA Mallet model using gensim's Mallet wrapper
        """
        self.lda_model_mallet = Wrappers.LdaMallet(self.mallet_path, \
            corpus=self.corpus, num_topics=self.num_topics, \
            alpha=self.alpha, id2word=self.id2word, \
            workers=self.workers, prefix=self.prefix, \
            optimize_interval=self.optimize_interval, \
            iterations=self.iterations, \
            topic_threshold=self.topic_threshold)

    def topics(self, num_topics=100, num_words=10):
        """Return top <num_words> words for the first <num_topics> topics

        Args:
            num_topics: Number of topics to print
            num_words: Number of top words to print for each topic

        Returns:
            List of topics and top words
        """
        return self.lda_model_mallet.print_topics(num_topics, num_words)

    def save(self, output_path):
        """Save the Mallet lDA model

        Also, save the document_topic distribution, corpus and inferencer

        Args:
            output_path: Location with filename to save the LDA model
        Raises:
            IOError: Error with output_path / File already exists
        """
        doctopic = self.lda_model_mallet.fdoctopics()
        inferencer = self.lda_model_mallet.finferencer()
        corpus = self.lda_model_mallet.fcorpusmallet()

        try:
            copy2(doctopic, output_path + "_doctopic")
            copy2(inferencer, output_path + "_inferencer")
            copy2(corpus, output_path + "_corpus")
        except:
            raise IOError('Error with output path / File already exists')
        self.lda_model_mallet.save(output_path)

    def get_coherence(self):
        """Compute Coherence Score of the model

        NOTE: You cannot compute the coherence score of a saved model

        Returns:
            Float value
        """
        coherence_model_lda = CoherenceModel(model=self.lda_model_mallet, \
                texts=self.tokens, dictionary=self.id2word, \
                coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda

    def optimum_topic(self, start=10, limit=100, step=11):
        """Compute c_v coherence for various number of topics

        if you want to change the parameters of the model while training,
        call Model.params() first as it uses the same parameters.

        NOTE: You cannot compute the coherence score of a saved model.

        Args:
            dictionary: Gensim dictionary
            corpus: Gensim corpus
            texts: List of input texts
            limit: Max num of topics

        Returns:
            Dictionary of {num_topics, c_v}
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = Wrappers.LdaMallet(self.mallet_path, \
                corpus=self.corpus, num_topics=num_topics, \
                alpha=self.alpha, id2word=self.id2word, \
                workers=self.workers, prefix=self.prefix, \
                optimize_interval=self.optimize_interval, \
                iterations=self.iterations, \
                topic_threshold=self.topic_threshold)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, \
                texts=self.tokens, dictionary=self.id2word, \
                coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        x = range(start, limit, step)
        out = dict()
        for m, cv in zip(x, coherence_values):
            out["num_topics"] = m
            out["c_v"] = round(cv, 4)
        return out

    def load(self, saved_model):
        """Load a Mallet LDA model previously saved

        Args:
            saved_model: Location to saved model
        Raises:
            IOError: File already present or location does not exist
        """
        try:
            self.lda_model_mallet = utils.SaveLoad.load(saved_model)
        except IOError:
            raise IOError('File already present or location does not exist')

class ModelVanilla:
    """Adjust, train and optimize LDA model

    This class is responsible for traning the Topic Model using Gensim's
    LDA.

    Attributes:
        tokens: List of lists containing data index number and tokens
        id2word: Dictionary of the Corpus
        corpus: Term Document frequency
        alpha = Model alpha hyperparameter
        workers = Number of workers spawned while training the model
        prefix = prefix
        optimize_interval = Number of iterations after which to re-evaluate
            hyperparameters
        iterations = Number of iterations
        topic_threshold = topic threshold
        num_topics = Number of topics
        lda_model = Gensim LDA object
    """

    def __init__(self, tokens=None, input_path=None):
        """Inits Model, tokenized data or input path to open saved tokenized
        data.

        NOTE: If both input_path and tokens is given, tokens will always take
        higher preference

        Args:
            input_path: Location of saved preprocessed tokens file
            tokens: tokens of preprocessed data

        Raises:
            IOError: Tokens file not found or not in specified format
            Exception: Not in specified structure
        """
        self.tokens = []

        if (tokens is not None and input_path is None) or \
                            (tokens is not None and input_path is not None):
            # Use tokens list passed as an argument
            print('Using tokens passed as argument')
            try:
                for i, val in enumerate(tokens):
                    self.tokens.append(tokens[i][1:])
            except:
                raise Exception("Tokens list does not follow required " + \
                                                                "structure")

        elif tokens is None and input_path is not None:

            # Read the saved tokens file
            print('Opening tokens file')
            try:
                with codecs.open(input_path, 'r', encoding='utf8') as F:
                    for row in F:
                        token_in_row = row.split(",")
                        for i, val in enumerate(token_in_row):
                            token_in_row[i] = force_unicode(token_in_row[i])
                        self.tokens.append(token_in_row[1:])
            except IOError:
                raise IOError("File not found")
            except:
                raise Exception("Tokens list does not follow required " + \
                                                                "structure")
        elif tokens is None and input_path is None:
            print("Assuming load model from saved file, use Model.load()")
        else:
            print("Missing tokens data")

    def fit(self):
        """Generate the id2word dictionary and term document frequency of
        the given tokens

        NOTE: Should be called only after making sure that the tokens
        have been properly read

        Raises:
            Exception: self.tokens empty or not in required format
        """
        try:
                # Create Dictionary
                self.id2word = corpora.Dictionary(self.tokens)
                # Term Document Frequency
                self.corpus = \
                        [self.id2word.doc2bow(text) for text in self.tokens]
        except:
            raise Exception('tokens not compatible')

    def params(self, alpha='symmetric', num_topics=100, distributed=False, \
                    chunksize=2000, passes=1, update_every=1, iterations=50):
        """Model parameters

        NOTE: These are the same parameters used while traning models
        for coherence computation. Call this function to re-initialize
        parameter values in that case

        Args:
            alpha: Can be set to an 1D array of length equal to the number of
            expected topics that expresses our a-priori belief for the each
            topics’ probability. Alternatively default prior selecting
            strategies can be employed by supplying a string:

                ’asymmetric’: Uses a fixed normalized asymmetric prior
                    of 1.0 / topicno.
                ’auto’: Learns an asymmetric prior from the corpus
                    (not available if distributed==True).
            num_topics: Number of topics
            distributed: Whether distributed computing should be used to
                accelerate training.
            chunksize: Number of documents to be used in each training chunk.
            passes: Number of passes through the corpus during training.
            update_every: Number of documents to be iterated through for each
                update. Set to 0 for batch learning, > 1 for online iterative
                learning.
            iterations: Number of iterations
        """
        self.alpha = alpha
        self.num_topics = num_topics
        self.distributed = distributed
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.iterations = iterations

    def train(self):
        """Train LDA model using gensim's LDA object
        """
        self.lda_model = LdaModel(corpus=self.corpus, \
                        num_topics=self.num_topics, alpha=self.alpha, \
                        id2word=self.id2word, distributed=self.distributed, \
                        chunksize=self.chunksize, passes=self.passes, \
                        update_every=self.update_every, \
                                                iterations=self.iterations)

    def topics(self, num_topics=100, num_words=10):
        """Return top <num_words> words for the first <num_topics> topics

        Args:
            num_topics: Number of topics to print
            num_words: Number of top words to print for each topic

        Returns:
            List of topics and top words
        """
        return self.lda_model.print_topics(num_topics, num_words)

    def save(self, output_path):
        """Save the lDA model

        Args:
            output_path: Location with filename to save the LDA model
        Raises:
            IOError: Error with output_path / File already exists
        """
        self.lda_model.save(output_path)

    def get_coherence(self):
        """Compute Coherence Score of the model

        NOTE: You cannot compute the coherence score of a saved model

        Returns:
            Float value
        """
        coherence_model_lda = CoherenceModel(model=self.lda_model, \
                texts=self.tokens, dictionary=self.id2word, \
                coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda

    def optimum_topic(self, start=10, limit=100, step=11):
        """Compute c_v coherence for various number of topics

        if you want to change the parameters of the model while training,
        call Model.params() first as it uses the same parameters.

        NOTE: You cannot compute the coherence score of a saved model.

        Args:
            start: Starting number of topics
            limit: Limit number of topics
            step: Step size

        Returns:
            Dictionary of {num_topics, c_v}
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = LdaModel(corpus=self.corpus, \
                        num_topics=self.num_topics, alpha=self.alpha, \
                        id2word=self.id2word, distributed=self.distributed, \
                        chunksize=self.chunksize, passes=self.passes, \
                        update_every=self.update_every, \
                                                iterations=self.iterations)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, \
                texts=self.tokens, dictionary=self.id2word, \
                coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        x = range(start, limit, step)
        out = dict()
        for m, cv in zip(x, coherence_values):
            out["num_topics"] = m
            out["c_v"] = round(cv, 4)
        return out

    def load(self, saved_model):
        """Load a LDA model previously saved

        Args:
            saved_model: Location to saved model
        Raises:
            IOError: File already present or location does not exist
        """
        try:
            self.lda_model = utils.SaveLoad.load(saved_model)
        except IOError:
            raise IOError('File already present or location does not exist')
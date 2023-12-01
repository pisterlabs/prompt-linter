"""
Gensim implementation of Topic Extraction from texts corpus using LDA algorithm
Including Mallet's implementation of LDA algorithm

Installation
Run in python console : import nltk; nltk.download('stopwords')

Run in terminal or command prompt : python3 -m spacy download en

Download Mallet from http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
Unzip preferably to C:/Mallet/
Add an environment variable MALLET_HOME with the path to Mallet directory
You might have to update smart_open library

Usage:
gmte = Gensim_Mallet_Topic_Extractor('english')
gmte.extract_topics(data, num_topics) <- data being a list of texts
                                         and num_topics a positive integer

To compute optimal number of topics
gmte.compute_coherence_values(start=2, limit=40, step=1)

To get most representative documents for each topic
gmte.compute_coherence_values(start=2, limit=40, step=1)
df_topic_sents_keywords = gmte.format_topics_sentences()
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic',
                             'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

To get topic distribution across documents
gmte.get_topic_distribution()
"""

import os
import pandas as pd
from pprint import pprint

# Gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore
from gensim.models.wrappers import LdaMallet

# Spacy for lemmatization
import spacy

# NLTK for stopwords
from nltk.corpus import stopwords

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
import warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GensimMalletTopicExtractor:
    def __init__(self, language='english', stopwords_extent=None):
        self.language2la = {
            'english': 'en',
            'french': 'fr',
            'spanish': 'es'
        }
        if language not in self.language2la:
            raise ValueError('Language must be "english", "french" or "spanish"')
        self.language = language
        self.stop_words = stopwords.words(self.language)
        if stopwords_extent is str or stopwords_extent is list:
            self.stop_words.extend(stopwords_extent)
        self.df_topic_sents_keywords = None
        self.bigram = None
        self.bigram_phraser = None
        self.trigram = None
        self.trigram_phraser = None
        self.vis = None
        self.data = None
        self.data_words = None
        self.data_words_nostops = None
        self.data_words_bigrams = None
        self.data_words_trigrams = None
        self.nlp = None
        self.data_lemmatized = None
        self.id2word = None
        self.texts = None
        self.corpus = None
        self.mallet_path = None
        self.lda_model = None
        self.coherence_model_lda = None
        self.coherence_lda = None
        self.coherence_values = []
        self.model_list = []
        self.optimal_number_of_topics = None
        self.optimal_model = None
        self.optimal_topics = None

    @staticmethod
    def sent_to_words(sentences, remove_punctuation=True):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(simple_preprocess(str(sentence), deacc=remove_punctuation))

    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in texts]

    def make_bigrams(self, texts):
        self.bigram = Phrases(self.data_words, min_count=5, threshold=100)
        self.bigram_phraser = Phraser(self.bigram)
        return [self.bigram_phraser[doc] for doc in texts]

    def make_trigrams(self, texts):
        tokens_ = self.bigram_phraser[texts]
        self.trigram = Phrases(tokens_, threshold=100)
        self.trigram_phraser = Phraser(self.trigram)
        return [self.trigram_phraser[self.bigram_phraser[doc]] for doc in texts]

    def lemmatization(self, texts, allowed_postags=None):
        if allowed_postags is None:
            allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def view_terms_frequency(self, text_id, first_words=20):
        # Human readable format of corpus (term-frequency)
        list_ = [[(self.id2word[id_], freq) for id_, freq in text[:first_words]] for text in self.corpus[text_id]]
        pprint(list_)

    def visualize_lda(self):
        # Visualize the topics
        # pyLDAvis.enable_notebook()
        self.vis = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.id2word)
        print(self.vis)

    def instanciate_model(self, num_topics, passes, iterations,
                          enable_mallet, optimize_interval, topic_threshold, show_topics_on_creation=False):
        if enable_mallet is True:
            # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
            os.environ.update({'MALLET_HOME': r'C:/mallet-2.0.8/'})
            self.mallet_path = 'C:\\mallet-2.0.8\\bin\\mallet'  # update this path
            self.lda_model = LdaMallet(self.mallet_path,
                                       corpus=self.corpus,
                                       num_topics=num_topics,
                                       id2word=self.id2word,
                                       iterations=iterations,
                                       optimize_interval=optimize_interval,
                                       topic_threshold=topic_threshold)
            print('Mallet LDA model built\n')
            if show_topics_on_creation is True:
                pprint(self.lda_model.show_topics(formatted=False))
        else:
            self.lda_model = LdaMulticore(corpus=self.corpus,
                                          id2word=self.id2word,
                                          num_topics=num_topics,
                                          random_state=100,
                                          chunksize=500,
                                          passes=passes,
                                          iterations=iterations,
                                          per_word_topics=True)
            print('LDA_MultiCore model built\n')
            if show_topics_on_creation is True:
                pprint(self.lda_model.print_topics())

    def extract_topics(self, data, num_topics, passes=10, iterations=500,
                       enable_mallet=True, optimize_interval=0,
                       topic_threshold=0.0):
        self.data = data
        print('\nEXTRACTING ' + str(num_topics) + ' TOPICS')
        self.data_words = list(self.sent_to_words(self.data, True))
        # Remove Stop Words
        print('\nRemoving stopwords')
        self.data_words_nostops = self.remove_stopwords(self.data_words)
        # Form Bigrams
        print('Looking for bigrams')
        self.data_words_bigrams = self.make_bigrams(self.data_words_nostops)
        # Form Trigrams
        print('Looking for trigrams')
        self.data_words_trigrams = self.make_trigrams(self.data_words_nostops)
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        print('Loading Spacy with ' + self.language + ' dictionary')
        self.nlp = spacy.load(self.language2la[self.language], disable=['parser', 'ner'])
        # Do lemmatization keeping only noun, adj, vb, adv
        print('Lemmatizing')
        self.data_lemmatized = self.lemmatization(self.data_words_trigrams,
                                                  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # Create Dictionary
        print('Creating dictionary')
        self.id2word = corpora.Dictionary(self.data_lemmatized)
        # Create Corpus
        print('Creating corpus')
        self.texts = self.data_lemmatized
        # Term Document Frequency
        print('Computing document frequency')
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]
        # Build LDA model
        print('\nEnable_mallet is', enable_mallet, '\n')
        self.instanciate_model(num_topics, passes, iterations,
                               enable_mallet, optimize_interval, topic_threshold,
                               show_topics_on_creation=True)
        # print(self.lda_model[self.corpus])
        # Compute Perplexity
        # a measure of how good the model is. lower the better.
        if hasattr(self.lda_model, 'log_perplexity'):
            print('\nPerplexity: ', self.lda_model.log_perplexity(self.corpus))

        # Compute Coherence Score
        print('\nComputing coherence model')
        self.coherence_model_lda = CoherenceModel(model=self.lda_model,
                                                  texts=self.data_lemmatized,
                                                  dictionary=self.id2word,
                                                  coherence='c_v')
        print('Getting coherence')
        self.coherence_lda = self.coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', self.coherence_lda)

        if enable_mallet is False:
            self.visualize_lda()

    def view_optimal_topics(self, num_words=20):
        pprint(self.optimal_model.print_topics(num_words=num_words))

    def compute_coherence_values(self, limit, start=2, step=3, passes=10,
                                 iterations=500, enable_mallet=True,
                                 optimize_interval=0, topic_threshold=0.0):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        for num_topics in range(start, limit, step):
            print('\n' + '*'*10 + ' COMPUTING COHERENCE FOR ' + str(num_topics) + ' TOPICS ' + '*'*10)
            self.instanciate_model(num_topics, passes, iterations,
                                   enable_mallet, optimize_interval, topic_threshold,
                                   show_topics_on_creation=False)
            self.model_list.append(self.lda_model)
            coherence_model = CoherenceModel(model=self.lda_model,
                                             texts=self.data_lemmatized,
                                             dictionary=self.id2word,
                                             coherence='c_v')
            self.coherence_values.append(coherence_model.get_coherence())

        # Show graph
        x = range(start, limit, step)
        plt.plot(x, self.coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.show()

        # Print the coherence scores
        for m, cv in zip(x, self.coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

        optimal_model_index = self.coherence_values.index(max(self.coherence_values))
        self.optimal_number_of_topics = start + optimal_model_index
        self.optimal_model = self.model_list[optimal_model_index]
        print('\nOptimal number of topics is ' + str(self.optimal_number_of_topics) +
              ' with coherence score : ' + str(self.coherence_values[optimal_model_index]))
        self.optimal_topics = self.optimal_model.show_topics(num_topics=self.optimal_number_of_topics,
                                                             num_words=20, formatted=False)
        self.view_optimal_topics()

    def format_topics_sentences(self, ldamodel=None):
        if ldamodel is None and self.optimal_model is not None:
            ldamodel = self.optimal_model
        elif ldamodel is None and self.lda_model is not None:
            ldamodel = self.lda_model
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num),
                                                                      round(prop_topic, 4),
                                                                      topic_keywords]),
                                                           ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(self.data)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    def get_most_representative_documents(self):
        # Group top 5 sentences under each topic
        sent_topics_sorteddf_mallet = pd.DataFrame()

        if self.df_topic_sents_keywords is None:
            self.df_topic_sents_keywords = self.format_topics_sentences()
        # Format
        df_dominant_topic = self.df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        sent_topics_outdf_grpd = self.df_topic_sents_keywords.groupby('Dominant_Topic')

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                     grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                    axis=0)

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
        # Format
        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
        # Show
        sent_topics_sorteddf_mallet.head()

        for i in range(len(sent_topics_sorteddf_mallet)):
            print(i, sent_topics_sorteddf_mallet.loc[i, 'Text'])

    def get_topic_distribution(self):
        if self.df_topic_sents_keywords is None:
            self.df_topic_sents_keywords = self.format_topics_sentences()
        # Number of Documents for Each Topic
        topic_counts = self.df_topic_sents_keywords['Dominant_Topic'].value_counts()
        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)
        # Topic Number and Keywords
        topic_num_keywords = self.df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
        # Show
        print(df_dominant_topics)

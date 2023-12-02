import os, email
import numpy as np 
import pandas as pd
import datetime
import random
import threading
from enum import IntEnum
import json
from collections import Counter

from EmailHelperFunctions import get_text_from_email, split_email_addresses, clean_email
from MDS import cmdscale

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Set the random seed for reproducability
random.seed(1)

# TODO: Set the sample sizes
optimum_sample_size = 1000
sample_size         = 15000

# Min and max number of topics
min_topic_size = 3
max_topic_size = 35

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
# Mallet is Java based, so make sure Java is installed


class DocumentTypeEnum(IntEnum):
    unknownType = 0
    emailType = 1           # 'emails'
    documentType = 2        # 'documents'

class createModelThread(threading.Thread):
    def __init__(self, tmo):
        threading.Thread.__init__(self)
        self.tm = tmo
        self.mallet_path = ''

    def run(self):
        print('Starting createModelThread: {}'.format(datetime.datetime.now()))

        #
        # These functions should set the following lists of strings:
        #       self.tm.text_clean
        #       self.tm.optimum_text_clean
        #
        if self.tm.documentType == DocumentTypeEnum.emailType:
            self.process_emails()
        elif self.tm.documentType == DocumentTypeEnum.documentType:
            self.process_documents()
        else:
            print('Error in createModelThread, document type not specified')
            return False

        # Create the dictionary for the model
        self.tm.dictionary = corpora.Dictionary(self.tm.optimum_text_clean)

        # Create the text_term_matrix
        self.tm.text_term_matrix = [self.tm.dictionary.doc2bow(text) for text in self.tm.optimum_text_clean]

        #
        # Automatically determine the number of topics if required
        #
        if self.tm.numberOfTopics <= 0:

            # Compute optimial number of topics only
            optimalTopicsOnly = False

            if self.tm.numberOfTopics == -1:
                optimalTopicsOnly = True

            # Set paths needed by Mallet
            self.mallet_distribution = os.environ["MALLET_HOME"]
            self.mallet_path = os.path.join(self.mallet_distribution, 'bin', 'mallet')

            # Compute the coherence values using Mallet
            model_list, coherence_values = self.compute_coherence_values(dictionary=self.tm.dictionary,\
                                                                         corpus=self.tm.text_term_matrix,\
                                                                         texts=self.tm.optimum_text_clean,\
                                                                         limit=max_topic_size,\
                                                                         start=5,step=5)

            # Find the optimal number of topics
            limit = max_topic_size
            start = 5
            step = 5
            x = list(range(start, limit, step))
            self.tm.numberOfTopics = x[np.argmax(coherence_values)]
            print('Optimum number of topics is: {}'.format(self.tm.numberOfTopics))

            if optimalTopicsOnly:
                return True

        # Create the dictionary and term matrix used by LDA
        self.tm.dictionary = corpora.Dictionary(self.tm.text_clean)
        self.tm.text_term_matrix = [self.tm.dictionary.doc2bow(text) for text in self.tm.text_clean]

        # Buid the Gensim LDA model
        Lda = gensim.models.ldamodel.LdaModel

        self.tm.ldamodel = Lda(self.tm.text_term_matrix, num_topics=self.tm.numberOfTopics, id2word = self.tm.dictionary, passes=30)

        #
        # Get token count proportion statistics for the plot.  Also add topic
        # category to sub_df
        #
        topic_token_count = [0 for i in range(0,self.tm.numberOfTopics)]
        topicSeries = []
        probabilitySeries = []

        # Note: Must use len(text_clean) because len(text_clean) <= sample_size because some documents may have been removed (e.g. were HTML)
        for i in range(0,len(self.tm.text_clean)):
            assignedTopic, topicProbability = self.assigned_topic(self.get_candidate_topics(i))
            topic_token_count[assignedTopic] += len(self.tm.text_term_matrix[i])
            topicSeries.append(assignedTopic)
            probabilitySeries.append(topicProbability)

        self.tm.token_count_proportions = np.array(topic_token_count) / sum(topic_token_count)
        self.tm.sub_df['topic'] = topicSeries
        self.tm.sub_df['probability'] = probabilitySeries

        # Sort by probability
        self.tm.sub_df.sort_values(by=['probability'], ascending=False, inplace=True)
        self.tm.sub_df.drop(columns=['probability'], inplace=True)

        # Write the data frame (with topic assignment) to disk so it can be read when the user switches the number of topics
        self.tm.sub_df.to_csv('./state/TopicData/topic_{0}.csv'.format(self.tm.numberOfTopics), index=False)

        self.tm.modelBuilt = True

        print('Finished createModelThread: {}'.format(datetime.datetime.now()))
        return True

    def createStopWordList(self, df):

        stop_words_path = './state/TopicData/stopwords.json'

        # Read stop words from a file if it already exists
        if os.path.isfile(stop_words_path):
            with open(stop_words_path, 'r') as f:
                return json.load(f)

        # Get word count vector
        cv = CountVectorizer(min_df = 0.01, max_df = 1.0)
        word_count_vector = cv.fit_transform(df.content)
        feature_names = cv.get_feature_names()

        # Calculate TF-IDF weights for words in documents
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        transformed_weights = tfidf_transformer.fit_transform(word_count_vector)

        # Create a count of least informative words from each document
        counter = Counter()

        def findLeastInformativeWordAndUpdateCounter(row):
            cols = row.nonzero()[1]
            vals = row.toarray().ravel()[cols].tolist()
            if len(vals) > 0:
                counter[cols[np.argmin(vals)]] += 1

        # Find least least informative word for each document
        for row in transformed_weights:
            findLeastInformativeWordAndUpdateCounter(row)

        # Now find the words (from the indicies)
        most_common_indicies = [x[0] for x in counter.most_common()]
        common_words = [feature_names[idx] for idx in most_common_indicies]

        # Remove English stop words
        final_list = []
        for word in common_words:
            if word not in set(stopwords.words('english')):
                final_list.append(word)

        final_list = final_list[:20]
        print('The stop words are: ', final_list)

        # Write to a file so it does not need to be computed each time
        with open(stop_words_path, 'w') as f:
            json.dump(final_list, f)

        return final_list

    def process_emails(self):
        global sample_size
        global optimum_sample_size

        # Load the entire email corpus
        emails_df = pd.read_csv('email_data.csv').dropna(subset=["body"])
        emails_df.columns = emails_df.columns.str.lower()

        # Adjust the sample size if necessary
        if sample_size > len(emails_df):

            # In case the hard coded sample size is greater than the number of
            # emails, then just use the number of emails
            sample_size = len(emails_df)

            # Use 10% of the sample size when determining the optimal number of
            # topics
            optimum_sample_size = round(sample_size * 0.1)

        # Sample emails from entire csv
        emails_df = emails_df.sample(n=sample_size)

        # Convert columns to the correct type
        emails_df['id'] = pd.to_numeric(emails_df['id'])
        emails_df['date'] = emails_df['date'].apply(lambda x: pd.to_datetime(str(x)))

        # Parse the emails into a list email objects
        messages = list(map(email.message_from_string, emails_df['body']))

        # Parse content from emails
        emails_df['content'] = list(map(get_text_from_email, messages))
        del messages
        emails_df = emails_df.drop(['body'], axis=1)

        # Remove emails that are HTML
        emails_df = emails_df[(emails_df['content'].str.lower()).str.find("<head>") == -1]

        self.tm.sub_df = emails_df

        # Create stop words
        stopWords = self.createStopWordList(emails_df)

        # Set the text_clean to be used to create the LDA model
        self.tm.text_clean = []
        for text in self.tm.sub_df['content']:
            self.tm.text_clean.append(clean_email(text, stopWords).split())

        # Use a smaller sample to find the coherence values in
        # compute_coherence_values()
        self.tm.optimum_text_clean = [
            self.tm.text_clean[i] for i in random.sample(range(len(self.tm.text_clean)), optimum_sample_size)
        ]

    def process_documents(self):
        # TODO
        self.tm.text_clean = []
        self.tm.optimum_text_clean = []

    def compute_coherence_values(self, dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Args:
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

        Returns:
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics

        Raises:
            None
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def assigned_topic(self, candidateTopics):
        largest = (0,0.0)
        for topic in candidateTopics:
            if topic[1] > largest[1]:
                largest = topic
        return largest

    def get_candidate_topics(self, index):
        return self.tm.ldamodel.get_document_topics(self.tm.text_term_matrix[index])


"""
Module containing model fitting code for a web application that implements a
research interests identification topic model using gensim(Choice between LDA, LSI, LDAMallet and HDP).
When run as a module, this will load a json dataset, train a decomposition
model using gensim, optimize the number of topics, and then pickle
the resulting optimum model object to disk.
"""
from cleaning import database_cleaner
from nlp_pipeline import clean_text, add_stopwords
from pyLDAvis_mallet import get_LDA_data

import numpy as np
import pandas as pd

# Gensim
import gensim
from gensim.models import CoherenceModel
from gensim import models, corpora, similarities
from gensim.test.utils import datapath

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

# NLTK
from nltk.corpus import stopwords
from nltk import word_tokenize

# spacy for lemmatization
import spacy

import gzip
import os
import pickle

import warnings
warnings.filterwarnings("ignore",category=UserWarning)

# mallet_path = '~/Documents/GitHub/capstone/mallet-2.0.8/bin/mallet' # update this path if needed
mallet_path = '../data/mallet-2.0.8/bin/mallet' # update this path if needed
stopwords_ = stopwords.words('english')
more_stopwords_ = add_stopwords()
stopwords_.extend(more_stopwords_)

class MyGenSimModel():
    """A gensim based topic model to identify research areas given information about papers:
        - cleans the json dataset
        - Vectorize the raw text into features.
        - Fit a topic model to the resulting features.
    """

    def __init__(self, num_topics=9, algorithm='LDAMallet', tf_idf=False, bigrams=False, trigrams=False, lemmatization=False):
        self.num_topics = num_topics
        self.algorithm = algorithm
        self.tf_idf = tf_idf
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.lemmatization = lemmatization

    def transform(self, data):
        """Transform training data."""
        # For gensim we need to tokenize the data and filter out stopwords
        self.tokens = [clean_text(doc, stopwords_) for doc in data]

        # bigrams
        if self.bigrams:
            bigram = models.Phrases(self.tokens, min_count=5, threshold=100) # higher threshold fewer phrases.
            bigram_mod = models.phrases.Phraser(bigram)
            self.tokens = make_bigrams(self.tokens, bigram_mod)

        # trigrams
        if self.trigrams:
            bigram = models.Phrases(self.tokens, min_count=5, threshold=100)
            bigram_mod = models.phrases.Phraser(bigram)
            trigram = models.Phrases(bigram[self.tokens], threshold=100)
            trigram_mod = models.phrases.Phraser(trigram)
            self.tokens = make_trigrams(self.tokens, bigram_mod, trigram_mod)

        # lemmatization
        if self.lemmatization:
            # Initialize spacy 'en_core_web_sm' model, keeping only tagger component (for efficiency)
            spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            # Do lemmatization keeping only noun, adj, vb, adv
            self.tokens = do_lemmatization(spacy_nlp=spacy_nlp, texts=self.tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # Again remove stopwords after doing lemmatization
        self.tokens = [[token for token in doc if token not in stopwords_] for doc in self.tokens]

        # Build a Dictionary - association word to numeric id
        self.dictionary = corpora.Dictionary(self.tokens)

        # Transform the collection of texts to a numerical form [(word_id, count), ...]
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokens]

        # tf-idf vectorizer
        if self.tf_idf:
            self._tfidf_model = models.TfidfModel(self.corpus, id2word=self.dictionary)
            self.corpus = self._tfidf_model[self.corpus]

    def fit(self):
        """Fit on transformed training data."""
        # topic model
        if self.algorithm == 'LDA':
            # Build a Latent Dirichlet Allocation Model
            self._model = models.ldamodel.LdaModel(corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary)

        elif self.algorithm == 'LDAMallet':
            # Build a Mallet Model (doesn't work with tf-idf)
            self._model = models.wrappers.LdaMallet(mallet_path, workers=8, corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary, prefix='../data/mallet_files/', optimize_interval=10, iterations=1000)

        elif self.algorithm == 'LSI':
            # Build a Latent Semantic Indexing Model
            self._model = models.LsiModel(corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary)
        
        elif self.algorithm == 'HDP':
             # Build a Hiearchical Dirichlet Process Model (doesn't need number of topics)
            self._model = models.hdpmodel.HdpModel(corpus=self.corpus, id2word=self.dictionary)

    def transform_new(self, search_text):
        """Return transformed new data."""
        bow = self.dictionary.doc2bow(clean_text(search_text, stopwords_))
        if self.tf_idf:
            return self._model[self._tfidf_model[bow]]
        return self._model[bow]

    def perplexity(self):
        """Returns perplexity for LDA model. Measures per-word likelihood bound, using a chunk of documents as evaluation corpus."""
        return self._model.log_perplexity(self.corpus)

    def coherence_score(self):
        """Returns topic coherence for topic models. This is the implementation of the four stage topic coherence pipeline."""
        coherence_model = CoherenceModel(model=self._model, texts=self.tokens, dictionary=self.dictionary, coherence='c_v')
        return coherence_model.get_coherence()

    def most_similar(self, search_text, top_n=5):
        """Returns top-n most similar professors for a given search text (cleaned and tokenized)."""
        self.lda_index = similarities.MatrixSimilarity(self._model[self.corpus])
        similarity_results = self.lda_index[self.transform_new(search_text)]
        similarity_results = sorted(enumerate(similarity_results), key=lambda item: -item[1])
        return np.array(similarity_results[:top_n])

    def visualize_lda_model(self, **kwargs):
        """Visualize LDA model using pyLDAvis"""
        vis = pyLDAvis.gensim.prepare(self._model, self.corpus, self.dictionary,sort_topics=False, **kwargs)
        return vis

    def visualize_lda_mallet(self, **kwargs):
        """Visualize LDA model using pyLDAvis"""
        dataDir = "../data/mallet_files" # update this if needed
        statefile = 'state.mallet.gz'
        data = get_LDA_data(dataDir, statefile)
        vis = pyLDAvis.prepare(sort_topics=False, **data, **kwargs)
        return vis

    def format_document_topics(self, top_n=12):
        """Returns a dataframe with top-n dominant topics, contribution of these topics to document
        and keywords for them"""
        # Init output
        doc_topics_df = pd.DataFrame()

        # Get main topic in each document
        for idx, row in enumerate(self._model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topics, Perc Contributions and Keywords list for each document
            prop_topics, topic_nums, topic_keywords_list = [], [], []
            for j, (topic_num, prop_topic) in enumerate(row):
                if j in list(range(top_n)):  # top-n dominant topics
                    wp = self._model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    prop_topics.append(round(100 * prop_topic, 2))
                    topic_nums.append(int(topic_num) + 1)# To match pyLDAvis topic numbers
                    topic_keywords_list.append(topic_keywords)
                else:
                    break
            doc_topics_df = doc_topics_df.append(pd.Series([topic_nums, prop_topics, topic_keywords_list]), ignore_index=True)

        doc_topics_df.columns = ['Dominant_Topics', 'Perc_Contributions', 'Topic_Keywords_List']

        return doc_topics_df

def get_data(filename):
    """Load raw data from a file and return paper abstracts and titles.
    Parameters
    ----------
    filename: The path to a json file containing the university database.
    Returns
    -------
    corpus: A numpy array containing abstracts.
    """
    df_cleaned = database_cleaner(filename)
    # For nlp, only retaining faculty_name, research_areas, paper_titles, abstracts
    df_filtered = df_cleaned[['faculty_name', 'research_areas', 'paper_titles', 'abstracts']]
    missing = df_filtered['paper_titles'] == ''
    df_nlp = df_filtered[~missing]
    # Choosing abstracts and paper_titles to predict topics for a professor
    df_nlp['research_areas'] = df_nlp['research_areas'].apply(lambda x: " ".join(x))
    data = (df_nlp['paper_titles'] + df_nlp['abstracts'] + df_nlp['research_areas']).values
    return data

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def do_lemmatization(spacy_nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = spacy_nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, limit, start=5, step=1, algorithm='LDAMallet'):
    """
    Compute c_v coherence for various number of topics for a LDA/LDAMallet given model.

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    coherence_values : Coherence values corresponding to the model with respective number of topics
    """
    coherence_values = []

    for num_topics in range(start, limit, step):
        if algorithm == 'LDAMallet':
            model = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        else:
            model = models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return coherence_values

def coherence_plot(list_num_topics, coherence_values, title):
    """ Generates coherence plot given the range of topics and corresponding coherence values."""
    plt.plot(list_num_topics, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.title(title)
    plt.savefig('plots/' + title + '.png', bbox_inches='tight')
    plt.close()

def compare_models(data, model_list):
    """Return the best model on the basis of coherence score for a given number of topics."""
    coherence_scores = []
    for model in model_list:
        model.transform(data)
        model.fit()
        coherence_scores.append(model.coherence_score())
    return coherence_scores, model_list[np.argmax(np.array(coherence_scores))]

if __name__ == '__main__':
    data = get_data('../data/json/majors_database.json')

    # Fit LDA to training data(doesn't work with tf-idf)
    lda_model = MyGenSimModel(num_topics=11, algorithm='LDA', tf_idf=False, bigrams=True, trigrams=False, lemmatization=True)
    lda_model.transform(data)
    lda_model.fit()
    print(lda_model.coherence_score())

    # Fit LDAMallet to training data(doesn't work with tf-idf)
    ldamallet_model = MyGenSimModel(num_topics=11, algorithm='LDAMallet', tf_idf=False, bigrams=True, trigrams=False, lemmatization=True)
    ldamallet_model.transform(data)
    ldamallet_model.fit()
    print(ldamallet_model.coherence_score())

    # Append to pge_database with updated predicted_research_areas based on top-10 features
    pge_df = database_cleaner('../data/json/majors_database.json')
    doc_topics_df_LDA = lda_model.format_document_topics()
    doc_topics_df_LDAMallet = ldamallet_model.format_document_topics()
    pge_df_updated_LDA = pd.concat([pge_df, doc_topics_df_LDA], axis=1)
    pge_df_updated_LDAMallet = pd.concat([pge_df, doc_topics_df_LDAMallet], axis=1)
    pge_df_updated_LDA.to_json(path_or_buf='../data/json/final_gensim_database_LDA.json')
    pge_df_updated_LDAMallet.to_json(path_or_buf='../data/json/final_gensim_database_LDAMallet.json')

    # Save html for the pyLDAvis visualization of LDAMallet model
    vis_LDA = lda_model.visualize_lda_model()
    vis_LDAMallet = ldamallet_model.visualize_lda_mallet()
    pyLDAvis.save_html(data=vis_LDA, fileobj="templates/LDA.html") 
    pyLDAvis.save_html(data=vis_LDAMallet, fileobj="templates/LDAMallet.html") 

    # Pickle model (has associated dictionary)
    with open('../data/pickle/pge_gensim_LDA.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    with open('../data/pickle/pge_gensim_LDAMallet.pkl', 'wb') as f:
        pickle.dump(ldamallet_model, f)

"""
Author = Cory J. Cascalheira
Date = 06/17/2023

The purpose of this script is to create features for the LGBTQ MiSSoM dataset.

The core code is heavily inspired by the following resources:
- https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
- https://radimrehurek.com/gensim/

Issues with importing pyLDAvis.gensim, solved with: https://github.com/bmabey/pyLDAvis/issues/131

Resources for working with spaCy
- https://spacy.io/models
- https://stackoverflow.com/questions/51881089/optimized-lemmitization-method-in-python

# Regular expressions in Python
- https://docs.python.org/3/howto/regex.html
"""

#region LOAD AND IMPORT

# Load core dependencies
import os
import pandas as pd
import numpy as np
import time

# Load plotting tools
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

# Import tool for regular expressions
import re

# Import NLTK
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('amp')

# Load Gensim libraries
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.downloader as api

# Initialize spaCy language model
# Must download the spaCy model first in terminal with command: python -m spacy download en_core_web_sm
# May need to restart IDE before loading the spaCy pipeline
import importlib_metadata
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Set file path
my_path = os.getcwd()

# Import data
missom_coded = pd.read_csv(my_path + '/data/cleaned/features/missom_coded_feat01.csv')
missom_not_coded = pd.read_csv(my_path + '/data/cleaned/features/missom_not_coded_feat01.csv')

#endregion

#region WORD2VEC MODEL ------------------------------------------------------------------

# MISSOM CODED DATASET ------------------------------------------------------------------

# Create empty list
corpus_coded = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in missom_coded['text'].astype(str).tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_coded.append(filtered_sentence)

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_coded:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

# Add average word embeddings to the MiSSoM coded data set
missom_coded1 = pd.concat([missom_coded, embedding_df], axis=1)

# MISSOM NOT CODED DATASET --------------------------------------------------------

# Create empty list
corpus_not_coded = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in missom_not_coded['text'].astype(str).tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_not_coded.append(filtered_sentence)

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_not_coded:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

# Add average word embeddings to the MiSSoM not coded data set
missom_not_coded1 = pd.concat([missom_not_coded, embedding_df], axis=1)

# Export files
missom_coded1.to_csv(my_path + '/data/cleaned/features/missom_coded_feat02a.csv')
missom_not_coded1.to_csv(my_path + '/data/cleaned/features/missom_not_coded_feat02a.csv')

#endregion

#region TOPIC MODELING ----------------------------------------------------------

# HELPER FUNCTIONS --------------------------------------------------------------

def transform_to_words(sentences):

    """
    A function that uses Gensim's simple_preprocess(), transforming sentences into tokens of word unit size = 1 and removing
    punctuation in a for loop.

    Parameters
    -----------
    sentences: a list
        A list of text strings to preprocess
    """

    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(word_list):

    """
    A function to remove stop words with the NLTK stopword data set. Relies on NLTK.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_list]


def make_bigrams(word_list):
    """
    A function to transform a list of words into bigrams if bigrams are detected by gensim. Relies on a bigram model
    created separately (see below). Relies on Gensim.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [bigram_mod[doc] for doc in word_list]


def make_trigrams(word_list):
    """
    A function to transform a list of words into trigrams if trigrams are detected by gensim. Relies on a trigram model
    created separately (see below). Relies on Gensim.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [trigram_mod[bigram_mod[doc]] for doc in word_list]


def lemmatization(word_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']):
    """
    A function to lemmatize words in a list. Relies on spaCy functionality.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    allowed_postags: a list
        A list of language units to process.
    """
    # Initialize an empty list
    texts_out = []

    # For everyone word in the word list
    for word in word_list:

        # Process with spaCy to lemmarize
        doc = nlp(" ".join(word))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # Returns a list of lemmas
    return texts_out


def get_optimal_lda(dictionary, corpus, limit=30, start=2, step=2):
    """
    Execute multiple LDA topic models and computer the perplexity and coherence scores to choose the LDA model with
    the optimal number of topics. Relies on Gensim.

    Parameters
    ----------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    limit: an integer
        max num of topics
    start: an integer
        number of topics with which to start
    step: an integer
        number of topics by which to increase during each model training iteration

    Returns
    -------
    model_list: a list of LDA topic models
    coherence_values: a list
        coherence values corresponding to the LDA model with respective number of topics
    perplexity_values: a list
        perplexity values corresponding to the LDA model with respective number of topics
    """
    # Initialize empty lists
    model_list = []
    coherence_values = []
    perplexity_values = []

    # For each number of topics
    for num_topics in range(start, limit, step):

        # Train an LDA model with Gensim
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                                                update_every=1, chunksize=2000, passes=10, alpha='auto',
                                                per_word_topics=True)

        # Add the trained LDA model to the list
        model_list.append(model)

        # Compute UMass coherence score and add to list  - lower is better
        # https://radimrehurek.com/gensim/models/coherencemodel.html
        # https://www.os3.nl/_media/2017-2018/courses/rp2/p76_report.pdf
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        coherence_values.append(coherence)

        # Compute Perplexity and add to list - lower is better
        perplex = model.log_perplexity(corpus)
        perplexity_values.append(perplex)

    return model_list, coherence_values, perplexity_values

# PREPROCESS THE TEXT --------------------------------------------------------------------

# Select the columns
missom_coded2 = missom_coded[['tagtog_file_id', 'post_id', 'how_annotated', 'text']]
missom_not_coded2 = missom_not_coded[['tagtog_file_id', 'post_id', 'how_annotated', 'text']]

# Combine the two data frames
missom_full = pd.concat([missom_coded2, missom_not_coded2])

# Convert text to list
missom_text_original = missom_full['text'].astype(str).values.tolist()

# Remove emails, new line characters, and single quotes
missom_text = [re.sub('\\S*@\\S*\\s?', '', sent) for sent in missom_text_original]
missom_text = [re.sub('\\s+', ' ', sent) for sent in missom_text]
missom_text = [re.sub("\'", "", sent) for sent in missom_text]

# Remove markdown links with multiple words
missom_text = [re.sub("\\[[\\S\\s]+\\]\\(https:\\/\\/[\\D]+\\)", "", sent) for sent in missom_text]

# Remove markdown links with single words
missom_text = [re.sub("\\[\\w+\\]\\(https:\\/\\/[\\D\\d]+\\)", "", sent) for sent in missom_text]

# Remove urls
missom_text = [re.sub("https:\\/\\/[\\w\\d\\.\\/\\-\\=]+", "", sent) for sent in missom_text]

# Transform sentences into words, convert to list
missom_words = list(transform_to_words(missom_text))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(missom_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[missom_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stop words
missom_words_nostops = remove_stopwords(missom_words)

# Form bigrams
missom_words_bigrams = make_bigrams(missom_words_nostops)

# Lemmatize the words, keeping nouns, adjectives, verbs, adverbs, and proper nouns
missom_words_lemma = lemmatization(missom_words_bigrams)

# Remove any stop words created in lemmatization
missom_words_cleaned = remove_stopwords(missom_words_lemma)

# CREATE DICTIONARY AND CORPUS ------------------------------------------------------------------

# Create Dictionary
id2word = corpora.Dictionary(missom_words_cleaned)

# Create Corpus
texts = missom_words_cleaned

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# EXECUTE THE TOPIC MODELS WITH VANILLA LDA ----------------------------------------------------

# Get the LDA topic model with the optimal number of topics
start_time = time.time()
model_list, coherence_values, perplexity_values = get_optimal_lda(dictionary=id2word, corpus=corpus,
                                                                  limit=50, start=2, step=2)
end_time = time.time()
processing_time = end_time - start_time
print(processing_time / 60)
print((processing_time / 60) / 15)

# Plot the coherence scores
# Set the x-axis valyes
limit = 50
start = 2
step = 2
x = range(start, limit, step)

# Create the plot
plt.figure(figsize=(6, 4), dpi=200)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("UMass Coherence Score")
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.axvline(x=20, color='red')
plt.savefig('results/plots/lda_coherence_plot.png')
plt.show()

# From the plot, the best LDA model is when num_topics == 20
optimal_lda_model = model_list[10]

# Visualize best LDA topic model
# https://stackoverflow.com/questions/41936775/export-pyldavis-graphs-as-standalone-webpage
vis = pyLDAvis.gensim_models.prepare(optimal_lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'results/plots/lda.html')

# Get the Reddit post that best represents each topic
# https://radimrehurek.com/gensim/models/ldamodel.html

# Initialize empty lists
lda_output = []
topic_distributions = []

# For each post, get the LDA estimation output
for i in range(len(missom_text_original)):
    lda_output.append(optimal_lda_model[corpus[i]])

# For each output, select just the topic distribution
for i in range(len(missom_text_original)):
    topic_distributions.append(lda_output[i][0])

# Initialize empty dataframe
# https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
list_topic_names = list(range(0, 22))
list_topic_names = [str(i) for i in list_topic_names]
list_topic_probs = [0] * 22
topic_dict = dict(zip(list_topic_names, list_topic_probs))
topic_df = pd.DataFrame(topic_dict, index=[0])

# For each post, assign topic probabilities as features
for i in range(len(topic_distributions)):

    # Initialize list of zeros
    post_topic_probs = [0] * len(topic_df.columns)

    # For each tuple holding topic probabilities
    for tup in range(len(topic_distributions[i])):

        # Get the topic in the tuple
        tup_topic = topic_distributions[i][tup][0]

        # Get the topic probability in the tuple
        tup_prob = topic_distributions[i][tup][1]

        # Change the list element for the post
        post_topic_probs[tup_topic] = tup_prob

    # Add the list as a new row in the dataframe
    topic_df.loc[len(topic_df)] = post_topic_probs
    print('Percent done: ', str(round(i / len(topic_distributions) * 100, 4)), '%')

# Extract top words
# https://stackoverflow.com/questions/46536132/how-to-access-topic-words-only-in-gensim
lda_top_words = optimal_lda_model.show_topics(num_topics=22, num_words=3)
lda_tup_words = [lda_tup_words[1] for lda_tup_words in lda_top_words]

# Initialize empty list
lad_topic_names = []

# For each topic
for topic in range(len(lda_tup_words)):

    # Extract the top 3 words
    my_words = re.findall("\\w+", lda_tup_words[topic])
    my_elements = [2, 5, 8]

    # Concatenate the top 3 words together and save to list
    my_name = ''.join([my_words[i] for i in my_elements])
    my_name1 = 'lda_' + my_name
    lad_topic_names.append(my_name1)

# Rename the LDA features
# https://sparkbyexamples.com/pandas/rename-columns-with-list-in-pandas-dataframe/?expand_article=1
topic_df.set_axis(lad_topic_names, axis=1, inplace=True)

# Join the two data frames by index
missom_full = missom_full.join(topic_df)

# Filter the dataframes
missom_coded2 = missom_full[missom_full['how_annotated'] == 'human']
missom_not_coded2 = missom_full[missom_full['how_annotated'] == 'machine']

# Export
missom_coded2.to_csv(my_path + '/data/cleaned/features/missom_coded_feat02b.csv')
missom_not_coded2.to_csv(my_path + '/data/cleaned/features/missom_not_coded_feat02b.csv')

#endregion

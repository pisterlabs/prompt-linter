"""
@author: Cory J Cascalheira
Created: 2022-10-30

The purpose of this script is to generate topic models of the monkeypox conversation among LGBTQ+ people using Reddit
text.

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

#region LIBRARIES AND IMPORT

# Load core libraries
import numpy as np
import pandas as pd
import time

# Import tool for regular expressions
import re

# Load Gensim libraries
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Initialize spaCy language model
# Must download the spaCy model first in terminal with command: python -m spacy download en_core_web_sm
# May need to restart IDE before loading the spaCy pipeline
import importlib_metadata
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load plotting tools
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns

# Load NLTK stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Improve NLTK stopwords
new_stop_words = [re.sub("\'", "", sent) for sent in stop_words]
stop_words.extend(new_stop_words)
stop_words.extend(['ish', 'lol', 'non', 'im', 'like', 'ive', 'cant', 'amp', 'ok', 'gt'])

# Load GSDMM - topic modeling for short texts (i.e., social media)
from gsdmm import MovieGroupProcess

# Import data
mpx = pd.read_csv('data/combined_subreddits/all_subreddits_mpx_data.csv')

#endregion

#region HELPER FUNCTIONS


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


def top_words(cluster_word_distribution, top_cluster, values):
    """
    Print the top words associated with the GSDMM topic modeling algorithm.

    Parameters
    ----------
    cluster_word_distribution: a GSDMM word distribution
    top_cluster: a list of indices
    values: an integer
    """

    # For each cluster
    for cluster in top_cluster:

        # Sort the words associated with each topic
        sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]

        # Print the results to the screen
        print('Cluster %s : %s' % (cluster, sort_dicts))
        print('-' * 120)

#endregion

#region PREPROCESS THE TEXT

# Convert text to list
mpx_text_original = mpx['body'].values.tolist()

# Remove emails, new line characters, and single quotes
mpx_text = [re.sub('\\S*@\\S*\\s?', '', sent) for sent in mpx_text_original]
mpx_text = [re.sub('\\s+', ' ', sent) for sent in mpx_text]
mpx_text = [re.sub("\'", "", sent) for sent in mpx_text]

# Remove markdown links with multiple words
mpx_text = [re.sub("\\[[\\S\\s]+\\]\\(https:\\/\\/[\\D]+\\)", "", sent) for sent in mpx_text]

# Remove markdown links with single words
mpx_text = [re.sub("\\[\\w+\\]\\(https:\\/\\/[\\D\\d]+\\)", "", sent) for sent in mpx_text]

# Remove urls
mpx_text = [re.sub("https:\\/\\/[\\w\\d\\.\\/\\-\\=]+", "", sent) for sent in mpx_text]

# Transform sentences into words, convert to list
mpx_words = list(transform_to_words(mpx_text))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(mpx_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[mpx_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stop words
mpx_words_nostops = remove_stopwords(mpx_words)

# Form bigrams
mpx_words_bigrams = make_bigrams(mpx_words_nostops)

# Lemmatize the words, keeping nouns, adjectives, verbs, adverbs, and proper nouns
mpx_words_lemma = lemmatization(mpx_words_bigrams)

# Remove any stop words created in lemmatization
mpx_words_cleaned = remove_stopwords(mpx_words_lemma)

#endregion

#region CREATE DICTIONARY AND CORPUS

# Create Dictionary
id2word = corpora.Dictionary(mpx_words_cleaned)

# Create Corpus
texts = mpx_words_cleaned

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

#endregion

#region EXECUTE THE TOPIC MODELS WITH VANILLA LDA

# Get the LDA topic model with the optimal number of topics
start_time = time.time()
model_list, coherence_values, perplexity_values = get_optimal_lda(dictionary=id2word, corpus=corpus,
                                                                  limit=30, start=2, step=2)
end_time = time.time()
processing_time = end_time - start_time
print(processing_time / 60)
print((processing_time / 60) / 15)

# Plot the coherence scores
# Set the x-axis valyes
limit = 30
start = 2
step = 2
x = range(start, limit, step)

# Create the plot
plt.figure(figsize=(6, 4), dpi=200)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("UMass Coherence Score")
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.axvline(x=10, color='red')
plt.savefig('plots/lda_coherence_plot.png')
plt.show()

# From the plot, the best LDA model is when num_topics == 10
optimal_lda_model = model_list[4]

# Visualize best LDA topic model
# https://stackoverflow.com/questions/41936775/export-pyldavis-graphs-as-standalone-webpage
vis = pyLDAvis.gensim_models.prepare(optimal_lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'plots/lda.html')

# Get the Reddit post that best represents each topic
# https://radimrehurek.com/gensim/models/ldamodel.html

# Initialize empty lists
lda_output = []
topic_distributions = []

# For each post, get the LDA estimation output
for i in range(len(mpx_text_original)):
    lda_output.append(optimal_lda_model[corpus[i]])

# For each output, select just the topic distribution
for i in range(len(mpx_text_original)):
    topic_distributions.append(lda_output[i][0])

# Initialize empty lists
dominant_topics = []
dominance_strength = []

# For each post, extract the dominant topic from the topic distribution
for i in range(len(mpx_text_original)):

    # Sort the tuple by the topic probability (2nd tuple item), largest to smallest
    # https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
    topic_distributions[i].sort(key = lambda x: x[1], reverse=True)

    # Extract the dominant topic
    dominant_topic = topic_distributions[i][0][0]
    dominant_topics.append(dominant_topic)

    # Extract the probability of the dominant topic
    how_dominant = topic_distributions[i][0][1]
    dominance_strength.append(how_dominant)

# Prepare to merge with original dataframe
new_mpx_df = mpx.loc[:, ['author', 'body', 'permalink']]

# Add the dominant topics and strengths
new_mpx_df['dominant_topic'] = dominant_topics
new_mpx_df['topic_probability'] = dominance_strength

# Sort the data frame
new_mpx_df = new_mpx_df.sort_values(by=['dominant_topic', 'topic_probability'], ascending=[True, False])

# Percent of posts for each topic
posts_per_topic = new_mpx_df.groupby(['dominant_topic'])['dominant_topic'].count()
posts_per_topic = pd.DataFrame(posts_per_topic)
posts_per_topic['percent_posts'] = posts_per_topic['dominant_topic'] / len(new_mpx_df.index)
print(posts_per_topic)

# Select the 10 most illustrative posts per topic
topics_to_quote = new_mpx_df.groupby(['dominant_topic']).head(10)

# Save the data frame for easy reading
topics_to_quote.to_csv("data/results/lda_topics_to_quote.csv")

#endregion

#region EXECUTE THE TOPIC MODELS WITH GSDMM

# Get the number of words per post
words_per_post = []

for i in range(len(mpx_words_cleaned)):
    words_per_post.append(len(mpx_words_cleaned[i]))

# Histogram of words per post
plt.hist(x=words_per_post)
plt.show()

# Descriptive statistic of words per post
print(np.mean(words_per_post))
print(np.std(words_per_post))
print(len([num for num in words_per_post if num <= 50]) / len(words_per_post))

# GSDMM ALGORITHM

# Create the vocabulary
vocab = set(x for doc in mpx_words_cleaned for x in doc)

# The number of terms in the vocabulary
n_terms = len(vocab)

# Train the GSDMM models, changing the value of beta given its meaning (i.e., how similar topics need to be to cluster
# together). K is 30, the same number of topic to consider as the above vanilla LDA. Alpha remains 0.1, which reduces
# the probability that a post will join an empty cluster

# Train the GSDMM model, beta = 1.0
mgp_10 = MovieGroupProcess(K=30, alpha=0.1, beta=1.0, n_iters=40)
gsdmm_b10 = mgp_10.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_10 = np.array(mgp_10.cluster_doc_count)
print('Beta = 1.0. The number of posts per topic: ', post_count_10)

# Train the GSDMM model, beta = 0.9
mgp_09 = MovieGroupProcess(K=30, alpha=0.1, beta=0.9, n_iters=40)
gsdmm_b09 = mgp_09.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_09 = np.array(mgp_09.cluster_doc_count)
print('Beta = 0.9. The number of posts per topic: ', post_count_09)

# Train the GSDMM model, beta = 0.8
mgp_08 = MovieGroupProcess(K=30, alpha=0.1, beta=0.8, n_iters=40)
gsdmm_b08 = mgp_08.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_08 = np.array(mgp_08.cluster_doc_count)
print('Beta = 0.8. The number of posts per topic: ', post_count_08)

# Train the GSDMM model, beta = 0.7
mgp_07 = MovieGroupProcess(K=30, alpha=0.1, beta=0.7, n_iters=40)
gsdmm_b07 = mgp_07.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_07 = np.array(mgp_07.cluster_doc_count)
print('Beta = 0.7. The number of posts per topic: ', post_count_07)

# Train the GSDMM model, beta = 0.6
mgp_06 = MovieGroupProcess(K=30, alpha=0.1, beta=0.6, n_iters=40)
gsdmm_b06 = mgp_06.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_06 = np.array(mgp_06.cluster_doc_count)
print('Beta = 0.6. The number of posts per topic: ', post_count_06)

# Train the GSDMM model, beta = 0.5
mgp_05 = MovieGroupProcess(K=30, alpha=0.1, beta=0.5, n_iters=40)
gsdmm_b05 = mgp_05.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_05 = np.array(mgp_05.cluster_doc_count)
print('Beta = 0.5. The number of posts per topic: ', post_count_05)

# Train the GSDMM model, beta = 0.4
mgp_04 = MovieGroupProcess(K=30, alpha=0.1, beta=0.4, n_iters=40)
gsdmm_b04 = mgp_04.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_04 = np.array(mgp_04.cluster_doc_count)
print('Beta = 0.4. The number of posts per topic: ', post_count_04)

# Train the GSDMM model, beta = 0.3
start_time = time.time()
mgp_03 = MovieGroupProcess(K=30, alpha=0.1, beta=0.3, n_iters=40)
gsdmm_b03 = mgp_03.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_03 = np.array(mgp_03.cluster_doc_count)
print('Beta = 0.3. The number of posts per topic: ', post_count_03)
end_time = time.time()
processing_time = end_time - start_time
print(processing_time / 60)

# Train the GSDMM model, beta = 0.2
mgp_02 = MovieGroupProcess(K=30, alpha=0.1, beta=0.2, n_iters=40)
gsdmm_b02 = mgp_02.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_02 = np.array(mgp_02.cluster_doc_count)
print('Beta = 0.2. The number of posts per topic: ', post_count_02)

# Train the GSDMM model, beta = 0.1
mgp_01 = MovieGroupProcess(K=30, alpha=0.1, beta=0.1, n_iters=40)
gsdmm_b01 = mgp_01.fit(docs=mpx_words_cleaned, vocab_size=n_terms)
post_count_01 = np.array(mgp_01.cluster_doc_count)
print('Beta = 0.1. The number of posts per topic: ', post_count_01)

# Remove topics with 0 posts assigned
beta_01 = [x for x in post_count_01 if x > 0]
beta_02 = [x for x in post_count_02 if x > 0]
beta_03 = [x for x in post_count_03 if x > 0]
beta_04 = [x for x in post_count_04 if x > 0]
beta_05 = [x for x in post_count_05 if x > 0]
beta_06 = [x for x in post_count_06 if x > 0]
beta_07 = [x for x in post_count_07 if x > 0]
beta_08 = [x for x in post_count_08 if x > 0]
beta_09 = [x for x in post_count_09 if x > 0]
beta_10 = [x for x in post_count_10 if x > 0]

# Make lists the same size, transform in array
beta_01 = np.sort(np.array(beta_01))
beta_02 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_02)]), beta_02))
beta_03 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_03)]), beta_03))
beta_04 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_04)]), beta_04))
beta_05 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_05)]), beta_05))
beta_06 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_06)]), beta_06))
beta_07 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_07)]), beta_07))
beta_08 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_08)]), beta_08))
beta_09 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_09)]), beta_09))
beta_10 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_10)]), beta_10))

# Append all topics
n_posts = np.append(beta_01, [beta_02, beta_03, beta_04, beta_05, beta_06, beta_07, beta_08, beta_09, beta_10])

# Create list of topic numbers
topic_numbers = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 10

# Create a list of beta values
beta_list = [[0.1] * 17] + [[0.2] * 17] + [[0.3] * 17] + [[0.4] * 17] + [[0.5] * 17] + [[0.6] * 17] + [[0.7] * 17] + [[0.8] * 17] + [[0.9] * 17] + [[1.0] * 17]
beta_values = [item for sublist in beta_list for item in sublist]

# Double check that the betas are same length as topic numbers
print(len(topic_numbers) == len(n_posts) == len(beta_values))

# Create data frame for plotting
list_of_tuples = list(zip(beta_values, topic_numbers, n_posts))
gsdmm_df = pd.DataFrame(list_of_tuples, columns=['beta', 'topic_numbers', 'n_posts'])

# Make grid plot
sns.set_theme(style="white")
gsdmm_plot = sns.FacetGrid(gsdmm_df, col='beta', col_wrap=2)
gsdmm_plot.map(sns.barplot, 'topic_numbers', 'n_posts', color='cornflowerblue')
gsdmm_plot.set_axis_labels("Topic Numbers", "Number of Posts")
gsdmm_plot.savefig('plots/gsdmm_topics.png')

# Optimal number of topics?
print('The optimal number of topics in GSDMM, based on average, is: ', (6 + 4 + 3 + 4 + 2 + 2 + 1 + 2 + 2 + 1) / 10)

# Since optimal number of plots is GSDMM is 2.7, round to 3---use model where beta = 0.3

# Rearrange the topics in order of importance
top_index = post_count_03.argsort()[-17:][::-1]

# Get the top 15 words per topic
top_words(mgp_03.cluster_word_distribution, top_cluster=top_index, values=15)

# Initialize empty list
gsdmm_topics = []

# Predict the topic for each set of words in a Reddit post
for i in range(len(mpx_words_cleaned)):
    gsdmm_topics.append(mgp_03.choose_best_label(mpx_words_cleaned[i]))

# Initialize empty lists
topic_classes = []
topic_probs = []

# For each post, extract the dominant topic from the topic distribution
for i in range(len(mpx_text_original)):

    # Extract the dominant topic
    topic_class = gsdmm_topics[i][0]
    topic_classes.append(topic_class)

    # Extract the probability of the dominant topic
    topic_prob = gsdmm_topics[i][1]
    topic_probs.append(topic_prob)

# Prepare to merge with original dataframe
gsdmm_mpx_df = mpx.loc[:, ['author', 'body', 'permalink']]

# Add the dominant topics and strengths
gsdmm_mpx_df['topic'] = topic_classes
gsdmm_mpx_df['topic_probability'] = topic_probs

# Sort the data frame
gsdmm_mpx_df = gsdmm_mpx_df.sort_values(by=['topic', 'topic_probability'], ascending=[False, False])

# Select the 10 most illustrative posts per topic
topics_to_quote = gsdmm_mpx_df.groupby('topic').head(10)

# Save the data frame for easy reading
topics_to_quote.to_csv("data/results/gsdmm_topics_to_quote.csv")

# Percentage of posts with each top 3 topic
print(post_count_03[29] / len(mpx_words_cleaned))
print(post_count_03[13] / len(mpx_words_cleaned))
print(post_count_03[5] / len(mpx_words_cleaned))

#endregion
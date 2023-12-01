# Built from this tutorial - https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#2prerequisitesdownloadnltkstopwordsandspacymodelforlemmatization
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



def testMallet2Model(self):
    if not self.mallet_path:
        return

    tm1 = ldamallet.LdaMallet(self.mallet_path, corpus=corpus, num_topics=2, id2word=dictionary)
    tm2 = ldamallet.malletmodel2ldamodel(tm1)
    for document in corpus:
        element1_1, element1_2 = tm1[document][0]
        element2_1, element2_2 = tm2[document][0]
        self.assertAlmostEqual(element1_1, element2_1)
        self.assertAlmostEqual(element1_2, element2_2, 1)
        element1_1, element1_2 = tm1[document][1]
        element2_1, element2_2 = tm2[document][1]
        self.assertAlmostEqual(element1_1, element2_1)
        self.assertAlmostEqual(element1_2, element2_2, 1)
        logging.debug('%d %d', element1_1, element2_1)
        logging.debug('%d %d', element1_2, element2_2)
        logging.debug('%d %d', tm1[document][1], tm2[document][1])


# Import Dataset
csvFile = 'data_out.csv'
df = pd.read_csv(csvFile)

# print(df.target_names.unique())
df.head()


# Convert to list
data = df["Abstract"].values.tolist()

# print(data[:1])

# # Remove Emails
# data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# # Remove new line characters
# data = [re.sub('\s+', ' ', sent) for sent in data]

# # Remove distracting single quotes
# data = [re.sub("\'", "", sent) for sent in data]

print("cleaned data")
pprint(data[:1])


#Let’s tokenize each sentence into a list of words, removing punctuations and unnecessary characters altogether.

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print("data words")
print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print("trigram example")
print(trigram_mod[bigram_mod[data_words[0]]])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# # Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#     id2word=id2word,
#     num_topics=10, 
#     random_state=100,
#     update_every=1,
#     chunksize=100,
#     passes=10,
#     alpha='auto',
#     per_word_topics=True)




# # Print the Keyword in the 10 topics
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]

# Mallet topic model
# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = '../mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=75, id2word=id2word)

optimal_model = ldamallet



# # Visualize the topics
# lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model)
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# pyLDAvis.save_html(vis, 'output_filename.html')


# # Show Topics
# model_topics = ldamallet.show_topics(formatted=False)
# pprint(ldamallet.print_topics(num_words=10))


# # Compute Coherence Score
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: ', coherence_ldamallet)



# # Can take a long time to run.
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=20, limit=105, step=5)


# start=20; limit=105; step=5;
# x = range(start, limit, step)

# # Print the coherence scores
# for m, cv in zip(x, coherence_values):
#     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# # Show graph
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

##32 topics seems optimal here - adrian 2020 feb
##75 topics seems optimal here - adrian 2020 feb

def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
print("Dominant topic in each sentence")
pprint(df_dominant_topic.head(10))
df_dominant_topic.to_csv("sentence_for_topic.csv", index=False)


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
print("most representative document for each topic")
sent_topics_sorteddf_mallet.to_csv("document_for_topic.csv", index=False)
pprint(sent_topics_sorteddf_mallet)




# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)
\
# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
print("dominant topics")
pprint(df_dominant_topics)

df_dominant_topics.to_csv("dominant_topics.csv", index=False)
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# First, lets count how much data we have!

train_len, test_len = len(train_df.index), len(test_df.index)

print(f'train size: {train_len}, test size: {test_len}')
# also, lets take a quick look at what we have 

train_df.head(20)
# its always a good idea to count the amount of missing values before diving into any analysis

# Lets also see how many missing values (in percentage) we are dealing with

miss_val_train_df = train_df.isnull().sum(axis=0) / train_len

miss_val_train_df = miss_val_train_df[miss_val_train_df > 0] * 100

miss_val_train_df
# lets create a list of all the identities tagged in this dataset. This list given in the data section of this competition. 

identities = ['male','female','transgender','other_gender','heterosexual','homosexual_gay_or_lesbian',

              'bisexual','other_sexual_orientation','christian','jewish','muslim','hindu','buddhist',

              'atheist','other_religion','black','white','asian','latino','other_race_or_ethnicity',

              'physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness',

              'other_disability']
# getting the dataframe with identities tagged

train_labeled_df = train_df.loc[:, ['target'] + identities ].dropna()

# changing the value of identity to 1 or 0 only

identity_label_count = train_labeled_df[identities].where(train_labeled_df == 0, other = 1).sum()

# dividing the time each identity appears by the total number of comments

identity_label_pct = identity_label_count / len(train_labeled_df.index)
# now we will use seaborn to do a horizontal bar plot and visualize our result. 

# since it would be nicer to have it ordered by most frequent to least, we do a simple sort

identity_label_pct = identity_label_pct.sort_values(ascending=False)

# seaborn is more of a wrapper around matplotlib. So to edit size and give x, y labels; we use the plt that we imported earlier

plt.figure(figsize=(30,20))

sns.set(font_scale=3)

ax = sns.barplot(x = identity_label_pct.values * 100, y = identity_label_pct.index, alpha=0.8)

plt.ylabel('Demographics')

plt.xlabel('Total Percentage')

plt.title('Most Frequent Identities')

plt.show()
# First we multiply each identity with the target

weighted_toxic = train_labeled_df.iloc[:, 1:].multiply(train_labeled_df.iloc[:, 0], axis="index").sum() 

# then we divide the target weighted value by the number of time each identity appears

weighted_toxic = weighted_toxic / identity_label_count

weighted_toxic = weighted_toxic.sort_values(ascending=False)

# plot the data using seaborn like before

plt.figure(figsize=(30,20))

sns.set(font_scale=3)

ax = sns.barplot(x = weighted_toxic.values , y = weighted_toxic.index, alpha=0.8)

plt.ylabel('Demographics')

plt.xlabel('Weighted Toxicity')

plt.title('Weighted Analysis of Most Frequent Identities')

plt.show()
# total length of characters in the comment

train_df['total_length'] = train_df['comment_text'].apply(len)
# number of capital letters

train_df['capitals'] = train_df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
# ratio of capital characters vs length of the comment

train_df['caps_vs_length'] = train_df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
# count of special characters

train_df['num_exclamation_marks'] = train_df['comment_text'].apply(lambda comment: comment.count('!'))

train_df['num_question_marks'] = train_df['comment_text'].apply(lambda comment: comment.count('?'))

train_df['num_punctuation'] = train_df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))

train_df['num_symbols'] = train_df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
# word count

train_df['num_words'] = train_df['comment_text'].apply(lambda comment: len(comment.split()))
# number of unique words in the comment

train_df['num_unique_words'] = train_df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
# ratio of unique words and number of words

train_df['words_vs_unique'] = train_df['num_unique_words'] / train_df['num_words']
# number of smiley faces

train_df['num_smilies'] = train_df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
# let's create a list of columns with the new features we have just created

new_features = ['total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks','num_question_marks', 

            'num_punctuation', 'num_words', 'num_unique_words','words_vs_unique', 'num_smilies', 'num_symbols']

# the dataset is labeled with more information alongside the target. lets collect them in a list as well

labels = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'funny', 'wow', 'sad', 

           'likes', 'disagree', 'sexual_explicit','identity_annotator_count', 'toxicity_annotator_count']
# in pandas calculating correlation is simple. You can calculate the correlation value between two columns in the following way

train_df['total_length'].corr(train_df['funny'])
# lets loop over each feature and calculate its correlation with a label

rows = [{label:train_df[feature].corr(train_df[label]) for label in labels} for feature in new_features]

train_correlations = pd.DataFrame(rows, index=new_features)
# now we have our beautiful correlation matrix

train_correlations
# heatmaps are great for visualizing correlation matrix. and its very simple to do so in seaborn

plt.figure(figsize=(10, 6))

sns.set(font_scale=1)

ax = sns.heatmap(train_correlations, vmin=-0.1, vmax=0.1, center=0.0)
# imports for topic modeling and loading stopwords

import nltk; nltk.download('stopwords')

import re

import numpy as np

import pandas as pd

from pprint import pprint

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

import spacy

import pyLDAvis

import pyLDAvis.gensim

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'people', 'may', 'think'])
#convert to list

#data = train_df.comment_text.values.tolist()

#covert to list while subsetting

data_select = train_df[['comment_text',"severe_toxicity"]]

data_select.head(5)

data_query = data_select[data_select["severe_toxicity"] > 0.01]

data_query2 = data_query[['comment_text']]

data = data_query2.comment_text.values.tolist()
#tokenize words and Clean-up text

def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:31])
#9. Creating Bigram and Trigram Models

#Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)

print(bigram_mod[bigram_mod[data_words[0]]])

print(trigram_mod[trigram_mod[data_words[0]]])

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

# call the functions in right order



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
# Human readable format of corpus (term-frequency)

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
#Building the Topic Model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
#Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis
from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors



cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



cloud = WordCloud(stopwords=stop_words,

                  background_color='white',

                  width=2500,

                  height=1800,

                  max_words=10,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):

    fig.add_subplot(ax)

    topic_words = dict(topics[i][1])

    cloud.generate_from_frequencies(topic_words, max_font_size=300)

    plt.gca().imshow(cloud)

    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

    plt.gca().axis('off')





plt.subplots_adjust(wspace=0, hspace=0)

plt.axis('off')

plt.margins(x=0, y=0)

plt.tight_layout()

plt.show()
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        # print(row)

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





df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.head(10)
# lets take the dataset with identitiy tags, created date, and target column

with_date_df = train_df.loc[:, ['created_date', 'target'] + identities].dropna()

# next we will create a weighted dataframe for each identity tag (like we did before)

# first we divide each identity tag with the total value it has in the dataset

weighted_df = with_date_df.iloc[:, 2:] / with_date_df.iloc[:, 2:].sum()

# then we multiplty this value with the target 

target_weighted_df = weighted_df.multiply(with_date_df.iloc[:, 1], axis="index")

# lets add a column to count the number of comments

target_weighted_df['comment_count'] = 1

# now we add the date to our newly created dataframe (also parse the text date as datetime)

target_weighted_df['created_date'] = pd.to_datetime(with_date_df['created_date']).values.astype('datetime64[M]')

# now we can do a group by of the created date to count the number of times a identity appears for that date

identity_weight_per_date_df = target_weighted_df.groupby(['created_date']).sum().sort_index()
# importing plotly

import plotly

import plotly.plotly as py

import cufflinks as cf

import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='ekhtiar', api_key='NUzf7CKPlCmChi5aFEdY')

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
races = ['black','white','asian','latino','other_race_or_ethnicity']

identity_weight_per_date_df[races].iplot(title = 'Time Series Toxicity & Race', filename='Time Series Toxicity & Race' )
religions = ['atheist', 'buddhist', 'christian', 'hindu', 'muslim', 'other_religion']

identity_weight_per_date_df[religions].iplot(title = 'Time Series Toxicity & Religion', filename='Time Series Toxicity & Religion')
sexual_orientation = ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']

identity_weight_per_date_df[sexual_orientation].iplot(title = 'Time Series Toxicity & Sexual Orientation', filename='Time Series Toxicity & Sexual Orientation')
identity_weight_per_date_df['comment_count'].iplot(title = 'Time Series Total Tagged Comments', filename='Time Series Total Tagged Comments')
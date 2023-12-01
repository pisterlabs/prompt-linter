import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords


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
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# df1 = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
# print(df1.target_names.unique())
# df1.head()
# df1.columns
# df.columns = df1.columns 
# df1 = pd.concat([df, df1], ignore_index=True)
# df1.head()
# df1.info()

###
df1 = pd.read_csv('data.csv')
df1.info()
df1.tail()


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'line'])

stop_words

# Convert to list
data = df1.content.values.tolist()


#lowercase

data = [sent.lower() for sent in data]


# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[0])

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out



# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
texts = remove_stopwords(texts)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(len(data_lemmatized))
from gensim.corpora import Dictionary
dictionary = Dictionary(data_lemmatized)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.6)

# dictionary.save_as_text('dictionary.txt')

dictionary = Dictionary.load_from_text('dictionary.txt')






# Create Dictionary
temp = dictionary[0]
id2word = dictionary.id2token
# id2word = corpora.Dictionary(data_lemmatized)
print(len(id2word))
id2word

# Create Corpus
texts = data_lemmatized

import json
# # with open('texts.txt','w')as f:
# #     f.write(json.dumps(texts))

with open('texts.txt', 'r') as f:
    texts = json.loads(f.read())


dictionary['use']
for idx, i in enumerate(texts):
    if 'use' in i:
        print(idx)



# Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in texts]
corpus = [dictionary.doc2bow(doc) for doc in texts]

print(len(corpus))

from gensim.models.ldamulticore import LdaMulticore
num_topics = 35
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.



lda_multicore = LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    chunksize=2000,
    alpha='asymmetric',
    eta='auto',
    iterations=400,
    num_topics=15,
    passes=20,
    workers=3,
    eval_every=eval_every)

model =  LdaMulticore(
                corpus=corpus,
                id2word= id2word,
                alpha='asymmetric',
                eta='auto',
                num_topics=10,
                random_state=100,
                passes=10,
                workers=3,
                eval_every=None,
                per_word_topics=True)

# lda_multicore = LdaMulticore(
#     corpus=corpus,
#     id2word=id2word,
#     chunksize=2000,
#     alpha='asymmetric',
#     eta='auto',
#     iterations=400,
#     num_topics=num_topics,
#     passes=20,
#     workers=3,
#     eval_every=eval_every)

# lda_multicore = LdaMulticore(
#     corpus=corpus,
#     id2word=id2word,
#     chunksize=2000,
#     alpha='asymmetric',
#     eta='auto',
#     iterations=400,
#     num_topics=20,
#     passes=20,
#     workers=3,
#     eval_every=eval_every)

# Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=35, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)


pprint(model.print_topics())
doc_lda = model[corpus]
doc_lda[4]
model.get_document_topics(corpus)[1]

# Compute Perplexity
print('\nPerplexity: ', model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_multicore, corpus, dictionary)
vis

mallet_path = '/home/ubuntu/Signal/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=35, id2word=id2word)

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

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
        model = LdaMulticore(
                random_state=42
                id2word= id2word,
                alpha='asymmetric',
                eta='auto',
                num_topics=num_topics,
                passes=10,
                workers=3,
                per_word_topics=True,
                eval_every=None)
        # model = gensim.models.ldamodel.LdaModel(corpus=corpus,
        #                                    id2word=id2word,
        #                                    num_topics=num_topics, 
        #                                    random_state=100,
        #                                    update_every=1,
        #                                    passes=10,
        #                                    alpha='auto',
        #                                    )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=5, limit=41, step=5)
# model_list1, coherence_values1 = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data_lemmatized, start=5, limit=41, step=5)


limit=41; start=5; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
coherence_values

# for m, cv in zip(x, coherence_values):
#     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# optimal_model = model_list[1]
# for i, row in enumerate(optimal_model[corpus]):
#     print(i, row)

# optimal_model.save('lda.model')
# model.save('lda.model')
model = LdaMulticore.load('lda.model')


model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

# def format_topics_sentences(ldamodel, corpus=corpus, texts=texts):
#     # Init output
#     sent_topics_df = pd.DataFrame()

#     # Get main topic in each document
#     for i, row in enumerate(ldamodel[corpus]):
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)
#         # Get the Dominant topic, Perc Contribution and Keywords for each document
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:  # => dominant topic
#                 wp = ldamodel.show_topic(topic_num)
#                 topic_keywords = ", ".join([word for word, prop in wp])
#                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
#             else:
#                 break
#     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

#     # Add original text to the end of the output
#     contents = pd.Series(texts)
#     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
#     return(sent_topics_df)

def format_topics_sentences(ldamodel, corpus=corpus, texts=texts):
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=model, corpus=corpus, texts=texts)
df_topic_sents_keywords

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

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
sent_topics_sorteddf_mallet.tail()

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
topic_counts

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)
topic_contribution

# Topic Number and Keywords
topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]
topic_num_keywords

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

df_dominant_topics.to_csv('topics.csv', index=False)

# gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
# lda1 = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model, gamma_threshold=0.001, iterations=50)

#### modeling with articles

df = pd.concat([justin, lester, rakhi], ignore_index=True)
df.info()
def predict_prepro(df):
    # Convert to list
    data1 = df.content.values.tolist()

    #lowercase
    data1 = [sent.lower() for sent in data1]

    # Remove Emails
    data1 = [re.sub('\S*@\S*\s?', '', sent) for sent in data1]

    # Remove new line characters
    data1 = [re.sub('\s+', ' ', sent) for sent in data1]

    # Remove distracting single quotes
    data1 = [re.sub("\'", "", sent) for sent in data1]

    #sentence to words
    data_words1 = list(sent_to_words(data1))

    # Remove Stop Words
    data_words_nostops1 = remove_stopwords(data_words1)


    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    # nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    articles = lemmatization(data_words_nostops1, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return articles

justin_clean = predict_prepro(justin)
rakhi_clean = predict_prepro(rakhi)
lester_clean = predict_prepro(lester)
r_l_df = pd.concat([rakhi, lester], ignore_index=True)

r_l = predict_prepro(pd.concat([rakhi, lester], ignore_index=True))




# dictionary.save_as_text('dictionary.txt')

other_corpus = [dictionary.doc2bow(text) for text in articles]
justin_corp = [dictionary.doc2bow(text) for text in justin_clean]
rak_les_corp= [dictionary.doc2bow(text) for text in r_l]
rak_corp = [dictionary.doc2bow(text) for text in rakhi_clean]
les_corp = [dictionary.doc2bow(text) for text in lester_clean]


all_top_vecs = [model.get_document_topics(corpus[n], minimum_probability=0) \
                    for n in range(len(corpus))]

justin_vecs = [model.get_document_topics(justin_corp[n], minimum_probability=0) \
                    for n in range(len(justin_corp))]
justin_vecs[3]
justin_vecs[5]
justin_vecs[24]

rak_les_vecs = [model.get_document_topics(rak_les_corp[n], minimum_probability=0) \
                    for n in range(len(rak_les_corp))]

rak_vecs = [model.get_document_topics(rak_corp[n], minimum_probability=0) \
                    for n in range(len(rak_corp))]

les_vecs = [model.get_document_topics(les_corp[n], minimum_probability=0) \
                    for n in range(len(les_corp))]


def find_most_similar(sim_vec, all_top_vecs, title_lst, vec_in_corp='Y', n_results=7):                
    '''
    Calculates cosine similarity across the entire corpus and returns 
    the n_results number of most similar documents
    '''
    
    cos_sims = [gensim.matutils.cossim(sim_vec, vec) for vec in all_top_vecs]
    
    if vec_in_corp == 'N':
        most_similar_ind = np.argsort(cos_sims)[::-1][:n_results]
    if vec_in_corp == 'Y':
        most_similar_ind = np.argsort(cos_sims)[::-1][:n_results+1][1:]
        #exclude 'self', in the case that it is a book in the corpus
    
    for ind in most_similar_ind:
        print (title_lst[ind], cos_sims[ind])
    


def find_topic_books(topic_num, all_top_vecs, n_results=10):
    '''
    Finds the books that have the strongest match to the selected topic number
    '''
    one_topic_lst = [vec[topic_num][1] for vec in all_top_vecs]
    return np.argsort(one_topic_lst)[::-1][:n_results]

all_top_vecs

find_most_similar(justin_vecs[3], all_top_vecs, df1.content, vec_in_corp='N', n_results=3)
# Is MSG sensitivity superstition?  sci.med 0.9922154188798971
# Selective Placebo sci.med 0.9916461672869317
# sex education sci.med 0.9839967035006875
find_most_similar(justin_vecs[3], rak_les_vecs, r_l_df.target, vec_in_corp='N', n_results=3)
# The Math Equation That Tried to Stump the Internet  Rakhi math 0.9609225565188918
# The Fullest Look Yet at the Racial Inequality of Coronavirus Rakhi social justice 0.958747481306203
# You Really Want To Humanize Math Education? Build A New Ship Rakhi math education 0.9569022871356381

find_most_similar(justin_vecs[24], all_top_vecs, df1.target_names, vec_in_corp='N', n_results=4)
# Fractals? what good are they? comp.graphics 0.9760094438780984
# Eumemics (was: Eugenics)  sci.med 0.9598570501781843
# MC SBI mixer  sci.electronics 0.9455659664803201

find_most_similar(justin_vecs[24], rak_les_vecs, r_l_df.target, vec_in_corp='N', n_results=3)
# preventing suicide the modern way Rakhi mental health/suicide/design 0.909599279054611
# how to destroy surveillance capitalism Lester politics/economics 0.8918235108587547
# The Academy’s New Inclusion Requirements Won’t Color Correct Hollywood Rakhi social justice 0.8831585144004426




for i in range(len(justin_vecs)):
    print(find_most_similar(justin_vecs[i], rak_les_vecs, r_l_df.target, n_results=3))



ls = pd.DataFrame.from_dict(les_vecs)
for i in ls.columns:
    ls[i] = ls[i].apply(lambda x: x[1])

jv

for i in ls.columns:
    print(f'{ls[i].mean(): .4f}')




jv_avg = [(0,0.1793),(1,0.0007),(2,0.0792),(3,0.0382),(4,0.1750),(5,0.0628),(6,0.0770),(7,0.0147),(8,0.3506),(9,0.0226)]
rk_avg = [(0,0.0544),(1,0.0014),(2,0.0610),(3,0.0123),(4,0.2093),(5,0.0467),(6,0.1689),(7,0.0021),(8,0.4197),(9,0.0242)]
ls_avg = [(0,0.1349),(1,0.0009),(2,0.1084),(3,0.0072),(4,0.1119),(5,0.0581),(6,0.1487),(7,0.0282),(8,0.3402),(9,0.0616)]
gensim.matutils.cossim(ls_avg, rk_avg)



from collections import Counter
topics = model.show_topics(formatted=False)
data_flat = [w for w_list in texts for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])       

import matplotlib.colors as mcolors
# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(5, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()


doc_lens = [len(d) for d in df1.content]

# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins = 1000, color='navy')
plt.text(125, 1400, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(125,  1350, "Median : " + str(round(np.median(doc_lens))))
plt.text(125,  1300, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(125,  1250, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(125,  1200, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 3000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,3000,5))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.save_html(vis, 'vis.html')


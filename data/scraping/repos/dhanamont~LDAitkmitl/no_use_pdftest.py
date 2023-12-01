import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
from pprint import pprint

# pythainlp - word tokenize
from pythainlp.tokenize import word_tokenize
import pythainlp.corpus
from pythainlp.corpus import thai_words, thai_stopwords
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import string
import pandas
from gensim import corpora, models

import matplotlib.pyplot as plt
from no_use_clean_doc import clean_alphabet, clean_thaistopwords

# pdfreader
from pylexto import LexTo
from PDFreader.pdfReader import extract_pdf
#docx2xt
import docx2txt

from no_use_outlier import removing_outlier
from no_use_zipf_law import zipf_law

print("========== PART 1 : Input Dataset ==========")
data_file = []
data_file.append('FinalReport_Sample/pdf/RDG56A030_full.pdf') 
data_file.append('FinalReport_Sample/pdf/RDG60T0048V01_full.pdf') 
data_file.append('/Users/dhanamon/Google Drive/TRF_Y61_Mon/TRF_Mining/gs-mod/RDG6110019_full.pdf') 
data_file.append('/Users/dhanamon/Google Drive/TRF_Y61_Mon/TRF_Mining/gs-mod/RDG6110005_full.pdf') 
data_file.append('/Users/dhanamon/Google Drive/TRF_Y61_Mon/TRF_Mining/gs-mod/RDG6110022_full.pdf') 

# Checking format file
for i in range(len(data_file)):
    if data_file[i].endswith('.pdf'):
        print("Document",i+1, " is pdf file.")
        # print("filename:", data_file[i])
        # pdf document
        data_file[i] = extract_pdf(data_file[i])
    elif data_file[i].endswith('.docx'):
        print("Document",i+1, " is docx file.")
        # docx document
        # data_file[i] = docx2txt.process(data_file[i])


print("========== PART 2 : Split word ==========")
def split_word(data):
    #empty this for participants
    words = thai_stopwords()
    thaiwords = clean_thaistopwords()
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    
    tokens = word_tokenize(data, engine='newmm')
    #print(tokens)
    
    # remove stop words
    stopped_tokens = [i for i in tokens if not i in words and not i in en_stop and not i in thaiwords]
    #print(stopped_tokens)
    
    # stem words
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    #print(stemmed_tokens)
    
    # remove single alphabet and number
    single_alpha_num_tokens = [i for i in stemmed_tokens if not i in pythainlp.thai_consonants and not i.isnumeric()]
    #print(single_alpha_num_tokens)
    deletelist = [' ', '  ', '   ','none','    ','\n','\x0c']
    tokens = [i for i in single_alpha_num_tokens if not i in deletelist]
    return tokens

print("========== PART 3 : Clean Data ==========")
data_ready = []
for i in range(len(data_file)):
    print("------- Document",i+1,"-----------")
    # print(clean_alphabet(data_file[i]))
    data_ready.append(split_word(clean_alphabet(data_file[i])))
    print("-------------------------")


# We will turn this into a term dictionary for our model
# data_ready = []
# for i in range(len(data_file)):
#     text = data_file[i]
#     data_ready.append(split_word(text))
# print("Data ready:",data_ready)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(data_ready)
# dict2 = {dictionary[ID]:ID for ID in dictionary.keys()}
# print(dict2)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in data_ready]
# print("Corpus: ",corpus)

counts = {}
# process filtered tokens for each document
for doc_tokens in data_ready:
    for token in doc_tokens:
        # increment existing?
        if token in counts:
            counts[token] += 1
        # a new term?
        else:
            counts[token] = 1
# print("Found %d unique terms in this corpus" % len(counts))
# print(counts)

# sort frequency count of unique word
import operator
sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_counts)

# ------ Detecting and Removing Outlier ------
corpus_remove_outlier = removing_outlier(sorted_counts, dictionary)
# print(corpus_remove_outlier)

# ------ Zipf's Law -------
# corpus_remove_outlier = zipf_law(sorted_counts, dictionary)

print("========== PART 4 : Generate LDA Model ==========")
# generate LDA model
import gensim
num_top = 8
num_words = 8
num_it = 50
# lda_model = gensim.models.ldamodel.LdaModel(corpus,num_top, id2word = dictionary, random_state = 2, passes=num_it)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_remove_outlier,
                                           id2word=dictionary,
                                           num_topics=num_top, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=num_it,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)
lda_model.show_topics(num_top, num_words, log=True, formatted=False)
pprint(lda_model.print_topics())

# lda_model.save('HDTwork/lda' + str(num_top) + '_Topics_' + str(num_it) + '_Passes.model')

# Gensim
# import gensim, spacy, logging, warnings

# import gensim.corpora as corpora
# from gensim.utils import lemmatize, simple_preprocess
# from gensim.models import CoherenceModel


# # NLTK Stop words
# print("========== PART 1 : Stop Words ==========")
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

# stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

# # %matplotlib inline
# warnings.filterwarnings("ignore",category=DeprecationWarning)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# # 2. Import NewsGroups Dataset
# # Import Dataset
# print("========== PART 2 ==========")
# df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
# df = df.loc[df.target_names.isin(['soc.religion.christian', 'rec.sport.hockey', 'talk.politics.mideast', 'rec.motorcycles']) , :]
# print(df.shape)  #> (2361, 3)
# df.head()


# # 3. Tokenize Sentences and Clean
# print("========== PART 3 ==========")
# def sent_to_words(sentences):
#     for sent in sentences:
#         sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
#         sent = re.sub('\s+', ' ', sent)  # remove newline chars
#         sent = re.sub("\'", "", sent)  # remove single quotes
#         sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
#         yield(sent)  

# # Convert to list
# data = df.content.values.tolist()
# data_words = list(sent_to_words(data))
# print(data_words[:1])
# # [['from', 'irwin', 'arnstein', 'subject', 're', 'recommendation', 'on', 'duc', 'summary', 'whats', 'it', 'worth', 'distribution', 'usa', 'expires', 'sat', 'may', 'gmt', ...trucated...]]


# # 4. Build the Bigram, Trigram Models and Lemmatize
# print("========== PART 4 ==========")
# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# # !python3 -m spacy download en  # run in terminal once
# def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
#     texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
#     texts = [bigram_mod[doc] for doc in texts]
#     texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
#     texts_out = []
#     nlp = spacy.load("//anaconda3/lib/python3.7/site-packages/spacy/lang/en_core_web_sm/en_core_web_sm-2.1.0", disable=['parser', 'ner'])
# #     nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#     for sent in texts:
#         doc = nlp(" ".join(sent)) 
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     # remove stopwords once more after lemmatization
#     texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
#     return texts_out

# data_ready = process_words(data_words)  # processed Text Data!

# 5. Build the Topic Model
# print("========== PART 5 ==========")
# # Create Dictionary
# id2word = corpora.Dictionary(data_ready)

# # Create Corpus: Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in data_ready]

# # Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=4, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=10,
#                                            passes=10,
#                                            alpha='symmetric',
#                                            iterations=100,
#                                            per_word_topics=True)

# pprint(lda_model.print_topics())


# 6. What is the Dominant topic and its percentage contribution in each document
print("========== PART 6 ==========")
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data_ready):
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

# 7. The most representative sentence for each topic
print("========== PART 7 ==========")
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)


# 8. Frequency Distribution of Word Counts in Documents
print("========== PART 8 ==========")
doc_lens = [len(d) for d in df_dominant_topic.Text]

# Plot
# plt.figure(figsize=(16,7), dpi=160)
# plt.hist(doc_lens, bins = 1000, color='navy')
# plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
# plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
# plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
# plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
# plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

# plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
# plt.tick_params(size=16)
# plt.xticks(np.linspace(0,1000,9))
# plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
# plt.show()


# # 9. Word Clouds of Top N Keywords in Each Topic
# print("========== PART 9 ==========")
# # 1. Wordcloud of Top N words in each topic
# from matplotlib import pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

# cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

# cloud = WordCloud(stopwords=stop_words,
#                   background_color='white',
#                   width=2500,
#                   height=1800,
#                   max_words=10,
#                   colormap='tab10',
#                   color_func=lambda *args, **kwargs: cols[i],
#                   prefer_horizontal=1.0)

# topics = lda_model.show_topics(formatted=False)

# fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

# for i, ax in enumerate(axes.flatten()):
#     fig.add_subplot(ax)
#     topic_words = dict(topics[i][1])
#     cloud.generate_from_frequencies(topic_words, max_font_size=300)
#     plt.gca().imshow(cloud)
#     plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
#     plt.gca().axis('off')


# plt.subplots_adjust(wspace=0, hspace=0)
# plt.axis('off')
# plt.margins(x=0, y=0)
# plt.tight_layout()
# plt.show()


# 10. Word Counts of Topic Keywords
print("========== PART 10 ==========")
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data_ready for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

# df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# # Plot Word Count and Weights of Topic Keywords
# fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
# cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
# for i, ax in enumerate(axes.flatten()):
#     ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
#     ax_twin = ax.twinx()
#     ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
#     ax.set_ylabel('Word Count', color=cols[i])
#     ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
#     ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
#     ax.tick_params(axis='y', left=False)
#     ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
#     ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

# fig.tight_layout(w_pad=2)    
# fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
# plt.show()


# # 11. Sentence Chart Colored by Topic
# print("========== PART 11 ==========")
# # Sentence Coloring of N Sentences
# from matplotlib.patches import Rectangle

# def sentences_chart(lda_model=lda_model, corpus=corpus, start = 0, end = 13):
#     corp = corpus[start:end]
#     mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

#     fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       
#     axes[0].axis('off')
#     for i, ax in enumerate(axes):
#         if i > 0:
#             corp_cur = corp[i-1] 
#             topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
#             word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
#             ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
#                     fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

#             # Draw Rectange
#             topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
#             ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
#                                    color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

#             word_pos = 0.06
#             for j, (word, topics) in enumerate(word_dominanttopic):
#                 if j < 14:
#                     ax.text(word_pos, 0.5, word,
#                             horizontalalignment='left',
#                             verticalalignment='center',
#                             fontsize=16, color=mycolors[topics],
#                             transform=ax.transAxes, fontweight=700)
#                     word_pos += .009 * len(word)  # to move the word for the next iter
#                     ax.axis('off')
#             ax.text(word_pos, 0.5, '. . .',
#                     horizontalalignment='left',
#                     verticalalignment='center',
#                     fontsize=16, color='black',
#                     transform=ax.transAxes)       

#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
#     plt.tight_layout()
#     # plt.show()

# sentences_chart()    

# 12. What are the most discussed topics in the documents?
# print("========== PART 12 ==========")
# # Sentence Coloring of N Sentences
# def topics_per_document(model, corpus, start=0, end=1):
#     corpus_sel = corpus[start:end]
#     dominant_topics = []
#     topic_percentages = []
#     for i, corp in enumerate(corpus_sel):
#         topic_percs, wordid_topics, wordid_phivalues = model[corp]
#         dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
#         dominant_topics.append((i, dominant_topic))
#         topic_percentages.append(topic_percs)
#     return(dominant_topics, topic_percentages)

# dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

# # Distribution of Dominant Topics in Each Document
# df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
# dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
# df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# # Total Topic Distribution by actual weight
# topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
# df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# # Top 3 Keywords for each Topic
# topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
#                                  for j, (topic, wt) in enumerate(topics) if j < 3]

# df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
# df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
# df_top3words.reset_index(level=0,inplace=True)


# from matplotlib.ticker import FuncFormatter

# # Plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

# # Topic Distribution by Dominant Topics
# ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
# ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
# tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
# ax1.xaxis.set_major_formatter(tick_formatter)
# ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
# ax1.set_ylabel('Number of Documents')
# ax1.set_ylim(0, 1000)

# # Topic Distribution by Topic Weights
# ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
# ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
# ax2.xaxis.set_major_formatter(tick_formatter)
# ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))
# plt.show()

# 13. t-SNE Clustering Chart
# print("========== PART 13 ==========")
# # Get topic weights and dominant topics ------------
# from sklearn.manifold import TSNE
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import Label
# from bokeh.io import output_notebook

# Get topic weights
# topic_weights = []
# for i, row_list in enumerate(lda_model[corpus]):
#     topic_weights.append([w for i, w in row_list[0]])

# # Array of topic weights    
# arr = pd.DataFrame(topic_weights).fillna(0).values

# # Keep the well separated points (optional)
# arr = arr[np.amax(arr, axis=1) > 0.35]

# # Dominant topic number in each doc
# topic_num = np.argmax(arr, axis=1)

# # tSNE Dimension Reduction
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
# tsne_lda = tsne_model.fit_transform(arr)

# # Plot the Topic Clusters using Bokeh
# output_notebook()
# n_topics = 4
# mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
# plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
#               plot_width=900, plot_height=700)
# plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
# show(plot)


# 14. pyLDAVis
print("========== PART 14 ==========")
import pyLDAvis.gensim

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
pyLDAvis.save_html(vis, "LDA_test_outlier.html")
print("\n========== ****Create HTML Success**** ==========")
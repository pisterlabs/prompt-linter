import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models import LsiModel
import numpy as np
from collections import Counter
import pyLDAvis.gensim
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from numpy import random
from nltk.corpus import stopwords
from pathlib import Path
import os
random.seed(1)


def create_gensim_lda_model(doc_clean, number_of_topics, words, dictionary, corpus):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    # generate LSA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=number_of_topics,
                                                random_state=100, update_every=1, chunksize=100, passes=10,
                                                alpha='auto', per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return lda_model


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, model_lsa, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for number_of_topics in range(start, stop, step):
        # generate LSA model
        if model_lsa:
            model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
        else:
            model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix, id2word=dictionary,
                                                    num_topics=number_of_topics, random_state=100, update_every=1,
                                                    chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


# Finding the dominant topic in each sentence
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = pd.concat([sent_topics_df,
                                                pd.DataFrame([pd.Series([int(topic_num), round(prop_topic, 4),
                                                                         topic_keywords])])], ignore_index=True)
                # sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),
                # topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


# Finding the dominant topic in each sentence
def format_topics_sentences_prob(ldamodel, corpus, texts):
    # Init output
    perc_topics_df = pd.DataFrame(0.0, index=np.arange(len(texts)),
                                  columns=['perc_1', 'perc_2', 'perc_3', 'perc_4', 'perc_5', 'perc_6', 'perc_7', 'perc_8'])
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        # row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row[0]):
            prob = round(prop_topic, 4)
            perc_topics_df['perc_{}'.format(j+1)][i] = prob

    return perc_topics_df


def frequency_distribution_word_counts_in_documents(df_dominant_topic, project_key, num_topics):
    doc_lens = [len(d) for d in df_dominant_topic.Text]

    # Plot
    plt.figure(figsize=(16, 7), dpi=150)
    plt.hist(doc_lens, bins=800, color='navy')

    plt.text(600, 60, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(600, 50, "Median : " + str(round(np.median(doc_lens))))
    plt.text(600, 40, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(600, 30, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(600, 20, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 800), ylabel='Number of Issues', xlabel='Issues Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 800, 9))
    plt.title('Distribution of Issues Word Counts {}'.format(project_key), fontdict=dict(size=20))
    path = addPath(f'Master/Models/topic_model/{project_key}/distribution_issues_word_counts_{project_key}.png')
    plt.savefig(path)
    plt.close()

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(num_topics, figsize=(16, 14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins=1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: ' + str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, 1000, 9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic {}'.format(project_key), fontsize=22)
    path = addPath(f'Master/Models/topic_model/{project_key}/'
                   f'distribution_issues_word_counts_byDominantTopic_{project_key}.png')
    plt.savefig(path)
    plt.close()


def word_clouds_top_n_keywords_each_topic(stop_words, lda_model, project_key, num_topics):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(num_topics, figsize=(10, 10), sharex=True, sharey=True)

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

    path = addPath(f'Master/Models/topic_model/{project_key}/'
                   f'word_clouds_top_n_keywords_each_topic_{project_key}.png')
    plt.savefig(path)
    plt.close()


def word_clouds_of_topic_keywords(lda_model, data_words_bigrams, project_key, num_topics):
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_words_bigrams for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(num_topics, figsize=(16, 10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030);
        ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    path = addPath(f'Master/Models/topic_model/{project_key}/'
                   f'word_clouds_of_topic_keywords_{project_key}.png')
    plt.savefig(path)
    plt.close()


def vizu_word_in_topic(lda_model, corpus):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    vis


def create_topic_model(data_train, data_test, number_of_topics, project_key):
    """
    the function get the data and return the dominant topic to the train and test set
    """
    
    words = 10
    text_train_list = []
    for row in data_train['clean_text_new']:
        text_train_list.append(row)
    # Creating the term dictionary of our corpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(text_train_list)

    path = addPath(f'Master/Models/topic_model/{project_key}/'
                   f'dictionary_{project_key}')
    dictionary.save(path)
    # dictionary = corpora.Dictionary.load("dictionary_{}".format(project_key))
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    corpus = [dictionary.doc2bow(doc) for doc in text_train_list]
    # create lda model
    lda_model = create_gensim_lda_model(text_train_list, number_of_topics, words, dictionary, corpus)

    path = addPath(f'Master/Models/lda_models/{project_key}/lda_model_{project_key}')
    lda_model.save(path)

    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, text_train_list)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text']
    # test
    text_test_list = []
    for row in data_test['clean_text_new']:
        text_test_list.append(row)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    corpus_test = [dictionary.doc2bow(doc) for doc in text_test_list]
    df_topic_sents_keywords_test = format_topics_sentences(lda_model, corpus_test, text_test_list)
    # Format
    df_dominant_topic_test = df_topic_sents_keywords_test.reset_index()
    df_dominant_topic_test.columns = ['Document_No', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text']
    # plot:
    stop = stopwords.words('english')
    frequency_distribution_word_counts_in_documents(df_dominant_topic, project_key, number_of_topics)
    word_clouds_top_n_keywords_each_topic(stop, lda_model, project_key, number_of_topics)
    word_clouds_of_topic_keywords(lda_model, text_train_list, project_key, number_of_topics)

    return df_dominant_topic, df_dominant_topic_test


def addPath(path):
    return str(Path(os.getcwd()).joinpath(path))

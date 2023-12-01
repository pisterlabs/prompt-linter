import re
import os
import pwd
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['lines.markersize'] = 5
from wordcloud import WordCloud


topics_color_dict = {0: '#292F68', 1: '#FF6011', 2: '#97D89D', 3: '#338CCC', 4: '#687269',
                     5: '#A288E3', 6: '#1DD3B0', 7: '#8E9AAF', 8: '#5333AA', 9:'#C2F970', 10: '#8e0dbd',
                     11: '#00b294', 12: '#f60101', 13: '#0d48bd', 14: '#c8f042'}


def get_filename(filepath):
    files = [f.path for f in os.scandir(filepath) if f.is_file()]  # files in corpus folders
    return files


def text_to_df(file_path):
    """
    1. import text files from respective corpus folders
    2. parse the metadata and add as database columns
    3. clean and store raw input and add in text column
    4. standardize database structure and store { id | rawText | title }
    :param file_path: myth_corpus | tech_corpus folders
    :return: myth_corpus.csv | tech_corpus.csv files
    """
    doc_list = []
    layer = []
    title = []
    files = get_filename(f"{file_path}")
    for file in files:
        naming_list = file.split('/')  # parse corpus_folder/filename.txt to layer/title
        layer.append(naming_list[-2])  # layer = 'myth_corpus' | 'tech_corpus"
        title.append(naming_list[-1])  # title = 'TitleOfDoc.txt'
        with open(file) as f:
            string_raw = f.read()
            norm_string = re.sub(r'(5g|5G)', 'five_g', string_raw)
            norm_string = re.sub(r'(4g|4G)', 'four_g', norm_string)
            norm_string = re.sub(r'(3g|3G)', 'three_g', norm_string)
            norm_string = re.sub(r'(2g|2G)', 'two_g', norm_string)
            norm_sans_email = re.sub('\S*@\S*\s?', '', norm_string)  # remove emails
            norm_sans_url = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', norm_sans_email,
                                   flags=re.MULTILINE)  # remove url
            " ".join([x for x in norm_sans_url.split(" ") if not x.isdigit()])
            norm_sans_newline = re.sub('\s+', ' ', norm_sans_url)  # remove newline chars
            norm_sans_quotes = re.sub("\'", "", norm_sans_newline)  # remove single quotes
            s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', norm_sans_quotes)  # xThis -> xx. This
            s = s.lower()  # lower case
            s = re.sub(r'&gt|&lt', ' ', s)  # remove encoding format
            s = re.sub(r'([a-z])\1{2,}', r'\1', s)  # letter repetition (if more than 2)
            s = re.sub(r'([\W+])\1{1,}', r'\1', s)  # non-word repetition (if more than 1)
            s = re.sub(r'\W+?\.', '.', s)  # xxx[?!]. -- > xxx.
            s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)  # [.?!]xxx --> [.?!] xxx
            s = re.sub(r'\\x', r'-', s)  # 'x' --> '-'
            sentence = s.strip()  # remove padding spaces
            string_join = ' '.join(sentence.splitlines())
            doc_list.append(string_join)
            df = pd.DataFrame({'rawText': doc_list, 'layer': layer, 'title': title})
            df.to_csv(f"{file_path}.csv")  # directory for corpus in csv


def get_topic_words(token_lists, labels, k=None):
    """
    get top words within each topic from clustering results
    """
    if k is None:
        k = len(np.unique(labels))
    topics = ['' for _ in range(k)]
    for i, c in enumerate(token_lists):
        topics[labels[i]] += (' ' + ' '.join(c))
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    # get sorted word counts
    word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
    # get topics
    topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

    return topics


def get_coherence(model, token_lists, measure='c_v'):
    """
    Get model coherence from gensim.models.coherencemodel
    :param model: Topic_Model object
    :param token_lists: token lists of docs
    :param topics: topics as top words
    :param measure: coherence metrics
    :return: coherence score
    """
    if model.method == 'LDA':
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    else:
        topics = get_topic_words(token_lists, model.cluster_model.labels_)
        cm = CoherenceModel(topics=topics, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()


def get_silhouette(model):
    """
    Get silhouette score from model
    :param model: Topic_Model object
    :return: silhouette score
    """
    if model.method == 'LDA':
        return
    lbs = model.cluster_model.labels_
    vec = model.vec[model.method]
    return silhouette_score(vec, lbs)


def get_wordcloud(model, token_lists, topic):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """
    junk = pd.read_csv('stopwords_extended.txt', names=['common'], header=0)
    junk = [word.lower() for word in junk.common]
    token_lists[:] = [[word for word in token_lst if word not in junk] for token_lst in token_lists]
    if model.method == 'LDA':
        return
    print('Getting wordcloud for topic {} ...'.format(topic))
    lbs = model.cluster_model.labels_
    tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])
    print("tokens,", len(tokens))
    wordcloud = WordCloud(width=800, height=560,
                          background_color='black', collocations=False,
                          min_font_size=10).generate(tokens)
    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    dr = f"results/{model.method}/{model.id}/images"
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/Topic' + str(topic) + '_wordcloud')
    print('Getting wordcloud for topic {}. Done!'.format(topic))
    plt.close()
    return tokens


def plot_proj(embedding, lbs):
    """
    Plot UMAP embeddings
    :param embedding: UMAP (or other) embeddings
    :param lbs: labels
    """
    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.9,
                 label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.grid(b=True, which='major', color='#212020', linestyle='-')
    plt.legend()


def visualize(model):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """
    if model.method == 'LDA':
        return
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    vec_umap = reducer.fit_transform(model.vec[model.method])
    print('Calculating UMAP projection. Done!')
    plot_proj(vec_umap, model.cluster_model.labels_)
    dr = f"results/{model.method}/{model.id}/images"
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/2D_vis')
    plt.close()



def plot_embeddings(embedding, subcat_dict, subcat_labels):
    for word in subcat_labels:
        x, y = embedding[subcat_dict[word]]
        plt.text(x + .015, y + .015, word, fontsize=15, color="white")
    plt.grid(b=True, which='major', color='#212020', linestyle='-')


def visualize_test(model, lbsT, sentencesT, token_listsT, titlesT):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """
    if model.method == 'LDA':
        return
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    sub_cat_dict = {key: i for i, key in enumerate(titlesT)}
    tech_vec = model.vectorize(sentencesT, token_listsT)
    tech_embedding = reducer.fit_transform(tech_vec)
    colrs = [topics_color_dict[key] for key in lbsT]

    plt.scatter(tech_embedding[:, 0], tech_embedding[:, 1], s=60, c=colrs)
    plot_embeddings(tech_embedding, sub_cat_dict, titlesT)

    dr = f"results/{model.method}/{model.id}/images"
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/2D_visT')
    plt.close()



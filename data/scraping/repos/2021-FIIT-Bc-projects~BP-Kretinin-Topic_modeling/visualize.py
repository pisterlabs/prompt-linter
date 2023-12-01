# Copyright 2022 Mykyta Kretinin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#  Part of the module's functionality and algorithms was inspired by the code
#  from the article https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/


import numpy as np, pandas as pd

# Gensim
import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt



import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['include', 'standard', 'principle', 'thomson', 'reuter', 'oct', 'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

def show_docs_per_topic(lda_model, corpus):
    docs_counts = [0] * lda_model.num_topics

    # Count documents in each topic
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                docs_counts[topic_num] += 1
            else:
                break

    # Create dataframe
    docs_per_topics_df = pd.DataFrame()
    for topic_num in range(lda_model.num_topics):
        wp = lda_model.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        docs_per_topics_df = docs_per_topics_df.append(pd.Series([int(topic_num), docs_counts[topic_num],
                                                                  topic_keywords]), ignore_index=True)
    docs_per_topics_df.columns = ['Topic', 'Documents_assigned', 'Topic_Keywords']

    # Visualization
    plt.rcParams["figure.figsize"] = [16, 3]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    axs.axis('off')

    the_table = axs.table(cellText=docs_per_topics_df.values,
                          colLabels=docs_per_topics_df.columns,
                          loc='upper center',
                          cellLoc='center')

    the_table.auto_set_column_width(col=list(range(len(docs_per_topics_df.columns))))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    plt.title('Documents per topic statistics', fontdict=dict(size=22))
    plt.show()


def show_coherence_and_perplexity(model, corpus):

    # Tokenized text is needed for a coherence model
#    texts_BoW = get_corpus(model, texts)

    # Coherence model to get coherence score, based on currently used corpus
    coherence_model_lda = gensim.models.CoherenceModel(model=model, corpus=corpus, coherence='u_mass')

    print("Calculating coherence, it might take a while...")

    coherence_score = coherence_model_lda.get_coherence()
    coherence_per_topic = coherence_model_lda.get_coherence_per_topic()

    print("U_Mass Coherence value: " + str(coherence_score))
    print("U_Mass Coherence per-topic values: " + str(coherence_per_topic))

    perplexity_score = model.log_perplexity(corpus)

    print("Perplexity: " + str(perplexity_score))



def show_table_with_documents_stats(dataFrame, text_numbers = None):

    # Create temporary copy of data frame to not change original one
    if text_numbers is None:
        text_numbers = []
    temp_df = dataFrame.copy(deep=True)
    if ("Text" in temp_df):
        temp_df = temp_df.drop(columns="Text")

    # Limit number of docs to not overflow the window
    if not bool(text_numbers) or len(text_numbers) > 20:
        temp_df = temp_df.head(10)
    else:
        temp_df = temp_df.iloc[text_numbers]

    colLabels = temp_df.columns

    # Choose first 6 words (or less, if they aren't there) of the text
    #if ('Text' in temp_df):
    #    temp_df['Text'] = temp_df['Text'].apply(lambda x: x.rsplit(maxsplit=len(x.split()) - 6)[0])
    #elif ('Representative Text' in temp_df):
    #    temp_df['Representative Text'] = temp_df['Representative Text'].apply(
    #        lambda x: x.rsplit(maxsplit=len(x.split()) - 6)[0])
    #else:
    #    print("Wrong data frame")
    #    return

    plt.rcParams["figure.figsize"] = [16, 3]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    axs.axis('off')

    the_table = axs.table(cellText=temp_df.values,
                          colLabels=colLabels,
                          loc='upper center',
                          cellLoc='center')

    the_table.auto_set_column_width(col=list(range(len(colLabels))))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    plt.title('Documents statistics', fontdict=dict(size=22))
    plt.show()


def show_statistics(lda_model, corpus, texts, text_numbers):
    def format_topics_sentences(ldamodel=None, corpus=corpus, texts=texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', "Text"]


    text_numbers = list(set(text_numbers))


    show_table_with_documents_stats(df_dominant_topic, text_numbers)

#


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
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    sent_topics_sorteddf_mallet = sent_topics_sorteddf_mallet.drop(columns="Text")




    show_table_with_documents_stats(sent_topics_sorteddf_mallet)





    # Words can be counted as amount of " "(spacebar) + 1
    doc_lens = [d.count(' ')+1 for d in texts]

    # Calculate and delete outliers, if they are
    # Formula: |x - mean| < 2 * std
    mean = np.mean(doc_lens)
    standard_deviation = np.std(doc_lens)
    distance_from_mean = abs(doc_lens - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = []
    for j, is_not_outlier in enumerate(not_outlier):
        if is_not_outlier:
            no_outliers.append(doc_lens[j])
    doc_lens = no_outliers

    if (len(corpus) > 1):
        values, counts = np.unique(doc_lens, return_counts=True)

        # Plot
        plt.figure(figsize=(16, 7), dpi=160)

        ax = plt.gca()

        ax.set(xlim=(min(doc_lens), max(doc_lens)), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color="black")
        ax.set_xlabel('Words', color="black")

        sns.histplot(ax=ax, x=doc_lens, bins="auto", stat="count")
        sns.kdeplot(doc_lens, color="black", ax=ax.twinx())


        plt.text(0.85 * max(doc_lens), 0.9 * plt.ylim()[1], "Mean   : " + str(round(np.mean(doc_lens))))
        plt.text(0.85 * max(doc_lens), 0.8 * plt.ylim()[1], "Median : " + str(round(np.median(doc_lens))))
        plt.text(0.85 * max(doc_lens), 0.7 * plt.ylim()[1], "Stdev  : " + str(round(np.std(doc_lens))))
        plt.text(0.85 * max(doc_lens), 0.6 * plt.ylim()[1], "1%ile  : " + str(round(np.quantile(doc_lens, q=0.01))))
        plt.text(0.85 * max(doc_lens), 0.5 * plt.ylim()[1], "99%ile : " + str(round(np.quantile(doc_lens, q=0.99))))


        plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
        plt.show()





    if (len(corpus) > 1):
        cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS' / mcolors.TABLEAU_COLORS

        rows = int(np.sqrt(lda_model.num_topics))
        columns = int(lda_model.num_topics/rows)+1


        fig, axes = plt.subplots(rows, columns, figsize=(16, 14), dpi=160, sharex="none", sharey="none")


        for i, ax in enumerate(axes.flatten()):

            # delete unused subplots
            if i >= lda_model.num_topics:
                plt.delaxes(axes[int(i / columns)][i % columns])
                continue

            df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
            doc_lens = [d.count(' ')+1 for d in df_dominant_topic_sub.Text]

            mean = np.mean(doc_lens)
            standard_deviation = np.std(doc_lens)
            distance_from_mean = abs(doc_lens - mean)
            max_deviations = 2
            not_outlier = distance_from_mean < max_deviations * standard_deviation

            no_outliers = []
            for j, is_not_outlier in enumerate(not_outlier):
                if is_not_outlier:
                    no_outliers.append(doc_lens[j])
            doc_lens = no_outliers

            if bool(doc_lens):
                max_len = max(doc_lens)
                min_len = min(doc_lens)
                empty_docs = False
            else:
                max_len = 1
                min_len = 0
                empty_docs = False


            ax.set(xlim=(min_len, max_len), xlabel='Document Word Count')
            ax.set_xticks(np.linspace(start=min_len, stop=max_len, num=3))
            ax.set_ylabel('Number of Documents', color="black")
            ax.set_xlabel('Words', color="black")
            ax.set_title('Topic: ' + str(i+1), fontdict=dict(size=16, color=cols[i]))
            ax.tick_params(axis='y', labelcolor="black", color=cols[i])

            if not (empty_docs):
                sns.histplot(ax=ax, color=cols[i], x=doc_lens, bins="auto", stat="count")
                sns.kdeplot(doc_lens, color="black", ax=ax.twinx())
            else:
                ax.text(0.35, 0.5, "Empty")


        fig.tight_layout()
        plt.subplots_adjust()
        fig.subplots_adjust(top=0.90)
        fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
        plt.show()


def t_SNE_clustering(lda_model, corpus):
    if (len(corpus) < 2):
        print("Required 2 or more texts in the corpus")
        return
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        topic_weights.append([w for i, w in row])


    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    print("Choose perplexity (neigbors=3*perplexity):")
    perpl = int(input())

    if perpl < 0:
        perpl = 30

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca', n_jobs=7, perplexity=perpl)
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_file("reports/t-SNE_clusters.html")
    n_topics = lda_model.num_topics
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    show(plot)

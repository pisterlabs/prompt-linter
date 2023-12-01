#A very good reference here: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

#for data processing for Gensim
def Gensim_data_processing(lemmatized): # a list of m sentences, each sentence is a lemmatized string
    import gensim
    import gensim.corpora as corpora
    # We should create the data that fits the format of gensim
    lemmatized_for_gensim=[i.split(' ') for i in lemmatized]
    # There are two important inputs for LdaModel in gesim
    id2word = corpora.Dictionary(lemmatized_for_gensim) #id to word
    corpus = [id2word.doc2bow(text) for text in lemmatized_for_gensim] #m, features
    return lemmatized_for_gensim, corpus, id2word

#this block can test different number of topics, return the perplexity and coherence scores.
#this block can also return the final model after choosing the best number of topic
def LDA_Gensim_model(lemmatized_for_gesnim, corpus, id2word, start_topic, end_topic):
    import gensim
    from tqdm import tqdm
    per_list=[]
    coh_list=[]
    topic_list=[i for i in range(start_topic,end_topic+1)]
    for i in tqdm(topic_list): # Specify the number of topics for experimentation
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=i,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='symmetric',
                                                   iterations=100,
                                                   per_word_topics=True)

        # Compute Perplexity, a measure of how good the model is. Lower the better.
        perplexity_score=lda_model.log_perplexity(corpus)
        print(i,'Perplexity: ', perplexity_score)
        # Compute Coherence Score
        from gensim.models import CoherenceModel
        coherence_model_lda = CoherenceModel(model=lda_model, texts=lemmatized_for_gesnim, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(i,'Coherence Score: ', coherence_lda)

        per_list.append(perplexity_score)
        coh_list.append(coherence_lda)
    return per_list, coh_list, lda_model # return lists of perplexity score and coherence score, and the final model

#this block is for plotting perplexity score and coherence score of different models
def plot_score(per_list,coh_list, start_topic, end_topic):
    # importing the required module
    import matplotlib.pyplot as plt

    # x axis values
    x = [i for i in range(start_topic, end_topic+1)]
    # plt.style.use('dark_background')
    # corresponding y axis values
    # plotting the points
    plt.plot(x, per_list)
    plt.plot(x, coh_list)

    # naming the x axis
    plt.xlabel('Number of topics', fontsize=12, horizontalalignment='right', position=(1, 25))

    # naming the y axis
    plt.ylabel('Value', fontsize=12)

    # giving a title to my graph
    plt.title('LDA models by # of topics', fontsize=14)
    plt.legend(["Perplexity score", "Coherence score"], loc=3, frameon=True)
    plt.ylim([0.2,0.5])
    # function to show the plot
    plt.show()

#this block is for plotting in another style (two axes)
def plot_score2(per_list, coh_list, start_topic, end_topic):
    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # Creating dataset
    dataset_1 = per_list
    dataset_2 = coh_list
    x = [i for i in range(start_topic, end_topic + 1)]
    # Creating plot with dataset_1
    fig, ax1 = plt.subplots()

    color = 'tab:orange'
    ax1.set_xlabel('Number of topics', fontsize=12, horizontalalignment='right', position=(1, 25))
    ax1.plot(x, dataset_1, color=color, linewidth=2.1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
    plt.legend(["Perplexity score"], loc=3, frameon=True)

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.plot(x, dataset_2, color=color, linewidth=2.1)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Adding title
    plt.title('LDA models by # of topics', fontsize=14)
    plt.legend(["Coherence score"], loc=1, frameon=True)

    plt.plot(8, 0.483, marker="o", markersize=10, color='green')
    plt.text(8.5, 0.483, 'Topic:8')

    plt.plot(15, 0.431, marker="o", markersize=10, color='purple')
    plt.text(15.5, 0.431, 'Topic:15')

    plt.plot(30, 0.343, marker="o", markersize=10, color='grey')
    plt.text(28, 0.353, 'Topic:30')

    # Show plot
    plt.show()

# Look at the weights and keywords of the final model
print (lda_model.print_topics()) #return the weights and keywords of each topic

# Plot t-SNE Clustering Chart in html form
def plot_t_SNE_clustering(lda_model, corpus, n_topics):
    # Get topic weights and dominant topics
    import pandas as pd
    import numpy as np
    import matplotlib.colors as mcolors
    from sklearn.manifold import TSNE
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import Label
    output_file("figure1.html") #output the file using html format

    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    #Plot the Topic Clusters using Bokeh
    #output_notebook()
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=600, plot_height=400)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    show(plot)

# Plot the topic distribution html， bubbles
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'visualization1.html') #display using html

# This block can look at the dominant topic of each sentence
def Dominant_topic(lda_model, corpus, lemmatized_for_gensim):
    #Init output
    import pandas as pd
    from tqdm import tqdm
    sent_topics_df = pd.DataFrame()
    texts=lemmatized_for_gensim
    # Get main topic in each document
    for i, row_list in tqdm(enumerate(lda_model[corpus])):
        row = row_list[0] if lda_model.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df = sent_topics_df.reset_index()
    sent_topics_df.colums=['No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return (sent_topics_df)

# Group top k representative sentences under each topic
def Group_representatives(sent_topics_df, num_of_sentences):
    import pandas as pd
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = sent_topics_df.groupby('Dominant_Topic') #group by Dominant Topic


    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(num_of_sentences)],
                                                axis=0)
    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    sent_topics_sorteddf_mallet

    # Format
    sent_topics_sorteddf_mallet.columns = ['Index','Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    return sent_topics_sorteddf_mallet

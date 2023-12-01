from dataset_creation.mongo import get_post_description, get_post_comments
import gensim
import re
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import os


NUM_TOPICS = 4


def lda(category, influencers_text_data, text_type):
    """
    Implements lda in order to detect the emerging topics
    :param category: influencer category e.g. yoga, pilates, nutrition
    :param influencers_text_data: list of text data
    :param text_type: either post's decription or post's comments
    :return: -
    """

    tokens = []
    # Create a dictionary
    # ----------------------------------
    print('\nCreating dictionary..')
    if text_type == 'Description':
        for influencer_posts in influencers_text_data:
            for post_description in influencer_posts:
                #print(post_description)
                try:
                    if post_description:
                        tokens.append(post_description.split(' '))
                except:
                    print("An exception occurred")
    elif text_type == 'Comments':
        for influencer_posts in influencers_text_data:
            for post in influencer_posts:
                for comment in post:
                    if comment:
                        try:
                            tokens.append(comment.split(' '))
                        except:
                            print("An exception occurred")
    else:
        return



    dictionary = gensim.corpora.Dictionary(tokens)
    #print(len(id2word))

    dictionary.filter_extremes(no_below=2, no_above=.99) # Filtering Extremes
    #print(len(id2word))
    # ----------------------------------


    # Creating a corpus object
    # ----------------------------------
    print('\nCreating corpus..')
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # ----------------------------------
    if not corpus:
        print("Corpus is empty " + category + " " + text_type)
        return
    if not dictionary:
        print("Dictionary is empty " + category + " " + text_type)
        return


    # LDA model
    # ----------------------------------
    print('\nBuilding LDA model..')
    LDA_model = gensim.models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=5)  # Instantiating a Base LDA model
    # ----------------------------------


    # Create Topics
    # ----------------------------------
    print('\nTopics:')
    words = [re.findall(r'"([^"]*)"', t[1]) for t in LDA_model.print_topics()]   # Filtering for words
    topics = [' '.join(t[0:10]) for t in words]

    for id, t in enumerate(topics): # Getting the topics
        print(f"------ Topic {id} ------")
        print(t, end="\n\n")
    # ----------------------------------


    # Print topics with propabilities
    # ----------------------------------
    print('\nTopics with propabilities:')
    for i in LDA_model.print_topics():
        for j in i: print(j)
    # ----------------------------------


    # Get most frequent words of each topic
    # ----------------------------------
    print('\nMost frequent words by topic:')
    topic_words = []
    for i in range(NUM_TOPICS):
        tt = LDA_model.get_topic_terms(i, 20)
        topic_words.append([dictionary[pair[0]] for pair in tt])

    # output
    for i in range(NUM_TOPICS):
        print(f"\n------ Topic {i} ------")
        print(topic_words[i])
    # ----------------------------------


    # Compute Coherence and Perplexity
    # ----------------------------------
    #Compute Perplexity, a measure of how good the model is. lower the better
    print('\nComputing Coherence and Perplexity..')
    base_perplexity = LDA_model.log_perplexity(corpus)
    print('\nPerplexity: ', base_perplexity)

    # Compute Coherence Score
    coherence_model = CoherenceModel(model=LDA_model, texts=tokens,
                                   dictionary=dictionary, coherence='c_v')
    coherence_lda_model_base = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence_lda_model_base)
    # ----------------------------------


    # Creating Topic Distance Visualization
    # ----------------------------------
    print('\nCreating visualization..')
    visualisation = pyLDAvis.gensim_models.prepare(LDA_model, corpus, dictionary)

    directory = 'output/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pyLDAvis.save_html(visualisation, directory + text_type + '_LDA_Visualization_' + category + '.html')
    # ----------------------------------


if __name__ == "__main__":
    posts_description_df = get_post_description()
    #print(posts_description_df)
    for index, row in posts_description_df.iterrows():
        print(row['posts_description'])
        #lda(row['_id'], row['posts_description'], 'Description')

    comments = get_post_comments()
    #for index, row in comments.iterrows():
        #lda(row['_id'], row['comments'], 'Comments')

    # -----------------------------------------------
    # Create LDA for all categories
    descr_list = []
    print(posts_description_df['posts_description'])
    for index, row in posts_description_df.iterrows():
        for post in row['posts_description']:
            descr_list.append(post)
    print(descr_list)

    lda('all_categories', descr_list, 'Description')

    comments_list = []
    print(comments['comments'])
    for index, row in comments.iterrows():
        for post in row['comments']:
            comments_list.append(post)
    print(descr_list)

    lda('all_categories', comments_list, 'Comments')







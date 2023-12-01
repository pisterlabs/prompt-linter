import spacy
import numpy as np
from spacy.tokens import Doc
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import pdb
import time
import pdb
from py2neo import Graph

from collections import Counter
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk
from nltk.corpus import stopwords
import pickle
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import pdb

nltk.download('vader_lexicon')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_lg")
stop_words = stopwords.words('english')
sentiment_analyzer = SentimentIntensityAnalyzer()

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password")) 


def strip_tweets(tweet):
    '''Process tweet text to remove retweets, mentions,links and hashtags.'''
    retweet = r'RT:? ?@\w+:?'
    tweet= re.sub(retweet,'',tweet)
    mention = r'@\w+'
    tweet= re.sub(mention,'',tweet)
    links = r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$'
    tweet= re.sub(links,'',tweet)
    tweet_links = r'https:\/\/t\.co\/\w+|http:\/\/t\.co\/\w+'
    tweet=re.sub(tweet_links,'',tweet)  
    tweet_link = r'http\S+'
    tweet=re.sub(tweet_link,'',tweet)
    hashtag = r'#\w+'
    tweet= re.sub(hashtag,'',tweet)
    return tweet

def polarity_scores(doc):
    return sentiment_analyzer.polarity_scores(doc.text)

def graph_sentiment(text):
    tweet = nlp(strip_tweets(text))
    return tweet._.polarity_scores['compound'],tweet.vector


Doc.set_extension('polarity_scores', getter=polarity_scores)

def encode_sentiment(tweet):
    if tweet['truncated']:
    	if isinstance(tweet['extended_tweet'],str):
        	sentiment,embedding = graph_sentiment(tweet['extended_tweet'])
    	else:
    		return
    else:
    	if isinstance(tweet['text'],str):
        	sentiment,embedding = graph_sentiment(tweet['text'])
    	else:
    		return
    sentiment=float(sentiment)
    embedding = np.array2string(embedding,separator=',')
    t_id=tweet['id_str']
    query = '''MERGE (t:Tweet {id_str: $id})
    ON CREATE SET t.stranded = 1 
    ON MATCH SET t.sentiment = $sentiment,
        t.embedding = $embedding
    '''
    graph.run(query,id=t_id,sentiment=str(sentiment),embedding=embedding)

if __name__ == '__main__':
	query = '''MATCH (t:Tweet)
	WHERE NOT exists(t.sentiment)
	RETURN t
	'''
	tweets = graph.run(query)
	for tweet in tweets:
		encode_sentiment(tweet['t'])

# Topic Modeling

def model_topics(df):
    """ Go through a corpus of documents and perform topic modeling, code sourced from the 
    excellent tutorial at Machine Learning Plus found at the link below.
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

    Parameters:
        df (DataFrame): Pandas DF containing a column labeled text on which the topic
            modeling is performed.
    Returns:
        model_list ([LdaMulticore]): List of gensim LdaMulticore models trained for different
            numbers of topics.
        coherence_values ([float]): List of float values with Coherence scores for the 
            corresponding models in the above list.
        id2word (Dictionary): Word to Id mapping for the words appearing in the corpus.
        corpus ([[int]]): Documents converted into a bag of words referenced by id in 
            Dictionary above.
    """

    data = df.text.values.tolist()
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Perform Topic Modeling for number of topics ranging from 5 to 50 in steps of 5
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=5, limit=50, step=5)

    return model_list,coherence_values,corpus,id2word

def sent_to_words(sentences):
    """ Process sentences in the tweets into a list of tokens. """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    """ Process the words in each tweet of the corpus to remove stopwords as defined by NLTK.  """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    """ Create list of bigrams from the tweets in the corpus. """
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,trigram_mod,bigram_mod):
    """ Create list of trigrams from the tweets in the corpus. """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """ Create lemmas through spaCy https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, limit, start=5, step=5):
    """ Compute coherence values for various number of topics.

    Parameters:
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
        start : Int of lowest number of topics to model
        step : Int of increment to change number of topics to model

    Returns:
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    pdb.set_trace()
    for num_topics in range(start, limit, step):
        start=time.time()
        model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           chunksize=10000,
                                           passes=1,
                                           per_word_topics=True)
        
        print(f'Topic modeling for {num_topics} topics took {time.time()-start} seconds.')
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def graph_coherence(coherence_values):
    """ Graph a list of coherence values to determine the optimal number of topics to model. """
    limit=50; start=5; step=5;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Print the coherence scores    
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

def find_dominant_topic(df_topic_sents_keywords):
    """ Discover the most representative document in a corpus for each topic, code sourced from the 
    excellent tutorial at Machine Learning Plus found at the link below.
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    """

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Group top 5 sentences under each topic
    sent_topics_sorteddf = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    start = time.time()
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)
        print(f'Group done. Total time {time.time() - start} seconds.')

    # Reset Index    
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    return sent_topics_sorteddf

def topic_stats(df_topic_sents_keywords):
    """ Compute aggregations and statistics on the topics modeled in a corpus, 
    code sourced from the excellent tutorial at Machine Learning Plus found at the link below.
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    """

    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    # Show
    df_dominant_topics

def group_topics(sent_topics_sorteddf):
    """ Relabel and aggregate documents under handlabeled and binned categories."""
    new_topics=pd.concat([sent_topics_sorteddf.groupby('Topic_Num').head()[['Keywords']],
           topic_contribution.sort_index(),
          pd.Series(['Economy','Immigration','Environment','Event',
          'Civil Rights','Civil Rights','Healthcare',
          'Defense','Trump','Community','Event','Event',
          'Thanks','Legislation','Trump','Community',
          'Community','Trump','Defense',
          'Legislation','Thanks','Economy','Thanks','Healthcare',
          'Legislation'])],axis=1).groupby(0).sum()
    plt.pie(new_topics,labels=new_topics.index,autopct='%.0f',pctdistance=.8)
    plt.title('Topic Share %');

    new_topic_words = pd.concat([sent_topics_sorteddf.groupby('Topic_Num').head()[['Keywords']],
           topic_contribution.sort_index(),
          pd.Series(['Economy','Immigration','Environment','Event',
          'Civil Rights','Civil Rights','Healthcare',
          'Defense','Trump','Community','Event','Event',
          'Thanks','Legislation','Trump','Community',
          'Community','Trump','Defense',
          'Legislation','Thanks','Economy','Thanks','Healthcare',
          'Legislation'])],axis=1).groupby(0)['Keywords'].sum()
    [print(f'{topic}: ' + words) for topic,words in zip(new_topic_words.index,new_topic_words)]

def format_topics_sentences(ldamodel, corpus, texts):
    """ Discover the dominant topics for each document in a corpus, code sourced from the 
    excellent tutorial at Machine Learning Plus found at the link below.
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    """

    # Init output
    sent_topics_df = pd.DataFrame()
    start = time.time()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
        if i%10000==0:
            print(f'{i} rows done in {time.time()-start} seconds.')
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def topic_wordcloud(top_model):
    """ Produce a wordcloud for each topic in the model. """

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=20,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = top_model.show_topics(formatted=False)

    fig, axes = plt.subplots(3, 3, figsize=(10,10), sharex=True, sharey=True)

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

def topic_sne(top_model,sample_corpus):
    """ Produce a visualization of the seperation of the topics and documents in the corpus. """

    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(top_model[sample_corpus]):
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
    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = 25
    mycolors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    show(plot)



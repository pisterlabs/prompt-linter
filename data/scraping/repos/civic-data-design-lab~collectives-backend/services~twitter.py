import sys
import os
from twython import Twython
from twython import TwythonRateLimitError
from twython import TwythonAuthError
import pandas as pd
import json
import random
import numpy as np
from pprint import pprint
from datetime import datetime
from datetime import timedelta
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation, NMF
from apscheduler.schedulers.background import BackgroundScheduler
from Server.logger import logger
# Add parent directory to system path to be able to resolve Server folder
sys.path.append('../')
from Server import database

# Constants to be used throughout the file
SCHEDULE_AFTER = 1  # In minutes
TWITTER_NAME = 'mitsap'  # Twitter user whose followers are to be queried
FOLLOWERS_COUNT = 190  # Number of followers to be queried
BATCH_COUNT = 100  # Number of followers to be queried in one batch
TWEETS_COUNT = 500  # Maximum number of tweets to be scrapped from one follower
RATE_LIMIT_ERROR_CODE = 429  # Error code when twython hits limit
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)


# Load credentials from json file
with open(PROJECT_DIR+"/twitter_credentials.json", "r") as file:
    creds = json.load(file)

# Instantiate a twython object with authentication token
twitter = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
print('loading followers')

scheduler = BackgroundScheduler()
followers_list = []
previous_followers = []


def time_to_schedule ():
    '''
    Helper function to get the time when the next batch of scrapping is to be done.
    :return: time after 15 minutes from now
    '''
    return datetime.now() + timedelta(minutes=SCHEDULE_AFTER)


def get_new_tweets ():
    global followers_list
    global previous_followers
    try:
        followers_list = twitter.get(endpoint='https://api.twitter.com/1.1/followers/ids.json',
                                        params={'screen_name': TWITTER_NAME, 'count': FOLLOWERS_COUNT})['ids']
    except (TwythonRateLimitError, TwythonAuthError) as error:
        logger.info('Error while getting the user timeline', error.msg)

    previous_followers = database.get_followers()
    scrap_tweets(0)


def scrap_tweets (start_index=0):
    end_index = start_index + BATCH_COUNT
    print('starting from', start_index, 'at time ', datetime.now())
    print('loading tweets...')
    for index, follower in enumerate(followers_list[start_index:end_index]):
        if index % 10 == 0:
            print('follower number ', start_index+index)
        try:
            if follower in previous_followers:
                last_tweet = previous_followers[follower]
                timeline = twitter.get(endpoint='https://api.twitter.com/1.1/statuses/user_timeline.json',
                                       params={'user_id': follower, 'count': TWEETS_COUNT, 'since_id': last_tweet,
                                               'exclude_replies': True, 'include_rts': False})
                # print(timeline)
                database.append_raw_tweet(follower, timeline)
            else:
                timeline = twitter.get(endpoint='https://api.twitter.com/1.1/statuses/user_timeline.json',
                                       params={'user_id': follower, 'count': TWEETS_COUNT, 'exclude_replies': True,
                                               'include_rts': False})
                # print(timeline)
                database.add_raw_tweet(follower, timeline)
        except (TwythonAuthError, TwythonRateLimitError) as e:
            logger.info('Error while getting the user timeline' + e.msg)
            if e.error_code == RATE_LIMIT_ERROR_CODE:
                end_index = start_index + index
                break
        except Exception as e:
            logger.info('Error while getting the user timeline' + str(e))

    print('done upto', end_index)
    if end_index < FOLLOWERS_COUNT:
        print('scheduling next batch of tweets scrapping')
        print(scheduler.get_jobs())
        scheduler.add_job(scrap_tweets, 'date', args=[end_index], run_date=time_to_schedule())
        if not scheduler.running:
            scheduler.start()
    else:
        process_tweets()


def process_tweets ():
    tweets_df = None
    raw_data = database.get_collection('collectives')
    print('processing tweets')
    for index, follower in enumerate(raw_data[:200]):
        if index % 100 == 0:
            print('processing follower', index)
        if index == 0:
            tweets_df = pd.DataFrame(follower['tweets'])
            tweets_df['user'] = follower['follower']
        else:
            temp_df = pd.DataFrame(follower['tweets'])
            temp_df['user'] = follower['follower']
            tweets_df = tweets_df.append(temp_df)
    return
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    doc_term_matrix = count_vect.fit_transform(tweets_df['text'].values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=10, random_state=42)
    LDA.fit(doc_term_matrix)

    # Print 10 random topics
    for i in range(10):
        random_id = random.randint(0, len(count_vect.get_feature_names()))
        print(count_vect.get_feature_names()[random_id])

    first_topic = LDA.components_[0]
    top_topic_words = first_topic.argsort()[-10:]  # Get the indices of the 10 most frequent words in the first topic
    # Print the top 10 words in the first topic
    for i in top_topic_words:
        print(count_vect.get_feature_names()[i])

    # Print the top 10 words for all the topics generated by CountVectorizer
    for i, topic in enumerate(LDA.components_):
        print(f'Top 10 words for topic #{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')

    topic_values = LDA.transform(doc_term_matrix)
    tweets_df['Topic'] = topic_values.argmax(axis=1)  # Assign each tweet to the topic which most likely represents it


    tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
    doc_term_matrix = tfidf_vect.fit_transform(tweets_df['text'].values.astype('U'))

    nmf = NMF(n_components=10, random_state=42)
    nmf.fit(doc_term_matrix )


    for i in range(10):
        random_id = random.randint(0,len(tfidf_vect.get_feature_names()))
        print(tfidf_vect.get_feature_names()[random_id])

    # In[222]:


    first_topic = nmf.components_[0]
    top_topic_words = first_topic.argsort()[-10:]

    # In[223]:


    for i in top_topic_words:
        print(tfidf_vect.get_feature_names()[i])

    # In[35]:


    topic_list = []

    for i,topic in enumerate(nmf.components_):
        topic_list.append([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])

    # In[36]:


    topic_list

    # In[19]:


    topic_values = nmf.transform(doc_term_matrix)
    tweets_df['Topic'] = topic_values.argmax(axis=1)
    tweets_df.head()

    # In[226]:


    tweets_df['user'].unique()

    # In[20]:


    topic_frequency = tweets_df.groupby(['user','Topic']).size().reset_index().rename(columns={0:'Topic Frequency'})

    # In[21]:


    topic_frequency.head()

    # In[22]:


    from bokeh.io import output_notebook
    from bokeh.plotting import show, figure
    from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, HoverTool
    from bokeh.palettes import Viridis256
    from bokeh.transform import transform
    # output_notebook()

    # In[23]:


    d = {'users': topic_frequency['user'].unique()
    }

    # In[28]:


    p = figure(x_range =[-2,11], y_range = list(topic_frequency['user']))

    p.scatter(x = list(topic_frequency['Topic']), y = list(topic_frequency['user']), size=list(topic_frequency['Topic Frequency']))

    show(p)


    # In[232]:


    import sys
    # !{sys.executable} -m spacy download en
    import re

    # Gensim
    import gensim, spacy, logging, warnings
    import gensim.corpora as corpora
    from gensim.utils import lemmatize, simple_preprocess
    from gensim.models import CoherenceModel
    import matplotlib.pyplot as plt

    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


    # In[233]:



    def sent_to_words(sentences):
        for sent in sentences:
            sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
            yield(sent)

    # Convert to list
    data = tweets_df['text'].values.tolist()
    data_words = list(sent_to_words(data))
    print(data_words[:1])

    # In[234]:


    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # !python3 -m spacy download en  # run in terminal once
    # or do
    # !conda install -c conda-forge spacy-model-en_core_web_md
    # and use nlp=spacy.load('en_core_web_sm') instead in below function.
    def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
        texts_out = []
        nlp = spacy.load('en', disable=['parser', 'ner'])
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
        return texts_out

    data_ready = process_words(data_words)  # processed Text Data!

    # In[251]:


    len(data_ready)

    # In[235]:


    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=4,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=10,
                                               passes=10,
                                               alpha='symmetric',
                                               iterations=100,
                                               per_word_topics=True)

    pprint(lda_model.print_topics())

    # In[236]:


    import seaborn as sns
    import matplotlib.colors as mcolors

    # In[260]:


    len(arr)

    # In[264]:


    from bokeh.plotting import figure, show, output_file
    from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label, WheelZoomTool


    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # In[265]:


    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # In[267]:


    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = 4

    # In[273]:


    tsne_lda

    # In[270]:


    colormap = {0: '#440154', 1: '#39568C', 2: '#1F968B', 3: '#73D055'}

    # colormap = {0: '#CF3E45', 1: '#FF8F94', 2: '#F4636A', 3: '#FFB3B3'}
    # colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}

    colors = [colormap[x] for x in list(topic_num)]

    source = ColumnDataSource(data=dict(x=tsne_lda[:,0],
                                        y=tsne_lda[:,1],
                                        tweets=list(tweets_df['text']),
                                        topic_num=topic_num,
                                        color=colors,
                                        topic=topic_num

                                       ))

    TOOLTIPS = [
        ("Topic", "@topic_num"),
        ("Tweet", "@tweets")
    ]

    TOOLTIPS = """
        <div>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@topic_num</span><br>
            </div>
            <div style="width:150px">
                <span style="font-size: 15px;">Tweet</span><br>
                <span style="font-size: 10px;">@tweets</span>
            </div>
        </div>
    """


    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=1900, plot_height=900, background_fill_color = "black", tooltips=TOOLTIPS, active_scroll = "auto")
    plot.circle(x='x', y='y', source=source, color='color', radius=0.25)
    plot.axis.visible = False
    plot.grid.visible = False
    plot.add_tools(WheelZoomTool())
    # labels = LabelSet(x='x', y='y', text='tweets', source=source, x_offset=5, y_offset=5)

    # plot.add_layout(labels)
    show(plot)


if __name__ == "__main__":
    get_new_tweets()


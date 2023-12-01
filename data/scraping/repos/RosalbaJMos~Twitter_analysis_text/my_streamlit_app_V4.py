import base64
import io
import logging
import re
import warnings
import webbrowser
from pprint import pprint
from datetime import datetime

# Gensim
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt

# NLTK Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import plotly.express as px
import spacy
import streamlit as st
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#  Bar_chart race
import bar_chart_race as bcr


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])



# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


st.image('Twitter_Bird.png', width=100)
st.title("Analysis of tweets")

url = 'https://github.com/RosalbaJMos/Twitter_analysis_text'

if st.button('My GitHub'):
    webbrowser.open_new_tab(url)


with st.sidebar.header('2. Set Parameters'):
    n_samples=st.sidebar.slider('No. samples', 500, 1000, 2000)
    n_features=st.sidebar.slider('No. features', 500, 1000)
    n_components = st.sidebar.slider('No. Topics', 4, 8, 12)
    n_top_words = st.sidebar.slider('No. Tokes per topic', 2, 5, 10)

####################
# ALL MY FUNCTIONS
####################
# 1. Funtions to Download CSV data
def filedownload(df, filename):
    csv = pd.DataFrame(df).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

# 2. Function to create a twitter Wordcloud
def Titter_Wordclouds(X):
    from PIL import Image
    from wordcloud import WordCloud
    mask = np.array(Image.open('Twitter_mask.png'))
    font_path = "PICOWA__.TTF" 
    
    wordcloud = WordCloud(
                      stopwords=stopwords,
                      background_color='white',
                      mask=mask, font_path=font_path, colormap = 'tab20').generate_from_frequencies(X.T.sum(axis=1))
    plt.figure(figsize = (12, 12), facecolor = None) 
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'twitter_comments.pdf'), unsafe_allow_html=True)

# funtion to display the list (top 30) of the most re-tweeted users
def top_cited_users(Y):
    top_users = df.groupby('username').count().reset_index()\
                .sort_values('text', ascending=False)[['username', 'text']]\
                .rename(columns={"text": "count"}).set_index('username')
    return top_users

# 3. Function to create a race char
def bar_race():
    df['date'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
    dfp = df.pivot_table(values = 'retweet_count', index = 'date', columns = 'username', aggfunc = 'sum')
    dfp.fillna(0, inplace = True)
    dfp.sort_values(list(dfp.columns), inplace = True)
    dfp.sort_index(inplace = True)
    dfp.iloc[:,:] = dfp.iloc[:,:].cumsum()
    top = set()
    for index, row in dfp.iterrows():
        # for each row (year)
        top |= set(row[row > 0].sort_values(ascending=False).head(10).index) # top 10 fighter each 
    
    bcr.bar_chart_race(df = dfp,n_bars = 8, title = "Retweet count", sort = 'asc',
                    orientation = 'h', filename = 'retweets_video.mp4', period_length = 1000,
                    steps_per_period = 20,  filter_column_colors = True, cmap = ('dark24'), 
                    period_label={'x': .99, 'y': .1, 'ha':'right','size':20}, dpi = 480, figsize=(5,5))

# 4. Functions to clean:
def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent) 

# 5. Function to tokenize and do the lematization :
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts_out = []
    nlp = spacy.load("en_core_web_sm")
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

# 6. Function to plot topics
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 40})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=30)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    st.pyplot(fig)
    st.markdown(imagedownload(plt,'plot_top_words_per_topic.pdf'), unsafe_allow_html=True)


# MY MAIN FUNCTION 
def build_model(df):
    df = df.loc[:] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    
    st.markdown('***1.2. The number of tweets in your data set : ***')
    st.info(len(df.index))

    st.markdown('**1.3. This is the list of the top 10 users**')
    top_cited_users(df)
    #st.write(top_cited_users(df).head(10))
    st.table(top_cited_users(df).head(10))

    ### Show the video of retweets 
    st.markdown('**1.4. This plot shows the most retweeted users in your data**')
    bar_race()
    video_file = open('retweets_video.mp4', 'rb') #enter the filename with filepath
    video_bytes = video_file.read() #reading the file
    st.video(video_bytes) #displaying the video

    ### clean, tokenize and do the lematization
    data = df.text.values.tolist()
    data_words = list(sent_to_words(data))
    data_ready = process_words(data_words)

    corpus = [' '.join(t) for t in data_ready]

    #---------------------------CREATE TOPICS USING LDA USING TFIDF ---------------------------------------------------------------
    from time import time

    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import (CountVectorizer,
                                                 TfidfVectorizer)

    tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b',) 
    dtm_tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    dense=dtm_tfidf.todense()
    lst1 = dense.tolist()
    df2 = pd.DataFrame(lst1, columns=tfidf_feature_names)
    
    st.markdown('**1.5. The most used words in your dataset are displayed here**')
    Titter_Wordclouds(df2)

    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method="online", learning_offset=50.0,
    random_state=0,) 
    lda.fit(dtm_tfidf)
    
    st.markdown('**1.6. These are most used words per topic **')
    plot_top_words(lda, tfidf_feature_names, n_top_words, 'Top words per topic')


#--------------------------------------------LDA CORPORA DICTIONARY MODEL---------------------------------------------
    ## Build the topic model 
    #id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    #corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    #lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, random_state=100, 
    #                                        update_every=1, chunksize=10, passes=10, alpha='symmetric', 
    #                                        iterations=100, per_word_topics=True)

    #st.write(lda_model.print_topics())

#----------------------------------------------------------------------------------------------------------------

######################
# CALL MY MODELS
######################
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(10))
    build_model(df)

else:        
    st.markdown('***This is an example of an analysis done for a data set about tweets on "chemistry". If you wish to analyse your own data set, please upload it***')
    df = pd.read_csv('tweets_about_chemistry_vsm.csv')
    st.write(df.head(10))
    build_model(df)







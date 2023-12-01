#streamlit run "C:\Users\yeosi\Documents\Python MAGES\07 DS106\Capstone\Streamlit\streamlit_app_capstone.py"

# Run the relevant libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

sns.set_style('whitegrid')

import re
import unicodedata

import contractions
import gensim
import gensim.downloader as api
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvi
from cleantext import clean
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.keyedvectors import KeyedVectors
from gensim.summarization import keywords
from gensim.test.utils import common_texts
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from textblob import TextBlob
from wordcloud import WordCloud


# Loading list of stop words from NLTK
stop_words = set(stopwords.words('english'))

# Remove word 'not' in stopwords as Not can depict emotions
stop_words.remove('not')

# Create function to define and remove custom stopwords 
def remove_custom_stopwords(text):
    stopset = set(stopwords.words("english"))
    for word in ['via', 'etc','very', 'hotel', 'room', 'stay', 'square', 'would', 'also', 'singapore', 'clarke', 'quay','ibis', 'ruby', 'sapphire','fragrance']:
        stopset.add(word)
  
    text = ' '.join([word for word in text.split() if word not in stopset])
    return text


# Create a function to map Part of Speech tags to the tokens
def pos_tag_wordnet(tagged_tokens):
    tag_map = {'j': wordnet.ADJ, 'v': wordnet.VERB, 'n': wordnet.NOUN, 'r': wordnet.ADV}
    new_tagged_tokens = [(word, tag_map.get(tag[0].lower(), wordnet.NOUN))
                            for word, tag in tagged_tokens]
    return new_tagged_tokens

# Create a function to process and clean texts 
def text_processing(review):
    
    #tokenize
    text = nltk.word_tokenize(review)
    
    #conver to lowercase 
    text = [t.lower() for t in text]

    # Remove all emoticons
    text = clean(text, no_emoji=True)
    
    # remove symbol
    text = [re.sub(r'^\d\w\s','',t) for t in text]
    
    #remove brackets
    text = [t.replace("(","").replace(")","") for t in text]
     
    # remove punctuation
    text = [t for t in text if t.isalnum()]
    
    # remove stopwords
    text  = [t for t in text if t not in  stop_words]
    
    #remove contractions
    text = [contractions.fix(t) for t in text]
    
    #Remove numbers
    text = [re.sub('\w*\d\w*', '', t) for t in text]
    
    #Remove ascii char
    text = [unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8', 'ignore') for t in text]
     
    #Remove empty tokens
    text = [t for t in text if t]
        
    #POS tagging
    text = nltk.pos_tag(text)
    text = pos_tag_wordnet(text)
    
    # Lemmatization
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(t,tag) for t,tag in text]
    
    #Join tokens
    text = ' '.join(text)
    
    return text


topic_mapping = {0:'Location', 1: 'Service', 2: 'Environment', 3: 'Room Amenities'}

# Create wordclouds   
def wordcloud(review_df, color, title):
    '''    
    INPUTS:
        reivew_df - dataframe, positive or negative reviews
        review_colname - column name, positive or negative review
        color - background color of worldcloud
        title - title of the wordcloud
    OUTPUT:
    Wordcloud visuazliation
    '''  
    text = review_df['Review'].to_string()
    #text_str = ' '.join(lemmatized_tokens(' '.join(text))) #call function "lemmatized_tokens"
    #text = review_df
    stopwords = ['fragrance']
    wordcloud = WordCloud(collocations = False,
                          background_color = color,
                          width=1600, 
                          height=800, 
                          margin=2,
                          min_font_size=20, stopwords= stopwords).generate(text)

    plt.figure(figsize = (10, 8))
    plt.imshow(wordcloud)
    #plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.figtext(.5,.8,title,fontsize = 20, ha='center')
    plt.show()
    st.pyplot()

# Create a function to analysis sentiments of reviews using TextBlob
def sentiment_analysis(df):
  
 #Create two new columns ‘Polarity’)
    df['TextBlob_Polarity'] = df['Review'].apply(lambda x:TextBlob(x).sentiment.polarity )
    
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score >= 0.15:
            return 'Positive'
        else:
            return 'Neutral'
     
    df['Sentiments'] = df['TextBlob_Polarity'].apply(getAnalysis )

    return  df

# Create a function to plot the countplots for each topic
def topic_sentiments(df, title):
    plt.figure(figsize = (12, 6))
    sns.countplot( x = 'Hotel_Name', data = df, hue = 'Sentiments', palette = 'Set2')
    plt.xticks(rotation=90, fontsize = 12)
    plt.title(title, fontsize = 18)
    plt.legend(loc = 2)

# Create a function for the computational part of the model
#@st.experimental_memo (suppress_st_warning = True)
def compute (df):
    df['Cleaned_Review'] = df['Review'].apply(text_processing)
    df['Cleaned_Review'] = df['Review'].apply(remove_custom_stopwords)
    df['Topic'] = model.predict(df['Cleaned_Review'])
    df['Topic'] = df['Topic'].map(topic_mapping)
    sentiment_analysis(df)
    return df

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

#Load model
#model = pickle.load(open(r'C:\Users\yeosi\Documents\Python MAGES\07 DS106\Capstone\Streamlit\lsvc_pipe_model.pkl','rb'))
model = pickle.load(open('lsvc_pipe_model.pkl','rb'))


def main():
    global df
    global room_amen
    #if 'df' not in st.session_state:
       # st.session_state.df = df

    # state session to ensure that if the user reloads the page, the data is still there
    if "load_state" not in st.session_state:
        st.session_state.load_state = False 

    # To turn off warning from st.pyplot()   
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #image = Image.open('Logo_IBISBudget_CMJN.png')
    #st.image(image)
    #st.image(r'C:\Users\yeosi\Documents\Python MAGES\07 DS106\Capstone\Logo_IBISBudget_CMJN')

    # Create user interface on Streamlit
    st.title('ibis Budget Singapore')
    st.subheader('Topic Classification and Sentiment Analysis of Hotel Reviews')
    st.write('This app can be used to process hotel reviews and assign them to the most appropriate topic using supervised machine learning model. Sentiment analysis will then be done for the reviews to determine the overall sentiment of each topic. You may access the sidebar to upload a .CSV file containing hotel reviews and run the app')

    # To upload CSV
    with st.sidebar:
        uploaded_file = st.file_uploader('Choose a file in .csv format', type=['csv'])
        
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
        read = st.checkbox('Show dataframe')
       
        if read:
            st.write('**The dataframe has been uploaded. The following are the columns:**')     
            if uploaded_file is not None:
                st.dataframe(df)
            else:
                st.write('No dataframe uploaded')

        st.write('You may ignore the following warning and proceed to run the model.')

    # To Pre-process text and run model with sentiment analysis
        with st.sidebar:
            if st.button('Run Topic Allocation and Sentiment Analysis', key = '1') or st.session_state.load_state:
                st.session_state.load_state = True 
                compute(df)

        # Create data subsets for each topic 
            room_amen = df.loc[df['Topic']== 'Room Amenities', (['Hotel_Name', 'Review', 'Sentiments'])]
            service = df.loc[df['Topic']== 'Service', (['Hotel_Name','Review', 'Sentiments'])]
            location = df.loc[df['Topic']== 'Location', (['Hotel_Name','Review', 'Sentiments'])]
            environment = df.loc[df['Topic']== 'Environment', (['Hotel_Name', 'Review', 'Sentiments'])]
    
    # Show results for topic allocation and sentiment analysis
        with st.expander("Results"):
            st.dataframe(df)

    # Allow choice to download csv
        download = st.download_button(label="Download results as CSV", data= convert_df(df), file_name = 'ibis_budget_results.csv')

    # Show metrics for sentiments 
        st.write('Sentiment Metrics')
        col1, col2, col3 = st.columns(3)
        positive = len(df.loc[df['Sentiments']== 'Positive'])
        negative = len(df.loc[df['Sentiments']== 'Negative'])
        neutral = len(df.loc[df['Sentiments']== 'Neutral'])
        col1.metric("Positive", positive)
        col2.metric("Neutral", neutral)
        col3.metric("Negative", negative)

    # Show overall numbe rof reviews by hotel branch
        st.write('Overview of Reviews by Hotel Branch')
        plt.figure(figsize = (12, 6))
        df.groupby('Hotel_Name')['Review'].count().sort_values(ascending = False).plot(kind = 'bar')
        plt.xticks(rotation=90, fontsize = 12)
        plt.title('No. of Reviews across Hotel Branches', fontsize = 18)
        st.pyplot()

    # Show overall sentiments across topics
        st.write('Sentiments across Topics')
        plt.figure(figsize = (8, 4))
        sns.countplot( x = 'Topic', data = df, hue = 'Sentiments', palette = 'Set2' ).set_title('Sentiments across Topics', fontsize = 15)
        st.pyplot()

    #Create Wordclouds
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with st.expander("See Wordclouds"):
            st.write(wordcloud (room_amen, 'white', 'Room Amenities'))
            st.write(wordcloud (service, 'white', 'Service'))
            st.write(wordcloud (environment, 'white', 'Environment'))
            st.write(wordcloud (location, 'white', 'Location'))
    
    # View sentiments across hotel branches
        with st.expander("View Sentiments across Hotel Branches"):
            plt.figure(figsize = (15, 6))
            sns.countplot( x = 'Hotel_Name', data = df, hue = 'Sentiments', palette = 'Set2')
            plt.xticks(rotation=90, fontsize = 12)
            plt.title('Sentiments across Hotel Branches', fontsize = 18)
            plt.legend(loc = 2)
            st.pyplot()

    # View topics across hotel branches
        with st.expander("View Topics across Hotel Branches"):
            plt.figure(figsize = (15, 6))
            sns.countplot( x = 'Hotel_Name', data = df, hue = 'Topic', palette = 'Set2')
            plt.xticks(rotation=90, fontsize = 12)
            plt.title('Topics across Hotel Branches', fontsize = 18)
            plt.legend(loc = 2)
            st.pyplot()

        if 'df' not in st.session_state:
            st.session_state.df = df 

    # Create radio buttons for topic selection
        with st.sidebar:
            option = st.radio('Select Topic',('Room Amenities', 'Service', 'Environment', 'Location'))
            #if 'df' not in st.session_state:
               # st.session_state.df = df 

    # Show sentiment analysis for selected topic
        st.write('Sentiments for the Selected Topic across Hotel Branches')
        if option == 'Room Amenities':
            st.pyplot(topic_sentiments(room_amen, 'Room Amenities'))
        elif option == 'Service':
            st.pyplot(topic_sentiments(service, 'Service'))
        elif option == 'Environment':
            st.pyplot(topic_sentiments(environment, 'Environment'))
        else: 
            st.pyplot(topic_sentiments(location, 'Location'))
    
    #View reviews for the selected topic
        with st.sidebar:
            box = st.checkbox('Show Reviews for Selected Topic')
        
        if box:
            st.write('Reviews for the Selected Topic')
            if option == 'Room Amenities':
                st.write(room_amen)
            elif option == 'Service':
                st.write(service)
            elif option == 'Environment':
                st.write(environment)
            else: 
                st.write(location)
        
        Review = st.sidebar.select_slider('View Reviews for Sentiment Groups for Selected Topic', options=['Positive', 'Neutral', 'Negative'])
        if Review:
            st.write('Reviews for the Selected Sentiment Group for Selected Topic:')
            st.write('You may hover over the text to see the full review.')
       
        if option == 'Room Amenities' and Review == 'Positive':
            st.write(room_amen.loc[room_amen['Sentiments']== 'Positive'])
        elif option == 'Room Amenities' and Review == 'Neutral':
            st.write(room_amen.loc[room_amen['Sentiments']== 'Neutral'])
        elif option == 'Room Amenities' and Review == 'Negative':
            st.write(room_amen.loc[room_amen['Sentiments']== 'Negative'])
        elif option == 'Service' and Review == 'Positive':
            st.write(service.loc[service['Sentiments']== 'Positive'])
        elif option == 'Service' and Review == 'Neutral':
            st.write(service.loc[service['Sentiments']== 'Neutral'])
        elif option == 'Service' and Review == 'Negative':
            st.write(service.loc[service['Sentiments']== 'Negative'])
        elif option == 'Location' and Review == 'Positive':
            st.write(location.loc[location['Sentiments']== 'Positive'])
        elif option == 'Location' and Review == 'Neutral':
            st.write(location.loc[location['Sentiments']== 'Neutral'])
        elif option == 'Location' and Review == 'Negative':
            st.write(location.loc[location['Sentiments']== 'Negative'])
        elif option == 'Environment' and Review == 'Positive':
            st.write(value.loc[value['Sentiments']== 'Positive'])
        elif option == 'Environment' and Review == 'Neutral':
            st.write(value.loc[value['Sentiments']== 'Neutral'])
        elif option == 'Environment' and Review == 'Negative':
            st.write(value.loc[value['Sentiments']== 'Negative'])
    
    else:
        st.write('No Data Found. Pls upload a valid file in the sidebar.')

if __name__=='__main__':
    main()

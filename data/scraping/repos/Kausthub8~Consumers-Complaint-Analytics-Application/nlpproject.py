import streamlit as st
import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image

# Set the title of the app
st.title('Huge Text Analyzer using Topic Modelling techniques')

# Create a file input widget to accept the text file
file = st.file_uploader('Upload a text file', type='csv')

# Define a function to preprocess the text and train the NMF model
def train_models(num_topics,model_type):
    df = pd.read_csv(file)
    def lemma_stem(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    def preprocess(text):
        result=[]
        for token in gensim.utils.simple_preprocess(text) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemma_stem(token))
        return result
    stemmer = SnowballStemmer("english")
    processed_docs = []
    for doc in df['Customer Complaint'].fillna('').astype(str):
        processed_docs.append(preprocess(doc))
    dictionary = gensim.corpora.Dictionary(processed_docs)
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
    # We now filter words that occur less than 5 times and those occurring more than half the time.
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if model_type == "LDA":
        model = gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    elif model_type == "NMF":
        model =  gensim.models.Nmf(bow_corpus,num_topics = num_topics,id2word = dictionary,passes = 10)
    else:
        model = gensim.models.LsiModel(bow_corpus, num_topics=num_topics, id2word=dictionary)
    
    # Calculate coherence score
    coherence_model = CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    stats = [coherence_score]
    return model.show_topics(num_topics=num_topics),stats
def team_details():
    # Define team member information
    team_members = [
        {
            "name": "Gowtham Thatavarthi",
            "bio": "A software techie..."
        }
        # Add more team members here...
    ]

    # Define function to display team member information
    def display_team_member_info(member):
        st.sidebar.write(f"## {member['name']}")
        st.sidebar.write(member['bio'])

    # Define sidebar contents
    st.sidebar.title("About The Team")

    # Display selected team member's information
    for member in team_members:
        display_team_member_info(member)
# Define the main function to run the app
def main():
    team_details()
    st.warning('Please make sure the data file has Customer Complaint column with complaints')
        # Check if a file has been uploaded
    if file is not None:
        num_topics = st.slider('Select the number of topics', min_value=2, max_value=16, step=1, value=6)    
        # Train the NMF model
        # Add a button to start training the model
        model_type = st.selectbox("Select Model Type", ["LDA", "NMF", "LSA"])
        if st.button("Train "+ model_type  +" model"):
            # Display a loading spinner while the model is being trained
            with st.spinner("Training the models..."):
                topics,stats = train_models(num_topics,model_type)
                st.subheader(model_type + ' Model Stats')
                st.metric(label="Coherence Score of the Model: ", value=stats[0], delta="Higher Value is better")
                stopwords = set(STOPWORDS)
                for tup in topics:
                    string = tup[1].split('"')[1::2]
                    string = ' '.join(string)
                    # Generate a wordcloud object
                    wordcloud = WordCloud(
                        background_color='white',
                        stopwords=stopwords,
                        max_words=200,
                        max_font_size=40, 
                        scale=3,
                        random_state=42
                    ).generate(string)
                    # Display the wordcloud using matplotlib
                    fig, ax = plt.subplots()
                    plt.title('Word Cloud for Issue ' +str(tup[0]+1))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)       
    else:
        st.write('Please upload a text file.')
    

# Run the app
if __name__ == '__main__':
    main()

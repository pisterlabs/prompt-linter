import streamlit as st
import cohere
from dotenv import load_dotenv
import os
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from scrape_functions import scrape_URL
from collections import Counter
import math
load_dotenv()

# initialise the Co:here client
co = cohere.Client('') ## set your api key here
# start the web driver
driver = webdriver.Chrome(ChromeDriverManager().install())
# read in the stories with tags and urls
df = pd.read_csv('reedsy.csv')

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = 'Output:'
    
# Predict tags for the input url
def predict_tags(index, df, scraped_url):
    number_chars = 1500
    response = co.generate(
    model='large',
    prompt=f"Passage: {df['Body'].values[0][:number_chars]}\n\nTags:{df['Tags'][0]}\n--\nPassage:{df['Body'].values[1][:number_chars]}\n\nTLDR:{df['Tags'][1]}\n--\nPassage:{df['Body'].values[2][:number_chars]}\n\nTLDR:{df['Tags'][2]}\n--\nPassage:{scraped_url['Body'].values[0][:number_chars]}\n\Tags:",
    max_tokens=50,
    temperature=0.8,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')
    return response.generations[0].text
    
# Cosine similarity formula
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

# Calculate cosine similarity scores
def calculate_similarirties(scraped_url, df):
    counterA = Counter(scraped_url['Tags'][0])
    for i in range(1, df.shape[0]):
        counterB = Counter(df['Tags'][i])
        df.at[i, 'cosine_similarity'] = counter_cosine_similarity(counterA, counterB) * 100

# Get top 3 urls by similarity score
def get_top_3(df):
    list_of_highest = df['cosine_similarity'].nlargest(3).index
    list_of_urls = df['URL'][list_of_highest].values
    list_of_similar = df['cosine_similarity'][list_of_highest].values
    list_of_recs = []
    for url, similaity in zip(list_of_urls, list_of_similar):
        list_of_recs.append(f'{url} : {similaity}')
    return list_of_recs

# Main function
def do_all(input_url):
    # scrape the text from the input url
    scraped_url = scrape_URL(input_url, driver)
    # predict tags of input (pass df to predict_tags)
    scraped_url.at[0, 'Tags'] = predict_tags(0, df, scraped_url)
    # calculate similarity scores of input to all other posts
    calculate_similarirties(scraped_url, df)
    # return top 3 urls and similarity scores
    list_of_recs = get_top_3(df)
    st.session_state['output'] = list_of_recs ## this is the output
    st.balloons()

st.title('Better Tags recommender')
st.subheader('Get better recommendations for lovers of short stories!')
st.write('''This **Streamlit** app takes a url to a short storty on reedsy and returns suggested recommendations for similar reads!''')

input = st.text_area('Enter your reedsy url here', height=100)
st.button('Generate recommendations', on_click = do_all(input))
st.write(st.session_state.output)
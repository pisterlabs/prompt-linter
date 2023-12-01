import streamlit as st
import openai
import os
import re
import requests
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Reading the prompt from a text file
with open('long_prompt.txt', 'r') as f:
    long_prompt = f.read().strip()

# Initialize OpenAI API
# Note: Use environment variables for API keys for security reasons.
openai.api_key = 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


def summarize_article_with_openai(url):
    # Scrape the article
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.text for p in paragraphs])

    # Extract most relevant sentences using Sentence Transformers
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = article_text.split('. ')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    query_embedding = model.encode("Summary", convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=10)

    summarized_text = ' '.join([sentences[i] for i in top_results.indices])


    # Refine summary using OpenAI GPT
    prompt = f"{long_prompt} {summarized_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=4000
    )
   
    return response['choices'][0]['message']['content']

def score_article(arguments):
    prompt = f"read this sumamry of an article and score it 1-100 in political neutrality: {arguments} (return only the number)"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=4000
    )
   
    return response['choices'][0]['message']['content']

# Title with large font and style
st.markdown("<h1 style='text-align: center; color: blue;'>🔬 Thoughts Laboratory 🔬</h1>", unsafe_allow_html=True)

# Dynamic Textbox for Surveyor
statement = st.text_input('Enter your URL:', '')


# Generate arguments and chart only when a statement is entered
if statement:
    arguments = summarize_article_with_openai(statement)  # Assuming you have this function defined
    
    try:
        score = float(score_article(arguments))*0.7
    except:
        score = 0.0

    # Display the gauge
    st.write(f'Neutrality Score: {score}%')
    
    # Color code
    if score < 33:
        color = 'red'  # Biased
    elif score < 66:
        color = 'yellow'  # Somewhat neutral
    else:
        color = 'green'  # Neutral

    gauge_html = f'''
    <div style="width:100%; background-color:lightgray;">
        <div style="width:{score}%; background-color:{color}; text-align:center;">
            {score}%
        </div>
    </div>
    '''
    st.markdown(gauge_html, unsafe_allow_html=True)
    
    st.title("Summary:")   
    st.write(arguments)

import streamlit as st
import pandas as pd
import openai
import numpy as np
import faiss
import re
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from dotenv import load_dotenv
import os


load_dotenv()

openai_api_key = os.getenv("openai")
openai.api_key = openai_api_key
DATA = [
    {
        "id": 1,
        "Name": "2 Hour Open Bar LDW Party & Games",
        "Description": "A 2 Hour Open Bar Labor Day Weekend Party at SPiN, Flatiron (a $55 value)",
        "Price": "29.0"
    },
    {
        "id": 2,
        "Name": "The New York City Wine Festival",
        "Description": "2023 New York City Wine Festival: 3.5 Hours To Enjoy 100+ Wines (a $70 Value)",
        "Price": "29.0"
    },
    {
        "id": 3,
        "Name": "3 Treatment Spa Day",
        "Description": "$33 For An 60-Minute Swedish Massage or Deep Tissue Massage & $59 For A 3 Treatment Spa Day at Taiji Jubest Spa",
        "Price": "33.0"
    },
    {
        "id": 4,
        "Name": "J.Crew 40th Anniversary Celebration",
        "Description": "Celebrate 40 Years of J.Crew at The Seaport",
        "Price": "N/A"
    },
    {
        "id": 5,
        "Name": "Barbara G. Mensch Photo Book Launch",
        "Description": "Photography Book Launch: \"A Falling-Off Place\" by Barbara G. Mensch",
        "Price": "N/A"
    },
    {
        "id": 6,
        "Name": "El Coco's Seasonal Opening Event",
        "Description": "Party with El Coco in Their New Event Space and Enjoy Free Drinks and Bites",
        "Price": "N/A"
    },
    {
        "id": 7,
        "Name": "Adult Coloring",
        "Description": "Relax, Color, and Make New Friends",
        "Price": "N/A"
    },
    {
        "id": 8,
        "Name": "Eataly Flatiron's 13th Birthday",
        "Description": "Celebrate 13 Years of Eataly Flatiron with Great Food, Live Music, and More",
        "Price": "N/A"
    },
    {
        "id": 9,
        "Name": "Open Bar Bubbly Brunch Sail",
        "Description": "$93 Ticket To An Open Bar Bubbly Brunch Sail Around NYC Aboard The Shearwater (a $124 Value)",
        "Price": "93.0"
    },
    {
        "id": 10,
        "Name": "Thursday Night Wine Tasting",
        "Description": "Start Your Weekend Early At Bottlerocket Wine & Spirit's Thursday Tasting",
        "Price": "N/A"
    },
    {
        "id": 11,
        "Name": "Cetaphil x Think Coffee",
        "Description": "Stop by Cetaphil's Healthy Renew Launch Event with Think Coffee",
        "Price": "N/A"
    },
    {
        "id": 12,
        "Name": "Cookbook Launch with James Park",
        "Description": "Catch James Park Launching \"Chili Crisp\" at Books Are Magic with Irene Yoo",
        "Price": "N/A"
    },
    {
        "id": 13,
        "Name": "Priv\u00e9 Sample Sale",
        "Description": "Shop Staple Men's and Women's Summer Pieces at the Priv\u00e9 Sample Sale",
        "Price": "N/A"
    },
    {
        "id": 14,
        "Name": "Brooklyn Unlimited Oyster Festival",
        "Description": "$45 Ticket To The Brooklyn Oyster Festival: Unlimited Oysters + 1 Beer At BK Backyard Bar (a $56 value)",
        "Price": "45.0"
    }
]

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    
    return text

# Define a function to get embeddings
def get_embeddings(texts):
    # Preprocess the texts
    texts = [preprocess(text) for text in texts]
    # Get the embeddings from OpenAI
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

# Define a function to get the most similar events to a given query
def getResponse(question):
    # Define the number of similar events to retrieve
    k = 10
    
    # Load the data
    data = pd.DataFrame(DATA)
    data = data[["Name", "Description"]]
    data = data.dropna()
    data = data.reset_index(drop=True)
    
    # Get the embeddings for the names and descriptions
    name_embeddings = get_embeddings(data["Name"].tolist())
    description_embeddings = get_embeddings(data["Description"].tolist())
    
    # Concatenate the name and description embeddings
    concatenated_embeddings = np.concatenate((name_embeddings, description_embeddings), axis=1)
    
    # Index the concatenated embeddings using FAISS
    index = faiss.IndexFlatL2(concatenated_embeddings.shape[1])
    index.add(concatenated_embeddings)
    
    # Preprocess the question and get its embedding
    query_embedding = get_embeddings([question])
    # Duplicate the query_embedding to match the dimensionality of the concatenated embeddings
    query_embedding = np.repeat(query_embedding, 2, axis=1)
    
    # Search for the most similar events to the query
    D, I = index.search(query_embedding, k)
    
    print("Query:", question)
    print("Top 4 most similar events:")
    print(data.iloc[I[0]])


    content = "Question: " + question
    # content = ""
    content = content + "\n" + "Here are the top most similar events:"
    content = content + "\n" + data.iloc[I[0]].to_string()
    # print(content)

        
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [
            {"role": "system", "content": "You are a helpful assistant that filters events based on the question and summarizes them"},
            {"role": "user", "content": content}
        ]
    )
        
    # print(response)
        # print(cb)
    return response.choices[0].message.content



with st.sidebar:

    st.markdown("""
# MVP to show how chat can be implemented in PULSD app
### Not made for public use    
#### Here you can see 15 example events from PULSD app             
                """)
        
    st.json(DATA)


st.markdown("""
## Here you can chat about events from PULSD app               
                """)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("user"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = getResponse(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



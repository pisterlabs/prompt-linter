import os
import random

import openai
import pinecone
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from PIL import Image

default_msg = "You are a helpful assistant that suggests creative and unique product names, taglines, categories and domain names to help list on product listing websites like producthunt"
solution_types = ['Web App','iOS','SaaS','Android','Mac','Chrome Extensions','API','iPad','Windows','Wearables','Hardware','Apple','Browser Extensions']
domains= ['Marketing','User Experience','Messaging','Analytics','Education','Social Media','Growth Hacking','Fintech','Photography','Writing','Email','Task Management','Web3','Health & Fitness','Sales','E-Commerce','Social Network','Streaming Services','Hiring','Robots','Customer Communication','No-Code','Software Engineering','Travel','Newsletters','Email Marketing','Prototyping','Meetings','News','Music','Home','Investing','Search','Books','Calendar','Privacy','Payments','Global Nomad','Branding','SEO','Games','Notes','GitHub','Remote Work','Startup Books','Venture Capital','Sketch','Freelance','Art','Funny','Internet of Things','Maker Tools','Advertising','Notion','Icons','Spreadsheets','Augmented Reality','Crypto','Startup Lessons']

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'records' not in st.session_state:
    st.session_state['records'] = []

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = 'us-west4-gcp-free'
index_name = 'lamaidx'
model = 'gpt-3.5-turbo'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV) 
index = pinecone.Index(index_name)
openai.api_key = OPENAI_API_KEY
embed = OpenAIEmbeddings(
    model=model,
    openai_api_key=OPENAI_API_KEY
)
vectorstore = Pinecone(
    index, embed.embed_query, "text"
)

def query_pinecone(query):
    results = vectorstore.similarity_search(
        query,  # our search query
        k=5  # return 3 most relevant docs
    )
    return results

def query_openai(messages):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.6
    )
    return response.choices[0]['message']['content']

def generate_options(query):
    results = query_pinecone(query)
    examples = []
    for row in results:
        topics = '\n'.join(row.metadata['topics'])
        examples.append(f"product name: {row.metadata['name']}\ntagline: {row.metadata['tagline']}\ncategories: {topics}\ndomain: {row.metadata['host']}")
    example_prompt = '\n'.join(examples)
    messages = [
        {"role": "system", "content": default_msg},
        {"role": "user", "content": f"here are few of the most popular and related listings on producthunt related to the idea:\n{example_prompt}"},
        {"role": "user", "content": f"generate 5 listing options each containing product name, tagline, categories and domain name for idea below:"},
        {"role": "user", "content": query}
    ]
    response = query_openai(messages)
    return response

def fetch_records():
    return index.query(
        vector=[0]*1536,
        top_k=1000,
        include_metadata=True
    )

def select_random_categories():
    return random.sample(solution_types, 1) + random.sample(domains, 2)

def generate_product_idea(categories):
    if len(st.session_state['records']) == 0:
        records = fetch_records()
        st.session_state['records'] = records['matches']
    random5 = random.sample(st.session_state['records'], 5)
    examples = []
    for record in random5:
        topics = ', '.join(record['metadata']['topics'])
        examples.append(f"categories: {topics}\nidea: {record['metadata']['name']}, {record['metadata']['tagline']}")
    example_prompt = '\n\n'.join(examples)
    category_prompt = ', '.join(categories)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant that suggests new product ideas given categories"}, 
                    {"role": "user", "content": f"here are few of the most popular categories and taglines on product hunt:\n{example_prompt}"},
                    {"role": "user", "content": f"generate 1 new product idea in 20 words for:\n categories: '{category_prompt}'"}],
        temperature=0.6
    )
    return response['choices'][0]['message']['content'].replace('Idea: ', '')

# Set up the Streamlit app
st.set_page_config(
    page_title="Product Lama: Let's make awesome ProductHunt listing for your idea",
    page_icon=":robot_face:"
)
st.markdown('<h1 style="text-align:center">Product Lama</h1>', unsafe_allow_html=True)


input_container = st.container()
with input_container:
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Idea:", key='input', placeholder='search through your documents using semantic search')
        col1, col2 = st.columns([1,1])
        with col1:
            submit_button = st.form_submit_button(label='Send')
        with col2:
            lucky_button = st.form_submit_button(label="I'm feeling lucky")

    if submit_button and user_input:
        st.write(f"Generating producthunt listing options for '{user_input}'")
        output = generate_options(user_input)
        st.text(output)
    if lucky_button:
        st.write("Generating a producthunt listing for a random AI generated idea:")
        categories = select_random_categories()
        st.write("1. Generated random categories for product ideas:", ", ".join(categories))
        idea = generate_product_idea(categories)
        st.write("2. Generated product idea:", idea)
        output = generate_options(idea)
        st.write("3. Generated producthunt listing options:")
        st.text(output)

image = Image.open('product-lama.png')
st.image(image, caption='Let the Lama guide you to your perfect producthunt listing', use_column_width=True)
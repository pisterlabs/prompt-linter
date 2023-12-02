import streamlit as st
import pandas as pd
import time
import gcsfs
import asyncio
import os

import chromadb
from chromadb.utils import embedding_functions

import langchain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma

import ast  # Import the ast module for literal evaluation

# SINGLE RESPONSE GENERATION
async def get_email_response_personalized(sender_id,
                                    replier_id,
                                    sender_email,
                                    email_retrieval_dataset,
                                    num_emails,
                                    vector_db_client,
                                    llm_endpoint,
                                    template_string):
    
    # First getting retrieved emails to understand conversation --------
    sender_replier_id='-'.join([sender_id, replier_id])
    previous_emails=(email_retrieval_dataset[email_retrieval_dataset.sender_replier_thread==sender_replier_id]['Sender_Receiver_Emails_list']).to_list()[0][-num_emails:]
    
    # Second, getting ranked responses as per context ------------------
        
        # Building the Langchain vectorstore using chroma collections
    user_vector_store = Chroma(
        client=vector_db_client, 
        collection_name='user'+str(replier_id),
        embedding_function=OpenAIEmbeddings())
        # Getting ranked responses using MMR
    found_rel_emails = await user_vector_store.amax_marginal_relevance_search(sender_email, k=num_emails, fetch_k=num_emails)
    list_rel_emails=[]
    for i, doc in enumerate(found_rel_emails):
        list_rel_emails.append(doc.page_content)    
    # Setting up LangChain
    prompt_template = ChatPromptTemplate.from_template(template=template_string)    
    llm_chain=LLMChain(llm=llm_endpoint, prompt=prompt_template)
    return llm_chain.run(sender_email=sender_email, prev_emails=previous_emails, relevant_emails=list_rel_emails),previous_emails, list_rel_emails

async def main():
    await get_email_response_personalized()
    
# https://medium.com/@faizififita1/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-ba776487c5fe
# gcloud builds submit --tag gcr.io/msca310019-capstone-49b3/streamlit-app

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/msca310019-capstone-49b3-af1bf51fb3e1.json"

# Create a GCS filesystem instance
fs = gcsfs.GCSFileSystem(project='msca310019-capstone-49b3')

# Define the path to your CSV file in the GCS bucket
file_path = 'user-scripts-msca310019-capstone-49b3/data/20231026_Emails_Deduped.csv'

# Read the CSV file from Google Cloud Storage
with fs.open(file_path, 'rb') as file:
    # Use Pandas to read the CSV content into a DataFrame
    df_messages_deduped = pd.read_csv(file)

df_messages_deduped['Sender_Receiver_Emails_list'] = df_messages_deduped['Sender_Receiver_Emails'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# Find senders with more than 10 rows
repliers_with_more_than_10_rows = df_messages_deduped['reply_sender'].value_counts()[df_messages_deduped['reply_sender'].value_counts() > 100].index

# Filter the DataFrame to include only the rows with those senders
df_messages_deduped = df_messages_deduped[df_messages_deduped['reply_sender'].isin(repliers_with_more_than_10_rows)]

# users_more_than_50 = pd.read_csv('users_more_than_50.csv')

# # Define the path to your CSV file in the GCS bucket
# data_path = 'user-scripts-msca310019-capstone-49b3/data/data_message_reply_pairs_cleaned.csv'


# # Read the CSV file from Google Cloud Storage
# with fs.open(data_path, 'rb') as file:
#     # Use Pandas to read the CSV content into a DataFrame
#     data = pd.read_csv(file)
    
# data = pd.read_csv('/Users/scottsmacbook/Hedwig/Hedwig/00_Data/data_data_message_reply_pairs_cleaned.csv')

# Define a function to generate a random email and its sender ID
def generate_random_email():
    st.session_state.random_email = df_messages_deduped.sample(n=1)

# # Define a function to type out a string letter by letter
def type_string(text):
    t = st.empty()
    
    for i in range(len(text) + 1):
        t.markdown("## %s" % text[0:i])
        time.sleep(0.005)

# # Set the title image path
title_image_path = 'Hedwig Logo.jpeg'  # Replace with the actual path

# # Display the title and image side by side
# st.title("Hedwig.AI")

col1, mid, col2 = st.columns([1,1,20])

with col1:
    st.image('Hedwig Logo.jpeg', width=60)
with col2:
    st.title("Hedwig.AI")

# Use Streamlit's 'columns' layout to display buttons side by side
col1, mid,col2 = st.columns(3)

with col1:
    st.subheader("Incoming Email")

with col2:
    st.subheader("Generated Email")

with mid: 
    st.subheader("Retrieval and Ranking")
    

# Button to get a random email
if col1.button("Get Random Email Reply Pair"):
    generate_random_email()

if 'random_email' not in st.session_state:
    generate_random_email()
   
random_email = st.session_state.random_email

with col1:
    st.write(f"Sender ID: {str(list(random_email['sender'])[0])}")

    # Create a larger text area for user input (e.g., 10 rows)
    user_input = st.text_area("Enter Email:", height=500)
    
    # st.write(f"Email: {user_input}")

# Input field for email response
# replier_id = st.text_input("Enter your user :")

# Button to generate the response
if col2.button("Generate Response") and st.session_state.random_email is not None:
    # st.write("Response:")
    random_email = st.session_state.random_email        
    client = chromadb.PersistentClient(path="vectorstores")

    sender_id = str(list(random_email['sender'])[0])
    replier_id = str(list(random_email['reply_sender'])[0])
    sender_email = user_input

    num_emails = 10 #FOR RETRIEVEL + RANKING
    email_retrieval_dataset = df_messages_deduped # FOR RETRIEVAL DATABASE
    vector_db_client = client 

    openai_api_key = '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'

    api_key=openai_api_key
    llm_model='gpt-3.5-turbo-0301' # CAN CHANGE
    llm_endpoint=ChatOpenAI(temperature=0.1, model=llm_model, openai_api_key=openai_api_key) # CAN CHANGE

     
    template_string="""You are the person recieving this email enclosed in the angle brackets: <{sender_email}>,

    Write a reply to the email as the person who recieved it,
    
    deriving context and writing style and email length from previous relevant emails from the person given in the angle brackets : <{relevant_emails}>
    
    Make sure to use salutation and signature style similar to the revelant emails above."""

    os.environ['OPENAI_API_KEY'] = openai_api_key

    personalized_response,previous_emails, list_rel_emails = asyncio.run(get_email_response_personalized(sender_id,replier_id,sender_email,df_messages_deduped,
                                                            num_emails,client,llm_endpoint,template_string))
    with mid: 
        # st.write("Previous Emails: ")
        # for email in previous_emails:
        #     st.write(email)
        st.write("Ranked Emails: ")
        for i in range(0,5):
            st.write((f"{i+1}. {list_rel_emails[i]}"))
        
    with col2: 
        st.write(f"Replier ID: {str(list(random_email['reply_sender'])[0])}")
        st.text_area("Generated Response: ", value=personalized_response, key='response_area', height=500)
        # type_text_in_textarea(personalized_response)
        


    
    
import streamlit as st
from cohere.responses.classify import Example
import cohere
import time
from transformers import pipeline
import streamlit as st
from weaviate import Client as WeaviateClient
from weaviate.auth import AuthApiKey
import cohere
import numpy as np
from transformers import pipeline
import pandas as pd
import os
import streamlit as st
from weaviate import *
import weaviate
from weaviate.schema import Schema
import cohere
from langchain.schema import Document
from langchain.document_transformers import GoogleTranslateTransformer
import os 
import asyncio
from transformers import pipeline
import time
from weaviate.util import generate_uuid5
import csv
import base64




def get_image_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

icon1_base64 = get_image_as_base64("icon1.png")
lang_base64 = get_image_as_base64("lang.png")
cohere_base64 = get_image_as_base64("cohere.png")


# Basic setup
st.set_page_config(page_title="Customer Service Assistant", layout="wide")


def generate_response(label):
    co = cohere.Client(st.secrets["COHERE_API_KEY"]) # This is your trial API key
    response = co.generate(
    model='command',
    prompt='Generate a Solution as a customer service chatbot for the following question:' + str(label),
    max_tokens=330,
    temperature=0.9,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    print('Prediction: {}'.format(response.generations[0].text))
    
    return response.generations[0].text
  

def gigachad_summarizer(prompt):
    co = cohere.Client(st.secrets["COHERE_API_KEY"]) # This is your trial API key
    response = co.summarize( 
        text=prompt,
        length='short',
        format='bullets',
        model='command',
        additional_command='',
        temperature=0.3,
    ) 
  
    print('Summary:', response.summary)

    return response.summary


def Initilalize_Weaviate_feedback():
    CLUSTER_URL = st.secrets['COHERE_FEEDBACK_CLUSTER_URL']
    WEAVIATE_API = st.secrets['COHERE_FEEDBACK_API_KEY']
    auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API)
    client = weaviate.Client(url=CLUSTER_URL, auth_client_secret=auth_config)
    # Define the schema
    schema = Schema(client)
    print("Feedback Server Initialized ...")  
    
    feedback_class = {
        "class": "Feedback",
        "vectorizer": "text2vec-cohere",
        "vectorIndexConfig": {"distance": "dot"},  # for multilingual models
        "properties": [
            {"name": "name", "dataType": ["string"]},
            {"name": "username", "dataType": ["string"]},
            {"name": "feedback", "dataType": ["int64"]}
        ],
    }    
    
    # Create some sample data
    feedbacks = [
        { "username": "Ron", "feedback": 3},
        { "username": "Joe", "feedback": 4}
    ]
    
   #for feedback in feedbacks:
   #    client.data_object.create(feedback,"Feedback")
        
    print("Feedback Server Running .... ")
    
    return {
        "schema": schema,
        "classes": [feedback_class],
        "sample_data": {
            "Feedback": feedbacks
        }
        
    } , client    
   

def welcome_message_from_bot():
  co = cohere.Client(st.secrets["COHERE_API_KEY"])

  response = co.generate(
    prompt='Generate a Welcome Message Sentence as a customer service assistance with a Proper Emoji',
  )
  generated_text = response.generations[0].text
  # print(generated_text)
  return generated_text






   
def load_data_from_feedback_weaviate(client):
    """
    Loads data from Weaviate and returns it as a Pandas DataFrame.

    Parameters:
    - client: Weaviate client instance.
    
    Returns:
    - df: A Pandas DataFrame containing the inquiries.
    """
    # Define your GraphQL query
    query = """
        {
            Get {
                Feedback {
                    username
                    feedback
                }
            }
        }
    """

    # Execute the query
    result = client.query.raw(query)

    # Check if the query returned any results
    if 'data' in result and 'Get' in result['data'] and 'Feedback' in result['data']['Get']:
        # Extract the relevant data from the query result
        inquiries = result['data']['Get']['Feedback']
    else:
        print("No inquiries found.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

    # Transform the data into a Pandas DataFrame
    df = pd.DataFrame(inquiries)

    return df   
   
   
   
    
# Function to handle "Where is my order?" query
def handle_order_query():
    # You can customize this response or fetch it dynamically
    response = "Your order is being processed and will be delivered soon. You can track your order using the link sent to your email."
    add_message_to_chat("assistant", response)

# Function to handle "Return Policy" query
def handle_return_policy_query():
    # Customize or dynamically generate this response
    response = "Our return policy allows returns within 30 days of receiving your order. Please visit [Return Policy](link-to-policy) for more details."
    add_message_to_chat("assistant", response)

# Function to add messages to the chat interface
def add_message_to_chat(role, content):
    st.session_state.messages.append({"role": role, "content": content})



  
def process_intent(prompt):
  co = cohere.Client(st.secrets["COHERE_API_KEY"])
  examples=[
  Example("Do you offer same day shipping?", "Shipping and handling policy"),
  Example("Can you ship to Italy?", "Shipping and handling policy"),
  Example("How long does shipping take?", "Shipping and handling policy"),
  Example("Can I buy online and pick up in store?", "Shipping and handling policy"),
  Example("What are your shipping options?", "Shipping and handling policy"),
  Example("My order arrived damaged, can I get a refund?", "Start return or exchange"),
  Example("You sent me the wrong item", "Start return or exchange"),
  Example("I want to exchange my item for another colour", "Start return or exchange"),
  Example("I ordered something and it wasn't what I expected. Can I return it?", "Start return or exchange"),
  Example("What's your return policy?", "Start return or exchange"),
  Example("Where's my package?", "Track order"),
  Example("When will my order arrive?", "Track order"),
  Example("What's my shipping number?", "Track order"),
  Example("Which carrier is my package with?", "Track order"),
  Example("Is my package delayed?", "Track order"),
  Example("where can i change my shipping address? ","Shipping Address"),
  Example("Is there a policy on changing shipping address?","Shipping Address"),    
  Example("what is the policy for lost package ?","Terms and Conditions"),
  Example("what if i dont get my package for over 30 days?","Terms and Conditions"),
  Example("Do you Accept my Credit Card? ","Payment & Credit Card"),
  Example("Do you Accept Debit Card? ","Payment & Credit Card"),
  Example("Do you Accept PayPal? ","Payment & Credit Card"),
  Example("How can i Reset My Password? ","Login & Account Setting"),
  Example("I cant login ", "Login & Account Setting")
  ]
  
  response = co.classify(
  inputs=prompt,
  examples=examples,
  )
  


  # Extract the highest score classification and its confidence
  highest_score_classification = max(response.classifications[0].labels, key=lambda x: response.classifications[0].labels[x].confidence)
  highest_score_confidence = response.classifications[0].labels[highest_score_classification].confidence

  return highest_score_classification, highest_score_confidence

def generate_bot_dialouge(input):
  co = cohere.Client(st.secrets["COHERE_API_KEY"]) # This is your trial API key
  response = co.generate(
    model='command',
    prompt= 'Write a customer service chatbot response with the following information, your response should be as short as possible , use proper emojis: '+ f"{input}" ,
    max_tokens=244,
    temperature=0.9,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')  
  return response.generations[0].text


def detect_language(user_input):
  co = cohere.Client(st.secrets['COHERE_API_KEY']) # This is your trial API key
  response = co.classify(
    model='embed-english-v2.0',
    inputs=[f"{user_input}"],
    examples=[Example("Hello , Good Morning, Whats Up ?", "English"), Example("This is a Sample Sentence , Refund Policy", "English"), Example("Shipping Cost, Reset Password", "English"), Example("Customer Service, Start", "English"), Example("Hallo, guten Morgen, was ist los?", "Deutsch"), Example("Dies ist ein Beispielsatz und eine Rückerstattungsrichtlinie", "Deutsch"), Example("Versandkosten, Passwort zurücksetzen", "Deutsch"), Example("Kundenservice, Start", "Deutsch"), Example("Ciao, buongiorno, come stai?", "Italian"), Example("Costo di spedizione, reimposta password", "Italian"), Example("Questa è una frase di esempio, politica di rimborso", "Italian"), Example("Servizio clienti, inizio", "Italian")])
  #print('The confidence levels of the labels are: {}'.format(response.classifications))

  highest_score_classification = max(response.classifications[0].labels, key=lambda x: response.classifications[0].labels[x].confidence)
  highest_score_confidence = response.classifications[0].labels[highest_score_classification].confidence
  
  return highest_score_classification,highest_score_confidence


def translate_text(text, source_lang, target_lang):
    language_pairs = {
            'en-de': 'Helsinki-NLP/opus-mt-en-de',
            'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
            'es-en': 'Helsinki-NLP/opus-mt-es-en',
            'en-it': 'Helsinki-NLP/opus-mt-en-it',
            'it-en': 'Helsinki-NLP/opus-mt-it-en',
            'de-en': 'Helsinki-NLP/opus-mt-de-en'
        }
    model_key = f'{source_lang}-{target_lang}'
    if model_key in language_pairs:
        model_name = language_pairs[model_key]
        translator = pipeline("translation", model=model_name)
        translation = translator(text, max_length=512)
        return translation[0]['translation_text']
    else:
        raise ValueError(f"No translation model found for {model_key}")


# Custom CSS for a more sophisticated look and fee
# Header with language selection
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.write("")  # Placeholder for potential logo or icon
with col2:
    st.title("Multi Lingual Assistant")
with col3:
    language = st.selectbox("Language", ["Auto"])
    
# Sidebar for FAQs and Contact Info
with st.sidebar:
    st.header("FAQs")
    st.write("Here are some frequently asked questions...")
    Initilalize_Weaviate_feedback()
    # Quick Replies and Additional Functionalities
    with st.expander("Quick Replies & More"):
        if st.button("Where is my order? 🧾", key="qr1"):
            # Add functionality here
            handle_order_query()

        if st.button("Return Policy 💵", key="qr2"):
            # Add functionality here
            handle_return_policy_query()

        # Additional interactive elements
        num = st.slider("Rate your experience", 1, 5)
        print(num)
        if st.button("Leave a comment"):
            # Add functionality to handle comments
            _, client = Initilalize_Weaviate_feedback()
            
                # Create some sample data
            feedbacks = [
                { "username": "Sarah", "feedback": num}
            ]
            
            for feedback in feedbacks:
                client.data_object.create(feedback,"Feedback")
           
            #df = load_data_from_feedback_weaviate(client=client)
            #st.dataframe(df) 
            st.success("Thank you for your feedback! sent to weaviate...")
    st.header("Contact Information")
    st.write("https://github.com/homanmirgolbabaee/CohereCode-Quest")
    st.text("LablabAI Hackathon Product © 2023\nCohereCode Quest Team\nPowered by:\nCohere,Weaviate,Langchain")   
    # Adding logos at the bottom of the sidebar
    st.markdown(
        f'<img src="data:image/png;base64,{icon1_base64}" style="width:100px;height:100px;">', 
        unsafe_allow_html=True)
    st.markdown(
        f'<img src="data:image/png;base64,{lang_base64}" style="width:100px;height:100px;">', 
        unsafe_allow_html=True)
    st.markdown(
        f'<img src="data:image/png;base64,{cohere_base64}" style="width:100px;height:100px;">', 
        unsafe_allow_html=True)  
# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    
            
# Main Chat Interface

st.header("Customer Service Chatbot")


st.markdown('<p style="font-size: 16px; font-style: italic;">YOU CAN TYPE ANYTHING BUT IF YOU NEED Sample Prompts in English,Deutsch,Italian to test functionalities ...  ere are some examples ...</p>', unsafe_allow_html=True)

# Bullet points with smaller italic text
st.markdown('* <span style="font-size: 12px; font-style: italic;">Track My Order...</span>', unsafe_allow_html=True)
st.markdown('* <span style="font-size: 12px; font-style: italic;">Traccia Il Mio Ordine?</span>', unsafe_allow_html=True)
st.markdown('* <span style="font-size: 12px; font-style: italic;">Where is my Parcel?</span>', unsafe_allow_html=True)

# Container for displaying chat history
chat_history_container = st.container()

# Display chat messages from history inside the container
with chat_history_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



# Accept user input and process it
prompt = st.chat_input("Type your message here...")
            

if prompt:
    
    lan,score= detect_language(prompt)
    print("prompt was"+str(prompt))
    print(lan)
    
    
    
    
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process user's message and get bot response
    try:

        bot_response = " "
        if lan == "Deutsch":
            prompt = translate_text(prompt,"de","en")
        if lan == "Italian":
            prompt = translate_text(prompt,"it","en")
        inputs=[f"{prompt}"]
        label,score = process_intent(inputs)
        print(label,score)
        
        ## detecting intent 
        
        if label == "Track order":
            su = generate_response("Track Order")
            bot_response = su + bot_response
            bot_response = gigachad_summarizer(bot_response) 
            bot_response = generate_bot_dialouge(bot_response)
            if lan == "Deutsch":

                bot_response = translate_text(bot_response, 'en', 'de')
            if lan == "Italian":

                bot_response = translate_text(bot_response, 'en', 'it')
                
        if label == "Shipping and handling policy":
            su1 = generate_response("Shipping and handling policy")
            bot_response = su1 + bot_response
            bot_response = gigachad_summarizer(bot_response)            
            bot_response = generate_bot_dialouge(bot_response)
            if lan == "Deutsch":
 
                bot_response = translate_text(bot_response, 'en', 'de')
            if lan == "Italian":

                bot_response = translate_text(bot_response, 'en', 'it')
                                
        if label == "Start return or exchange":
            
            su2 = generate_response("Start return or exchange")
            bot_response = su2 + bot_response
            bot_response = gigachad_summarizer(bot_response)  
            bot_response = generate_bot_dialouge(bot_response)
            if lan == "Deutsch":
 
                bot_response = translate_text(bot_response, 'en', 'de')
            if lan == "Italian":

                bot_response = translate_text(bot_response, 'en', 'it')
                
                
        if label == "Shipping Address":
            su3 = generate_response("Shipping Address")
            bot_response = su3+bot_response
            bot_response = gigachad_summarizer(bot_response)  
            bot_response = generate_bot_dialouge(bot_response)
            if lan == "Deutsch":
 
                bot_response = translate_text(bot_response, 'en', 'de')
            if lan == "Italian":

                bot_response = translate_text(bot_response, 'en', 'it')            
            
        if label == "Terms and Conditions":
            su4 = generate_response("Terms and Conditions")
            bot_response = su4+bot_response
            bot_response = gigachad_summarizer(bot_response)  
            bot_response = generate_bot_dialouge(bot_response)
            if lan == "Deutsch":
 
                bot_response = translate_text(bot_response, 'en', 'de')
            if lan == "Italian":

                bot_response = translate_text(bot_response, 'en', 'it')        
                            
        bot_response = bot_response +"\n✍️Score: " + str(score) + "\n🏷️Label: "+str(label)

        # Add bot response to chat history and display it
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
    except Exception as e:
        st.error("Error in processing: " + str(e))

# Footer with additional links and information
st.markdown("---")



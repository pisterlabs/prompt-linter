import streamlit as st
import time
import os
import json
from dotenv import load_dotenv
import pandas as pd
import openai
import tiktoken
import time
from datetime import datetime
from PIL import Image
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import streamlit as st


load_dotenv() # Load environment variables from the .env file

# get the current index name from the environment variable
index_name = os.getenv("ES_INDEX")

favicon = Image.open("images/robot-icon2.png")
st.set_page_config(page_title='AI-augments', page_icon=favicon, layout="wide")

# Read the contents from the CSS file
with open("css/styles.css", "r") as f:
    css = f.read()

# set collections_path
collections_path = "./collections"

# Get all the JSON files from the "collections" subfolder
#collection_files = [f for f in os.listdir(collections_path) if f.endswith('.json')]

# List the collection files and sort them by last modified time (newest first)
collection_files = sorted([f for f in os.listdir(collections_path) if f.endswith('.json')], 
                          key=lambda x: os.path.getmtime(os.path.join(collections_path, x)), 
                          reverse=True)

# Load the collections into a dictionary
collections = {}
for file in collection_files:
    with open(os.path.join(collections_path, file), 'r') as f:
        collection = json.load(f)
        collections[collection['collectionid']] = collection


def get_usermessage_title(collection_id, usermessage_id, collections):
    collection = collections.get(collection_id)
    if not collection:
        return None
    
    user_messages = collection.get("usermessages", [])
    
    for user_message in user_messages:
        if user_message.get("id") == usermessage_id:
            return user_message.get("title")
            
    return None


# Include the CSS in the Streamlit app
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def connect_to_elasticsearch():
    # Load environment variables from the .env file
    load_dotenv()
    
    # Retrieve environment variables
    es_ca_cert_path = os.getenv("ES_CA_CERT_PATH")
    es_user = os.getenv("ES_USER")
    es_password = os.getenv("ES_PASSWORD")
    
    # Connect to the Elasticsearch cluster
    es = Elasticsearch("https://localhost:9200", 
                       ca_certs=es_ca_cert_path,
                       basic_auth=(es_user, es_password))
    
    try:
        # Try to get info from the Elasticsearch cluster
        info = es.info()
        print("Successfully connected to Elasticsearch!")
        print("Cluster Info:", info)
        return es, True
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, False


def disconnect_from_elasticsearch(es):
    if es is not None:
        try:
            es.transport.connection_pool.close()
            print("Disconnected from Elasticsearch.")
            return True
        except Exception as e:
            print(f"An error occurred while disconnecting: {e}")
            return False
    else:
        print("No active Elasticsearch client to disconnect.")
        return False

# Get Windows username
username = os.getlogin()

st.title("Search Memory")
st.info("This is a demo of the Search Memory feature with Elasticsearch. If elasticsearch connection fails, you may not see the search box.")

st.sidebar.markdown("""
### What is Search Memory?

Search Memory is your personal search assistant that stores and retrieves your previous generative AI queries and results. It helps you:

- Keep track of your past searches
- Reduce API calls and costs
- Quickly find and revisit important information
- Gain insights by comparing new and old search results

Simply enter your search query to get started!
""")

st.sidebar.markdown("""
### How to Use the Search UI

#### Search Box
The search box at the top is where you can type in your query. As you type, the system will automatically display the top 5 most relevant results from your search history.

#### Results Section
Each result is displayed in the following format:

- **System and User Prompts**: Shows the context of the conversation where the response was generated.
- **Edited Response**: The model's human edited response (when corrected by the user) based on the prompt.
- **Prompt Category**: Indicates the type of query, such as 'programming', 'general knowledge', etc.
- **Similarity Score**: A numerical score between 0 and 1 that represents how similar the model's response is to the edited response. A higher score indicates greater similarity.
- **Feedback**: A thumbs up or thumbs down button that allows you to provide feedback on the quality of the response. This feedback is used to improve the model's performance.
""")


es = None
if os.getenv("ES_INSTALLED") == "True":
    # Connect to Elasticsearch  
    es, es_connected = connect_to_elasticsearch()
else:
    # Disconnect from Elasticsearch
    es_connected = disconnect_from_elasticsearch(es)

# Now, you can use `es` for queries if `es_connected` is True
print(f"es_connected: {es_connected}")
if es_connected:
    st.sidebar.success("Connected to Elasticsearch.")
    # 1. Check if the user index exists
    if not es.indices.exists(index=index_name):
        st.sidebar.warning(f"You have not yet stored user prompts under your user index {index_name}. You can do that if you connect to Elasticsearch and save a first user prompt in your search memory.")
    else:
        # Get the document count for the index
        document_count = es.count(index=index_name)['count']

        # Display the count in Streamlit
        st.write(f"Total number of documents on this user memory: {document_count} document(s).")

        
        # 2. Create a search box for the user to enter their query
        user_query = st.text_input("🔍 __Enter your search query:__", "")
        if user_query:
            # Perform Elasticsearch search query
            search_result = es.search(
                index=index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": user_query,
                            "fields": [
                                "conversation_piece.original_prompt.user",
                                "conversation_piece.original_prompt.system",
                                "edited_response",
                                "promptcategory",
                                "usernote",
                                "feedback",
                                "model",
                                "userskillfocus"
                            ]
                        }
                    },
                    "size": 5
                }
            )
            
            promptcolor = "#D3D3D3" # light gray
            responsecolor = "#FAFAD2" # light goldenrod yellow
            footercolor = "#272829" # grayish purple
            fontcolor = "#D8D9DA" # black

            # Display search results
            #for hit in search_result['hits']['hits']:
            # Check if there are any hits
            if len(search_result['hits']['hits']) == 0:
                st.warning("No search results found.")
            else:
                for hit_num, hit in enumerate(search_result['hits']['hits'], 1):
                    # Display original_prompt in plain format
                    st.markdown(f'## :orange[Prompt Result {hit_num}]')
                    # Display the timestamp if available
                    timestamp = hit["_source"].get("timestamp", "no publishing date")  # Using .get() to avoid KeyError if the field is not present
                    
                    collection_id = hit["_source"].get("collection_id", "unknown")
                    usermessage_id = hit["_source"].get("usermessage_id", "unknown")
                    usermessage_title = get_usermessage_title(collection_id, usermessage_id, collections)
                    
                    # If usermessage_title is None, it won't display anything
                    display_usermessage_title = f"{usermessage_title}" if usermessage_title else ""
                    
                    # Display the timestamp if available
                    timestamp_str = hit["_source"].get("timestamp", None)
                    if timestamp_str:
                        # Assuming the timestamp is in ISO 8601 format
                        timestamp_dt = datetime.fromisoformat(timestamp_str)
                        
                        # Format the date as "mmm d, yyyy"
                        formatted_timestamp = timestamp_dt.strftime('%b %d, %Y')
                        
                        # Extract the Elasticsearch document ID
                        doc_id = hit.get("_id", "unknown")

                        # Display the timestamp, username, collection ID, user message ID, and Elasticsearch document ID
                        st.markdown(f'Published on {formatted_timestamp} | Username: {hit["_source"].get("username", "unknown")} | Collection ID: {collection_id} | User Message ID: {usermessage_id} | Doc ID: {doc_id}')
                        
                        # Extract the model information
                        model_info = hit["_source"].get("model", "unknown")
                        
                        st.markdown(f'#### {display_usermessage_title} ({model_info})')
                        # Insert a thick horizontal line to separate each result
                        st.markdown('<hr style="border:1px solid orange;padding:0rem;margin:0rem">', unsafe_allow_html=True)


                    else:
                        st.markdown(f'')

                    # Display promptcategory and similarity score with different colors
                    st.markdown(f'<div style="background-color:{footercolor}; padding:10px; color:{fontcolor}; border-radius: 0px;"><b>Prompt Category:</b> {hit["_source"].get("promptcategory", "None")} | <b>Similarity Score:</b> {hit["_source"]["similarityscore"]} | <b>Feedback:</b> {hit["_source"]["feedback"]} | <b>Skill Focus:</b> {hit["_source"]["userskillfocus"][0]} | <b>Costs USD:</b> {hit["_source"].get("total_cost", "unknown")}</div>', unsafe_allow_html=True)

                    # Display edited_response in markdown format
                    st.markdown("")
                    st.markdown(f'{hit["_source"]["edited_response"]}')
                    
                    usernote = hit["_source"].get("usernote")
                    if usernote and usernote != "None":  # Check if usernote exists and is not the string "None"
                        st.markdown(f':memo:__Revision__: {usernote}')
                    else:
                        st.markdown(f'')
                    
                    with st.expander("Show Original Prompt"):
                        st.markdown(f':orange_book: __Prompt:__ {hit["_source"]["conversation_piece"]["original_prompt"]["system"]} {hit["_source"]["conversation_piece"]["original_prompt"]["user"]}')
                    
                    # Create an expander for showing the JSON object
                    #with st.expander("Show JSON"):
                    #    st.json(hit)
                    st.code({hit["_source"]["edited_response"]})                    
                    
                    # Insert a thick horizontal line to separate each result
                    st.markdown('<hr style="border:2px solid gray;margin-top:0rem">', unsafe_allow_html=True)
else:
    st.sidebar.error("Not Connected to Elasticsearch.")
    st.markdown("""
                ## Install Elasticsearch Locally Using Docker
                If you see these instructions, this means that you have not installed elasticsearch on your local machine OR you have not started elasticsearch on your local machine.
                * If you are not running docker on your machine, install it from here https://docs.docker.com/get-docker/.
                * Then  follow instructions here https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html
                * For help, there is a sample output on elasticsearch installation in this file __./config/elasticsearch.txt__
                * After installation of elasticsearch and kibana make sure to change the following in the .env file
                    * ES_INSTALLED=True
                    * ES_USER=elastic
                    * ES_PASSWORD=changeme
                    * ES_CA_CERT_PATH=certs/ca/ca.crt
                    Note: The ES_CA_CERT_PATH is the path to the ca.crt file in the certs folder of the elasticsearch installation
                    In docker under elasticsearch container the http_ca.crt is located under /usr/share/elasticsearch/config/certs/
                    Take a copy of it and store it in your local machine and change the path of ES_CA_CERT_PATH in the .env file.
                * Then start elasticsearch and kibana in your docker container.
                * When elasticsearch is running, you should see in the sidebar the message "Connected to Elasticsearch".

    """)
    

import openai
import os
import streamlit as st

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector

# Load secrets and config from .env file
load_dotenv()

# OpenAI API
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
gpt35_model = os.getenv("OPENAI_GPT35_MODEL")
gpt35_16k_model = os.getenv("OPENAI_GPT35_16K_MODEL")
gpt4_model = os.getenv("OPENAI_GPT4_MODEL")
gpt4_32k_model = os.getenv("OPENAI_GPT4_32K_MODEL")

# Azure Search API
search_service_name = os.getenv("SEARCH_SERVICE_NAME")
search_service_key = os.getenv("SEARCH_SERVICE_KEY")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
search_endpoint = "https://{}.search.windows.net/".format(search_service_name)
search_vector_config_name = os.getenv("SEARCH_VECTOR_CONFIG_NAME")
search_semantic_config_name = os.getenv("SEARCH_SEMANTIC_CONFIG_NAME")

# Instantiate a client
class CreateClient(object):
    def __init__(self, endpoint, key, index_name):
        self.endpoint = endpoint
        self.index_name = index_name
        self.key = key
        self.credentials = AzureKeyCredential(key)

    # Create a SearchClient
    def create_search_client(self):
        return SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credentials,
        )

# Embeddings API
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def generate_embeddings(text):
    response = openai.Embedding.create(input=text, engine=embedding_model)
    embeddings = response["data"][0]['embedding']
    return embeddings

# Text Generation API
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def generate_text(prompt, model=gpt4_model, messages=[], max_tokens=600, temperature=0.5, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
    _messages = []
    _messages.extend(messages)
    _messages.append({"role": "user", "content": prompt})
    
    print("\n\n============================ PROMPT ============================\n")
    for message in _messages:
        print(f"{message['role']}: {message['content']}")
        
    response = openai.ChatCompletion.create(
        engine=model,
        messages=_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    print(response["choices"][0]["message"]["content"])
    return response

# Azure Search API
def azure_search(search_client, user_query, k, search_type='simple'):
    # Simple Search
    if search_type == 'simple':
        results = search_client.search(
            search_text=user_query,  
            select=[],
            top=k
        )

    # Semantic Search
    elif search_type == 'semantic':
        results = search_client.search(
            query_type='semantic', 
            query_language='es-es', 
            semantic_configuration_name=search_semantic_config_name,
            search_text=user_query,
            select=[], 
            query_caption='extractive',
            top=k
        )

    # Simple + Vector Search
    elif search_type == 'simple_vector':
        vector = Vector(value=generate_embeddings(user_query), k=k, fields="embeddings")
        results = search_client.search(
            search_text=user_query,  
            vectors= [vector],
            select=[],
            top=k
        )

    # Semantic + Vector Search
    elif search_type == 'semantic_vector':
        vector = Vector(value=generate_embeddings(user_query), k=k, fields="embeddings")
        results = search_client.search(
            query_type='semantic', 
            query_language='es-es', 
            semantic_configuration_name=search_semantic_config_name,
            search_text=user_query,
            vectors= [vector],
            select=[], 
            query_caption='extractive',
            top=k
        )

    else:
        raise Exception('Invalid search type. Valid options are: simple, semantic, simple_vector, semantic_vector')

    print("\n\n============================ SEARCH RESULTS ============================\n")
    _results = []
    for result in results:
        _results.append(
            {
                "score": result["@search.score"],
                "reranker_score": result["@search.reranker_score"],
                "key": result["id"],
                "content": result["content"],
                "filename": result["filename"],
                "page": result["page_number"]
            }
        )
        print(f"Score: {result['@search.score']} | Re-ranker Score: {result['@search.reranker_score']} | Filename: {result['filename']} | Page: {result['page_number']}")
    return _results

###############################################################
##################### Streamlit App ###########################
############################################################### 

# Method to add message to chat
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    st.chat_message(role).write(content)

# Set page title
st.set_page_config(page_title="🤖 Wisely - Chatbot with Data")
st.title('🤖 Wisely - Chatbot with Data')

# Model selection
model_options = [gpt35_model, gpt35_16k_model, gpt4_model, gpt4_32k_model]
selected_model = st.sidebar.selectbox("Select Model for Text Generation", model_options, index=0)

# Temperature
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Max tokens
max_tokens = st.sidebar.slider("Max Tokens", min_value=500, max_value=2000, value=600, step=50)

# Search type
search_type_options = ['simple', 'semantic', 'simple_vector', 'semantic_vector']
selected_search_type = st.sidebar.selectbox("Select Search Type", search_type_options, index=3)

# Top results
top_results = st.sidebar.slider("Top Results", min_value=1, max_value=10, value=2)

# Create welcome message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Create Azure Search Client 
if "search_client" not in st.session_state:
    base_client = CreateClient(search_endpoint, search_service_key, search_index_name)
    st.session_state["search_client"] = base_client.create_search_client()

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# When user submits a message
if user_input := st.chat_input():
    # Add user input to chat
    add_message("user", user_input)
    
    # Get Client for search and get relevant documents
    search_client = st.session_state.search_client
    relevant_docs = azure_search(search_client, user_input, k=top_results, search_type=selected_search_type)
    
    # Prepare prompt
    system_message = "You are trying to answer the [QUESTION] from the user. Based only in the [CONTEXT] information and the conversation history, create a high-quality answer to the user's question. Be brief and precise. Remember to cite your sources. Answer in Spanish."
    context = "\n\n".join([f"Page {doc['page']} of {doc['filename']}:\n{doc['content']}\n\n--------------------------" for doc in relevant_docs])
    prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{user_input}\n\n[ANSWER]"
    _messages = []
    _messages.append({"role": "system", "content": system_message})
    _messages.extend(st.session_state.messages)
    
    # Generate answer and add to chat
    answer = generate_text(prompt, messages=_messages, model=selected_model, max_tokens=max_tokens, temperature=temperature)
    add_message("assistant", answer["choices"][0]["message"]["content"])
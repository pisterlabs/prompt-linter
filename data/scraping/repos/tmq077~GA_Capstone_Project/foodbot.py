# Import the required libraries
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.llms import OpenAI
import openai
from llama_index import StorageContext, load_index_from_storage
from PIL import Image

# Set OpenAI API key through the streamlit app's secrets
openai.api_key = st.secrets.openai_key

# Display image banner
image = Image.open('streamlit/banner.jpg')
st.image(image, use_column_width=True)

# Sidebar contents
with st.sidebar:
    st.title('Do you know..')
    st.markdown('''
    Singapore generated **813 million kg** of waste in 2022, accounting for **12%** of total waste generated.
    \nAnnually, each household throws away **$258** worth of food, equivalent of **52 plates of nasi lemak**! 
    \nAmount of food waste has grown by **30%** over the past 10 years and is expected to rise further.
    \nAt the current rate of waste disposal, Singapore will need a new incineration plant every **7-10 years** and a new landfill every **30-35 years**.
    \nYou can help by donating your excess food to organizations who need them.
    \nDo note that information of the organisations is accurate as of 1 November 2023.
    ''')

# Session state to keep track of chatbot's message history
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help?"}
    ]

# Load and index data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading – hang tight! This should take 1-2 minutes."):
        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(persist_dir="streamlit/data/index.vecstore")

        # Load the index
        index = load_index_from_storage(storage_context)

        # Load the model 
        gpt_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0), context_window=2048, system_prompt="You are an expert on food donation in Singapore. Your role is to provide detailed information about the organisation and help individuals looking to donate. For each inquiry, answer with name of organisation, include any relevant information such as expiry dates and delivery instructions if available, and provide details on how to donate it and where to donate it. Ensure your responses are based on the documents and resources you have access to.")
        return index, gpt_context

index, gpt_context= load_data()

# Create chat engine
query_engine = index.as_query_engine(service_context=gpt_context)
chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine, verbose=True)

# Prompt for user input and display message history
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


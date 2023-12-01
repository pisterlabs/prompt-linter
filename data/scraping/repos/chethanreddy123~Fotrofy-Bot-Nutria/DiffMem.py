import streamlit as st
from langchain.llms import GooglePalm
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
import json

st.sidebar.title("Select Conversation Memory Type and Temperature")

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_string = json_file.read()
    return json_string

temperature = st.sidebar.number_input("Temperature", value=0.1, min_value=0.0, max_value=1.0, step=0.1)

llm = GooglePalm(
    model='models/chat-bison-001',
    temperature=temperature,
    max_output_tokens=1024,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)




# Add a dropdown menu to select memory type
memory_type = st.sidebar.selectbox("Memory Type", ["Buffer Memory", "Summary Memory", "Buffer Window Memory", "KG Memory"])

if memory_type == "Buffer Memory":
    conversation_memory = ConversationBufferMemory()
elif memory_type == "Summary Memory":
    conversation_memory = ConversationSummaryMemory()
elif memory_type == "Buffer Window Memory":
    conversation_memory = ConversationBufferWindowMemory()
elif memory_type == "KG Memory":
    conversation_memory = ConversationKGMemory(llm=llm)

st.title("Fotrofy Bot - Nutria Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


conversation_buf = ConversationChain(
    llm=llm,
    memory=conversation_memory
)

user_data = load_json_file('CUST_PROFILE.json')

initial_message = f'''Act like a expert Indian diet chatbot only use given formulas and user data to continue the chat:

Calculations :

### Recommended Calories Calculations:

...

UserData : {user_data}'''

initial_message = conversation_buf(initial_message)

# Function to get response from Google Palm
def get_google_palm_response(query):
    response = count_tokens(conversation_buf, query)
    return response

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_google_palm_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(f"Assistant: {response}")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": f"Assistant: {response}"})

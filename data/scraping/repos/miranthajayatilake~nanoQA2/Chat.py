# Import necessary libraries
import os
import streamlit as st
import pinecone
from PIL import Image
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains import ChatVectorDBChain

# Initialize Pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])

# Load logo and display it
image = Image.open('utils/logo.jpg')
st.image(image, width=200)

# Set the title
st.title(f'{os.environ["EGPTNAME"]}')

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
index = pinecone.Index(os.environ["INDEX_NAME"])
vectorstore = Pinecone(index, embeddings.embed_query, text_key='text', namespace=os.environ["NAMESPACE"])

# Define the system message template
system_template = """Use the following pieces of context to answer the users question. 
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
----------------
{context}"""

# Create the chat prompt templates
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

# Initialize the ChatVectorDBChain
qa = ChatVectorDBChain.from_llm(ChatOpenAI(temperature=0), vectorstore, qa_prompt=prompt, return_source_documents=True)

# Initialize the chat history
chat_history = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Define the function to execute a query
def execute_query(query):
    with st.spinner('Thinking...'):
        for i in range(len(st.session_state['chat_history'])-1, -1, -1):
            chat_history.append(st.session_state['chat_history'][i])

        result = qa({"question": query, "chat_history": chat_history})

        st.session_state.chat_history.append((query, result["answer"]))
        chat_history.append((query, result["answer"]))

    st.info(query)
    st.success(result['answer'])

    # Display the sources for the latest answer
    with st.expander("Sources for the latest answer"):
        sources = result['source_documents']
        for idx, i in enumerate(sources):
            st.markdown(f"**Source number {idx + 1}** \n")
            st.markdown(i)
            st.write('-'*10)

    # Display the previous chat history
    if len(chat_history) > 1:
        for query, answer in chat_history[:-1]:
            st.info(query)
            st.success(answer)

# Create input field and sample question buttons
query = st.text_input("Ask a question or tell what to do:", key="input")

if query:
    execute_query(query)

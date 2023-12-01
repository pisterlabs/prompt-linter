import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os
load_dotenv()

# 1. Vectorise company csv data
loader = CSVLoader(file_path="data.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
template= """
You are an assistant who helps me understand your company. I will provide a question
and you will give me the best answer based on some data about your company.

Below is my question:
{message}

Here is a list of related data about your company:
{company_data}
Please write the best response.

IMPORTANT NOTE: If the question is not related to your company, don't give direct answer but let me know that question is out-of-scope for this conversation and then get my attention back to the context of your company.
"""

prompt = PromptTemplate(
    input_variables=["message", "company_data"],
    template=template
)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Get augmented generation
def generate_response(message):
    company_data = retrieve_info(message)
    response = chain.run(message=message, company_data=company_data)
    return response

# Sidebar content
with st.sidebar:
    st.title('💬 LLM chatbot 📄')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    "[View source code](https://github.com/Timothy1102/company-assistant-bot)"

st.title("💬 Kobizo Assistant 🤖")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    kobizoRes = generate_response(prompt)
    st.session_state.messages.append({"role": "bot", "content": kobizoRes})
    st.chat_message("assistant").write(kobizoRes)

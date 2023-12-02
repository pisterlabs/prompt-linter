# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
Retrieval Chain.

# **utils.py**

import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
		"""
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """

		# Load OpenAI embedding model
		embeddings = OpenAIEmbeddings()
		
		# Load OpenAI chat model
		llm = ChatOpenAI(temperature=0)
		
		# Load our local FAISS index as a retriever
		vector_store = FAISS.load_local("faiss_index", embeddings)
		retriever = vector_store.as_retriever(search_kwargs={"k": 3})
		
		# Create memory 'chat_history' 
		memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")
		
		# Create system prompt
		template = """
    You are an AI assistant for answering questions about the Blendle Employee Handbook.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know ... 😔. 
    Don't try to make up an answer.
    If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.
    
    {context}
    Question: {question}
    Helpful Answer:"""
		
		# Create the Conversational Chain
		chain = ConversationalRetrievalChain.from_llm(llm=llm, 
				                                          retriever=retriever, 
				                                          memory=memory, 
				                                          get_chat_history=lambda h : h,
				                                          verbose=True)
		
		# Add systemp prompt to chain
		# Can only add it at the end for ConversationalRetrievalChain
		QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
		chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
		
		return chain
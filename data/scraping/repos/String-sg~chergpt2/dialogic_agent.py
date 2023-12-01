import streamlit as st
import wikipedia
import openai
import os
import pymongo
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.agents import load_tools, initialize_agent, Tool, tool, ZeroShotAgent, AgentExecutor
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from typing import List, Dict
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from agent_tools import wikipedia_to_json_string, document_search, google_search_serp, bing_search_internet



@st.cache_resource
def ailc_agent_serp():
	
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	tools = [
		Tool(
		name="Wikipedia_to_JSON_String",
		func=wikipedia_to_json_string,
		description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
		#return_direct=True
		),
		Tool(
		name = "Google Search Results",
		func=google_search_serp,
		description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
		#return_direct=True
		),
	]

	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain

@st.cache_resource
def ailc_agent_bing():
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	tools = [
		Tool(
		name="Wikipedia_to_JSON_String",
		func=wikipedia_to_json_string,
		description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
		#return_direct=True
		),
		Tool(
		name = "Bing Search Results",
		func=bing_search_internet,
		description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
		#return_direct=True
		),
	]

	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain



@st.cache_resource
def ailc_resource_agent():
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	if st.session_state.document_exist == True:
		subject, description = st.session_state.doc_tools_names
		#st.write(st.session_state.doc_tools_names)

		tools = [
			Tool(
			name = f"{subject} material search",
			func=document_search,
			description=f"A tool to search for {description}, this tool should be used more than the rest of the tool especially for relating topics on {subject}",
			#return_direct=True
			),
		
		]

	else:

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			#return_direct=True
			),

		]
	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
	#memory=ConversationTokenBufferMemory(llm=llm, max_token_limit=200, return_messages=True),

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain



@st.cache_resource
def sourcefinder_agent_serp():

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key
	st.session_state.tool_use = False
	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	if st.session_state.document_exist == True:
		subject, description = st.session_state.doc_tools_names

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			#return_direct=True
			),
			Tool(
			name = "Google Search Results",
			func=google_search_serp,
			description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
			#return_direct=True
			),
			Tool(
			name = f"{subject} material search",
			func=document_search,
			description=f"A tool to search for {description}, this tool should be used more than the rest of the tools especially for relating topics on {subject}",
			#return_direct=True
			),
		
		]

	else:

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			#return_direct=True
			),
			Tool(
			name = "Google Search Results",
			func=google_search_serp,
			description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
			#return_direct=True
			),

		]

	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain


@st.cache_resource
def chergpt_agent():

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key
	st.session_state.tool_use = False
	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	if st.session_state.document_exist == True:
		subject, description = st.session_state.doc_tools_names

		tools = [
			Tool(
				name = f"{subject} material search",
				func=document_search,
				description=f"A tool to search for {description}, this tool should be used more than the rest of the tool especially for relating topics on {subject}",
				),
		
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			),
		]

	else:

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			),

		]

	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain

@st.cache_resource
def metacog_agent(): #to be further upgraded by the base agent base class

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key
	st.session_state.tool_use = False
	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	if st.session_state.document_exist == True:
		subject, description = st.session_state.doc_tools_names

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			),
			Tool(
			name = f"{subject} material search",
			func=document_search,
			description=f"A tool to search for {description}, this tool should be used more than the rest of the tool especially for relating topics on {subject}",
			),
		
		]

	else:

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			),

		]

	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain




@st.cache_resource
def load_instance_index(_tch_code):
	embeddings = OpenAIEmbeddings()
	vectordb = FAISS.load_local(_tch_code, embeddings)

	return vectordb

def ailc_resources_bot(_query): #not in use for now 
	# if 'chat_history' not in st.session_state:
	# 	st.session_state.chat_history = [] 

	chat_history = []

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()
	cb_engine = st.session_state.engine_key
	
	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)


	vectorstore = load_instance_index(st.session_state.teacher_key)

	question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
	doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

	qa = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True
	)

	result = qa({"question": _query, "chat_history": chat_history})

	return result














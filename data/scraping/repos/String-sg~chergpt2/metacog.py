import streamlit as st
import pymongo
import os
import openai
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationEntityMemory, ConversationBufferWindowMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import json
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key  = st.session_state.api_key


def load_metacog_prompts():
	metacog_dir = "metacog"
	metacog_file = "metacog.json"

	if os.path.exists(metacog_dir) and os.path.isfile(os.path.join(metacog_dir, metacog_file)):
		with open(os.path.join(metacog_dir, metacog_file), "r") as f:
			prompts = json.load(f)
	else:
		st.write("here")
		metacog_str = config.get('metacog_samples', 'question_examples')
		prompts = ast.literal_eval(metacog_str)

	return prompts


@st.cache_resource
def load_instance_index():
	embeddings = OpenAIEmbeddings()
	vectordb = Chroma(collection_name=st.session_state.teacher_key, embedding_function=embeddings, persist_directory=st.session_state.teacher_key)
	return vectordb


@st.cache_resource
def conversation_starter_resources(_query): #not in use for now 
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = [] 

	openai.api_key  = st.session_state.api_key
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

	vectordb = load_instance_index()
	#question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
	qa = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), return_source_documents=True)
	result = qa({"question": _query, "chat_history": st.session_state.chat_history})
	return result

@st.cache_resource
def conversation_starter():
	
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()
	cb_engine = st.session_state.engine_key

	_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = f"""
	Welcome! I am a virtual teacher designed to help you learn and clarify about any educational topic. 
	To start, please provide a brief description of the topic you would like to explore. 
	As we discuss, I must provide you with a short paragraph or key points of not more than 80 words related to your query. I must create a list of 3 to 4 related questions to deepen your understanding for every query that you have provided. 

    At any point of the conversation during a topic discussion, if you want to end the topic or you want to change the topic or you attempted to change the topic, 
    I must create a list of 3 reflective questions below specifically on the topic before we move to a new topic but I must not show the list of reflective questions during the discussion. The sample list of reflective questions are 
    What did you previously know about the topic we just discussed? What is something new that you have learned about this topic during our conversation? What is an area or aspect of the topic that you would like to explore further or find out more about?

	After you have answered any of the reflective questions above and you want to end or change the topic, I must end the discussion by analysing and summarising the discussion thread of the topic and then ask you what would you like to learn next?

	Feel free to ask questions or provide feedback throughout our conversation. Let's begin!
	Context:
	{{entities}}
	Current conversation:
	{{history}}
	Last line:
	Human: {{input}}
	You:"""

	ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
	input_variables=["entities", "history", "input"],
	template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
	)

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)


	conversation = ConversationChain(
									llm=llm, 
									verbose=True,
									prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
									memory=ConversationEntityMemory(llm=llm)
									)
	return conversation






import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import SystemMessage

import pinecone
from langchain.vectorstores import Pinecone

import os
import uuid
from dotenv import load_dotenv

# load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_1 = os.getenv('PINECONE_INDEX_1')
PINECONE_INDEX_2 = os.getenv('PINECONE_INDEX_2')

chat_history_map = {}


def init_pinecone():
	# init pinecone index
	pinecone.init(
		api_key=PINECONE_API_KEY,
		environment=PINECONE_ENVIRONMENT,
	)

	for PINECONE_INDEX in [PINECONE_INDEX_1, PINECONE_INDEX_2]:
		if PINECONE_INDEX not in pinecone.list_indexes():
			pinecone.create_index(PINECONE_INDEX, dimension=1536, metric="cosine")

	[pinecone.Index(PINECONE_INDEX) for PINECONE_INDEX in [PINECONE_INDEX_1, PINECONE_INDEX_2]]


def init_vectorstore():
	# init embedding function
	embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002",
	                                      disallowed_special=())

	vectorstore_reglement = Pinecone.from_existing_index(
		index_name=PINECONE_INDEX_1,
		embedding=embedding_function,
		# namespace=namespace
	)

	vectorstore_cours = Pinecone.from_existing_index(
		index_name=PINECONE_INDEX_2,
		embedding=embedding_function,
		# namespace=namespace
	)

	return vectorstore_reglement, vectorstore_cours


def init_retriever(k):
	# init vectorstore
	vectorstore_reglement, vectorstore_cours = init_vectorstore()

	# init retriever
	retriever_reglement = vectorstore_reglement.as_retriever(search_kwargs={"k": k}, return_source_documents=True)
	retriever_cours = vectorstore_cours.as_retriever(search_kwargs={"k": k}, return_source_documents=True)

	return retriever_reglement, retriever_cours


class JSON_output(BaseModel):
	label: str = Field(description="label")
	comment: str = Field(description="comment")


class JSON_output_sources(BaseModel):
	doc_ids: list = Field(description="list of doc ids")


def llm_routing(query):
	llm = ChatOpenAI(
		openai_api_key=OPENAI_API_KEY,
		model_name='gpt-4-1106-preview',
		temperature=0,
		max_tokens=512
	)

	# ReAct prompt
	template = """
You will be presented with a user query in french. Your task is to classify the user query into one final category.

Approach this task step by step, take your time and do not skip any steps.

1. Read the user query.
2. Determine whether the query is related to university regulation OR university courses.
- If the query is related to general university regulations or faculty regulations, assign a label of "reglement".
- If the query is related to course details (eg. question about a specific course, a timetable, a professor, the contents of a course, etc.), assign a label of "cours".
3. Output a response as JSON with keys as follows:
	- "label": allowable values are ["reglement", "cours"]
	- "comment": any suitable comment based on the classification you performed, if required.

Input query: {query}
"""
	prompt_template = PromptTemplate(input_variables=["query"], template=template)

	parser = PydanticOutputParser(pydantic_object=JSON_output)

	prompt = prompt_template.format(query=query)
	res = llm([SystemMessage(content=prompt)])

	label = parser.parse(res.content).label

	return label


def update_retriever(label, qa):
	retriever_reglement, retriever_cours = init_retriever(k=5)

	if label == "reglement":
		retriever = retriever_reglement
		print(">>> Routing to reglement DB")
	elif label == "cours":
		retriever = retriever_cours
		print(">>> Routing to cours DB")
	else:
		retriever = MergerRetriever(retrievers=[retriever_reglement, retriever_cours])
		print(">>> Routing failed")

	qa.retriever = retriever

	return qa


def init_retrievalqa_chain():
	init_pinecone()
	retriever_reglement, retriever_cours = init_retriever(k=5)
	retriever = MergerRetriever(retrievers=[retriever_reglement, retriever_cours])

	# init llm
	llm = ChatOpenAI(
		openai_api_key=OPENAI_API_KEY,
		model="gpt-4-1106-preview",
		temperature=0
	)

	# init prompt template
	template = """
You are an assistant answering questions about the University of Geneva, based in Switzerland.
Use the following context and chat history to answer the questions. 
Your answer must be related to the University of Geneva only. If the question does not appear in the context or history of the chat, answer "I'm sorry, I don't know the answer".
Answers should be detailed but concise and short. 
Answers should be in english.
Take a deep breath and work step by step.

History: {chat_history}

Context: {context}

Question: {question}
Answer:
"""

	prompt = PromptTemplate(input_variables=["context", "question"], template=template)

	# init conversation memory
	conversational_memory = ConversationBufferWindowMemory(
		memory_key='chat_history',
		input_key="question",
		k=3,
		return_messages=True
	)

	# init retrievalQA chain
	qa = RetrievalQA.from_chain_type(
		llm=llm,
		chain_type="stuff",
		retriever=retriever,
		chain_type_kwargs={
			"prompt": prompt,
			"memory": conversational_memory
		},
		return_source_documents=True,
		verbose=False,
	)

	return qa


def isolate_sources(source_docs, answer):
	# init llm
	llm = ChatOpenAI(
		openai_api_key=OPENAI_API_KEY,
		model="gpt-4-1106-preview",
		temperature=0)

	template = """
	You will be presented with a list of retrieved source documents and an LLM generated answer. Your task is to determine which source documents contributed to the answer.
	
	Approach this task step by step, take your time and do not skip any steps.

	1. Read the generated LLM answer.
	2. Read the source documents.
	3. Determine which source documents in the list of source documents contributed to the answer.
	4. Output a response as JSON with keys as follows:
		- "doc_ids": allowable values are a list of integers (eg. [0, 1, 3])

	Input source documents: {source_docs}

	LLM generated answer: {answer}
"""

	prompt_template = PromptTemplate(input_variables=["source_docs", "answer"], template=template)

	parser = PydanticOutputParser(pydantic_object=JSON_output_sources)

	prompt = prompt_template.format(source_docs=source_docs, answer=answer)
	res = llm([SystemMessage(content=prompt)])

	doc_ids = parser.parse(res.content).doc_ids

	relevant_sources = list(set([source_docs[i].metadata["url"] for i in doc_ids]))

	#
	#
	# if len(relevant_sources) > 1:
	# 	relevant_sources = "\n- ".join([x for x in relevant_sources])
	# else:
	# 	relevant_sources = list(relevant_sources) if len(relevant_sources) > 0 else []

	return relevant_sources


def run_query(qa, query, labels, session_id=None):
	global chat_history_map

	# init conversation memory
	if session_id is None or session_id not in chat_history_map:
		session_id = uuid.uuid4()
		current_chat_history = ConversationBufferWindowMemory(
			memory_key='chat_history',
			input_key="question",
			k=3,
			return_messages=True
		)
		chat_history_map[session_id] = current_chat_history

	#  if session_id already exists, retrieve chat history
	else:
		current_chat_history = chat_history_map.get(session_id)

	if len(labels) >= 2:
		# if query is not related to the same topic as the previous query, update retriever
		if labels[-1] != labels[-2]:
			qa = update_retriever(labels[-1], qa)
			print(">>> Retriever updated")

			res = qa({"query": query, "chat_history": current_chat_history})

		# else, run query with same retriever
		else:
			res = qa({"query": query, "chat_history": current_chat_history})

	else:
		qa = update_retriever(labels[-1], qa)

		res = qa({"query": query, "chat_history": current_chat_history})

	# update chat history
	chat_history_map[session_id] = qa.combine_documents_chain.memory

	#  isolate relevant sources
	relevant_sources = isolate_sources(res["source_documents"], res["result"])
	if isinstance(relevant_sources, str):
		relevant_sources = [relevant_sources]
	if not relevant_sources:
		relevant_sources = ["**No relevant sources found**"]
	return {
		"answer": res["result"],  # + "\n\n" + "Source documents:\n- " + relevant_sources,
		# "source_documents": res["source_documents"],
		"source_documents": "\n".join(relevant_sources),
		"session_id": session_id
	}


"""
# classify query into reglement or cours for vectorstore routing

# run query
res = run_query(qa, query, labels, session_id)
"""

qa = None
labels = None


def startup_function():
	global qa, chat_history_map, labels
	qa = init_retrievalqa_chain()
	chat_history_map = dict()
	labels = []


if __name__ == "__main__":
	# qa = init_retrievalqa_chain()
	# session_id = None
	# while True:
	# 	question = input('>')
	# 	res = run_query(qa, question, session_id)
	# 	print(res["answer"])
	pass

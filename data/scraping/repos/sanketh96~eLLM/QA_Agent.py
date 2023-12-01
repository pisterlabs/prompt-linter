import os
from collections import defaultdict

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader, TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.vectorstores import Chroma
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = "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"


class QAModule:

	def __init__(self):
		self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
		self.vector_store = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
		self.agent = None
		self.text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=25)
		self.username_to_chat_history_mapping = defaultdict(list)
		self._template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
						Chat History:
						{chat_history}
						Follow Up Input: {question}
						Standalone question:"""
		self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self._template)
		self.template = """You are a helpful educator who answers queries based on provided context. Do not any use additional knowledge. 
		Answer in {style}. You are given the following extracted parts of a long document and a question. Provide a conversational answer that is faithful to the provided context.
		Do not mention that you are an AI language model or answering in that style . Try not to be very verbose.
		{context}
		Question: {question}
		Helpful answer: """
		self.followup_question_template = """You are a helpful educator who answers queries based on provided context. Do not any use additional knowledge. 
		Answer in {style}. You are given the following extracted parts of a long document and a question. Provide a conversational answer that is faithful to the provided context. Also ask a thoughtful follow up question in a conversational manner after the explanation that is related to the given context. Let it be easy and nor repititive.
		Do not mention that you are an AI language model or answering in that style . Try not to be very verbose.
		{context}
		Question: {question}
		Helpful answer: """
		self.style_template = "a format that is easy for students to understand"
		self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=['context', 'question', 'style'], template=self.template)
		self.QA_CHAIN_FOLLOW_UP_PROMPT = PromptTemplate(input_variables=['context', 'question', 'style'], template=self.followup_question_template)
		self.text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=25)

	def add_word_document_to_index(self, document_uri):
		loader = UnstructuredWordDocumentLoader(document_uri, mode="elements", strategy="fast")
		self.update_vector_store(loader)

	def add_pdf_document_to_index(self, document_uri):
		loader = PyPDFLoader(document_uri)
		self.update_vector_store(loader)

	def add_text_document_to_index(self, document_uri):
		loader = TextLoader(document_uri)
		self.update_vector_store(loader)

	def get_source(self, question, chat_history):
		llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
		question_generator = LLMChain(llm=llm, prompt=self.CONDENSE_QUESTION_PROMPT)
		doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
		chain = ConversationalRetrievalChain(
			retriever=self.vector_store.as_retriever(search_kwargs={"k": 1}),
			question_generator=question_generator,
			combine_docs_chain=doc_chain,
		)
		result = chain({"question": question, "chat_history": chat_history})
		return result['answer']

	def query_index(self, question, username, return_source_docs=False, character=None, followup_qa=False):
		style = self.style_template
		qa_chain_prompt = self.QA_CHAIN_PROMPT
		if followup_qa:
			qa_chain_prompt = self.QA_CHAIN_FOLLOW_UP_PROMPT
		if character:
			style = "in the style of {character}".format(character=character)
		self.agent = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
														   self.vector_store.as_retriever(search_kwargs={"k": 1}),
														   condense_question_llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
														   combine_docs_chain_kwargs={'prompt': qa_chain_prompt},
														   condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
														   output_key='answer',
														   return_source_documents=True)
		chat_history = self.get_chat_history_for_user(username)
		source = self.get_source(question, chat_history)
		source_idx = source.index("SOURCES")
		source = source[source_idx:]
		result = self.agent({"question": question, "chat_history": chat_history, "style":style})
		self.update_chat_history_for_user(username, question, result['answer'])
		if return_source_docs:
			return result['answer'], [source_doc.page_content for source_doc in result['source_documents']]
		return result['answer'] + '\n' + source

	def update_vector_store(self, loader):
		document = loader.load()
		document = self.text_splitter.split_documents(document)
		self.vector_store.add_documents(document)

	def get_chat_history_for_user(self, username):
		return self.username_to_chat_history_mapping[username]

	def reset_user_conversation(self, username):
		self.username_to_chat_history_mapping[username] = []

	def update_chat_history_for_user(self, username, question, answer):
		chat_history = self.username_to_chat_history_mapping[username]
		chat_history.append((question, answer))

	def add_document_to_index(self, document_uri):
		if "docs.google" in document_uri:
			pass
		else:
			extension = document_uri.split(".")[1]
			if extension == "txt":
				self.add_text_document_to_index(document_uri)
			elif extension == "docx":
				self.add_word_document_to_index(document_uri)
			elif extension == "pdf":
				self.add_pdf_document_to_index(document_uri)
			else:
				return -1


if __name__ == "__main__":
	# example usage.
	with get_openai_callback() as cb:
		agent = QAModule()
		agent.add_document_to_index("docs/class_6_chapter_4.pdf")
		# print(agent.query_index("What is a petiole?", "yash", character="jack sparrow", followup_qa=True))
		# print(agent.query_index("What is a petiole?", "yash", return_source_docs=True, followup_qa=True))
		print(agent.query_index("Can you tell photo synthsis", "yash", followup_qa=True, character="peppa pig"))
		print(agent.query_index("No", "yash", followup_qa=False, character="peppa pig"))
		# print(agent.query_index("No I don't know, can you explain", "yash", character="jack sparrow", followup_qa=False))
		print(cb)
# print(agent.query_index("What was my previous problem?", "yash"))
# print(agent.query_index("I am still stuck with the previous problem", "yash"))
# print(agent.query_index("Thank you!", "yash"))
# print(agent.query_index("I have one more query!", "yash"))
# print(agent.query_index("How do I download Slack on my phone?", "yash"))

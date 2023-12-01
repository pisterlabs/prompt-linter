from PIL import Image
import streamlit as st
import openai
#exercise 11
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
#exercise 12
from langchain.memory import ConversationBufferWindowMemory
#exercise 13
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
import os
import tempfile

# os.environ["OPENAI_API_KEY"] = st.secrets["openapi_key"]
# openai.api_key = st.secrets["openapi_key"]

#Global ex 13
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)
	
def ex11a():  # change in ex11a
	# langchain prompt template
	prompt = PromptTemplate(
		input_variables=["subject", "topic"],
		template="""Design a lesson plan on {subject} on the topic of {topic} for primary 1 students""",
	)

	# openai_api_key = st.secrets["openapi_key"]
	llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

	# creating a LLM chain with the langchain call and prompt template
	chain = LLMChain(llm=llm, prompt=prompt)
	if st.button("Run my chain"):
		input_prompt = prompt.format(subject="English", topic="Verbs")
		# Showing what is sent to LLM Chain
		st.write("Input prompt: ", input_prompt)
		# Showing the output from LLM Chain
		st.write(chain.run({"subject": "English", "topic": "Verbs"}))
	
def prompt_inputs_form(): #Using st.form, create the starting prompt to your prompt template, this is an expert on a topic that is talking to a user of a certain age
	#langchain prompt template
	with st.form("Prompt Template"):
		occupation = st.text_input("Enter the occupation:")
		topic = st.text_input("Enter the topic:")
		age = st.text_input("Enter the age:")

	# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
	#return a dictionary of the values
	if submitted:
		return {
			'occupation': occupation,
			'topic': topic,
			'age': age
		}

def ex11b():
	# create your template
	prompt_template = PromptTemplate(
		input_variables=["occupation", "topic", "age"],
		template="""Imagine you are a {occupation} who is an expert on the  topic of {topic} , you are going to help , teach and provide information to the person who is {age} years old, if you do not not know the answer, you must tell the person , do not make any answer up""",
	)
	# create a langchain function call to openai
	llm = ChatOpenAI(
		model_name="gpt-3.5-turbo",
		temperature=0.9,
	)
	# create a LLM chain with the langchain call and prompt template
	chain = LLMChain(llm=llm, prompt=prompt_template)
	# call the prompt_inputs_form()
	dict_inputs = prompt_inputs_form()
	if dict_inputs:
		st.write(chain.run(dict_inputs))
		
def ex12():
	memory = ConversationBufferWindowMemory(k=3)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))
   
	memory = ConversationBufferWindowMemory( k=3, return_messages=True)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))

#exercise 13 - loading
def upload_file_streamlit():
	def get_file_extension(file_name):
		return os.path.splitext(file_name)[1]

	st.subheader("Upload your docs")

	# Streamlit file uploader to accept file input
	uploaded_file = st.file_uploader("Choose a file", type=["docx", "txt", "pdf"])

	if uploaded_file:
		# Reading file content
		file_content = uploaded_file.read()

		# Determine the suffix based on uploaded file's name
		file_suffix = get_file_extension(uploaded_file.name)

		# Saving the uploaded file temporarily to process it
		with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
			temp_file.write(file_content)
			temp_file.flush()  # Ensure the data is written to the file
			temp_file_path = temp_file.name
		return temp_file_path
	
#exercise 13 - split and chunk, embeddings and storing in vectorstores for reference
def vectorstore_creator():
	# WORKING_DIRECTORY set above in the main.py
	# Process the temporary file using UnstructuredFileLoader (or any other method you need)
	embeddings = OpenAIEmbeddings()
	db = lancedb.connect(WORKING_DIRECTORY)
	table = db.create_table(
		"my_table",
		data=[
			{
				"vector": embeddings.embed_query("Query unsuccessful"),
				"text": "Query unsuccessful",
				"id": "1",
			}
		],
		mode="overwrite",
	)
	# st.write(temp_file_path)
	temp_file_path = upload_file_streamlit()
	if temp_file_path:
		loader = PyPDFLoader(temp_file_path)
		documents = loader.load_and_split()
		db = LanceDB.from_documents(documents, embeddings, connection=table)
		return db

def ex13():
	if "vectorstore" not in st.session_state:
		st.session_state.vectorstore = False
	db = vectorstore_creator()
	st.session_state.vectorstore = db
	if st.session_state.vectorstore:
		query = st.text_input("Enter a query")
		if query:
			st.session_state.vectorstore = db
			docs = db.similarity_search(query)
			st.write(docs[0].page_content)

def chat_completion_stream_prompt(prompt):
	MODEL = "gpt-3.5-turbo" #consider changing this to session_state
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template},
			{"role": "user", "content": prompt},
		],
		temperature= 0, # temperature
		stream=True #stream option
	)
	return response
	
# save the vectorstore in st.session_state
# add semantic search prompt into memory prompt
# integrate back into your chatbot
def ex14_basebot():
	# Prompt_template form from ex11
	prompt_template = PromptTemplate(
		input_variables=["occupation", "topic", "age"],
		template="""Imagine you are a {occupation} who is an expert on the  topic of {topic} , you are going to help , teach and provide information
						to the person who is {age} years old, if you do not not know the answer, you must tell the person , do not make any answer up""",
	)
	dict_inputs = prompt_inputs_form()
	if dict_inputs:
		input_prompt = prompt_template.format(
			occupation=dict_inputs["occupation"],
			topic=dict_inputs["topic"],
			age=dict_inputs["age"],
		)
		st.session_state.input_prompt = input_prompt

	if "input_prompt" not in st.session_state:
		st.session_state.input_prompt = "Speak like Yoda from Star Wars"

	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=5)

	# step 1 save the memory from your chatbot
	# step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write(memory_data)
	st.session_state.prompt_template = f"""
st.session_state.input_prompt: {st.session_state.input_prompt}

This is the last conversation history
{memory_data}

"""
	st.write("new prompt template: ", st.session_state.prompt_template)

	st.session_state.vectorstore = vectorstore_creator()

	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			# query information
			if st.session_state.vectorstore:
				docs = st.session_state.vectorstore.similarity_search(prompt)
				docs = docs[0].page_content
				# add your query prompt
				vs_prompt = f"""You should reference this search result to help your answer,
								{docs}
								if the search result does not anwer the query, please say you are unable to answer, do not make up an answer"""
			else:
				vs_prompt = ""
			# add query prompt to your memory prompt and send it to LLM
			st.session_state.prompt_template = (
				st.session_state.prompt_template + vs_prompt
			)
			# set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state.memory.save_context(
				{"input": prompt}, {"output": full_response}
			)

	except Exception as e:
		st.error(e)
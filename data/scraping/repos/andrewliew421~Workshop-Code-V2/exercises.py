import streamlit as st
from openai import OpenAI
from authenticate import return_api_key
import sqlite3
import os
import pandas as pd
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
import streamlit as st
import openai
from authenticate import return_api_key
from langchain.tools import YouTubeSearchTool
from kb_module import display_vectorstores
from users_module import vectorstore_selection_interface
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import tool
import json

cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

# os.environ.get("OPENAI_API_KEY")
# api_key=return_api_key()
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=return_api_key(),
)

#Only modify the code below for the exercises and challenges

#python exercises
def hello_world():
	st.write("Hello World")
	st.write("My name is ... ")

#exrcise 1
def input_exercise():
	name = st.text_input("Enter your name:", key = 1)
	st.write("Hello " + name)

#exercise 2
def button_exercise():
	if st.button("Say Hello"):
		st.write("Hello")
	else:
		st.write("Goodbye")

#exercise 3
def using_if_else():
	if st.button("Print Hello"):
		st.write("Hello")
	elif st.button("Say Goodbye"):
		st.write("Goodbye")
	else:
		st.write("Press a button")		

#challenge 1 is to modify the code below to create a button that says "Say Goodbye"
def button_input_exercise():
	name = st.text_input("Enter your name:", key=2)
	#add buton here
	#if button...
	# st.write("something" + name)
	#else...
	#do something else


#exercise 3
def using_session_state():
	st.write("Normal Variable")
	count = 0
	if st.button("Increment Variable"):
		count += 1
		st.write("Variable Count = ", count)
	st.write("Using Session State")
	#initialize session state
	if "count" not in st.session_state:
		st.session_state.count = 0
	#increment session state
	if st.button("Increment Session State"):
		st.session_state.count += 1
	st.write("Session State Count = ", st.session_state.count)

#exercise 4
def rule_based_question_answering():
	#create a rule based question answering system with the following questions and answers
	question = st.text_input("Enter your question:")
	if question == "What is your name?":
		st.write("My name is EAI, an electronic artificial being.")
	elif question == "How old are you?":
		st.write("Today is my birthday!")
	else:
		st.write("I am sorry, I am unable to help you with your query.")

#challenge 4 modify the code below to create a rule based question answering system using session state and show the previous query
def rule_based_question_answering_challenge():
	if "previous_query" not in st.session_state:
		st.session_state.previous_query = ""
	st.write("Previous query is " + st.session_state.previous_query)
	question = st.text_input("Enter your query:")
	#modify below to use session state
	if question == "What is your name?":
		#st.write...
		#st.session_state.previous_query.....
		pass
	elif question == "How old are you?":
		#st.write...
		#st.session_state.previous_query.....
		pass
	else:
		#st.write...
		#st.session_state.previous_query.....
		pass

#exercise 5
def simple_data_structure():
	#this is a list
	list = ["apple", "banana", "cherry"]
	st.write(list)
	#this is a dictionary
	dict = {"name": "John", "age": 36}
	st.write(dict)
	#this is a list of dictionaries
	dict1 = {"name": "John", "age": 36}
	dict2 = {"name": "Mary", "age": 25}
	dict3 = {"name": "Peter", "age": 42}
	list_of_dict = [dict1, dict2, dict3]
	st.write(list_of_dict)

#exercise 6
def display_dictionary_in_dataframe():
	dict1 = {"name": "John", "age": 36}
	dict2 = {"name": "Mary", "age": 25}
	dict3 = {"name": "Peter", "age": 42}
	list_of_dict = [dict1, dict2, dict3]
	df = pd.DataFrame(list_of_dict)
	st.write(df)

#exercise 7
def loop_exercise():
	#loop through a list
	list = ["apple", "banana", "cherry"]
	for x in list:
		st.write(x)
	for x in range(3):
		st.write(x)
	#loop through a list of dictionaries
	dict1 = {"name": "John", "age": 36}
	dict2 = {"name": "Mary", "age": 25}
	dict3 = {"name": "Peter", "age": 42}
	list_of_dict = [dict1, dict2, dict3]
	#for loop
	for x in list_of_dict:
		st.write(x)
		st.write("name", x["name"])
		st.write("age", x["age"])
	
#exercise 8
def streamlit_form_exercise():
	with st.form("new form"):
		st.write("Inside the form")
		name = st.text_input("Enter your name:")
		age = st.number_input("Enter your age:", value=int(18))
		gender = st.selectbox("Select Gender", ("Male", "Female"))
		submit_button = st.form_submit_button("Submit")
	if submit_button:
		st.write("Outside the form")
		st.write("Name:", name)
		st.write("Age:", age)
		st.write("Gender:", gender)
		person = {
			"Name": name,
			"Age": age,
			"Gender": gender
		}
		st.write(person)

#challenge 9
# Append the form data from a dictionary to a session state list and display the list of dictionaries
def append_form_data_to_list():
	if "list_of_dict" not in st.session_state:
		st.session_state['list_of_dict'] = []
		#st.session_state.list_of_dict = [] ( Same as above)

	with st.form("my_form2"):
		st.write("Inside the form")
		name = st.text_input("Enter your name:")
		age = st.number_input("Enter your age:", value=int(18))
		gender = st.selectbox("Select Gender", ("Male", "Female"))
		submit_button = st.form_submit_button("Submit")
	if submit_button:
		st.write("Name:", name)
		st.write("Age:", age)
		st.write("Gender:", gender)
		person = {
			"Name": name,
			"Age": age,
			"Gender": gender
		}
		#append the person dictionary to the list_of_dict session state
		
		#display the list_of_dict session state using a for loop
	





#Streamlit App challenge - python recap and streamlit
#create your first streamlit app
def streamlit_app():
	st.write("My first Streamlit App")


#Streamlit App challenge - example
def streamlit_app_example():
	st.title("Streamlit App Exercise Example")
	# Initialize session state
	if 'participants' not in st.session_state:
		st.session_state['participants'] = []

	# Function to add participant
	def add_participant():
		participant = {
			"Name": name,
			"Age": age,
			"Gender": gender,
			"Email": email,
			"Phone Number": phone,
			"Department": department
		}
		st.session_state['participants'].append(participant)

	# Streamlit form
	with st.form("participant_form", clear_on_submit=True):
		st.write("Participant Registration Form")

		# Form fields
		name = st.text_input("Name")
		age = st.number_input("Age", min_value=16, max_value=99, step=1)
		gender = st.selectbox("Gender", ["Male", "Female", "Other"])
		email = st.text_input("Email")
		phone = st.text_input("Phone Number")
		department = st.selectbox("Department", ["Sales", "Marketing", "IT", "HR", "Finance", "Operations"])

		# Submit button
		submit_button = st.form_submit_button("Submit")

	# Process form submission
	if submit_button:
		if len(st.session_state['participants']) < 5:
			add_participant()
			st.success("Registration Successful!")
		else:
			st.error("Registration Full")

	# Display participants in a DataFrame
	if st.session_state['participants']:
		df = pd.DataFrame(st.session_state['participants'])
		st.write(df)



#Challenge 1 is to modify the code below to create a rule based bot and creating a dictionary data to store the prompts and replies
def get_reply(prompt):
	# Dictionary of prompts and replies
	replies = {
		"Hello": "Hi there, what can I do for you?",
		"What is your name?": "My name is EAI, an electronic artificial being.",
		"How old are you?": "Today is my birthday!"
	}

	# Return the reply for the given prompt, or a default response
	return replies.get(prompt, "I am sorry, I am unable to help you with your query.")

#Exercise and challenge 1
def rule_based_chatbot():

	st.title("Echo Bot to Rule Based Bot")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

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
		
		#modify the code below to create a rule based bot ( challenge 2), replace f"Echo: {prompt}" with get_reply(prompt)
		response = f"Echo: {prompt}"
		#response = get_reply(prompt)

		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})

#Exercise 2
def api_call_exercise():
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	st.title("Api Call Exercise")
	MODEL = "gpt-3.5-turbo"
	with st.status("Calling the OpenAI API..."):
		response = client.chat.completions.create(
			model=MODEL,
			messages=[
				{"role": "system", "content": "Speak like a pirate"},
				{"role": "user", "content": "Tell me the story about Singapore in the 1970s in 50 words."},
			],
			temperature=0,
		)

		st.markdown("**This is the raw response:**") 
		st.write(response)
		st.markdown("**This is the extracted response:**")
		st.write(response.choices[0].message.content.strip())
		completion_tokens = response.usage.completion_tokens
		prompt_tokens = response.usage.prompt_tokens
		total_tokens = response.usage.total_tokens

		st.write(f"Completion Tokens: {completion_tokens}")
		st.write(f"Prompt Tokens: {prompt_tokens}")
		st.write(f"Total Tokens: {total_tokens}")
		#s = str(response["usage"]["total_tokens"])
		#st.markdown("**Total tokens used:**")
		#st.write(s)




#challenge 2 is to create a function call_api to pass the prompt design and variables to call the OpenAI API 
def call_api_challenge():
	st.title("Api Call Challenge")
	prompt_design = st.text_input("Enter your the prompt design for the API call:", value="You are a helpful assistant.")
	prompt_query = st.text_input("Enter your prompt query:", value="Tell me about Singapore in the 1970s in 50 words.")
	if st.button("Call the API"):
		#modify the code below to create a function call_api to pass the prompt_design and prompt_query to call the OpenAI API
		if prompt_design and prompt_query:
			#call your api_call function here
			st.write("Call the api_call function here")
		else:
			st.warning("Please enter a prompt design and prompt query.")
#challenge 2 this function below is called by the call_api_challenge function
def api_call(p_design, p_query):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	
	MODEL = "gpt-3.5-turbo"
	with st.status("Calling the OpenAI API..."):
		response = client.chat.completions.create(
			model=MODEL,
			messages=[
				{"role": "system", "content": p_design},
				{"role": "user", "content": p_query},
			],
			temperature=0,
		)

		st.markdown("**This is the raw response:**") 
		st.write(response)
		st.markdown("**This is the extracted response:**")
		st.write(response.choices[0].message.content)
		completion_tokens = response.usage.completion_tokens
		prompt_tokens = response.usage.prompt_tokens
		total_tokens = response.usage.total_tokens

		st.write(f"Completion Tokens: {completion_tokens}")
		st.write(f"Prompt Tokens: {prompt_tokens}")
		st.write(f"Total Tokens: {total_tokens}")


#Exercise 3 is to simplify the api call function and create the chat completion function
def chat_completion(prompt_design, prompt):
	MODEL = "gpt-3.5-turbo"
	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": prompt_design},
			{"role": "user", "content": prompt},
		],
		temperature=0,
	)
	return response.choices[0].message.content.strip()

#Challenge 3 - Modifying the rule based chatbot to include the OpenAI API
def ai_chatbot():

	st.title("Rule Based Bot to AI Chatbot")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

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
		
		#remove get_reply function and replace with chat_completion function and pass in two parameters
		response = get_reply(prompt)
		

		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})


# Exercise 4 - Customising the chat completion with streaming
def chat_completion_stream(prompt_design, prompt):
	openai.api_key = return_api_key()
	MODEL = "gpt-3.5-turbo"
	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": prompt_design},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
		stream=True,  # stream option
	)
	return response

#Exercise 4 integration chat completion with streaming into streamlit chat components
def basebot():
	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	# Showing Chat history
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			# set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				# Note the code change in streaming function below
				full_response = ""
				for response in chat_completion_stream("You are a helpful assistant",prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)

# Exercise 5 - Set the prompt design for the chatbot
def prompt_design():

	st.title("Prompt Design")
	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "You are a helpful assistant."
	name = st.text_input("Enter your name:", value="John Doe")
	prompt_design = st.text_input("Enter your the prompt design for the chatbot:", value="You are a helpful assistant.")
	if prompt_design and name:
		st.session_state.prompt_template = prompt_design + f" .You are talking to a person called {name}."
		st.success("Prompt Design: " + prompt_design + " . \n\n You are talking to a person called " + name + ".")

#Challenge 5 - Set the prompt design for the chatbot for the AI Chatbot
#Hint Replace You are a helpful assistant with the prompt design variable #st.session_state.prompt_template
def basebot_prompt_design():
	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	# Showing Chat history
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			# set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# streaming function
				#replace the prompt design "You are a helpful assistant" with the prompt design variable st.session_state.prompt_template
				for response in chat_completion_stream("You are a helpful assistant", prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)

#Exercise 6 - Create a function that returns the memory variables
def return_memory():
	memory = ConversationBufferWindowMemory(k=3)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))

#Challenge 6 - Integrate this function into the chatbot so that the chatbot has memory of the conversation add to prompt_design function
def memory_variables():
	if "memory_variables" not in st.session_state:
		st.session_state.memory_variables = {}
		st.session_state.memory_variables = ConversationBufferWindowMemory(k=3)
	memory = st.session_state.memory_variables.load_memory_variables({})
	return memory['history']


#Challenge 6 - Save the context of the conversation into the memory variables using the save_context function
def basebot_prompt_design_memory():
	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	#include memory variables
	memory = memory_variables()
	memory_context = "\n\n Previous conversation" + memory

	# Showing Chat history
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			# set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# Add teh memory context after the st.session_state.prompt_template
				for response in chat_completion_stream(st.session_state.prompt_template, prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				#modify the code below by calling save_context function found in exercise 6
				#st.session_state.memory_variables ( save context to memory_variables) )
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)

#Using RAG for the chatbot
#Exercise 7 - Showing the capabilties of RAG
def show_rag_results():
	prompt = st.text_input("Enter your search query:")
	if st.button("Search"):
		if st.session_state.vs:
			docs = st.session_state.vs.similarity_search(prompt)
			resource = docs[0].page_content
			source = docs[0].metadata
			st.write("Resource", resource)
			st.write("Source", source)

#Challenge 7 - Integrate RAG into the chatbot, add the RAG search results function to the chatbot where the prompt is the user input

def rag_results(prompt):
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		resource = docs[0].page_content
		source = docs[0].metadata
		results = "\n\nResource from knowledge base " + resource + "\n\n Source: " + source['source']
		return results
	else:
		return "No results found"


#Challenge 7 - Set the prompt design for the chatbot for the AI Chatbot
#Hint Replace You are a helpful assistant with the prompt design variable
def basebot_prompt_design_memory_rag():
	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	# Showing Chat history
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	#include memory variables
	memory = memory_variables()
	memory_context = "\n\n Previous conversation" + memory

	try:
		#
		if prompt := st.chat_input("What is up?"):
			# set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# Call rag_results and pass the prompt variable into the function
				rag = rag_results(prompt)
				# add the rag variable after the memory context
				for response in chat_completion_stream(st.session_state.prompt_template + memory_context , prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				st.session_state.memory_variables.save_context({"input": prompt}, {"output": full_response})	
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)



#Exercise 8 - Creating a database
def create_db():
	# Create or check for the 'database' directory in the current working directory
	# Set DB_NAME to be within the 'database' directory at the top of main.py
	# Connect to the SQLite database
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	# Conversation data table
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS data_test_table (
			id INTEGER PRIMARY KEY,
			date TEXT NOT NULL UNIQUE,
			username TEXT NOT NULL,
			chatbot_ans TEXT NOT NULL,
			user_prompt TEXT NOT NULL,
			tokens TEXT
		)
	"""
	)
	conn.commit()
	conn.close()

#Exercise 8 - Collecting data
def collect(username, chatbot_response, prompt):
	# collect data from bot
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()
	now = datetime.now()  # Using ISO format for date
	tokens = len(chatbot_response + prompt) * 1.3
	cursor.execute(
		"""
		INSERT INTO data_test_table (date, username,chatbot_ans, user_prompt, tokens)
		VALUES (?, ?, ?, ?, ?)
	""",
		(now, username, chatbot_response, prompt, tokens),
	)
	conn.commit()
	conn.close()

#Exercise 8 - Displaying data
# implementing data collection and displaying
def initialise():
	# initialise database first
	create_db()
	# collect some data
	collect("yoda", "I am Yoda. The Force is strong with you", "Who are you?")
	# display data
	# Connect to the specified database
	conn = sqlite3.connect(WORKING_DATABASE)
	cursor = conn.cursor()

	# Fetch all data from data_table
	cursor.execute("SELECT * FROM data_table")
	rows = cursor.fetchall()
	column_names = [description[0] for description in cursor.description]
	df = pd.DataFrame(rows, columns=column_names)
	st.dataframe(df)
	conn.close()



#Challenge 9 - Collect your data from your chatbot
#Hint Replace You are a helpful assistant with the prompt design variable
def basebot_prompt_design_memory_rag_data():
	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	# Showing Chat history
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			# set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				rag = rag_results(prompt)
				# streaming function
				for response in chat_completion_stream(st.session_state.prompt_template + rag, prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				st.session_state.memory_variables.save_context({"input": prompt}, {"output": full_response})
				# modify and add the code below by calling collect function
				#insert here hint collect()
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)

#Exercise 10 - Smart Agent with Tools
def agent_bot():
	st.title("Smart Agent with Tools - One Tool")

	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs,
		return_messages=True,
		memory_key="chat_history",
		output_key="output",
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(
					f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
				):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(
			model_name=st.secrets["default_model"], openai_api_key=return_api_key(), streaming=True
		)
		tools = [DuckDuckGoSearchRun(name="Search")]
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response[
				"intermediate_steps"
			]

#Challenge 10 - Add new tools to the smart agent

@tool("Document search")
def document_search(query: str) -> str:
	# this is the prompt to the tool itself
	"Use this function first to search for documents pertaining to the query before going into the internet"
	if st.session_state.vs:	
		docs = st.session_state.vs.similarity_search(query)
		docs = docs[0].page_content
	else:
		docs = "No results found"
	json_string = json.dumps(docs, ensure_ascii=False, indent=4)
	return json_string

@tool("Wiki search")
def wiki_search(query: str) -> str:
	"Use this function to search for documents in Wikipedia"
	wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
	results = wikipedia.run(query)
	return results

def agent_bot_with_more_tools():
	st.title("Smart Agent with Tools - 2 or more Tools")

	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs,
		return_messages=True,
		memory_key="chat_history",
		output_key="output",
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(
					f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
				):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(
			model_name=st.secrets["default_model"], openai_api_key=return_api_key(), streaming=True
		)
		#Modify the code below to add more tools to the smart agent
		tools = tools = [DuckDuckGoSearchRun(name="Internet Search")]
		
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response[
				"intermediate_steps"
			]


def prototype_application():
	#insert the code
	st.write("Prototype Application")
	pass



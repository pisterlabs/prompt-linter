from PIL import Image
import streamlit as st
import openai
#exercise 11
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#exercise 12
from langchain.memory import ConversationBufferWindowMemory
#exercise 13
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
import os
import tempfile
#exercise 15
import sqlite3
import pandas as pd
from datetime import datetime
#exercise 16
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
#Exercise 17
from langchain.agents import tool
import json
#Exercise 18
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

os.environ["OPENAI_API_KEY"] = st.secrets["openapi_key"]
openai.api_key = st.secrets["openapi_key"]

#Global ex 13
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)
#ex15
DB_NAME = os.path.join(WORKING_DIRECTORY, "default_db")

def template():
	st.subheader("Template")
	st.write("Instruction lines.")

	st.markdown("**:blue[Code]**")
	st.code('''
#exercise code here
''')
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
#challenge code here
''')	  
	st.markdown("**:red[Code Output]**")
	# Actual code here

def class1_prep():
	# st.subheader("Course Slides")
	# st.markdown("https://go.gov.sg/itdchatbotprototype")
	st.divider()
	st.subheader("Pre-workshop Setup")
	# st.divider()
	st.markdown("""1. Visual Studio (VS Code): this is the Integrated Development Environment (IDE) of choice by many coders and will make it easier for us to code our app.""")
	st.markdown("""2. Python (3.10 release or later): this is the coding language of choice for many data science related apps.""")
	st.write("""
			3. Once you have installed the above, we will need to set up a virtual environment and install the libraries in that environment.\n
				Create a folder named “chatbot” anywhere, e.g. in your Desktop.\n 
				Open VS Code and navigate to the folder in a new terminal window.\n 
				Create a virtual environment and activate it by entering the following commands in the terminal:
			 """)
	st.markdown("   **a) Mac**")
	st.code('''
		python3 -m venv venv
		source venv/bin/activate 
			''')
	st.markdown("(You should see a (venv) appear in your terminal window prompt)")
	st.markdown("#")
	st.markdown("   **b) Windows**")
	st.code('''
		python -m venv venv
		cd venv\Scripts
		activate
			''')
	st.markdown("4. While in your virtual environment, install the libraries using pip which should already be installed together with Python:")
	st.code('''
		pip install streamlit openai
			''')
	st.markdown(" To test if Streamlit is installed properly, run this command:")
	st.code('''
	streamlit hello
			''')
	st.markdown(" You should see a Streamlit application running at http://localhost:8501")
	st.markdown(" Type Ctrl + C in VS Code terminal to stop the Streamlit app")
	pass

def class1_hw1():
	st.subheader("My first Hello World app")
	st.divider()
	st.markdown("""1. Create a new file called 'main.py'.""")
	st.markdown("""2. Copy the code below and paste it in the newly created helloworld.py file.""")
	st.markdown("**:blue[Code]**")
	st.code('''
			import streamlit as st
			#my first Hello World app
			st.write("Hello World")
		''')
	st.markdown("Install the watchdog module by running the command below in the terminal.")
	st.code("pip install watchdog")
	st.markdown("Now you don't have to keep restarting the app to see the changes you make to the code. Just refresh the browser.")
	st.write("Save your file and run the app by typing the following command in the terminal:")
	st.code('''
			streamlit run main.py
		''')
	st.markdown("""3. You should see a Streamlit application running at http://localhost:8501""")
	st.markdown("""4. Type Ctrl + C in VS Code terminal to stop the Streamlit app""")
	st.markdown("**:red[Code Output]**")
	st.write("Hello World")
	pass

def objectives():
	st.subheader("Objectives")
	st.markdown("1. Learn how to use Python and Streamlit library to create an interactive web app.")
	st.markdown("2. Learn how to integrate and use OpenAI's API in their streamlit application to create a simple chatbot.")
	st.markdown("3. Learn how to apply basic prompt engineering to enhance the interaction with the chatbot.")

def workshop_outline():
	st.subheader("Outline")
	st.markdown("Part 0: Workshop introduction and rules")
	st.markdown("Part 1: Introduction to Python and Streamlit")
	st.markdown("Part 2: Creating a rule-based chatbot")
	st.markdown("Part 3: Creating a chatbot using OpenAI's API")
	st.markdown("Part 4: Modifying your chatbot with prompt engineering")

def team_introduction():
	st.write("Do introduce yourself to your teammates:\n", "1) name\n", "2) division\n", "3) role")
	st.write("Please also share your favourite Star Wars character and why!")
	image = Image.open('team_introductions.jpeg')
	st.image(image, caption='Hello there!')

def workshop_rules():
	st.subheader("Workshop Rules")
	st.write("1. Ask if you have questions.")
	st.write("2. Be open to different ways to solve the problem.")
	st.write("3. Try. Fail. Learn. Repeat.")
	st.write("4. Seek help from other team members.")
	st.write("5. Collaborate, if possible, for the challenges.")
	st.write("6. Approach facilitators if your team cannot solve the problem.")
	st.write("7. Toilet break is own-time-own-target.")
	st.write("8. Have fun!")

def vscode_ui():
	st.subheader("Navigating the VS Code interface")
	image = Image.open('VSCode_interface.png')
	st.image(image, caption='VS Code UI layout')
	st.markdown("**A: Activity Bar: this is where you can see the different activities you can do in VS Code.**")
	st.markdown("	Explorer: this is where you can see all the files and folders in your project.")
	st.markdown("	Source Control: this is where you can see the changes you have made to your project.")
	st.markdown("	Extensions: this is where you can install extensions to VS Code.")
	st.markdown("	Run and Debug: this is where you can debug your code.")
	st.markdown("**B: Side Bar: this is where you can see the different views of your project.**")
	st.markdown("**C: Editor: this is where you can see the code you have written in your project.**")
	st.markdown("**D: Panel: this is where you can see the different panels you have opened in your project.**")
	st.markdown("	Terminal: this is where you can run commands in your project.")
	st.markdown("	Output: this is where you can see the output of your code.")
	st.markdown("	Problems: this is where you can see the errors in your code.")
	st.markdown("**E. Status Bar: this is where you can see the status of your project.**")

def command_palette_indent():
	st.markdown("Python is very particular about indentation.\nUse the command palette to automatically indent your code.\n\nWindows: Ctrl-Shift-P  \nMac: Command-Shift-P\n\nSelect the option to *Convert Indentation to Tabs*")
	image = Image.open('command_palette_command.png')
	st.image(image, caption='Command Palette auto-indent command')

def final_product():
	st.write("This is what we will working towards and building by the end of the workshop today.")
	st.write("Do try out the chatbot below!")
	st.subheader("**:green[Feel the force! Yoda Chatbot]**")
	image = Image.open('yoda.jpg')
	st.image(image, caption='Master Yoda at your service')
	st.divider()

	openai.api_key = st.secrets["openapi_key"]

	prompt_template = """
	"Speak like Yoda from Star Wars for every question that was asked, 
	do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"
	"""

	if "openai_model" not in st.session_state:
		st.session_state["openai_model"] = "gpt-3.5-turbo"

	if "msg_bot" not in st.session_state:
		st.session_state.msg_bot = []

	for message in st.session_state.msg_bot:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	
	try:

		if prompt := st.chat_input("What is up?"):
			st.session_state.msg_bot.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in openai.ChatCompletion.create(
					model=st.session_state["openai_model"],
					messages=[
								{"role": "system", "content": prompt_template},
								{"role": "user", "content": prompt},
							],
					stream=True,
				):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg_bot.append({"role": "assistant", "content": full_response})

	except Exception as e:
		st.error(e)
	pass

def resources():
	st.subheader("Useful resources and references")
	st.markdown("1. [Streamlit documentation](https://docs.streamlit.io/en/stable/)")
	st.markdown("2. [OpenAI API documentation](https://beta.openai.com/docs/introduction)")
	st.markdown("3. [VS Code documentation](https://code.visualstudio.com/docs)")
	st.markdown("4. [Python documentation](https://docs.python.org/3/)")
	st.markdown("5. [Python cheatsheet](https://www.pythoncheatsheet.org/)")
	st.markdown("6. [Python for beginners](https://www.python.org/about/gettingstarted/)")
	st.markdown("7. [ChatGPT](https://chat.openai.com/) - you can ask ChatGPT to generate code for you!")
	st.markdown("**Notes for this workshop course:**  \n- you may do a single click to copy all the code  \n- challenge code is hidden, click reveal to see the code")
	st.markdown("Python is very particular about indentation.\nUse the command palette to automatically indent your code.\n\nWindows: Ctrl-Shift-P  \nMac: Command-Shift-P\n\nSelect the option to *Convert Indentation to Tabs*")
	image = Image.open('command_palette_command.png')
	st.image(image, caption='Command Palette auto-indent command')

def part1_intro1():
	st.subheader("Streamlit")
	st.markdown("""
		 * an open-source Python library
		 * used extensively for machine learning and data science
		 * helps to create interactive web apps in just a few lines of code
		 * highly flexible and supports complex interactive apps with highly customisable UI
		 * Some real world examples:
		 	* CherGPT in String
		 	* Metacog for CotF MOE
		 	* AILC prototype for MOE
		 """)

def ex1():
	# Exercise 1 : Functions
	st.write("Hello World")
	# only prints the Hello {name} if input box is not empty
	name = st.text_input("Enter your name")
	if name:
		st.write("Hello " + name)

def class1_ex1():
	st.subheader("Exercise 1: Functions")
	st.markdown("In the ***main.py*** file, the code below is already in ***ex1()*** in the ***part1.py*** file.")
	st.write("The code for *helloworld* is inside what you call a Python function.")
	st.write("The ***def main()*** function and ***if _ _ name _ _ == '_ _ main _ _'*** statement are coding conventions for any Python programme.")
	st.write("You need to include an import statement in ***main.py*** to import the ***ex1()*** function from the other file.")
	st.code('''
from part1 import ex1
''')
	st.write("You will need to do the same for all the other exercises and challenges for the rest of the workshop. The code exercises are already in the respective files: ***part1.py, part2.py, part3.py, part4.py***.")
	st.markdown("**:blue[Code]**")
	st.code('''
import streamlit as st

from part1 import ex1
		 	 
#Exercise 1: Functions
def ex1():
	st.write("Hello World")
	name = st.text_input("Enter your name")
	if name:
		st.write("Hello " + name)
	 
def main():
	ex1()
	 
if __name__ == "__main__":
	main()		
	 ''')
	st.markdown("Run the code by typing the following into the terminal:")
	st.code("streamlit run main.py")
	st.markdown("You should see the following behaviour in your browser window:")
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex1()

def ch1():
	name = st.text_input("Enter your name")
	gender = st.selectbox("State your gender", ["Male", "Female"])
	age = st.text_input("State your age", 18)

	if name and gender and age:
		st.text(f"Hello {name}, you are {gender} and this year you are {age} years old")

def class1_ch1():
	st.subheader("Challenge 1: Input, Output and Variables")
	st.write("Create a new function called ***ch1()*** in ***part1.py*** and call it in the main function.")
	st.write("Create three variables *name*, *age* and *gender*, and obtain these from the user.")
	st.write("Once the user filled up the input boxes, display back the information to the user.")
	st.write("Code hint: the following piece of code checks if *name* has been filled, and if so, displays it back to the user.")
	st.code('''
name = st.text_input("Enter your name")
if name:
	st.text(f"Hello {name}")
''')
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
def ch1():
        name = st.text_input("Enter your name")
        gender = st.selectbox("State your gender", ["Male", "Female"])
        age = st.text_input("State your age", 18)

        if name and gender and age:
            st.text(f"Hello {name}, you are {gender} and this year you are {age} years old")
''')	  
	st.markdown("**:red[Code Output]**")
	# Actual code here
	ch1()

def ex2():
    gender = st.selectbox("State your gender", ["Male", "Female"])
    age = int(st.text_input("State your age", 18))
    photo = st.camera_input("Smile! take a picture here.")

    # conditional logic to run different statements
    if age >= 21 and gender == "Male":
        st.write("You are a male adult")
    elif age < 21 and gender == "Male":
        st.write("You are a young boy")
    elif age >= 21 and gender == "Female":
        st.write("You are a female adult")
    elif age < 21 and gender == "Female":
        st.write("You are a young girl")

    if photo:
        st.write("Here is your photo: ")
        st.image(photo)
    else:
        st.write("No photo taken")

def class1_ex2():
	st.subheader("Exercise 2: If-else logical conditionals")
	st.markdown("If-else statements help control the flow and logic of our application.")
	st.markdown("**:blue[Code]**")
	st.code('''
def ex2():
    gender = st.selectbox("State your gender", ["Male", "Female"])
    age = int(st.text_input("State your age", 18))
    photo = st.camera_input("Smile! take a picture here.")

    # conditional logic to run different statements
    if age >= 21 and gender == "Male":
        st.write("You are a male adult")
    elif age < 21 and gender == "Male":
        st.write("You are a young boy")
    elif age >= 21 and gender == "Female":
        st.write("You are a female adult")
    elif age < 21 and gender == "Female":
        st.write("You are a young girl")

    if photo:
        st.write("Here is your photo: ")
        st.image(photo)
    else:
        st.write("No photo taken")
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex2()

def ex3():
	# Data list
	fruits = ["apple", "banana", "orange"]

	# For loop to show list
	for fruit in fruits:
		st.write(fruit)

	# Dictionary
	person = {"name": "John", "age": 30, "gender": "Male", "city": "New York"}

	# Print out the items in the dictionary
	st.write("Here is your *person* dictionary: ")
	st.write(person)

	# for loop to show dictionary list
	st.write("You can also show individual items in the dictionary like this: ")
	for key, value in person.items():
		st.write(key + ": " + str(value))

	# get user input to update the dictionary
	name = st.text_input("Enter your name", "John")
	age = st.text_input("State your age", 30)
	gender = st.selectbox("State your gender", ["Male", "Female"])
	city = st.text_input("State your city", "New York")
	person["name"] = name
	person["age"] = age
	person["gender"] = gender
	person["city"] = city
	
	st.write("Here is your updated *person* dictionary: ")
	st.write(person)

def class1_ex3():
	st.subheader("Exercise 3: Data and Loops ")
	st.write("We can store data in a list or dictionary and display the data using a for loop.")
	st.write("Append the following code to the ***main.py*** file. Refresh the browser to see the changes.")
	st.write("You should see output similar to the *Code Output* below.")
	st.markdown("**:blue[Code]**")
	st.code('''
#Data and Loops
def ex3():
	# Data list
	fruits = ["apple", "banana", "orange"]

	# For loop to show list
	for fruit in fruits:
		st.write(fruit)

	# Dictionary
	person = {"name": "John", "age": 30, "city": "New York"}

	# Print out the items in the dictionary
	st.write("Here is your *person* dictionary: ")
	st.write(person)

	# for loop to show dictionary list
	st.write("You can also show individual items in the dictionary like this: ")
	for key, value in person.items():
		st.write(key + ": " + str(value))

	# get user input to update the dictionary
	name = st.text_input("Enter your name", "John")
	age = st.text_input("State your age", 30)
	gender = st.selectbox("State your gender", ["Male", "Female"])
	city = st.text_input("State your city", "New York")
	person["name"] = name
	person["age"] = age
	person["gender"] = gender
	person["city"] = city
	
	st.write("Here is your updated *person* dictionary: ")
	st.write(person)
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex3()

def ex4a():
	#Exercise 4a: Creating Session Data
	if "session_data" not in st.session_state:
		st.session_state.session_data = ["alpha", "omega"]
	
	if "name" not in st.session_state:
		st.session_state.name = ""
	
	if "age" not in st.session_state:
		st.session_state.age = ""

	if "gender" not in st.session_state:
		st.session_state.gender = ""
	
	# For loop to show list
	for data in st.session_state.session_data:
		st.write("session_data: ", data)

	st.write("name: ", st.session_state.name)
	st.write("age: ", st.session_state.age)
	st.write("gender: ", st.session_state.gender)

def class1_ex4a():
	st.subheader("Exercise 4a: Session Data")
	st.write("We can create variables to store data in a user session. Session data persist within a user session.")

	st.markdown("**:blue[Code]**")
	st.code('''
# Exercise 4: Session State
def ex4a():
	st.subheader("Session Data:")
	if "session_data" not in st.session_state:
		st.session_state.session_data = ["alpha", "omega"]
	
	if "name" not in st.session_state:
		st.session_state.name = ""
	
	if "age" not in st.session_state:
		st.session_state.age = ""

	if "gender" not in st.session_state:
		st.session_state.gender = ""
	
	# For loop to show list
	for data in st.session_state.session_data:
		st.write("session_data: ", data)

	st.write("name: ", st.session_state.name)
	st.write("age: ", st.session_state.age)
	st.write("gender: ", st.session_state.gender)
''')
		 
	st.markdown("**:red[Code Output]**")
	ex4a()

def ex4b():
	#Exercise 4b: Session Data with User Input

	user_name = st.text_input("Enter your name")
	user_age = st.text_input("State your age")
	user_gender = st.selectbox("State your gender", ["", "Male", "Female"])

	if user_name:
		st.session_state.name = user_name
		st.write("name: ", st.session_state.name)
	if user_age:
		st.session_state.age = int(user_age)
		st.write("age: ", st.session_state.age)
	if user_gender:
		st.session_state.gender = user_gender
		st.write("gender: ", st.session_state.gender)

def class1_ex4b():
	st.subheader("Exercise 4b: Session Data with User Input")
	st.write("Lets now get input from the user and store it in the session data.")
	st.write("Now run *ex4a()* again to check the session data. Note that it persists.")

	st.markdown("**:blue[Code]**")
	st.code('''
def ex4b():
	st.subheader("Session Data:")
	userName = st.text_input("Enter your name")
	userAge = st.text_input("State your age")
	userGender = st.selectbox("State your gender", ["", "Male", "Female"])

	if userName:
		st.session_state.name = userName
		st.write("name: ", st.session_state.name)
	if userAge:
		st.session_state.age = int(userAge)
		st.write("age: ", st.session_state.age)
	if userGender:
		st.session_state.gender = userGender
		st.write("gender: ", st.session_state.gender)
''')
	st.markdown("**:red[Code Output]**")
	ex4b()

def ch4():
	if "name" not in st.session_state:
		st.session_state.name = "Yoda"

	if "age" not in st.session_state:
		st.session_state.age = 999

	if "gender" not in st.session_state:
		st.session_state.gender = "male"

	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"

	st.write("session_state.name: ", st.session_state.name)
	st.write("session_state.age: ", st.session_state.age)
	st.write("session_state.gender: ", st.session_state.gender)
	st.write("session_state.prompt_template: ", st.session_state.prompt_template)

def class1_ch4():
	st.subheader("Challenge 4: Session Data")
	st.markdown("""
		 Add a new function called ***ch4()*** to the ***part1.py*** file and call it in the main function.\n
		 In *ch4()*, modify the code in Exercise 4b to include the following:
		 * Create session data for ***name***, ***age*** and ***gender***
		 * Create session data for ***prompt_template*** with the following value:
			 "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"
		 * Include this code in ***main()*** as well, because we need the session data for later exercises. Omit the ***st.write*** functions, since we do not want to see this output every time we run ***main()***. \n
		 Hint:
		 * In ***ch4()***, to check that the session data is created, you can print out the session data using ***st.write()***:
		 """)
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
def ch4():
	if "name" not in st.session_state:
		st.session_state.name = "Yoda"

	if "age" not in st.session_state:
		st.session_state.age = 999

	if "gender" not in st.session_state:
		st.session_state.gender = "male"

	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"

	st.write("session_state.name: ", st.session_state.name)
	st.write("session_state.age: ", st.session_state.age)
	st.write("session_state.gender: ", st.session_state.gender)
	st.write("session_state.prompt_template: ", st.session_state.prompt_template)

def main():
	# initialize session state, from ch4
	if "name" not in st.session_state:
		st.session_state.name = "Yoda"

	if "age" not in st.session_state:
		st.session_state.age = 999

	if "gender" not in st.session_state:
		st.session_state.gender = "male"

	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"
		  
	#the rest of your code is below
''')
		  
	st.markdown("**:red[Code Output]**")
	#actual code here
	ch4()

def ex5():
	st.title("My first chatbot")

	if "store_msg" not in st.session_state:
		st.session_state.store_msg = []

	prompt = st.chat_input("Say something")
	if prompt:
		st.write(f"User has sent the following prompt: {prompt}")
		st.session_state.store_msg.append(prompt)
		for message in st.session_state.store_msg:
			with st.chat_message("user"):
				st.write(message)
			with st.chat_message("assistant"):
				st.write("Hello human, what can I do for you?")

def class1_ex5():
	st.subheader("Exercise 5: Elements of a chatbot")
	st.write("We will start creating a user interface for our first chatbot.")
	st.write("Call the following code from ***part1.py*** in your **main()**.")
	st.write("You should see the output below when you run your programme.")
	st.markdown("**:blue[Code]**")
	st.code('''
#Exercise 5 : Chatbot UI
def ex5():
	st.title("My first chatbot")

	if "store_msg" not in st.session_state:
		st.session_state.store_msg = []

	prompt = st.chat_input("Say something")
	if prompt:
		st.write(f"User has sent the following prompt: {prompt}")
		st.session_state.store_msg.append(prompt)
		for message in st.session_state.store_msg:
			with st.chat_message("user"):
				st.write(message)
			with st.chat_message("assistant"):
				st.write("Hello human, what can I do for you?")
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex5()

def ex6():
	st.markdown("**Echo Bot**")

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

		response = f"Echo: {prompt}"
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})
		
def class1_ex6():
	st.subheader("Exercise 6: Building a simple echo chatbot")
	st.write("We will now build a simple echo chatbot.")
	st.write("Call the following code from **part1.py** in your ***main()***.")
	st.write("You should see the output below when you run your programme.")
	st.markdown("**:blue[Code]**")
	st.code('''
#Exercise 6 : Rule-based Echo Chatbot 
def ex6():
	st.title("Echo Bot")

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

		response = f"Echo: {prompt}"
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex6()

def ch6():
	st.markdown("**Rule Based Bot**")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# # Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("Enter your query"):
		if prompt == "Hello":
			reply = "Hi there what can I do for you"

		elif prompt == "What is your name?":
			reply = "My name is EAI , an electronic artificial being"

		elif prompt == "How old are you?":
			reply = "Today is my birthday!"
		
		else:
			reply = "I am sorry, I am unable to help you with your query"

		with st.chat_message("user"):
			st.write(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("assistant"):
			st.write(reply)
			st.session_state.messages.append({"role": "assistant", "content": reply})

def class1_ch6():
	st.subheader("Challenge 6: Rule based chatbot ")
	st.markdown("""
		 Create a new function called ***ch6()*** in **part1.py** and modify the ***ex6()*** function to create the following rule based chatbot:\n
		 * Human : “Hello”,  Assistant: “Hi there what can I do for you”\n
		 * Human : “What is your name?”,  Assistant: “My name is EAI , an electronic artificial being”\n	
		 * Human : “How old are you?”,  Assistant: “Today is my birthday!”\n
		 For other queries, it will reply “I am sorry, I am unable to help you with your query”\n
		 Use *if / elif / else* statements to create the chatbot behaviour logic.\n 
		 You should see the output below when you run your programme.\n
		 """)
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
#Challenge 6 : Rule-based If-Else Chatbot
def ch6():
	st.markdown("**Rule Based Bot**")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# # Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("Enter your query"):
		if prompt == "Hello":
			reply = "Hi there what can I do for you"

		elif prompt == "What is your name?":
			reply = "My name is EAI , an electronic artificial being"

		elif prompt == "How old are you?":
			reply = "Today is my birthday!"
		
		else:
			reply = "I am sorry, I am unable to help you with your query"

		with st.chat_message("user"):
			st.write(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("assistant"):
			st.write(reply)
			st.session_state.messages.append({"role": "assistant", "content": reply})
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ch6()

def class1_ex7():
	st.subheader("Exercise 7: Secrets- Shhh ")
	st.write("In this exercise, we will learn how to hide your API key")
	st.markdown("""
	In your working directory (chatbot), create a directory called **.streamlit**\n
	Note the *dot* in front of the directory\n
	In this folder, create a file called **secrets.toml**\n
	Get an API key from your OpenAI account and type the following in **secrets.toml**:
	""")
	st.markdown("**:blue[Code]**")
	st.code('''
	openapi_key = "xxxxxx"
	''')
	st.write("Include the following global variables in your ***main.py*** file under the import statements:")
	st.code('''
os.environ["OPENAI_API_KEY"] = st.secrets["openapi_key"]
openai.api_key = st.secrets["openapi_key"]
''')
	st.write("Create a .gitignore file and add .streamlit into it")
	#st.markdown("**:red[Code Output]**")
	pass

def class1_ch7():
	pass

def ex8():
	st.title("Api Call")
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Tell me about Singapore in the 1970s in 50 words."},
		],
		temperature=0,
	)
	st.markdown("**This is the raw response:**") 
	st.write(response)
	st.markdown("**This is the extracted response:**")
	st.write(response["choices"][0]["message"]["content"].strip())
	s = str(response["usage"]["total_tokens"])
	st.markdown("**Total tokens used:**")
	st.write(s)

def class1_ex8():
	st.subheader("Exercise 8: Calling the OpenAI LLM API")
	st.write("In this exercise, we will learn how to call the OpenAI LLM API")
	st.write("Note that there is a new import statement **import openai**")
	st.markdown("""
		 Call the following code in your **main()** and run it.\n
		 You should see the output as shown below.\n
		 """)
	st.markdown("**:blue[Code]**")
	st.code('''
import openai
	 
#Exercise 8 : Using the OpenAI API
def ex8():
	st.title("Api Call")
	MODEL = "gpt-3.5-turbo"

	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Tell me about Singapore in the 1970s in 50 words."},
		],
		temperature=0,
	)

	st.markdown("**This is the raw response:**") 
	st.write(response)
	st.markdown("**This is the extracted response:**")
	st.write(response["choices"][0]["message"]["content"].strip())
	s = str(response["usage"]["total_tokens"])
	st.markdown("**Total tokens used:**")
	st.write(s)
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex8()

def chat_completion(prompt):
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt},
		],
		temperature=0,
	)
	return response["choices"][0]["message"]["content"].strip()

def class1_ch8():
	st.subheader("Challenge 8: Incorporate your LLM API call into your chatbot")
	st.write("In this challenge, we will incorporate the LLM API call into our previous rule-based *Echo* chatbot")
	st.markdown("""
**Step1**\n
Create a new function **ch8()** in ***part1.py*** and copy the code from **ex6()** into it. Recall that **ex6()** shows the chat history and gets a chat input from the user, and echoes the user input back to the user. \n
**Step 2**\n
Next, copy the code from **ex8** into a new function named **chat_completion()**. Recall that **ex8()** is about making an API call.\n
Now, instead of echoing the user's input in **ex6()**, we will call the LLM API to generate a response. In particular, we are replacing this line of code with the response from the API call:\n
""")
	st.code('''
response = f"Echo: {prompt}"
''')
	st.markdown("""
**Step 3**\n
In **chat_completion()**, what we will do is to replace the previous *Tell me the history ..."* prompt from **ex8()** with the current user's input.\n
In order to do so, in **ch8()**, use the following code to call **chat_completion()**.\n
What we are doing now is to pass the prompt from the user to the API call instead of hard-coding the prompt as in **ex8()**.\n""")
	st.code('''
	 if prompt := st.chat.input("What's up?"):
	 	#display user messgae in chat message container
	 	reply = chat_completion(prompt) 
	 	st.chat_message("user").markdown(prompt)
		''')
	st.write("You should see the code output as shown below.")
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''	
#Challenge 8: Incorporating the API into your chatbot
def chat_completion(prompt):
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt},
		],
		temperature=0,
	)
	return response["choices"][0]["message"]["content"].strip()
	
def ch8():
	st.title("My first LLM Chatbot")

	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	# Display chat chat_msg from history on app rerun
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("What's up?"):
		# Display user message in chat message container
		reply = chat_completion(prompt)
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.chat_msg.append({"role": "user", "content": prompt})
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(reply)
		# Add assistant response to chat history
		st.session_state.chat_msg.append({"role": "assistant", "content": reply})
''')
	st.markdown("**:red[Code Output]**")
	st.title("My LLM Chatbot")

	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	# Display chat chat_msg from history on app rerun
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("What is up?"):
		# Display user message in chat message container
		reply = chat_completion(prompt)
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.chat_msg.append({"role": "user", "content": prompt})
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(reply)
		# Add assistant response to chat history
		st.session_state.chat_msg.append({"role": "assistant", "content": reply})

def chat_completion_stream(prompt):
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
		stream=True,  # stream option
	)
	return response

def ex9_basebot():
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
				for response in chat_completion_stream(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)
	except Exception as e:
		st.error(e)

def class1_ex9():
	st.subheader("Exercise 9: Building a ChatGPT-like clone with streaming responses")
	st.write("Now, we will incorporate a streaming response from the LLM API into our chatbot to mimic the behaviour of ChatGPT.")
	st.write("Copy and run the code below to see the streaming responses.")
	st.markdown("**:blue[Code]**")
	st.code('''
# Exercise 9 : Using the OpenAI API with streaming option
def chat_completion_stream(prompt):
	openai.api_key = st.secrets["openapi_key"]
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
		stream=True,  # stream option
	)
	return response

# integration API call into streamlit chat components
def ex9_basebot():
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
				for response in chat_completion_stream(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)
''')
	st.markdown("**:red[Code Output]**")
	ex9_basebot()

def class1_ch9():
	pass

def ex10():
	# prompt_template in session state already set in main()
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template},
			{
				"role": "user",
				"content": "Tell me about Singapore in the 1970s in 50 words",
			},
		],
		temperature=0,
	)
	st.markdown("**LLM Response:**")
	st.write(response["choices"][0]["message"]["content"].strip())
	st.markdown("**Total tokens:**")
	st.write(str(response["usage"]["total_tokens"]))

def class1_ex10():
	st.subheader("Exercise 10: Basic Prompt Engineering")
	st.markdown("""
		 Now, we are going to create a chatbot with a personality by using a default prompt for our chatbot. \n
		 This is the default prompt that will be used for every conversation.\n
		 Let's make it a chatbot that speaks like Yoda from Star Wars.\n
		 We will use the ***prompt_template*** that is already in our ***main()*** for this.
		 """)
	st.code('''
if "prompt_template" not in st.session_state:
	st.session_state.prompt_template = "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"
	''')
	st.markdown("""
		 Run the code below. You should get the same chatbot behaviour as the code output below.\n
		 Try varying the temperature setting (0.0 to 1.0) to see how it affects the chatbot's response.\n
		 """)
	st.markdown("**:blue[Code]**")
	st.code('''
# Exercise 10: Basic prompt engineering
def ex10_basebot():
	#prompt_template in session state already set in main()
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template},
			{
				"role": "user",
				"content": "Tell me about Singapore in the 1970s in 50 words",
			},
		],
		temperature=0,
	)
	st.markdown("**LLM Response:**")
	st.write(response["choices"][0]["message"]["content"].strip())
	st.markdown("**Total tokens:**")
	st.write(str(response["usage"]["total_tokens"]))
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex10()

#Challenge 10
#mod chat complete stream function by replacing system content to session_state prompt template
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

# Challenge 10: Make the bot speak like someone you know
def ch10():
	#call the function in your base bot
	#Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	#Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			#set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})

	except Exception as e:
		st.error(e)

def class1_ch10():
	st.subheader("Challenge 10: Make your bot like someone you know!")
	st.write("Modify the ***prompt_template*** in your ***main()*** to your own liking.")
	st.write("Be imaginative!")
	st.write("Now, in new function called **chat_completion_stream_prompt()**, we are going to modify the earlier **streaming chat_completion** function to accept a user prompt input.")
	st.write("You will need to pass in a new input variable called **prompt** and replace the user content with the new **prompt** variable.")
	st.write("Replace the system prompt with **st.session_state.prompt_template**.")
	st.write("Before calling **chat_completion_stream_prompt()**, get a new prompt from the user like this to update the **st.session_state.prompt_template**:")
	st.code('''
if my_prompt_template := st.text_input("Enter a system prompt template. E.g. Speak like Yoda from Star Wars."):
	st.session_state.prompt_template = my_prompt_template
	st.write("new prompt template set! ", st.session_state.prompt_template)
''')
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
#Challenge 10
#mod chat complete stream function by replacing system content to session_state prompt template
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

# Challenge 10: Make the bot speak like someone you know
def ch10_basebot():
	# call the function in your base bot
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
			#set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})

	except Exception as e:
		st.error(e)
''')
	st.markdown("**:red[Code Output]**")
	st.title("ChatGPT-like clone with Prompt Engineering")
	ch10()

#https://python.langchain.com/docs/modules/chains/
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
		
def class1_ex11a():
	st.subheader("Exercise 11a: Prompt Template with LangChain")
	st.write("LangChain helps you to create a more complext prompt template for your chatbot.")

	st.markdown("**:blue[Code]**")
	st.code('''
#https://python.langchain.com/docs/modules/chains/
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
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex11a()

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

def class1_ex11b():
	st.subheader("Exercise 11b")
	st.write("Now, we will create a chatbot with a prompt template that is more complex.")
	st.write("We will use the ***prompt_inputs_form()*** function to get the user's input for the prompt template.")
	st.write("Run the code below to see the chatbot in action.")

	st.markdown("**:blue[Code]**")
	st.code('''
def prompt_inputs_form():  # Using st.form, create the starting prompt to your prompt template, this is an expert on a topic that is talking to a user of a certain age
	# langchain prompt template
	with st.form("Prompt Template"):
		occupation = st.text_input("Enter the occupation:")
		topic = st.text_input("Enter the topic:")
		age = st.text_input("Enter the age:")

		# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
	# return a dictionary of the values
	if submitted:
		return {"occupation": occupation, "topic": topic, "age": age}

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
''')
	
	st.markdown("**:red[Code Output]**")
	# Actual code here
	ex11b()

def ch11():
	# instead of running of the langchain, we are going to use the prompt template and run it the chatbot using format
	prompt_template = PromptTemplate(
		input_variables=["occupation", "topic", "age"],
		template="""Imagine you are a {occupation} who is an expert on the  topic of {topic} , you are going to help , teach and provide information to the person who is {age} years old, if you do not not know the answer, you must tell the person , do not make any answer up""",
	)
	dict_inputs = prompt_inputs_form()
	if dict_inputs:
		input_prompt = prompt_template.format(
			occupation=dict_inputs["occupation"],
			topic=dict_inputs["topic"],
			age=dict_inputs["age"],
		)
		# set session_state.prompt_template
		st.session_state.prompt_template = input_prompt
		st.write("New session_state.prompt_template: ", input_prompt)
	# call ch10() with the new session_state.prompt_template
	ch10()

def class1_ch11():
	st.subheader("Challenge 11: Prompt Template with LangChain")
	st.write("Now, let's incorporate the prompt template into our chatbot from the previous exercise.")
	st.write("We will use the ***prompt_inputs_form()*** function to get the user's input for the prompt template.")
	st.write("You can use the ***ch10*** function from the previous exercise to do the llm api call with the updated session_state.prompt_template.")
	st.write("Ignore the text input field that asks for a system prompt template from ch10(), since we will be using the prompt template from the user's input.")
	st.write("As you interact with the chatbot, observe that the prompt template is updated with the latest user input as seen from the code output.")

	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
def ch11():
	# instead of running of the langchain, we are going to use the prompt template and run it the chatbot using format
	prompt_template = PromptTemplate(
		input_variables=["occupation", "topic", "age"],
		template="""Imagine you are a {occupation} who is an expert on the  topic of {topic} , you are going to help , teach and provide information to the person who is {age} years old, if you do not not know the answer, you must tell the person , do not make any answer up""",
	)
	dict_inputs = prompt_inputs_form()
	if dict_inputs:
		input_prompt = prompt_template.format(
			occupation=dict_inputs["occupation"],
			topic=dict_inputs["topic"],
			age=dict_inputs["age"],
		)
		# set session_state.prompt_template
		st.session_state.prompt_template = input_prompt
		st.write("New session_state.prompt_template: ", input_prompt)
	# call the ch10() basebot with the new session_state.prompt_template
	ch10()
''')
	st.markdown("**:red[Code Output]**")
	# actual code here
	ch11()

def ex12():
	memory = ConversationBufferWindowMemory(k=3)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))
   
	memory = ConversationBufferWindowMemory( k=3, return_messages=True)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))

def class1_ex12():
	st.subheader("Exercise 12: Chatbot with memory")
	st.write("Now, we will create a chatbot with memory.")
	st.write("You can determine the number of previous messages to remember by setting the ***k*** parameter.")

	st.markdown("**:blue[Code]**")
	st.code('''
def ex12():
	memory = ConversationBufferWindowMemory(k=3)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))
   
	memory = ConversationBufferWindowMemory( k=3, return_messages=True)
	memory.save_context({"input": "hi"}, {"output": "whats up?"})
	memory.save_context({"input": "not much"}, {"output": "what can I help you with?"})

	st.write(memory.load_memory_variables({}))
''')  
	st.markdown("**:red[Code Output]**")
	#actual code here
	ex12()

def ch12():
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
	else:
		input_prompt = "You are a helpful assistant. "

	st.write("input prompt: ", input_prompt)

	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=3)

	# step 1 save the memory from your chatbot
	# step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write("Memory Data: ", memory_data)
	st.session_state.prompt_template = f"""
{input_prompt}										

Below is the conversation history between the AI and Users so far
										
{memory_data}

"""
	st.write("New prompt template:", st.session_state.prompt_template)
	# call the function in your base bot
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

def class1_ch12():
	st.subheader("Challenge 12: Chatbot with memory")
	st.write("Now, let's incorporate the memory into the session state prompt template.")
	st.write("The chatbot should remember the previous user input and use it as the prompt template for the next conversation.")
	st.write("Start with the following code and modify ex12() to create a chatbot with memory.")
	st.write("Get the *{input_prompt}* using *prompt_inputs_form()*.")
	st.write("As you interact with the chatbot, observe that the memory is updated with the latest k number of user input and output as seen from the code output.")
	st.markdown("**:blue[Code]**")
	st.code('''
if "memory" not in st.session_state:
	st.session_state.memory = ConversationBufferWindowMemory(k=5)

	#step 1 save the memory from your chatbot 
	#step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) 
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write(memory_data)
	st.session_state.prompt_template = f"""{input_prompt}\n\nBelow is the conversation history between the AI and Users so far\n\n{memory_data}"""
''')
	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
def ch12():
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
	else:
		input_prompt = "You are a helpful assistant. "

	st.write("input prompt: ", input_prompt)

	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=3)

	# step 1 save the memory from your chatbot
	# step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write("Memory Data: ", memory_data)
	st.session_state.prompt_template = f"""
{input_prompt}										

Below is the conversation history between the AI and Users so far
										
{memory_data}

"""

	st.write("New prompt template: ", st.session_state.prompt_template)
	# call the function in your base bot
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
''')
	st.markdown("**:red[Code Output]**")
	#actual code here
	ch12()

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

def ex13_vectorstore_creator():
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

def class1_ex13():
	st.subheader("Exercise 13: Create a vector store")
	st.write("Now, we will create a vector store to store the user's document.")
	st.write("This process uses OpenAI to generate embeddings and LanceDB for storing these embeddings.")
	st.write("For now, this only works for pdf files.")
	st.write("You may need to run the following commands in terminal to install new libaries:")
	st.code('''
pip install tiktoken
''')
	st.markdown("**:blue[Code]**")
	st.code('''
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

def ex13_vectorstore_creator():
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
''')
		  
	st.markdown("**:red[Code Output]**")
	ex13_vectorstore_creator()

# save the vectorstore in st.session_state
# add semantic search prompt into memory prompt
# integrate back into your chatbot
def ex14():
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

def class1_ex14():
	st.subheader("Exercise 14: Semantic search")
	st.write("In this exercise. we will do a semantic search on the vector store in our chatbot.")
	st.write("At the same time, the chatbot is able to remember its conversation history to some extent.")
	st.write("This code integrates advanced features like semantic search and context-aware prompts to provide a more engaging and helpful conversational experience.")
	st.write("Copy and run the code below to see the chatbot in action.")

	st.markdown("**:blue[Code]**")
	st.code('''
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
		{st.session_state.input_prompt}

		This is the last conversation history:
		{memory_data}"""
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
				vs_prompt = f"""
					You should reference this search result to help your answer,
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
''')

	st.markdown("**:red[Code Output]**")
	#actual code here
	ex14()

def ex15_initialise():
	# Create or check for the 'database' directory in the current working directory
	# Set DB_NAME to be within the 'database' directory at the top of main.py
	# Connect to the SQLite database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Conversation data table
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS data_table (
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

def ex15_collect(username, chatbot_response, prompt):
	# collect data from bot
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()
	now = datetime.now()  # Using ISO format for date
	tokens = len(chatbot_response) * 1.3
	cursor.execute(
		"""
		INSERT INTO data_table (date, username,chatbot_ans, user_prompt, tokens)
		VALUES (?, ?, ?, ?, ?)
	""",
		(now, username, chatbot_response, prompt, tokens),
	)
	conn.commit()
	conn.close()

# implementing data collection and displaying
def ex15():
	# initialise database first
	ex15_initialise()
	# collect some data
	ex15_collect("yoda", "I am Yoda. The Force is strong with you", "Who are you?")
	# display data
	# Connect to the specified database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Fetch all data from data_table
	cursor.execute("SELECT * FROM data_table")
	rows = cursor.fetchall()
	column_names = [description[0] for description in cursor.description]
	df = pd.DataFrame(rows, columns=column_names)
	st.dataframe(df)
	conn.close()

def class1_ex15():
	st.subheader("Exercise 15: Using a database")
	st.write("In this exercise, we will demonstrate how to create a database, as well as how to store and retrieve data from it.")
	st.markdown("**:blue[Code]**")
	st.code('''
def ex15_initialise():
	# Create or check for the 'database' directory in the current working directory
	# Set DB_NAME to be within the 'database' directory at the top of main.py
	# Connect to the SQLite database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Conversation data table
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS data_table (
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

def ex15_collect(username, chatbot_response, prompt):
	# collect data from bot
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()
	now = datetime.now()  # Using ISO format for date
	tokens = len(chatbot_response) * 1.3
	cursor.execute(
		"""
		INSERT INTO data_table (date, username,chatbot_ans, user_prompt, tokens)
		VALUES (?, ?, ?, ?, ?)
	""",
		(now, username, chatbot_response, prompt, tokens),
	)
	conn.commit()
	conn.close()

# implementing data collection and displaying
def ex15():
	# initialise database first
	ex15_initialise()
	# collect some data
	ex15_collect("yoda", "I am Yoda. The Force is strong with you", "Who are you?")
	# display data
	# Connect to the specified database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Fetch all data from data_table
	cursor.execute("SELECT * FROM data_table")
	rows = cursor.fetchall()
	column_names = [description[0] for description in cursor.description]
	df = pd.DataFrame(rows, columns=column_names)
	st.dataframe(df)
	conn.close()
''')
	st.markdown("**:red[Code Output]**")
	# Actual code here
	ex15()

def ch15_chatbot():
	#display ex15 table
	ex15()
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
	st.session_state.prompt_template = f"""{st.session_state.input_prompt}
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

			ex15_collect(st.session_state.name, full_response, prompt)

	except Exception as e:
		st.error(e)

def class1_ch15():
	st.subheader("Challenge 15: Using a database")
	st.write("For this challenge, we will incorporate using a database from our previous exercise.")
	st.write("Copy the code from ***ex14()*** and use the ***ex15()*** to collect and display the data.")

	st.markdown("**:blue[Code]**")
	with st.expander("Reveal Code"):
		st.code('''
def ch15_chatbot():
	#display ex15 table
	ex15()
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
	st.session_state.prompt_template = f"""{st.session_state.input_prompt}
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

			ex15_collect(st.session_state.name, full_response, prompt)

	except Exception as e:
		st.error(e)
''')
		  
	st.markdown("**:red[Code Output]**")
	# Actual code here
	ch15_chatbot()

# smart agents accessing the internet for free
# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/search_and_chat.py
def ex16():
	st.title("🦜 LangChain: Chat with internet search")

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
			model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, streaming=True
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

def class1_ex16():
	st.subheader("Exercise 16: Smart agent")
	st.write("In this exercise, we will configure a chatbot with an internet search tool that shows all intermediate steps and tool logs.")
	st.write("This overcomes the limitation of the training data that is only up to a certain point in time, by being able to access the current internet to search for answers.")

	st.write("You may need to run the following commands in terminal to install new libaries:")
	st.code('''
pip install duckduckgo-search
''')
	
	st.markdown("**:blue[Code]**")
	st.code('''
# smart agents accessing the internet for free
# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/search_and_chat.py
def ex16_agent_bot():
	st.title("🦜 LangChain: Chat with internet search")

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
			model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
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
''')
	st.markdown("**:red[Code Output]**")
	# Actual code here
	ex16()

# agents ,vectorstores, wiki
# https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval
# note tool
@tool("Document search")
def document_search(query: str) -> str:
	# this is the prompt to the tool itself
	"Use this function first to search for documents pertaining to the query before going into the internet"
	docs = st.session_state.vectorstore.similarity_search(query)
	docs = docs[0].page_content
	json_string = json.dumps(docs, ensure_ascii=False, indent=4)
	return json_string

# combine vector store and internet search
def ex17():
	st.title("🦜 LangChain: Chat with internet search")

	st.session_state.vectorstore = vectorstore_creator()

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
			model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, streaming=True
		)
		tools = [document_search, DuckDuckGoSearchRun(name="Internet Search")]
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

def class1_ex17():
	st.subheader("Exercise 17: Smart agent with vector store")
	st.write("In this exercise, we will combine the vector store with the smart agent.")
	st.write("This allows the chatbot to search for answers from the vector store and the internet.")
	st.write("The @tool(\"Document search\") function is an enhancement to the chatbot. It allows for an initial internal document search based on the user query before resorting to external internet searches. ")
	st.write("Copy and run the code below to see the chatbot in action.")

	st.markdown("**:blue[Code]**")
	st.code('''
# agents ,vectorstores, wiki
# https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval
# note tool
@tool("Document search")
def document_search(query: str) -> str:
	# this is the prompt to the tool itself
	"Use this function first to search for documents pertaining to the query before going into the internet"
	docs = st.session_state.vectorstore.similarity_search(query)
	docs = docs[0].page_content
	json_string = json.dumps(docs, ensure_ascii=False, indent=4)
	return json_string

# combine vector store and internet search
def ex17_agent_bot():
	st.title("🦜 LangChain: Chat with internet search")

	st.session_state.vectorstore = vectorstore_creator()

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
			model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
		)
		tools = [document_search, DuckDuckGoSearchRun(name="Internet Search")]
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
''')
	st.markdown("**:red[Code Output]**")
	# Actual code here
	ex17()

# PandasAI- A smart agent that can do visual analytics
def ex18():
	st.title("pandas-ai streamlit interface")

	# Upload CSV file using st.file_uploader
	uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	if "openai_key" not in st.session_state:
		st.session_state.openai_key = st.secrets["openapi_key"]
		st.session_state.prompt_history = []
		st.session_state.df = None

	if uploaded_file is not None:
		try:
			df = pd.read_csv(uploaded_file)
			st.session_state.df = df
		except Exception as e:
			st.write("There was an error processing the CSV file.")
			st.write(e)

	else:
		st.session_state.df = pd.DataFrame(
			{
				"country": [
					"United States",
					"United Kingdom",
					"France",
					"Germany",
					"Italy",
					"Spain",
					"Canada",
					"Australia",
					"Japan",
					"China",
				],
				"gdp": [
					19294482071552,
					2891615567872,
					2411255037952,
					3435817336832,
					1745433788416,
					1181205135360,
					1607402389504,
					1490967855104,
					4380756541440,
					14631844184064,
				],
				"happiness_index": [
					6.94,
					7.16,
					6.66,
					7.07,
					6.38,
					6.4,
					7.23,
					7.22,
					5.87,
					5.12,
				],
			}
		)
	chart_path = os.path.join("exports/charts")
	with st.form("Question"):
		question = st.text_input("Question", value="", type="default")
		submitted = st.form_submit_button("Submit")
		if submitted:
			with st.spinner():
				llm = OpenAI(api_token=st.session_state.openai_key)
				df = SmartDataframe(
					st.session_state.df,
					config={
						"llm": llm,
						"save_charts_path": chart_path,
						"save_charts": True,
						"verbose": True,
					},
				)
				response = df.chat(
					question
				)  # Using 'chat' method based on your context

				# Display the textual response (if any):
				if response:
					st.write(response)
				chart_path = os.path.join("exports/charts", "temp_chart.png")
				if os.path.exists(chart_path):
					st.image(
						chart_path, caption="Generated Chart", use_column_width=True
					)
				# Append the question to the history:
				st.session_state.prompt_history.append(question)

	if st.session_state.df is not None:
		st.subheader("Current dataframe:")
		st.write(st.session_state.df)

	st.subheader("Prompt history:")
	st.write(st.session_state.prompt_history)

	if st.button("Clear"):
		st.session_state.prompt_history = []
		st.session_state.df = None

def class1_ex18():
	st.subheader("Exercise 18: Data Analytics")
	st.write("In this exercise, we will use the Pandas AI library to perform data analytics.")
	st.write("The Pandas AI library is a smart agent that can perform data analytics on a dataframe.")
	st.write("You may need to install the following library:")
	st.code('''pip install bs4''')
	st.write("Copy and run the code below to see the chatbot in action.")

	st.markdown("**:blue[Code]**")
	st.code('''
# PandasAI- A smart agent that can do visual analytics
def ex18_pandas_AI():
	st.title("pandas-ai streamlit interface")

	# Upload CSV file using st.file_uploader
	uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	if "openai_key" not in st.session_state:
		st.session_state.openai_key = st.secrets["openapi_key"]
		st.session_state.prompt_history = []
		st.session_state.df = None

	if uploaded_file is not None:
		try:
			df = pd.read_csv(uploaded_file)
			st.session_state.df = df
		except Exception as e:
			st.write("There was an error processing the CSV file.")
			st.write(e)

	else:
		st.session_state.df = pd.DataFrame(
			{
				"country": [
					"United States",
					"United Kingdom",
					"France",
					"Germany",
					"Italy",
					"Spain",
					"Canada",
					"Australia",
					"Japan",
					"China",
				],
				"gdp": [
					19294482071552,
					2891615567872,
					2411255037952,
					3435817336832,
					1745433788416,
					1181205135360,
					1607402389504,
					1490967855104,
					4380756541440,
					14631844184064,
				],
				"happiness_index": [
					6.94,
					7.16,
					6.66,
					7.07,
					6.38,
					6.4,
					7.23,
					7.22,
					5.87,
					5.12,
				],
			}
		)
	chart_path = os.path.join("exports/charts")
	with st.form("Question"):
		question = st.text_input("Question", value="", type="default")
		submitted = st.form_submit_button("Submit")
		if submitted:
			with st.spinner():
				llm = OpenAI(api_token=st.session_state.openai_key)
				df = SmartDataframe(
					st.session_state.df,
					config={
						"llm": llm,
						"save_charts_path": chart_path,
						"save_charts": True,
						"verbose": True,
					},
				)
				response = df.chat(
					question
				)  # Using 'chat' method based on your context

				# Display the textual response (if any):
				if response:
					st.write(response)
				chart_path = os.path.join("exports/charts", "temp_chart.png")
				if os.path.exists(chart_path):
					st.image(
						chart_path, caption="Generated Chart", use_column_width=True
					)
				# Append the question to the history:
				st.session_state.prompt_history.append(question)

	if st.session_state.df is not None:
		st.subheader("Current dataframe:")
		st.write(st.session_state.df)

	st.subheader("Prompt history:")
	st.write(st.session_state.prompt_history)

	if st.button("Clear"):
		st.session_state.prompt_history = []
		st.session_state.df = None
''')
	st.markdown("**:red[Code Output]**")
	ex18()
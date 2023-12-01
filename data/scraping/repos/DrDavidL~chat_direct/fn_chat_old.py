import streamlit as st
import sys

import json
import requests
from datetime import datetime

import ast
import inspect

import openai
import os
import time
import re as regex



import sympy as sp
from sympy import *
from random import randint

st.set_page_config(page_title='Problem Solver', layout = 'centered', page_icon = ':face_palm:', initial_sidebar_state = 'auto')

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ''
    
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
    
if 'query' not in st.session_state:
    st.session_state.query = ''
    
if 'iteration_limit' not in st.session_state:
    st.session_state.iteration_limit = 5
    
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
    
if "done" not in st.session_state:
    st.session_state.done = False

def check_password():

    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.getenv("password"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        # fetch_api_key()
        return True

class FunctionWrapper:
	def __init__(self, func):
		self.func = func
		self.info = self.extract_function_info()

	def extract_function_info(self):
		source = inspect.getsource(self.func)
		tree = ast.parse(source)

		# Extract function name
		function_name = tree.body[0].name

		# Extract function description from docstring
		function_description = self.extract_description_from_docstring(self.func.__doc__)

		# Extract function arguments and their types
		args = tree.body[0].args
		parameters = {"type": "object", "properties": {}}
		for arg in args.args:
			argument_name = arg.arg
			argument_type = self.extract_parameter_type(argument_name, self.func.__doc__)
			parameter_description = self.extract_parameter_description(argument_name, self.func.__doc__)
			parameters["properties"][argument_name] = {
				"type": argument_type,
				"description": parameter_description,
			}

		# Extract function return type
		return_type = None
		if tree.body[0].returns:
			return_type = ast.get_source_segment(source, tree.body[0].returns)

		function_info = {
			"name": function_name,
			"description": function_description,
			"parameters": {
				"type": "object",
				"properties": parameters["properties"],
				"required": list(parameters["properties"].keys()),
			},
			"return_type": return_type,
		}

		return function_info

	def extract_description_from_docstring(self, docstring):
		if docstring:
			lines = docstring.strip().split("\n")
			description_lines = []
			for line in lines:
				line = line.strip()
				if line.startswith(":param") or line.startswith(":type") or line.startswith(":return"):
					break
				if line:
					description_lines.append(line)
			return "\n".join(description_lines)
		return None

	def extract_parameter_type(self, parameter_name, docstring):
		if docstring:
			type_prefix = f":type {parameter_name}:"
			lines = docstring.strip().split("\n")
			for line in lines:
				line = line.strip()
				if line.startswith(type_prefix):
					return line.replace(type_prefix, "").strip()
		return None

	def extract_parameter_description(self, parameter_name, docstring):
		if docstring:
			param_prefix = f":param {parameter_name}:"
			lines = docstring.strip().split("\n")
			for line in lines:
				line = line.strip()
				if line.startswith(param_prefix):
					return line.replace(param_prefix, "").strip()
		return None

	# Rest of the class implementation...
	def __call__(self, *args, **kwargs):
		return self.func(*args, **kwargs)

	def function(self):
		return self.info

def function_info(func):
    return FunctionWrapper(func)



def gen_response(prefix, history, gpt_model):
    history.append({"role": "system", "content": prefix})
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages = history,
        temperature=0.9,
    )
    # summary = response['choices'][0]['message']['content']
    # st.session_state.message_history.append(summary)
    # st.write(f'Here is the input summary: {summary}')
    return response

# message_history = []

def ai(query=st.session_state.query):
    available_functions = [calculate_expression, search_internet]

    # Add the new user message to the history
    st.session_state.message_history.append({"role": "user", "content": query})
    openai.api_key = st.session_state.openai_api_key

    # Check if message_history is not empty
    if st.session_state.message_history:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=st.session_state.message_history,
            functions=[func.function() for func in available_functions],
            function_call="auto",
        )





    message = response["choices"][0]["message"]
    first_answer = message["content"]
    if first_answer != st.session_state.last_result:
        st.write(first_answer)
        st.session_state.last_result = first_answer

    # Add the new system message to the history
    st.session_state.message_history.append(message)
    # Step 2, check if the model wants to call a function
    
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        # st.markdown(f'*No guessing - here is where we use the function call:* **{function_name}**')

        function_function = globals().get(function_name)

        # test we have the function
        if function_function is None:
            print("Couldn't find the function!")
            sys.exit()

        # Step 3, get the function information using the decorator
        function_info = function_function.function()

        # Extract function call arguments from the message
        function_call_args = json.loads(message["function_call"]["arguments"])

        # Filter function call arguments based on available properties
        filtered_args = {}
        for arg, value in function_call_args.items():
            if arg in function_info["parameters"]["properties"]:
                filtered_args[arg] = value

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = function_function(**filtered_args)
        # st.write(f'here is the function response: {function_response}')
        
        if function_name == 'search_internet':
            if isinstance(function_response, requests.Response):
                function_response = function_response.json()
                function_response['items'][0]['snippet'] = function_response['items'][0]['snippet'] + 'Now we are done.'


        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": query},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response)
                },
            ],
        )
        # st.write(f'here is the second response message: {second_response}')
        message2 = second_response["choices"][0]["message"]
        second_answer = message2["content"]
        if second_answer != st.session_state.last_result:
            st.write(second_answer)
            st.session_state.last_result = second_answer
        st.session_state.message_history.append(message2)
        return second_response
    else:
        return response



from sympy import sympify, solve, symbols



def calculate_expression(expression: str):
    """
    Calculates the result for an expression.
    Uses input expressions written for the sympy library.
    For example, cosine is cos (not math.cos) and pi is pi.

    :param expression: A mathematical expression written for the sympy library in python
    :type expression: string
    :return: A float or list of floats representing the result of the expression
    :rtype: float or list
    """

    # Check if the string contains the "solve" function
    if "solve" in expression:
        # Extract the equation and the variable to solve for
        equation_str = expression.split("solve(")[1].split(",")[0].strip()
        variable_str = expression.split(",")[1].split(")")[0].strip()

        # Convert the strings to symbolic expressions
        equation = sympify(equation_str)
        variable = symbols(variable_str)

        # Solve the equation
        solutions = solve(equation, variable)
        
        # If there's only one solution, return it as a float
        # Otherwise, return a list of floats
        if len(solutions) == 1:
            return float(solutions[0])
        else:
            return [float(sol) for sol in solutions]

    # If it's not a solve expression, simply evaluate it
    else:
        return float(sympify(expression))




@function_info
def calculate_expression_old(expression: str) -> float:
    """
    Calculates the result for an expression.
    Uses input expressions written for the sympy library.
    For example, cosine is cos (not math.cos) and pi is pi.

    :param expression: A mathematical expression written for the sympy library in python
    :type expression: string
    :return: A float representing the result of the expression
    :rtype: float
    """
    st.info(f'Our current equation: **{expression}**')
    result = float(sp.sympify(expression))
    
    return result

@function_info
def search_internet(web_query: str) -> float:
    """
    Obtains real-time search results from across the internet. 
    Supports all Google Advanced Search operators such (e.g. inurl:, site:, intitle:, etc).
    
    :param web_query: A search query, including any Google Advanced Search operators
    :type web_query: string
    :return: A list of search results
    :rtype: json
    
    """
    st.info(f'Our current search query has been called: **{web_query}**')
    url = "https://real-time-web-search.p.rapidapi.com/search"
    querystring = {"q":web_query,"limit":"10"}
    headers = {
	"X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
	"X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response_data = response.json()
    def display_search_results(json_data):
        data = json_data['data']
        for item in data:
            st.markdown(f"### [{item['title']}]({item['url']})")
            st.write(item['snippet'])
            st.write("---")
    # st.info('Searching the web using: **{web_query}**')
    display_search_results(response_data)
    st.session_state.done = True
    st.write('Done with websearch function')
    return response


def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENAI_API_KEY"]
        # os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.openai_api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        # st.write(f'Here is what we think the key is step 1: {api_key}')
    except:
        
        if st.session_state.openai_api_key != '':
            api_key = st.session_state.openai_api_key
            os.environ['OPENAI_API_KEY'] = api_key
            # If the API key is already set, don't prompt for it again
            # st.write(f'Here is what we think the key is step 2: {api_key}')
            return 
        else:        
            # If the secret is not found, prompt the user for their API key
            st.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
            api_key = st.text_input("Please, whisper your API key into my ears: ", key = 'warning2')
  
            st.session_state.openai_api_key = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            # Save the API key as a secret
            # st.secrets["my_api_key"] = api_key
            # st.write(f'Here is what we think the key is step 3: {api_key}')
            return 
    
    return 



def process_query(query):
    done_phrase = "Now we are done."
    st.session_state.done = False
    if st.button('Go'):
        query = """ You have access to two functions to assist responses:
        1. Use the 'calculate_expression' function to calculate any expression. For trig, use radians. (radians = degrees * pi/180). When your answer is complete, always include ```Now we are done.``` to indicate you are finished.
        2. Use the 'search_internet' function to search the internet for an answer. When your answer is complete, always include ```Now we are done.``` to indicate you are finished.
        3. If you receive input without a question to answer, summarize and include ```Now we are done.``` to indicate when you are finished. 
        
        Here is your query: """ + query 
        
        i = st.session_state.iteration_limit
        st.session_state.last_result = query
        delay = 1  # Start with a delay of 1 second
        should_break = False
        while not st.session_state.done:
            # st.write(f' here is last result: {st.session_state.last_result}')
            # if regex.search(r'\b{}\b'.format(done_phrase), st.session_state.last_result):
            #     st.info('We are done!')
            #     break
            if should_break:
                break
            try:
                response = ai(query=st.session_state.last_result)
                
                i -= 1
                if i == 0:
                    st.write('We are done here - complexity exceeded.')
                    should_break  = True
                    break
                if st.session_state.done == True:
                    st.info('We are done here.')
                    should_break = True
                    break
                if response:
                    if done_phrase in response["choices"][0]["message"]["content"]:
                        st.info('We are done here.')
                        should_break = True
                        break

                    # if response["function_call"]["name"] == 'search_internet':
                    #     break
                    # st.write(f' here is last result: {st.session_state.last_result}')
                if done_phrase in st.session_state.last_result:
                    st.info('We are done!')
                    should_break = True
                    break 
                for choice in response.get('choices'):
                    response_content = choice.get('message').get('content')
                    
                    # st.write(f'Here is the additional response: {response_content}')
                    # st.write(f'Here is the last result: {st.session_state.last_result}')
                    if response_content != st.session_state.last_result:
                        st.write(response_content)
                        st.session_state.last_result = response_content
                    if done_phrase in response_content:
                        st.info('We are done here.')
                        should_break = True
                        break
                        # if done_phrase in response_content:
                        #     st.info('We are done here.')
                        #     should_break = True
                        #     break
                    # else:
                    #     st.info('We are done here.')
                    #     should_break = True
                    #     break
                if should_break:
                    break
                     
            except:
                time.sleep(delay)
                delay *= 2  # Double the delay each time



# Streamlit functions
st.title('Natural Language Calculator and Story Problem Solver')
if check_password():
    fetch_api_key()
    st.info("""Welcome to the Natural Language Calculator and Story Problem Solver. This is a work in progress. Check out the GitHub. 
            As GPT-4 costs $$$ and many problems are multi-step, you have control here to limit the number of iterations.            
            """)
    st.session_state.iteration_limit = st.number_input('Iteration Limit', min_value=1, max_value=10, value=5)
    st.session_state.query = st.text_area("Type a natural language math problem (e.g, what is the area of a circle with a radius of 4cm), or expression (24 factorial). Or, even ask me: Create a story problem and solve it!")
    process_query(st.session_state.query)
    # conversation_text = '\n'.join([f"Role: {message['role']}, Content: {message['content']}" for message in st.session_state.message_history])
    # conversation_text = '\n'.join([f"{message['role']}: {message['content']} \n" for message in st.session_state.message_history])
    # conversation_text = '\n'.join([f"{message['role']}: {message['content']} \n" for message in st.session_state.message_history if message['content'] is not None and message['content'].lower() != 'none'])
    conversation_text = ''
    for message in st.session_state.message_history:
        if message['content'] is not None and message['content'].lower() != 'none':
            # If the keyword is in the message content, split the content at the keyword
            if "Use the 'calculate_expression' function call" in message['content']:
                parts = message['content'].split("Use the 'calculate_expression' function call", 1)
                content = parts[0]
            else:
                content = message['content']
            # Add the content (or the part before the keyword) to the conversation text
            conversation_text += f"{message['role']}: {content} \n\n"




    with st.expander('View Your Conversation History'):
        st.write(conversation_text)
        
    a1, b2, c3 = st.columns(3)
    
    with b2:
        st.download_button(
            label="Download Conversation History",
            data=conversation_text,
            file_name="conversation_history.txt",
            mime="text/plain",
            )
    
    with c3:
        if st.button('Clear History'):
            st.session_state.message_history.clear()
        
        
    # Check if the message history is too long (> 50) or has too many characters
    if st.session_state.message_history is not None:
        if len(st.session_state.message_history) == 50:
            st.write("The message history is getting long. The oldest messages will be summarized. Download now if you need a full record.")
        if len(st.session_state.message_history) > 50:
            # Summarize the message history
            summary_prefix = "Following this prompt is a text history of our conversation. Generate a summary of this message history: "
            summarized_history = gen_response(summary_prefix, st.session_state.message_history, "gpt-4")
            summary = summarized_history['choices'][0]['message']['content']
            # st.write(f'Thread is being summarized as follows: {summary}. Full details are still available for downloading.')
            # Keep the most recent 5 messages intact
            recent_messages = st.session_state.message_history[-5:]
            
            # Replace the message history with the summary and the recent messages
            st.session_state.message_history = [{'role': 'assistant', 'content': summary}] + recent_messages

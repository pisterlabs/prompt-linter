import streamlit as st
import openai
import requests
import time
import json
import os
from prompts import *
from functions import *
import re


st.set_page_config(page_title='Web Answers', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')

if "current_fn" not in st.session_state:
    st.session_state["current_fn"] = "websearch"
    
if "current_param" not in st.session_state:
    st.session_state["current_param"] = "topic"

def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except KeyError:
        
        try:
            api_key = os.environ["OPENAI_API_KEY"]
            # If the API key is already set, don't prompt for it again
            return api_key
        except KeyError:        
            # If the secret is not found, prompt the user for their API key
            st.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
            api_key = getpass.getpass("Please, whisper your API key into my ears: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Save the API key as a secret
            # st.secrets["my_api_key"] = api_key
            return api_key
    
    return api_key


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
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
        return True


def process_streamed_fn_call(content):
    """
    This function processes the content and returns a function call to websearch.
    It extracts the arguments from the "function_call" fields and concatenates them to form a string.
    This string is then passed to the websearch function.
    """
    # Find all arguments in the content using regex
    arguments = re.findall(r'"arguments": "(.*?)"', content)
    # st.write(f'Here are the inital arguments: {arguments}')

    # Filter out unwanted arguments
    arguments = [arg for arg in arguments if arg not in ['{', '}', '\n', ':', '', ' ', 'topic']]
    
    # st.write(f'Here are the filtered arguments: {arguments}')

    # Join the arguments with a space to form the search query
    search_query = ' '.join(arguments)
    # st.write(f'Here is the search query: {search_query}')
    
    # Remove leading and trailing whitespace
    search_query = search_query.strip()
    # st.write(f'Here is the search query after stripping: {search_query}')

    # Remove unwanted characters
    search_query = search_query.replace("{", "").replace("\n", "").replace("\\", "").strip()

    # If the string starts with "n ", remove it
    if search_query.startswith("n "):
        search_query = search_query[2:]
    
    # st.write(f'Here is the search query after replacing: {search_query}')

    # Return a function call to websearch with the search query
    return search_query




def standard_answer(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_key = os.environ['OPENAI_API_KEY']
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
    # model = 'gpt-3.5-turbo',
    model = st.session_state.model,
    messages = messages,
    temperature = temperature,
    stream = True,   
    )
    
    
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:        
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content





def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_key = os.environ['OPENAI_API_KEY']
    is_function_call = False
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
    # model = 'gpt-3.5-turbo',
    model = st.session_state.model,
    functions = function_descriptions,
    function_call="auto",
    messages = messages,
    temperature = temperature,
    stream = True,   
    )
    
    
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:        
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        if event_text is not None:
            answer += str(event_text)
        if "function_call" in event_text:
            is_function_call = True
            continue
            function_call = event_text["function_call"]
            st.write(f'HEre is full event_text: {event_text}')
            function_name = function_call["name"]
            arguments = function_call["arguments"]
            st.write(f"Function call detected: {function_call}")            
            if function_name == "websearch":
                st.write(f"Function called: {function_name}")
                st.write(f"Function parameters: {arguments}")
                # full_answer += websearch(arguments[0])
                
            continue
    # st.write(f'Now here is the full answer: {answer}')
    search_topic = process_streamed_fn_call(answer)
    search_output = websearch(search_topic)
    
    
                # return full_answer, completion
        # answer += event_text.get('content', '')
        # full_answer += event_text.get('content', '')
        # time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer, is_function_call, search_output # Change how you access the message content
# def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
#     openai.api_key = os.environ['OPENAI_API_KEY']
#     messages = [{'role': 'system', 'content': prefix},
#             {'role': 'user', 'content': sample_question},
#             {'role': 'assistant', 'content': sample_answer},
#             {'role': 'user', 'content': history_context + my_ask},]
#     # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
#     completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
#     # model = 'gpt-3.5-turbo',
#     model = st.session_state.model,
#     functions = function_descriptions,
#     function_call="auto",
#     messages = messages,
#     temperature = temperature,
#     stream = True,   
#     )
    
    
#     start_time = time.time()
#     delay_time = 0.01
#     answer = ""
#     full_answer = ""
#     c = st.empty()
#     for event in completion:        
#         c.markdown(answer)
#         event_time = time.time() - start_time
#         event_text = event['choices'][0]['delta']
#         answer += event_text.get('content', '')
#         full_answer += event_text.get('content', '')
#         time.sleep(delay_time)
    
    
    # start_time = time.time()
    # delay_time = 0.01
    # answer = ""
    # full_answer = ""
    # c = st.empty()
    # function_call_detected = False
    # response_text =""
    # function_parameters = []
    # for event in completion:
        
    #     if "choices" in event:
    #         deltas = event["choices"][0]["delta"]
    #         if "function_call" in deltas:
    #             function_call_detected = True
    #             st.write(f"Function call detected: {deltas['function_call']}")
    #             if "name" in deltas["function_call"]:
    #                 function_called = deltas["function_call"]["name"]
    #                 st.write(f"Function called: {function_called}")
    #             if "arguments" in deltas["function_call"]:
    #                 function_parameters += deltas["function_call"]["arguments"]
    #                 st.write(f"Function parameters: {function_parameters}")
    #         # if (function_call_detected and event["choices"][0].get("finish_reason") == "function_call"):
    #             st.write("Letting finish to get parameters.")
    #             c.markdown(response_text)
    #                 # full_answer += answer
    #             event_time = time.time() - start_time
    #             event_text = event['choices'][0]['delta']
    #             answer += event_text.get('content', '')
    #             full_answer += event_text.get('content', '')
    #             time.sleep(delay_time)
    #             st.write(f'Now here is the full answer: {full_answer}')
    st.stop()

                
                
                
                
            #     function_response_generator = function_call(function_called)                
            #     for function_response_chunk in function_response_generator:
            #         if "choices" in function_response_chunk:
            #             deltas = function_response_chunk["choices"][0]["delta"]
            #             if "content" in deltas:
            #                 response_text += deltas["content"]
            #                 st.write(response_text)
            # elif "content" in deltas and not function_call_detected:
            #     response_text += deltas["content"]
            #     # yield response_text

            #     c.markdown(response_text)
            #         # full_answer += answer
            #     event_time = time.time() - start_time
            #     event_text = event['choices'][0]['delta']
            #     answer += event_text.get('content', '')
            #     full_answer += event_text.get('content', '')
            #     time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer, completion # Change how you access the message content

def websearch(web_query: str) -> float:
    """
    Obtains real-time search results from across the internet. 
    Supports all Google Advanced Search operators such (e.g. inurl:, site:, intitle:, etc).
    
    :param web_query: A search query, including any Google Advanced Search operators
    :type web_query: string
    :return: A list of search results
    :rtype: json
    
    """
    st.info(f'Here is the websearch input: **{web_query}**')
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
            st.sidebar.markdown(f"### [{item['title']}]({item['url']})")
            st.sidebar.write(item['snippet'])
            st.sidebar.write("---")
    # st.info('Searching the web using: **{web_query}**')
    # display_search_results(response_data)
    # st.session_state.done = True
    st.write('Done with websearch function')
    return response_data

def function_call(initial_response):
    st.write('Here is the initial response in functioncall fn: ', initial_response)
    # function_call = initial_response["choices"][0]["message"]["function_call"]
    # function_name = function_call["name"]
    # arguments = function_call["arguments"]
    if initial_response == "websearch":
        return websearch()

if 'history' not in st.session_state:
            st.session_state.history = []

if 'output_history' not in st.session_state:
            st.session_state.output_history = []
            
if 'mcq_history' not in st.session_state:
            st.session_state.mcq_history = []
            
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    
if 'model' not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"
    
if 'temp' not in st.session_state:
    st.session_state.temp = 0.3

if check_password():
    
    st.title("Web Answers")
    st.write("ALPHA version 0.3")
    os.environ['OPENAI_API_KEY'] = fetch_api_key()

    disclaimer = """**Disclaimer:** This is a tool to assist education regarding artificial intelligence. Your use of this tool accepts the following:   
    1. This tool does not generate validated medical content. \n 
    2. This tool is not a real doctor. \n    
    3. You will not take any medical action based on the output of this tool. \n   
    """
    with st.expander('About Web Answers - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.session_state.model = st.radio("Select model - leave default for now", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"), index=0)
        st.session_state.temp = st.slider("Select temperature", 0.0, 1.0, 0.3, 0.01)
        st.write("Last updated 8/12/23")
    
    my_ask = st.text_area('Ask away!', height=100, key="my_ask")
    
    # if st.button("Enter"):
    #     openai.api_key = os.environ['OPENAI_API_KEY']
    #     st.session_state.history.append(my_ask)
    #     history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    #     output_text, is_fn_call, search_output = answer_using_prefix(web_search_prefix, sample_question, sample_response, my_ask, st.session_state.temp, history_context=history_context)
    #     st.session_state.output_history.append((search_output))
    #     if is_fn_call:
    #         # st.write(f'Here is the output{search_output}')
    #         search_output = json.dumps(search_output)
    #         standard_answer(" ", " ", " ", my_ask + "Now use to answer:" + search_output, st.session_state.temp, history_context=history_context)
            
    #     # st.session_state.my_ask = ''
    #     # st.write("Answer", output_text)
        
    #     # st.write(st.session_state.history)
    #     # st.write(f'Me: {my_ask}')
    #     # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
    #     # st.write(list(output_text))
    #     # st.session_state.output_history.append((output_text['choices'][0]['message']['content']))
    #     st.session_state.output_history.append((output_text))
    
    # tab1_download_str = []
    
    if st.button("Assemble Web Content to Answer a Question"):
        openai.api_key = os.environ['OPENAI_API_KEY']
        st.session_state.history.append(my_ask)
        search_output = websearch(my_ask)
        search_output = json.dumps(search_output)
        history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
        output_text=standard_answer(" ", " ", " ", my_ask + "Now use to answer:" + search_output, st.session_state.temp, history_context="")
        st.session_state.output_history.append((output_text))

            
            
        # st.session_state.my_ask = ''
        # st.write("Answer", output_text)
        
        # st.write(st.session_state.history)
        # st.write(f'Me: {my_ask}')
        # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
        # st.write(list(output_text))
        # st.session_state.output_history.append((output_text['choices'][0]['message']['content']))
        st.session_state.output_history.append((output_text))
    
    tab1_download_str = []
        
        # ENTITY_MEMORY_CONVERSATION_TEMPLATE
        # Display the conversation history using an expander, and allow the user to download it
    with st.expander("View or Download Thread", expanded=False):
        for i in range(len(st.session_state['output_history'])-1, -1, -1):
            st.info(st.session_state["history"][i],icon="🧐")
            st.success(st.session_state["output_history"][i], icon="🤖")
            tab1_download_str.append(st.session_state["history"][i])
            tab1_download_str.append(st.session_state["output_history"][i])
        tab1_download_str = [disclaimer] + tab1_download_str 
        
        # Can throw error - requires fix
        tab1_download_str = '\n'.join(tab1_download_str)
        if tab1_download_str:
            st.download_button('Download',tab1_download_str, key = "Conversation_Thread")
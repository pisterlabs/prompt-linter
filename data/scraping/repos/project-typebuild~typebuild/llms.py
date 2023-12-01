import os
import re
import streamlit as st
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(st.session_state.typebuild_root)

def last_few_messages(messages):
    """
    Long messages get rejected by the LLM.  So,
    - Will keep the system message, which is the first message
    - Will keep the last 3 user and system messages
    """
    last_messages = []
    if messages:
        # Get the first message
        last_messages.append(messages[0])
    # Get the last 3 user or assistant messages
    user_assistant_messages = [i for i in messages if i['role'] in ['user', 'assistant']]
    last_messages.extend(user_assistant_messages[-7:])
    return last_messages

def extract_message_to_agent(content):
    """
    Extracts the message to the agent from the content.
    This is found within <<< and >>>.
    There will at least be one set of triple angle brackets
    for this function to be invoked.
    """
    pattern = r"<<<([\s\S]*?)>>>"
    matches = re.findall(pattern, content)
    if len(matches) == 1:
        message_to_agent = matches[0].strip()
    else:
        message_to_agent = '\n'.join(matches)
    
    # Add it to the session state
    st.session_state.message_to_agent = message_to_agent
    return message_to_agent

def get_llm_output(messages, max_tokens=2500, temperature=0.4, model='gpt-4', functions=[]):
    """
    This checks if there is a custom_llm.py in the plugins directory 
    If there is, it uses that.
    If not, it uses the openai llm.
    """
    # Check if there is a custom_llm.py in the plugins directory
    # If there is, use that
    # Get just the last few messages
    messages = last_few_messages(messages)
    st.session_state.last_request = messages
    typebuild_root = st.session_state.typebuild_root
    if os.path.exists(os.path.join(typebuild_root, 'custom_llm.py')):
        from custom_llm import custom_llm_output

        content = custom_llm_output(messages, max_tokens=max_tokens, temperature=temperature, model=model, functions=functions)
    # If claude is requested and available, use claude
    elif model == 'claude-2' and 'claude_key' in st.session_state:
        content = get_claude_response(messages, max_tokens=max_tokens)
    else:
        model = 'gpt-3.5-turbo'
        msg = get_openai_output(messages, max_tokens=max_tokens, temperature=temperature, model=model, functions=functions)
        content = msg.get('content', None)
        if 'function_call' in msg:
            func_call = msg.get('function_call', None)
            st.session_state.last_function_call = func_call
            st.sidebar.info("Got a function call from LLM")
            content = func_call.get('content', None)
    
    # progress_status.info("Extracting information from response...")
    if content:
        st.session_state.last_response = content
    # We can get back code or requirements in multiple forms
    # Look for each form and extract the code or requirements

    # Recent GPT models return function_call as a separate json object
    # Look for that first.
    # If there are triple backticks, we expect code
    if '```' in str(content) or '|||' in str(content):
        # NOTE: THERE IS AN ASSUMPTION THAT WE CAN'T GET BOTH CODE AND REQUIREMENTS
        extracted, function_name = parse_func_call_info(content)
        func_call = {'name': function_name, 'arguments': {'content':extracted}}
        st.session_state.last_function_call = func_call


    # Stop ask llm
    st.session_state.ask_llm = False
    # progress_status.success('Response generated!')
    return content


def get_openai_output(messages, max_tokens=3000, temperature=0.4, model='gpt-4', functions=[]):
    """
    Gets the output from GPT models. default is gpt-4. 

    Args:
    - messages (list): A list of messages in the format                 
                messages =[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}],

                system_instruction is the instruction given to the system to generate the response using the prompt.

    - model (str): The model to use.  Default is gpt-4.
    - max_tokens (int): The maximum number of tokens to generate, default 800
    - temperature (float): The temperature for the model. The higher the temperature, the more random the output
    """
    if functions:
        response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages = messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1,
                    functions=functions,
                )
    else:
        response = openai.ChatCompletion.create(
                    model=model,
                    messages = messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1,
                )
    msg = response.choices[0].message
    
    # Stop ask llm
    st.session_state.ask_llm = False    
    return msg

def get_claude_response(messages, max_tokens=2000):
    anthropic = Anthropic(
        api_key=st.session_state.claude_key,
    )
    # Since claude has a higher max_tokens, let's increase the limit
    max_tokens = int(max_tokens * 2)
    prompt = ""
    for i in messages:
        if i['role'] == 'assistant':
            prompt += f"{AI_PROMPT} {i['content']}\n\n"
        else:
            prompt += f"{HUMAN_PROMPT} {i['content']}\n\n"

    prompt += AI_PROMPT
    response = anthropic.completions.create(
        prompt=prompt,
        stop_sequences = [anthropic.HUMAN_PROMPT],
        model="claude-2",
        temperature=0.4,
        max_tokens_to_sample=max_tokens,
    )
    return response.completion

def parse_func_call_info(response):
    """
    The LLM can return code or requirements in the content.  
    Ideally, requirements come in triple pipe delimiters, 
    but sometimes they come in triple backticks.

    Figure out which one it is and return the extracted code or requirements.
    """
    # If there are ```, it could be code or requirements
    function_name = None
    if '```' in response:
        # If it's python code, it should have at least one function in it
        if 'def ' in response:
            extracted = parse_code_from_response(response)
            function_name = 'save_code_to_file'
        elif 'data_agent' in response:
            extracted = parse_modified_user_requirements_from_response(response)
            function_name = 'data_agent'
        # If it's not python code, it's probably requirements
        else:
            extracted = parse_modified_user_requirements_from_response(response)
            function_name = 'save_requirements_to_file'
    # If there are |||, it's probably requirements
    elif '|||' in response:
        extracted = parse_modified_user_requirements_from_response(response)
        function_name = 'save_requirements_to_file'
    else:
        extracted = None
    return extracted, function_name
            



def parse_code_from_response(response):

    """
    Returns the code from the response from LLM.
    In the prompt to code, we have asked the LLM to return the code inside triple backticks.

    Args:
    - response (str): The response from LLM

    Returns:
    - matches (list): A list of strings with the code

    """

    pattern = r"```python([\s\S]*?)```"
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        matches = '\n'.join(matches)
    else:
        matches = matches[0]
    return matches

def parse_modified_user_requirements_from_response(response):
    
    """
    Returns the modified user requirements from the response from LLM. 
    In the prompt to modify, we have asked the LLM to return the modified user requirements inside triple pipe delimiters.

    Args:
    - response (str): The response from LLM

    Returns:
    - matches (list): A list of strings with the modified user requirements

    """
    if '|||' in response:
        pattern = r"\|\|\|([\s\S]*?)\|\|\|"
    if '```' in response:
        # It shouldnt have ```python in it
        pattern = r"```([\s\S]*?)```"

    matches = re.findall(pattern, response)
    # if there are multiple matches, join by new line
    if len(matches) > 0:
        matches = '\n'.join(matches)
    else:
        matches = matches[0]
    return matches

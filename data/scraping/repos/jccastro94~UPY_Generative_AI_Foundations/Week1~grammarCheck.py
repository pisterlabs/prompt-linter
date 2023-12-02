import streamlit as st
import openai
import os

from dotenv import load_dotenv

openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def grammar_check(essay):
    jsonPrompt = f"""
Perform the following actions on the text delimited by triple backticks: \
1. Check if the text has any grammatical errors.
2. If there are errors, get the sentence where the errors are. \
3. Using the following categories, get the type of error: \
    - Capitalization \
    - Punctuation \
    - Spelling \
    - Word Order \
    - Verb Tense \
    - Word Choice \
    - Sentence Structure \
    - Sentence Fragment \
    - Run-on Sentence \
3. Create a hint of what the error might be, for the student \
to correct the error themselves such as \
<error>: The light bulb on the cieling is broken. \
<errorType>: Spelling  \
<correction>: The light bulb on the ceiling is broken. \
<response>: You have a spelling error. Remember, I before E except after C.  \

<error>: I hate mondays. \
<errorType>: Capitalization \
<correction>: I hate Mondays.  \
<response>: You have a Capitalization error. Remember to always capitalize proper nouns \
, such as month names or days of the week. \

4. Translate the hint to Spanish. \

After following those steps, the only output you should provide is a JSON \
with the following format: \
"errors": [
"id": "Error number count. Integer", \
"sentence": "The sentence with the error.", \
"type": "Error type ", \
"hint": "The hint for the student to correct the error.", \
"translation": "The hint translated to Spanish.",
"correction": "The corrected sentence." ]

```{essay}```
"""
    
    created_dict = get_completion(jsonPrompt)

    checkPrompt = f"""
Please check if the text delimited by triple backticks \
has a valid json structure. It must contain the keys: \
id, sentence, type, hint, translation and correction. \
If it is a valid json with all the keys required,  \
simply write 'TRUE'. If it does not contain a valid json \
or it does not contain all the required keys, \
simply write 'FALSE'.

```{created_dict}```
"""   

    json_check = get_completion(checkPrompt)

    if json_check == "TRUE":
        if type(created_dict) is not dict:
            created_dict = eval(created_dict)
        return created_dict
    else:
        return "{error: 'Invalid JSON'}"

st.title('Grammar Checker')
essay = st.text_area('Write your essay here')

if 'error_dict' not in st.session_state:
    st.session_state.error_dict = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if st.button('Submit'):
    st.session_state.error_dict = grammar_check(essay)
    st.session_state.current_index = 0

if st.session_state.error_dict is not None:
    if 'errors' in st.session_state.error_dict:
        errors = st.session_state.error_dict['errors']

        st.text(f'Error: {st.session_state.current_index+1}/{len(errors)}')
        st.text(f'ID: {errors[st.session_state.current_index]["id"]}')
        st.text(f'Type: {errors[st.session_state.current_index]["type"]}')
        st.text(f'Sentence: {errors[st.session_state.current_index]["sentence"]}')

        corrected_sentence = st.text_input('Correct the sentence here')

        if st.button('Hint'):
            st.text(errors[st.session_state.current_index]["hint"])
        
        if st.button('Translate Hint'):
            st.text(errors[st.session_state.current_index]["translation"])
        
        if st.button('View Answer'):
            st.text(errors[st.session_state.current_index]["correction"])

        if st.button('Next'):
            if st.session_state.current_index < len(errors) - 1:
                st.session_state.current_index += 1
        
        if st.button('Previous'):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
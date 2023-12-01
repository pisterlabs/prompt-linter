import streamlit as st
from dotenv import load_dotenv
import os
import openai
from streamlit_chat import message
from titan_function_defination import openai_functions, make_ai_function_call
import json


@st.cache_resource
def get_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    return key


openai.api_key = get_api_key()


def make_open_ai_call_and_get_msgs(conversation_history):
    function_call = "none" if conversation_history[-1]['role'] == 'function' else 'auto'
    # consideringl last 3 if last msg if function call(none will be when last was function call) else choosing last 4.
    last_msgs_ct = 4 if function_call == 'none' else 3
    final_conversation_history = conversation_history if len(
        conversation_history) <= last_msgs_ct + 1 else conversation_history[:1] + conversation_history[-1*last_msgs_ct:]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=final_conversation_history,
        functions=openai_functions,
        function_call=function_call,
    )
    return response.choices[0].message


def talk_to_gpt3(prompt, conversation_history, role='user', function_name=None):
    msg = {"role": role, "content": prompt}
    if role == 'function':
        msg['name'] = function_name
    conversation_history.append(msg)

    print(f'{conversation_history=}')

    message = make_open_ai_call_and_get_msgs(
        conversation_history=conversation_history)
    print('gpt response {}'.format(message))
    if message['content']:
        answer = message['content']
        conversation_history.append({"role": "assistant", "content": answer})
    elif message['function_call']:
        function_name = message["function_call"]["name"]
        function_args = json.loads(message["function_call"]["arguments"])
        result = make_ai_function_call(function_name, function_args)
        return talk_to_gpt3(result, conversation_history=conversation_history,
                            role='function', function_name=function_name)

    st.session_state['chat'] = conversation_history
    return answer


def main():
    st.title('Titan AI Chat')

    if 'chat' not in st.session_state:
        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant which is being used in an email application, for example: gmail, your aim is to help user only with email related tasks and some other natural comverstaion if he wants"}]
    else:
        conversation_history = st.session_state['chat']
    user_input = st.text_input("You: ")
    if user_input:
        talk_to_gpt3(user_input, conversation_history)

    for msg in conversation_history:
        if msg["role"] == "user":
            message(msg["content"], is_user=True)
        elif msg["role"] == "assistant":
            message(msg["content"])


if __name__ == '__main__':
    main()

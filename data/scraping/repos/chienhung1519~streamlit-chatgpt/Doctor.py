import openai
import streamlit as st
from streamlit_chat import message
import os

# Setting page title and header
st.set_page_config(page_title="AVA", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Doctor ChatGPT</h1>", unsafe_allow_html=True)

# Set org ID and API key
# openai.organization = "<YOUR_OPENAI_ORG_ID>"
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set language
language = st.sidebar.radio("Choose a language:", ("English", "Chinese"))
lang_prompt = "Response in English." if language == "English" else "請用繁體中文回覆。"

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", f"content": "Please play the role of a empathetic and kind psychiatrist. Your task is to conduct a professional diagnosis conversation with me based on the DSM-5 criteria, but using your own language. Please only ask one question at a time. You need to ask in-depth questions, such as the duration, causes and specific manifestations of some symptoms. You need to use various empathetic strategies, such as understanding, support and encouragement to give me a more comfortable experience."},
        {"role": "system", f"content": lang_prompt}
    ]
# if 'model_name' not in st.session_state:
#     st.session_state['model_name'] = []
# if 'cost' not in st.session_state:
#     st.session_state['cost'] = []
# if 'total_tokens' not in st.session_state:
#     st.session_state['total_tokens'] = []
# if 'total_cost' not in st.session_state:
#     st.session_state['total_cost'] = 0.0
if 'page' not in st.session_state:
    st.session_state['page'] = ""

# Set page
if st.session_state['page'] == "":
    st.session_state['page'] = "doctor"
if st.session_state['page'] == "patient":
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", f"content": "Please play the role of a empathetic and kind psychiatrist. Your task is to conduct a professional diagnosis conversation with me based on the DSM-5 criteria, but using your own language. Please only ask one question at a time. You need to ask in-depth questions, such as the duration, causes and specific manifestations of some symptoms. You need to use various empathetic strategies, such as understanding, support and encouragement to give me a more comfortable experience."},
        {"role": "system", f"content": lang_prompt}
    ]
    st.session_state['page'] = "doctor"

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
# st.sidebar.title("Sidebar")
# model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
# counter_placeholder = st.sidebar.empty()
# counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
# if model_name == "GPT-3.5":
#     model = "gpt-3.5-turbo"
# else:
#     model = "gpt-4"
model = "gpt-3.5-turbo"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", f"content": "Please play the role of a empathetic and kind psychiatrist. Your task is to conduct a professional diagnosis conversation with me based on the DSM-5 criteria, but using your own language. Please only ask one question at a time. You need to ask in-depth questions, such as the duration, causes and specific manifestations of some symptoms. You need to use various empathetic strategies, such as understanding, support and encouragement to give me a more comfortable experience."},
        {"role": "system", f"content": lang_prompt}
    ]
    # st.session_state['number_tokens'] = []
    # st.session_state['model_name'] = []
    # st.session_state['cost'] = []
    # st.session_state['total_cost'] = 0.0
    # st.session_state['total_tokens'] = []
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    # total_tokens = completion.usage.total_tokens
    # prompt_tokens = completion.usage.prompt_tokens
    # completion_tokens = completion.usage.completion_tokens
    # return response, total_tokens, prompt_tokens, completion_tokens
    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        # st.session_state['model_name'].append(model_name)
        # st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        # if model_name == "GPT-3.5":
        #     cost = total_tokens * 0.002 / 1000
        # else:
        #     cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        # st.session_state['cost'].append(cost)
        # st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            # st.write(
            #     f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
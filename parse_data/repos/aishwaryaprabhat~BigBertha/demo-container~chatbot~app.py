import streamlit as st
import replicate
import os
import redis
from streamlit_feedback import streamlit_feedback
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from llama_cpp import Llama

# Connect to the local Redis server
def redis_conn():
    r = redis.Redis(host='localhost', port=6379, db=0)
    return r


def load_llm():
    llm_local_path = os.environ.get('MODEL_LOCAL_PATH')
    llm = CTransformers(
        model=llm_local_path,     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
        model_type='llama',
        stream=True,
        threads=8,
        config={'max_new_tokens': 50,
                'temperature': 0.01}
    )
    # llama_model_path = os.environ.get('MODEL_LOCAL_PATH')  # /home/llamauser/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/a17885f653039bd07ed0f8ff4ecc373abf5425fd/llama-2-13b-chat.ggmlv3.q5_1.bin' 
                                                       # path to downloaded model is printed out by download_model.py and set as env var in the entrypoint.sh                 
    # llm = Llama(
    #     model_path=llama_model_path,
    #     n_threads=os.cpu_count(),
    #     n_batch=512,
    # )
    return llm

if "llm" not in st.session_state.keys():
    st.session_state["llm"] = load_llm()

if "redis" not in st.session_state.keys():
    st.session_state["redis"] = redis_conn()

# Function to increment and display the thumbs-down count
def record_feedback(dict_message):
    r = st.session_state["redis"]
    metric = "thumbs_down_count" if dict_message['score'] == '👎' else "thumbs_up_count"
    r.incr(metric)
    st.success("You've recorded a thumbs-down!")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B'], key='selected_model')
    # if selected_model == 'Llama2-7B':
    #     llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    # elif selected_model == 'Llama2-13B':
    #     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    # output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
    #                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
    #                               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    # return output
    
    #Template for building the PROMPT
    template = """
    {string_dialogue} {prompt_input} Assistant: 
    """

    #Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["string_dialogue", "prompt_input"],
        template=template
    )

  
    #Generating the response using LLM
    llm = st.session_state["llm"]
    response = llm(prompt.format(string_dialogue=string_dialogue, prompt_input=prompt_input))

    return response


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


streamlit_feedback(
    feedback_type="thumbs",
    on_submit=record_feedback,
)
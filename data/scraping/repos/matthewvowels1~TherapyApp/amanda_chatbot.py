
import pandas as pd
from langchain.llms import CTransformers
from langchain import PromptTemplate,  LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

if 'is_authenticated' not in st.session_state:
    st.session_state['is_authenticated'] = False

if 'username' not in st.session_state:
    st.session_state.username = ''

# Splash Screen UI
def display_auth_page():
    st.title('Login to Amanda Chatbot')
    st.session_state.username = st.text_input('Username:')
    if st.button('Proceed'):
        # Authentication Logic
        if st.session_state.username:
            st.session_state.is_authenticated = True
        else:
            st.warning('Please enter a username to proceed.')

# Conditional Display
if not st.session_state.is_authenticated:
    display_auth_page()
else:
    ################################### LLM STUFF ####################################

    initial_bot_message = "Hi. I'm Amanda. Can you tell me what has brought you in today?"

    template = """You are a trained psychotherapist called Amanda specialising in working with relationship difficulties. Please provide short responses and try to help the user to think about the issue from multiple different angles.
        If they do not provide any context, assume you know nothing about their situation and ask them for more information.
        Conversation history: {chat_history}
        Current patient utterance: {patient_utterance}
        Only return the helpful response below.
        Helpful response:"""

    prompt = PromptTemplate(
            input_variables=["chat_history", "patient_utterance"], template=template)

    llm = CTransformers(model='models/llama-2-13b-chat.ggmlv3.q8_0.bin', model_type='llama',
                  config={'gpu_layers': 7, 'max_new_tokens': 128, 'temperature': 0.02})

    @st.cache_resource
    def get_memory():
        return ConversationBufferMemory(memory_key="chat_history")

    @st.cache_resource
    def get_llm_chain():
        return LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

    bot_icon_seed = 23
    st.set_page_config(page_title="Amanda - An LLM-powered Streamlit app for therapy")

    with st.sidebar:
        st.title('<3 Amanda')
        st.markdown('''
        ## About
        ''')
        add_vertical_space(1)
        st.write('Made by Matthew Vowels')

    # Generate empty lists for generated and past.
    # stores AI generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [initial_bot_message]

    if 'past' not in st.session_state:
        st.session_state['past'] = [' ']

    # Layout of input/response containers
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    if 'input_text' not in st.session_state:
        st.session_state.input_text = ''

    def submit():
        st.session_state.input_text = st.session_state.input_widget
        st.session_state.input_widget = ''
    # User input
    # Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("You: ", "", key="input_widget", on_change=submit)
        return st.session_state.input_text

    # Applying the user input box
    with input_container:
        user_input = get_text()

    # Conditional display of AI generated responses as a function of user provided prompts
    with response_container:

        memory = get_memory()
        llm_chain = get_llm_chain()

        if user_input:
            st.session_state.past.append(user_input)
            placeholder = st.empty()

            # Display something in the placeholder
            placeholder.text("Amanda is typing...")

            chat_placeholder = st.empty()
            with chat_placeholder.container():
                a = message(user_input, key='temp', is_user=True, avatar_style='identicon')

                if len(st.session_state['generated']) == 1:
                    print('doing this')
                    message(st.session_state["generated"][0], key='temp_gen', avatar_style='open-peeps', seed=bot_icon_seed)

            response = llm_chain.predict(patient_utterance=user_input).rstrip('\"')
            st.session_state.generated.append(response)
            placeholder.empty()
        if st.session_state['generated']:

            if 'chat_placeholder' in locals():
                chat_placeholder.empty()

            for i in range(len(st.session_state['generated'])-1, -1, -1):  # reverse ordering (old->new  = bottom -> top)
                message(st.session_state["generated"][i], key=str(i), avatar_style='open-peeps', seed=bot_icon_seed)
                if i > 0:
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='identicon')

    if len(st.session_state.past) < len(st.session_state.generated):
        st.session_state.past.append(None)
    elif len(st.session_state.generated) < len(st.session_state.past):
        st.session_state.generated.append(None)

    # Create a DataFrame with chat history
    df_chat = pd.DataFrame({
        'User Input': st.session_state.past,
        'Model Response': st.session_state.generated
    })

    # Save the DataFrame to a CSV file
    df_chat.to_csv(f'{st.session_state.username}_chat_history.csv', index=False)

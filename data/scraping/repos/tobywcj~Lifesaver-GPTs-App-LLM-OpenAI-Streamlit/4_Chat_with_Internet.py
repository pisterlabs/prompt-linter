import streamlit as st
import os
from streamlit_chat import message
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun



def validate_openai_api_key(api_key):
    import openai

    openai.api_key = api_key

    with st.spinner('Validating API key...'):
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt="This is a test.",
                max_tokens=5
            )
            # print(response)
            validity = True
        except:
            validity = False

    return validity


# clear the chat history from streamlit session state
def clear_history():
    if 'internet_cb_history' in st.session_state:
        del st.session_state.internet_cb_history
        st.session_state.internet_cb_history= [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can connect to the internet. How can I help you?"}
        ]


if __name__ == "__main__":

    ############################################################ System Configuration ############################################################

    if "internet_cb_history" not in st.session_state:
        st.session_state.internet_cb_history= [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can connect to the internet. How can I help you?"}
        ]

    ############################################################ SIDEBAR widgets ############################################################

    with st.sidebar:
        # Setting up the OpenAI API key via secrets manager
        if 'OPENAI_API_KEY' in st.secrets:
            api_key_validity = validate_openai_api_key(st.secrets['OPENAI_API_KEY'])
            if api_key_validity:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                st.success("✅ API key is valid and set via Encrytion provided by Streamlit")
            else:
                st.error('🚨 API key is invalid and please input again')
        # Setting up the OpenAI API key via user input
        else:
            api_key_input = st.text_input("OpenAI API Key", type="password")
            api_key_validity = validate_openai_api_key(api_key_input)

            if api_key_input and api_key_validity:
                os.environ['OPENAI_API_KEY'] = api_key_input
                st.success("✅ API key is valid and set")
            elif api_key_input and api_key_validity == False:
                st.error('🚨 API key is invalid and please input again')

            if not api_key_input:
                st.warning('Please input your OpenAI API Key')
        
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        st.divider()

        if st.button('Clear Chat History'):
            clear_history()


    ############################################################ MAIN PAGE widgets ############################################################

    st.title("🌐 Chat with Internet")

    st.divider()

    for message in st.session_state.internet_cb_history:
        st.chat_message(message["role"]).write(message["content"])

    question = st.chat_input(placeholder="Ask me anything currently on the web")

    if question:
        if api_key_validity:
            st.session_state.internet_cb_history.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key_validity=api_key_validity, streaming=True)
            search = DuckDuckGoSearchRun(name="Search")
            search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
            with st.chat_message("assistant"):
                chatBot = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(st.session_state.internet_cb_history, callbacks=[chatBot])
                st.session_state.internet_cb_history.append({"role": "assistant", "content": response})
                st.write(response)
        elif not api_key_validity:
            st.warning('Please enter your OpenAI API Key to continue.')
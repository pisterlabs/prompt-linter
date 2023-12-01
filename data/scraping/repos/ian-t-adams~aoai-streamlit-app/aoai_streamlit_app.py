'''
Using the https://github.com/kirkhofer/data-ai/blob/main/aoai/chatbot.py repo from Kirk Hofer as a starting point.
'''

import os
import openai
import streamlit as st
from dotenv import load_dotenv  
import aoai_helpers as helpers

load_dotenv()  

openai.api_type = "azure"  
openai.api_key = os.environ['APIM_KEY']  
openai.api_base = os.environ['APIM_ENDPOINT']  
openai.api_version = os.environ['AOAI_API_VERSION']

# Preload environment variables and sidebar settings
helpers.load_settings(reload_api_settings=True)

# Create containers for the header, chat window, and footer - will use sidebar for setting model parameters
header_container = st.container()
chat_container = st.container()
footer_container = st.container()
settings_container = st.sidebar.container()
standard_system_message = "You are an AI assistant that helps people."

# Top level title and description of the app
with header_container:
    st.title("Interact with an Azure OpenAI 🤖", anchor="top", help='''This demo showcases the Azure OpenAI Service, Azure API Management Service,
          and Azure Web Apps with Streamlit.''')

# Pull up the messages if they exist - needed
if 'messages' not in st.session_state:
    helpers.env_to_st_session_state('SYSTEM','system', standard_system_message)
    st.session_state.messages = []
    st.session_state.messages.append({"role":"system","content":st.session_state.system})


with st.sidebar.title("Model Parameters", anchor="top", help='''The model parameters are used to control the behavior of the model. 
                      Each parameter has its own tooltip.'''):
    with settings_container:

        # Create a dictionary containing the available models for each completion type 
        # ###UPDATE 07/12/2023 first round will only accomodate the Chat models until gpt-35-turbo-instruct is released
        available_models = {  
            "Chat": ["gpt-4", "gpt-4-32k", "gpt-35-turbo", "gpt-35-turbo-16k"],  
            #"Completion": ["text-davinci-003"],  
            #"Embedding": ["text-embedding-ada-002"]  
        }

        # ###UPDATE model_options hard-coded to Chat for now 07/12/2023
        model_options = available_models["Chat"]

        # Set a default value for the model
        default_index = model_options.index("gpt-35-turbo-16k")

        # model_options
        model = st.sidebar.selectbox("Choose a model:", model_options, key="modelkey", index=default_index,
                                    help='''Choose the model with which you wish to interact. Defaults to gpt-35-turbo-16k.
                                    You can select the original GPT-35-Turbo (ChatGPT) model with 4k or 16k token contexts.
                                    The GPT-4 models with 8k or 32k token contexts are also available.''')

        # Then, when a model is selected, load the parameters for that model
        if model is not None:
            params = helpers.model_params[model]
        else:
            params = helpers.model_params["gpt-35-turbo-16k"]
        
        # Create a system message box so users may supply their own system message
        system_message = st.sidebar.text_area("System Message",
                                            value=st.session_state.system,
                                            height=150,
                                            key="txtSystem",
                                            help='''Enter a system message here.
                                            This is where you define the personality, rules of behavior, and guardrails for your Assistant.
                                            Don't forget to click the "Save Settings" button after making changes.''')

        # Check if the system message has been updated and the save button has not been clicked
        if system_message != st.session_state.system:
            st.sidebar.warning('''WARNING: You have made changes to the system message.
                            Click the "SaveSettings" to save the new message!''')

        # Read in the appropriate model specific parameters for the streamlit sliders - these all come from the dictionary in aoai_helpers.py
        # These are passed into the appropriate helpers.generate_ function calls
        # Default values are set with value= and are not defined in the dictionary
        temperature = st.sidebar.slider(label="Set a Temperature:", min_value=params['temp_min'], max_value=params['temp_max'], value=st.session_state.temperature,
                                        step=params["temp_step"], help=params['temp_help'], key="tempkey")
        max_tokens = st.sidebar.slider(label="Set Max Tokens per Response:", min_value=params['tokens_min'], max_value=params['tokens_max'], value=st.session_state.maxtokens,
                                    step=params["tokens_step"], help=params['tokens_help'], key="tokenskey")
        top_p = st.sidebar.slider(label="Set a Top P:", min_value=params['top_p_min'], max_value=params['top_p_max'], value=st.session_state.topp,
                                step=params["top_p_step"], help=params['top_p_help'], key="top_pkey")
        frequency_penalty = st.sidebar.slider(label="Set a Frequency Penalty:", min_value=params['frequency_penalty_min'], max_value=params['frequency_penalty_max'], value=st.session_state.frequencypenalty,
                                            step=params["frequency_penalty_step"], help=params['frequency_penalty_help'], key="frequency_penaltykey") 
        presence_penalty = st.sidebar.slider(label="Set a Presence Penalty:", min_value=params['presence_penalty_min'], max_value=params['presence_penalty_max'], value=st.session_state.presencepenalty,
                                            step=params["presence_penalty_step"], help=params['presence_penalty_help'], key="presence_penaltykey")

        # Save the chosen parameters to the system state upon submission
        if st.sidebar.button("Save Settings",
                    on_click=helpers.save_session_state(),
                    key="saveButton",
                    help='''Save the model parameter settings to the session state.''',
                    type="primary"):    
            st.sidebar.success('Settings saved successfully!', icon="✅")
        
        if st.sidebar.button("Clear Chat History",
                    help='''Clear the chat history from the session state.''',
                    key="mainChatClear",
                    type="secondary"):
            st.session_state.messages = []
            st.session_state.messages.append({"role":"system","content":standard_system_message})
            st.session_state["user_message"] = ""
            st.session_state["assistant_message"] = ""
            helpers.save_session_state()

    token_counter_container = st.sidebar.container()
    with token_counter_container:
        st.empty()  # Clear the container before updating
        st.caption(f":red[_____________________________________]")
        model = helpers.translate_engine_to_model(st.session_state.engine)
        system_tokens = helpers.num_tokens_from_messages([st.session_state.messages[0]], model)
        user_tokens = helpers.num_tokens_from_messages([msg for msg in st.session_state.messages if msg["role"] == "user"], model)
        assistant_tokens = helpers.num_tokens_from_messages([msg for msg in st.session_state.messages if msg["role"] == "assistant"], model)
        total_tokens = system_tokens + user_tokens + assistant_tokens

        st.write(f"System tokens: {system_tokens}")
        st.write(f"User tokens: {user_tokens}")
        st.write(f"Assistant tokens: {assistant_tokens}")
        st.write(f"Total tokens: {total_tokens}")

        # Progress bar
        max_tokens_progress = params['tokens_max']
        progress = total_tokens / max_tokens_progress
        # Uncomment to check values for max tokens and progress
        # st.write(f"Max tokens: {max_tokens_progress}")
        # st.write(f"Progress: {progress}")
        st.progress(progress)

with chat_container:
    if st.session_state['messages']:
        for message in st.session_state.messages:
            if message["role"] != "system":  # Skip system message
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])  

    if prompt := st.chat_input("💬 Window - Go ahead and type!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for response in helpers.generate_chat_completion(engine=st.session_state.engine,
                                                            messages=[
                                                                {"role": m["role"], "content": m["content"]}
                                                                for m in st.session_state.messages
                                                            ],
                                                            temperature=st.session_state.temperature,
                                                            max_tokens=st.session_state.maxtokens,
                                                            top_p=st.session_state.topp,
                                                            frequency_penalty=st.session_state.frequencypenalty,
                                                            presence_penalty=st.session_state.presencepenalty,
                                                            stop=None,
                                                            stream=True):

                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

with footer_container:
    st.caption(f":red[______________________________________________________________________________________________]")
    st.caption(f":red[NOTE: ALL SYSTEM MESSAGES, PROMPTS, AND COMPLETIONS ARE LOGGED FOR THIS DEMO. DO NOT ENTER ANY SENSITIVE INFORMATION.]")

# # Path: aoai-streamlit-app\src\aoai_streamlit_app.py

# ###UPDATE 07/16/2023 - Add in ability to pass parameters for apim endpoint, key, and aoai_version
# '''
# import argparse
# import os
# import streamlit as st

# # Define the command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--apim-endpoint', type=str, required=True, help='The Azure OpenAI API Management endpoint')
# parser.add_argument('--apim-key', type=str, required=True, help='The Azure OpenAI API Management key')
# args = parser.parse_args()

# # Set the environment variables
# os.environ['APIM_ENDPOINT'] = args.apim_endpoint
# os.environ['APIM_KEY'] = args.apim_key

# # Define the Streamlit app
# def main():
#     # Your Streamlit app code here
#     pass

# if __name__ == '__main__':
#     main()
# '''
import openai
import streamlit as st
from schema import LlmModelType, ConversationSettings



st.subheader("Simulation")

openai.api_key = st.secrets["OPENAI_API_KEY"]

settings = ConversationSettings()
model_names = [enum.value for enum in LlmModelType.openAI_models()]

with st.sidebar:
    
    #import settings
    with st.expander("Import Settings"):
        uploaded_file = st.file_uploader("" , key="import_upload")
        if uploaded_file is not None:
            
            bytes_data = uploaded_file.read()
            settings = ConversationSettings.parse_raw(bytes_data)

    tab1, tab2 = st.tabs(["Simulation", "Evaluation"])

    with tab1:
    
        #temperature
        settings.temperature = st.slider("Temperature", 0.0 ,1.0  ,settings.temperature)

        #LLM model
        model_names = [enum.value for enum in LlmModelType.openAI_models()]
        model_name = st.selectbox('Model', model_names, index=model_names.index(settings.llm_model_type.value))
        selected_model = LlmModelType(model_name)
        settings.llm_model_type = selected_model

        #title
        settings.title = st.text_input('Title', settings.title  , key="title")

        #first message
        settings.first_message = st.text_area('First message', settings.first_message , height=100 , key="first_message_area")
        '''placeholders: <<title>>'''

        #content
        with st.expander("Content"):
            uploaded_file = st.file_uploader("", key="content_upload")
            if uploaded_file is not None:
                    settings.content = uploaded_file.read().decode('utf-8')
                    print("#############$$$$$$$$$$$$$$%%%%%%%%%%%",uploaded_file.name)
            
            settings.content = st.text_area('', settings.content , height=400 , key="content_area")
    

        #role
        with st.expander("Role"):
            uploaded_file = st.file_uploader("", key="role_upload")
            if uploaded_file is not None:
                    settings.role = uploaded_file.read().decode('utf-8')
            
            settings.role = st.text_area('', settings.role , height=400 , key="role_area")
            '''placeholders: <<content>> , <<title>>'''
        
        #instructions
        with st.expander("Instructions"):
            uploaded_file = st.file_uploader("", key="instructions_upload")
            if uploaded_file is not None:
                    settings.instructions = uploaded_file.read().decode('utf-8')
            
            settings.instructions = st.text_area('', settings.instructions , height=400 , key="instructions_area")
            '''placeholders: <<content>> , <<title>>'''

    with tab2:
        #role
        with st.expander("Evaluation Role"):
            uploaded_file = st.file_uploader("", key="eval_role_upload")
            if uploaded_file is not None:
                    settings.eval_role = uploaded_file.read().decode('utf-8')
            
            settings.eval_role = st.text_area('', settings.eval_role , height=400 , key="eval_role_area")
            '''placeholders: <<content>> , <<title>>'''
        
        #instructions
        with st.expander("Instructions"):
            uploaded_file = st.file_uploader("", key="eval_instructions_upload")
            if uploaded_file is not None:
                    settings.eval_instructions = uploaded_file.read().decode('utf-8')
            
            settings.eval_instructions = st.text_area('', settings.eval_instructions , height=400 , key="eval_instructions_area")
            '''placeholders: <<content>> , <<title>>'''


def start():

    role = settings.role
    instructions = settings.instructions
    first_message = settings.first_message

    for place_holder , injection in [("<<content>>" , settings.content),("<<title>>" , settings.title) ]:
        role = role.replace(place_holder, injection)
        instructions = instructions.replace(place_holder, injection)
        first_message = first_message.replace(place_holder, injection)

    st.session_state.messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": instructions},
        {"role": "assistant", "content": first_message}
        ]
    
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"


st.button("Start",on_click=start)



# if "messages" not in st.session_state:
#     st.session_state.messages = [
#          {"role": "system", "content": settings.role},
#          {"role": "user", "content": settings.instructions},
#          {"role": "assistant", "content": settings.first_message}
#          ]

if "messages" in st.session_state:

    for message in st.session_state.messages[2:]:
        if message["role"] == "system": continue

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            print(selected_model.value)
            for response in openai.ChatCompletion.create(
                model=selected_model.value,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

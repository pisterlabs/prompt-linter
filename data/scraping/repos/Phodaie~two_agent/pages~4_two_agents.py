import streamlit as st
import base64

import os
import openai
import time
from datetime import datetime
import json

from elevenlabs import clone, generate, play, set_api_key
from elevenlabs.api import History

from schema import Agent
from schema import LlmModelType, get_completion_from_messages
from schema import TwoAgentsSettings
from utility.file_import_export import create_download_link 


st.subheader('Two Agents')

settings = TwoAgentsSettings()

with st.sidebar:

    with st.expander("Import Settings"):
        uploaded_file = st.file_uploader("")
        if uploaded_file is not None:
            
            bytes_data = uploaded_file.read()
            settings = TwoAgentsSettings.parse_raw(bytes_data)



    # UI for agents
    for agent in [settings.agent1 , settings.agent2]:
        with st.expander(agent.title):
            agent.title = st.text_input('Title', agent.title  , key=agent.title)
            agent.role = st.text_area('Role Description', agent.role, height=400)
            if agent.first_message:
                agent.first_message = st.text_area('First Message', agent.first_message, height=100)
                    

    #UI of other settings      
    settings.temperature = st.slider("Temperature", 0.0 ,1.0  ,settings.temperature)

    model_names = [enum.value for enum in LlmModelType]
    model_name = st.selectbox('Model', model_names, index=model_names.index(settings.llm_model_type.value))
    selected_model = LlmModelType(model_name)


    settings.number_of_turns = st.number_input("Number of exchanges" , settings.number_of_turns , 10)

    start = st.button("Start")
    
    settings.llm_model_type = selected_model
    
    download_settings = create_download_link(settings.json(), 'settings.json', 'Click here to download settings')
    st.markdown(download_settings, unsafe_allow_html=True)



    
messages =  [  
{'role':'system', 'content': settings.agent2.role},    
{'role':'user', 'content': settings.agent1.first_message},  
] 


if start:
    total_cost = 0
    total_seconds = 0
    st.write(f"**{settings.agent1.title}**")
    st.write(messages[1]["content"])

    for i in range(1, settings.number_of_turns):

        st.markdown(f"**{settings.agent1.title if i%2 == 0 else settings.agent2.title}**")
        #st.write(messages)
        start_time = time.time()
        with st.spinner('...'):
            try:
                response , usage = get_completion_from_messages(messages, temperature=settings.temperature , model=selected_model)
            except openai.error.Timeout as e:
                st.error(f"OpenAI API request timed out: {e}")
                break
            except openai.error.APIError as e:
                st.error(f"OpenAI API returned an API Error: {e}")
                break
            except openai.error.APIConnectionError as e:
                st.error(f"OpenAI API request failed to connect: {e}")
                break
            except openai.error.InvalidRequestError as e:
                st.error(f"OpenAI API request was invalid: {e}")
                break
            except openai.error.AuthenticationError as e:
                st.error(f"OpenAI API request was not authorized: {e}")
                break
            except openai.error.PermissionError as e:
                st.error(f"OpenAI API request was not permitted: {e}")
                break
            except openai.error.RateLimitError as e:
                st.error(f"OpenAI API request exceeded rate limit: {e}")
                break
            except Exception as e:
                st.error(f"OpenAI API: {e}")
                break
        
            set_api_key("e78bc29cdc5b72d0760c84e57078786c")

            audio = generate(
                text=response,
                voice="Bella" if i%2 == 0 else "Adam" ,
                model='eleven_monolingual_v1'
            )

            #st.audio(data=audio)
            play(audio)
        
        end_time = time.time()
        execution_time = end_time - start_time
        total_seconds += execution_time

        

        st.write(response)
        
        cost = selected_model.cost(usage)
        total_cost += cost


        st.write(f'*{round(execution_time, 2)} sec , {round(cost, 2)} cents*')

        messages.append({'role':'assistant' if i%2 == 0 else "user" , 'content' : response })
        
        #switch roles
        new_messages = [{'role' :'system' , 'content' : settings.agent2.role if i%2 == 0 else settings.agent1.role}]
        for j in range(1,len(messages)):
            new_message = {'role':'user' if i%2 == 0 else "assistant" , 'content' : messages[j]['content'] }
            new_messages.append(new_message)

        messages = new_messages


    #total cost and time
    minutes, seconds = divmod(total_seconds, 60)
    time_format = f"{minutes:.0f}:{seconds:02.0f}"   
    st.write(f'**total cost : {round(total_cost,2)} cents** in *{time_format} seconds*')

    #download
    download_str = ""
    for i in range(1,len(messages)):
        role = settings.agent2.title if i%2 == 0 else settings.agent1.title
        download_str += f'''{role} : 

{messages[i]['content']}

'''

    filename = 'two agent' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.txt' 
    download_link = create_download_link(download_str, filename, 'Click here to download the conversation')
    st.markdown(download_link, unsafe_allow_html=True)



# def get_completion_from_messages(messages, 
#                                  model=LlmModelType, 
#                                  temperature=0, 
#                                  max_tokens=1000):
    
#     try:
#         response = openai.ChatCompletion.create(
#             model=model.value,
#             messages=messages,
#             temperature=temperature, 
#             max_tokens=max_tokens, 
#         )
#     except:       
#         raise

#     return (response.choices[0].message["content"] , response["usage"])

def create_download_link(string, filename, text):
    # Encode the string as bytes
    string_bytes = string.encode('utf-8')
    
    # Create a base64 representation of the bytes
    base64_str = base64.b64encode(string_bytes).decode('utf-8')
    
    # Create the download link
    href = f'<a href="data:file/txt;base64,{base64_str}" download="{filename}">{text}</a>'
    return href
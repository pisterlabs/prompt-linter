from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_csv_agent
from langchain.llms import AzureOpenAI
from streamlit_chat import message
import streamlit as st
import pandas as pd
import random 
import os



st.set_page_config(page_title="CSV investigator Chatbot", page_icon=":robot_face:")
st.title("CSV investigator Chatbot")
st.markdown("This is a chatbot that can answer questions about the csv")

runner = False
with st.sidebar:
    key = st.text_input('API Key', type='password')
    base = st.text_input('API Base', value='https://demoforisraeli1.openai.azure.com/')
    
    deployment_name = st.text_input('Deployment Name')
    model_name = st.text_input('Model Name')
    runner =  st.checkbox('Submit')
    if runner:
        st.success('This is a success message!', icon="✅")
    


if runner:
        
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2022-12-01"
    os.environ["OPENAI_API_BASE"] = base
    os.environ["OPENAI_API_KEY"] = key


    input_file = st.file_uploader("Upload CSV", type=["csv"])


    if input_file is not None:

        df = pd.read_csv(input_file)

        st.dataframe(df)
    
        files_data = f'file-{random.randint(1,1000)}.csv'
        
        df.to_csv(files_data,index=False)

        agent = create_csv_agent(AzureOpenAI(temperature=0 ,
                                            verbose=True,
                                            deployment_name=deployment_name,
                                            model_name=model_name, 
                                            max_tokens=1000),files_data)
        agent.agent.llm_chain.prompt.template


        st.session_state['generated'] = []
        st.session_state['past'] = []


        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
            
            

        user_input=st.text_input("You:",key='input')

        if user_input:
            output=agent.run(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user') 
                    st.session_state.generated = ''


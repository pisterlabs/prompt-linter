#import libraries
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

from specklepy.api.wrapper import StreamWrapper
from specklepy.api.client import SpeckleClient
from specklepy.api import operations

#functions
def chat_speckle(df,prompt):
    llm = OpenAI(api_token = OPENAI_API_KEY)
    pandas_ai = PandasAI(llm, conversational=False)
    result = pandas_ai.run(df, prompt=prompt)
    print(result)
    return result

# get parameter names
def get_parameter_names(commit_data, selected_category):
    parameters = commit_data[selected_category][0]["parameters"].get_dynamic_member_names()
    parameters_names = []
    for parameter in parameters:
        parameters_names.append(commit_data[selected_category][0]["parameters"][parameter]["name"])
    parameters_names = sorted(parameters_names)
    return parameters_names

#get parameter value by parameter name
def get_parameter_by_name(element, parameter_name, dict):
    for parameter in parameters:
        key = element["parameters"][parameter]["name"]
        if key == parameter_name:
            dict[key] = element["parameters"][parameter]["value"]
    return dict

#containers 📦

header = st.container()
input = st.container()
data = st.container()

#header 
with header:
    st.title('chatSpeckle 🗣️🔷')
    st.info('This web app allows you to chat with your AEC data using Speckle and OpenAI')

#inputs
with input:
    st.subheader('Inputs 📁')
    commit_url = st.text_input('Commit URL',"https://speckle.xyz/streams/70d2d5b6d4/commits/3abca7abfb")

#wrapper
wrapper = StreamWrapper(commit_url)
#client
client = wrapper.get_client()
#trasnport
transport = wrapper.get_transport()

#get speckle commit
commit = client.commit.get(wrapper.stream_id, wrapper.commit_id)
#get object id from commit 
obj_id = commit.referencedObject
#receive objects from commit
commit_data = operations.receive(obj_id, transport)

with input:
    selected_category = st.selectbox("Select category", commit_data.get_dynamic_member_names())

#parameters
parameters = commit_data[selected_category][0]["parameters"].get_dynamic_member_names()

with input:
    form = st.form("parameter_input")
    with form:
        selected_parameters = st.multiselect("Select Parameters", get_parameter_names(commit_data, selected_category))
        run_button = st.form_submit_button('RUN')

category_elements = commit_data[selected_category]

with data:
    st.subheader("Data 📚")
    result_data = []
    for element in category_elements:
        dict = {}
        for s_param in selected_parameters:
            get_parameter_by_name(element, s_param, dict)
        result_data.append(dict)
    result_DF = pd.DataFrame.from_dict(result_data)
    
    #show dataframe and add chatSpeckle feature 
    col1, col2 = st.columns([1,1])
    with col1:
        result = st.dataframe(result_DF)
        
    with col2:
        st.info("⬇️chatSpeckle⬇️")
        OPENAI_API_KEY = st.text_input('OpenAI key',"sk-...vDlY")

        input_text = st.text_area('Enter your query')
        if input_text is not None:
            if st.button ("Send"):
                st.info('Your query:'+ input_text)
                result = chat_speckle(result_DF, input_text)
                st.success(result)
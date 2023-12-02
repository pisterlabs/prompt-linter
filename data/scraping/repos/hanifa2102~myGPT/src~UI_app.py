import streamlit as st
import pandas as pd
import os
import openai
from st_aggrid import AgGrid,GridUpdateMode
import sys
sys.path.append('..')
from llm_api import LLM_API
from db import DB
from apikey import apikey
from utility import Utility

os.environ['OPENAI_API_KEY']=apikey
openai.api_key  = os.getenv('OPENAI_API_KEY')
st.set_page_config(layout="wide")
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

def sendLabelling(df,label,isUNSPSC=False):
    st.write("<font color='red'>Send Server.</font>", unsafe_allow_html=True)
    input=Utility.csv_to_openai(df) 

    with st.expander('Inputs'):
        st.write(input[0])
        AgGrid(input[1],theme='dark',key='gridsendServer0')
        AgGrid(input[2],theme='dark',key='gridsendServer1')

    st.divider()

    if isUNSPSC:
        result=LLM_API.do_UNSPSC_label(input[0],label)
    else:
        result=LLM_API.do_label(input[0],label)
    st.subheader('After Classifying with OpenAI Api.......')
    with st.expander('Response'):
        st.write(result)
    merged_df=pd.merge(input[2],result,on=['index'],how='inner')
    st.success(f"Data has been labelled with : {label}")
    st.dataframe(merged_df)
    # BUG : Not able to refresh the AgGrid tables 
    # AgGrid(merged_df,theme='dark',key='gridsendServer2')
  
# UI Components
def upload_file(key):
    uploaded_file = st.file_uploader("Upload Input File", type={"csv", "txt"},key=key)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# def text_input_callback():
#     st.sidebar.write("<font color='red'>Not Implemented.</font>", unsafe_allow_html=True)

def main():

    # Sidebar components
    # prompt_text_input = st.sidebar.text_input("Pls enter the label", on_change=text_input_callback,placeholder="...")
    st.title("DARC LLM Usage ")
    tab1,tab2=st.tabs(["Labelling tabular data : ","Natural Language Querying : "])
    
    with tab1:
        st.header("Labelling tabular data : ")
        st.caption("Load a csv file.Choose the label to annotate the data with. Click ChatGPT Label.")
        tab1_df=upload_file('label')

        # Create a row layout using the 'with' statement
        col1, col2,col3 = st.columns(3)
        with col1:
            btn=st.button("ChatGPT Label:")
        with col2:
            user_prompt_input = st.text_input(".",placeholder="Enter Label to annonate the data: ",label_visibility="collapsed")
        with col3:
            sp_btn=st.button("UNSPSC Button.")

        if tab1_df is not None:
            # Editable cells   
            grid_return=AgGrid(tab1_df,editable=True,height=180,theme='dark',key='gridmain0')
            tab1_df=grid_return['data']
         
        if btn and user_prompt_input and tab1_df is not None:
            sendLabelling(tab1_df,user_prompt_input)
        elif sp_btn and tab1_df is not None:
            sendLabelling(tab1_df,'Family(UNSPSC)',isUNSPSC=True)
        else:
            st.error("Label is required.")
        
        # if sp_btn and tab1_df is not None:
        #     sendLabelling(tab1_df,'Family',isUNSPSC=True)
        #     # st.success("UNSPSC Coding done.")

    with tab2 :
        st.header("Natural Language Querying : ")
        st.caption("Input file with text is loaded.")
        tab2_df=upload_file('query')
        if tab2_df is not None:
            st.success('File Uploaded. Loading to DB.')
            DB.pushDb(tab2_df)
            AgGrid(DB.readDB(),theme='dark',key='gridmain2')
            st.success("DataFrame saved to database successfully!")

        text_input = st.text_input("Enter Natural Query for the above table",placeholder="Find the number of fruits per Shape")
        if st.button("ChatGPT Query"):
            print(text_input)
            with st.expander('Table Schema'):
                st.write(DB.getColsDB())
            response=LLM_API.get_sql(text_input)
            st.success("Query Result displayed below!")
            temp_df=DB.readDB(response)
            with st.expander('Response'):
                st.write(response)
            st.dataframe(temp_df)    
            # AgGrid(temp_df,theme='dark',key='gridmain3',reload_data=False, update_mode=GridUpdateMode.MODEL_CHANGED)
            


if __name__ == "__main__":
    main()
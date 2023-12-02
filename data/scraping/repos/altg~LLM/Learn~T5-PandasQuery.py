import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Data App')
st.title('🦜🔗 Ask the Data App')

# Load CSV file
def load_csv(input_csv):
  df = pd.read_csv(input_csv)
  with st.expander('See DataFrame'):
    st.write(df)
  return df


def load_csv2():
  input_csv = "../Docs/idb_ops_data.csv"
  df = pd.read_csv(input_csv)
  return df[['ops_data_ID', 'operation_id', 'project_code', 'project_title',
       'profile', 'country', 'fund', 'is_member_country',
       'country_code', 'country_ldmc', 'country_class', 'country_region',
       'sector', 'status', 
       'main_mode_of_finance', 'sub_mode_of_finance', 
       'date_of_approval', 'date_of_signature', 
       'date_of_effective', 'date_of_first_disb',
       'date_of_last_disb', 
       'approval_amt_usd',  'disb_amt_usd'
       ]].sample(100)

# Generate LLM response
def generate_response(csv_file, input_query):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.0)
  #llm = ChatOpenAI(model_name='gpt-4', temperature=0.0)
  df = load_csv2()
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.success(response)

# Input widgets
#uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
uploaded_file = True
question_list = [
  'How many rows are there?',
  'What is the range of values for MolWt with logS greater than 0?',
  'How many rows have MolLogP value greater than 0.',
  'Other']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
#openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text is 'Other':
  query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...', disabled=not uploaded_file)

generate_response(uploaded_file, query_text)

# if not openai_api_key.startswith('sk-'):
#   st.warning('Please enter your OpenAI API key!', icon='⚠')
# if openai_api_key.startswith('sk-') and (uploaded_file is not None):
#   st.header('Output')
#   generate_response(uploaded_file, query_text)
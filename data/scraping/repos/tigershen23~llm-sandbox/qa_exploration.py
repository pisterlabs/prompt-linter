# cSpell:disable
# Backing functions for Question-Answering exploration

import os
path = os.path.dirname(__file__)

from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain import OpenAI
import streamlit as st

#region marketing site supporting code
# Set up document QA index
@st.experimental_singleton
def get_marketing_site_index():
  saved_path = path + "/gpt_indexes/website/welcome_marketing.json"
  if os.path.exists(saved_path):
    return GPTSimpleVectorIndex.load_from_disk(saved_path)
  else:
    welcome_marketing_data = SimpleDirectoryReader(
      path + "/data/website/welcome_marketing",
      recursive=True,
      required_exts=[".jsonl"],
    ).load_data()
    welcome_marketing_index = GPTSimpleVectorIndex(welcome_marketing_data)
    welcome_marketing_index.save_to_disk(saved_path)
    return welcome_marketing_index

# Query DB
def query_marketing_site_db(query: str):
  return get_marketing_site_index().query(query, verbose=True)

# Create LangChain agent
@st.experimental_memo
def get_marketing_site_agent():
  tools = [
    Tool(
      name="QueryingDB",
      func=query_marketing_site_db,
      description="Returns most relevant answer from document for query string",
    )
  ]
  llm = OpenAI(temperature=0.0)
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  return agent
#endregion

#region blog supporting code
# Set up document QA index
@st.experimental_singleton
def get_blog_index():
  saved_path = path + "/gpt_indexes/website/welcome_blog.json"
  if os.path.exists(saved_path):
    return GPTSimpleVectorIndex.load_from_disk(saved_path)
  else:
    welcome_blog_data = SimpleDirectoryReader(
      path + "/data/website/welcome_blog",
      recursive=True,
      required_exts=[".jsonl"],
    ).load_data()
    welcome_blog_index = GPTSimpleVectorIndex(welcome_blog_data)
    welcome_blog_index.save_to_disk(saved_path)
    return welcome_blog_index

# Query DB
def query_blog_db(query: str):
  return get_blog_index().query(query, verbose=True)

# Create LangChain agent
@st.experimental_memo
def get_blog_agent():
  tools = [
    Tool(
      name="QueryingDB",
      func=query_blog_db,
      description="Returns most relevant answer from document for query string",
    )
  ]
  llm = OpenAI(temperature=0.0)
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  return agent
#endregion

#region Zendesk supporting code
# Set up document QA index
@st.experimental_singleton
def get_zendesk_index():
  saved_path = path + "/gpt_indexes/website/welcome_zendesk.json"
  if os.path.exists(saved_path):
    return GPTSimpleVectorIndex.load_from_disk(saved_path)
  else:
    welcome_zendesk_data = SimpleDirectoryReader(
      path + "/data/website/welcome_zendesk/2023-02-06",
      required_exts=[".html"],
    ).load_data()
    welcome_zendesk_index = GPTSimpleVectorIndex(welcome_zendesk_data)
    welcome_zendesk_index.save_to_disk(saved_path)
    return welcome_zendesk_index

# Query DB
def query_zendesk_db(query: str):
  return get_zendesk_index().query(query, verbose=True)

# Create LangChain agent
@st.experimental_memo
def get_zendesk_agent():
  tools = [
    Tool(
      name="QueryingDB",
      func=query_zendesk_db,
      description="Returns most relevant answer from document for query string",
    )
  ]
  llm = OpenAI(temperature=0.0)
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  return agent
#endregion

#region transcripts supporting code
def get_webinar_name_from_filename(full_path):
  # Get the filename from the full path
  filename = full_path.split('/')[-1]

  # Split the filename by underscore and remove the first element (the number)
  parts = filename.split('_')[1:]

  # Remove the '.srt' extension from the last element
  parts[-1] = parts[-1].replace('.srt', '')

  # Join the remaining elements with a space to create the desired output
  webinar_name = ' '.join(parts)

  return { "webinar_name": webinar_name }

@st.experimental_singleton
def get_transcripts_index():
  saved_path = path + "/gpt_indexes/transcripts/org_243.json"
  if os.path.exists(saved_path):
    return GPTSimpleVectorIndex.load_from_disk(saved_path)
  else:
    transcripts_data = SimpleDirectoryReader(
      path + "/data/transcripts",
      required_exts=[".srt"],
      file_metadata=get_webinar_name_from_filename,
      recursive=True
    ).load_data()
    transcripts_index = GPTSimpleVectorIndex(transcripts_data)
    transcripts_index.save_to_disk(saved_path)

    return transcripts_index

def query_transcripts_db(query: str):
  return get_transcripts_index().query(query, verbose=True)

@st.experimental_memo
def get_transcripts_agent():
  tools = [
    Tool(
      name="QueryingDB",
      func=query_transcripts_db,
      description="Returns most relevant answer from document for query string",
    )
  ]
  llm = OpenAI(temperature=0.0)
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  return agent
#endregion

#region combined supporting code
# Set up document QA index
@st.experimental_singleton
def get_combined_index():
  saved_path = path + "/gpt_indexes/website/welcome_combined.json"
  if os.path.exists(saved_path):
    return GPTSimpleVectorIndex.load_from_disk(saved_path)
  else:
    welcome_combined_data = SimpleDirectoryReader(
      path + "/data/website",
      recursive=True,
      required_exts=[".jsonl"],
    ).load_data()
    welcome_combined_index = GPTSimpleVectorIndex(welcome_combined_data)
    welcome_combined_index.save_to_disk(saved_path)
    return welcome_combined_index

# Query DB
def query_combined_db(query: str):
  return get_combined_index().query(query, verbose=True)

# Create LangChain agent
@st.experimental_memo
def get_combined_agent():
  tools = [
    Tool(
      name="QueryingDB",
      func=query_combined_db,
      description="Returns most relevant answer from document for query string",
    )
  ]
  llm = OpenAI(temperature=0.0)
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  return agent
#endregion


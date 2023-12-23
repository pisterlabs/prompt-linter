from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import WikipediaAPIWrapper 

from dotenv import load_dotenv

import streamlit as st

load_dotenv()

st.title( 'Youtube Assistant' )

prompt = st.text_input( 'Ask a question about a youtube video' )

llm = OpenAI( temperature = 0.9 )

title_memory = ConversationBufferMemory(
  input_key = 'topic',
  memory_key = 'chat_history'
)

script_memory = ConversationBufferMemory(
  input_key = 'title',
  memory_key = 'chat_history'
)

title_template = PromptTemplate(
  input_variables = [ 'topic' ],
  template = 'write me a youtube video title about {topic}'
)

title_chain = LLMChain( 
  llm = llm, 
  prompt = title_template,
  output_key = 'title',
  memory = title_memory
)

script_template = PromptTemplate(
  input_variables = [ 'title', 'wikipedia_research' ],
  template = 'write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch: {wikipedia_research}'
)

script_chain = LLMChain( 
  llm = llm,
  prompt = script_template,
  output_key = 'script',
  memory = script_memory
)

wiki = WikipediaAPIWrapper()

if prompt:
  title = title_chain.run( prompt )
  wiki_research = wiki.run( prompt )

  script = script_chain.run( 
    title = title,
    wikipedia_research = wiki_research
  )

  st.write( title )
  st.write( script )

  with st.expander( 'Title History' ): 
      st.info( title_memory.buffer )

  with st.expander( 'Script History' ): 
      st.info( script_memory.buffer )

  with st.expander( 'Wikipedia Research' ): 
      st.info( wiki_research )
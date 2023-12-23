from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from dotenv import load_dotenv

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

load_dotenv()

st.set_page_config( page_title = 'Hugging Chat', page_icon = '🤗', layout = 'wide' )

with st.sidebar:
  st.title( 'Hugging Chat' )
  st.markdown( 
    '''
      ## About
      This app is an LLM-powered chatbot built using:
      - [Streamlit](https://streamlit.io/)
      - [LangChain](https://python.langchain.com/)
      - [OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5) LLM model
    ''' 
  )
  add_vertical_space( 3 )
  st.write( 'Made by recursive-reshy' )

st.header( 'Your Personal Assistant' )
    
def main():

  # Placeholder that we use at the start of script
  if 'generated' not in st.session_state:
    st.session_state[ 'generated' ] = [ 'I\'m Assistant, How may I help you?' ]

  # User question
  if 'user' not in st.session_state:
    st.session_state[ 'user' ] = [ 'Hi!' ]

  response_container = st.container()
  colored_header( 
    label = '', 
    description = '', 
    color_name = 'blue-30'
  )

  input_container = st.container()

  get_text = lambda : st.text_input( 'You', '', key = 'input' )

  with input_container:
    user_input = get_text()
  
  chain_setup = lambda : LLMChain(
    llm = HuggingFaceHub( 
      repo_id = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
      model_kwargs = { 'max_new_tokens': 1200 }
    ),
    prompt = PromptTemplate(
      input_variables = [ 'question' ],
      template = '''
        <|prompter|>{question}<|endoftext|>
        <|assistant|>
      '''
    )
  )

  generate_response = lambda question, llm_chain : llm_chain.run( question )

  llm_chain = chain_setup()

  with response_container:
    if user_input:
      st.session_state.user.append( user_input )
      st.session_state.generated.append( generate_response( user_input, llm_chain ) )
    
    if st.session_state[ 'generated' ]:
      for i in range( len( st.session_state[ 'generated' ] ) ):
        message( 
          st.session_state[ 'user' ][ i ],
          is_user = True,
          key = str( i ) + '_user'
        )
        message( 
          st.session_state[ 'generated' ][ i ],
          key = str( i )
        )

if __name__ == '__main__':
  main()
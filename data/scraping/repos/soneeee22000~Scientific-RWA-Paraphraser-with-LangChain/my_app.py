# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain # our langchain is going to allow us to run our topic through our prompt template and then run it through our llm chain and then run it through our sequential chain(generate our response)
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import create_csv_agent



#load_dotenv()

#filepath = r'C:\Users\pyaes\Downloads\Paraphrase-Langchain\ParaSCI\Data\train.csv'
#loader = CSVLoader(filepath, 'source', 'target')

# intialize the Vector index creator
#index = VectorstoreIndexCreator().from_loaders([loader])


# App framework
st.title('🦜 Seon\'s Scientific RWA Paraphraser 🔗')
prompt = st.text_input('Plug in your Prompt Sentence to be Paraphrased here!') 

#Prompt Templates
title_template = PromptTemplate(
    input_variables = ['sentences'],
    template = 'Paraphrase the following sentence for me in the most scientific research writing terms, please : {sentences}'
)

# Instantiate the OpenAI API the LLMS
llm = OpenAI(temperature=0.5)
title_chain = LLMChain(llm = llm ,prompt = title_template, verbose=True)

# Instantiate the prompt template # this will show stuff to the screen if there's a prompt 
if prompt:
    response = title_chain.run(sentences = prompt)
    st.write(response) 



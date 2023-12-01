from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import streamlit as st

llm = ChatOpenAI(temperature=0.0,openai_api_key=st.secrets["OPENAI_API_KEY"])

def categ_video( title: str, transcript: str) -> str:
    from prompts.chains import template_categ
    prompt = ChatPromptTemplate.from_template(template_categ)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = chain.run(title=title, transcript=transcript)
    print(response)
    return response
    
    
    
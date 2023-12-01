import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.agents import AgentType


st.title("EcosystemBuilderAgent")

st.write("This agent leverages the GPT large language model to automate the building of ecosystems for specific projects. Following the Value Blueprint Framework of Ron Adner (2012, The Wide Lens), this agent will (1) automatically identify important complementors and intermediaries for your project, (2) provide a description of the minimal viable ecosystem, and (3) analyze the most important ecosystem risks. This agent is developed by Dries Faems, Professor of Entrepreneurship, Innovation and Technological Transformation at the WHU Otto Beisheim School of Management.")

st.write("If you want to know more about the ecosystem value blueprint, check out the book of Ron Adner here: https://ronadner.com/book/the-wide-lens/. You can check out the first chapter for free!")
#define company and project

open_api_key = st.text_input('Enter your open api key. This information is not recorded or stored in any way', type = "password")

company = st.text_input('Provide the name of your company')
prompt = st.text_input('Provide a description of your ecosystem project')

clicked = st.button('Click me')

if clicked:
    llm = OpenAI(openai_api_key=open_api_key, temperature=0.8)
    # identification of complementors and intermediaries in the ecosystem
    complementors_template = PromptTemplate(
        input_variables = ['company', 'project_description'], 
        template='You are an ecosystem specialist that needs to identify the neccessary partners for {company} in the business ecosystem to realize {project_description}. Make a clear distinction between complementors and intermediaries. Complementors are entities that provide complementary goods or services that enhance the value of another companys product or service. They are not part of the direct supply chain but add value to the end product. Intermediaries, on the other hand, are entities that facilitate the connection between different stages of the value chain, often between the producer and the consumer. They do not produce the primary value but rather support its delivery.')
    complementors_chain = LLMChain(llm=llm, prompt = complementors_template, verbose=True)
    complementors = complementors_chain.run(company=company, project_description=prompt)
    st.markdown('**The most important complementors and intermediaries are**')
    st.write(complementors)
    #identfication of minimal viable ecosystem
    MVE_template = PromptTemplate(
        input_variables = ['company', 'project_description'],
        template = 'Provide a description of the minimal viable ecosystem that {company} needs to establish for realizing the following {project_description}.a Minimal Viable Ecosystem is the smallest possible configuration of elements that can successfully support the delivery and enhancement of a new innovation.')
    MVE_chain = LLMChain(llm=llm, prompt = MVE_template, verbose= True)
    MVE = MVE_chain.run(company=company, project_description= prompt)
    st.markdown('**Here is a description of the Minimal Viable Ecosystem**')
    st.write(MVE)
    #identification of ecosystem risks
    risks_template = PromptTemplate(
        input_variables = ['MVE'],
        template = 'Analyze the core risks of executing the following ecosytem: {MVE}.')
    Risk_chain = LLMChain(llm=llm, prompt = risks_template, verbose= True)
    risks = Risk_chain.run(MVE = MVE)
    st.markdown('**Here are the most important risks associated with executing the Minimal Viable Ecosystem**')
    st.write(risks)
else:
    st.write('Please click the button to perform an operation')

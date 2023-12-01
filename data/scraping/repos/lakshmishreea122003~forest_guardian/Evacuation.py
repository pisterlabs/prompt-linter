import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import joblib


st.set_page_config(
    page_title="Evacuation",
    page_icon="🌿",
)

st.markdown("<h1 style='color: green; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >Evacuation🚨🌲</h1> <h3 style='color: #00755E; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Use AI and ML to integrate fire prediction and deforestation data to recommend vital evacuations, ensuring safety from both wildfires and human impact. </h3>", unsafe_allow_html=True)

# #############
year = st.text_input("Enter the year")
fire_model = joblib.load(r"D:\llm projects\Forest-Amazon\models\fire_model.pkl")
deforestation_model = joblib.load(r"D:\llm projects\Forest-Amazon\models\forest_model.pkl")

evacuation_template = PromptTemplate(
        input_variables=['info'],
        template='{info}'
    )

llm = OpenAI(temperature=0.9)

evacuation_chain = LLMChain(llm=llm, prompt=evacuation_template, output_key ='evacuation')

if year is not None:
    ##################
    fire_res= fire_model.predict(year)
    deforestation_res= deforestation_model.predict(year)
    
    st.markdown("<p style='color: #004B49; font-style: italic; font-family: Comic Sans MS; ' >Fire prediction model</p>", unsafe_allow_html=True)
    st.write(fire_res)
    st.markdown("<p style='color: #004B49; font-style: italic; font-family: Comic Sans MS; ' >Amazon Deforestation prediction model</p>", unsafe_allow_html=True)
    st.write(deforestation_res)
    info  = f'In palces ACRE	AMAPA	AMAZONAS	MARANHAO	MATO GROSSO	PARA	RONDONIA	RORAIMA	TOCANTINS the firespotes in the year {year} is {fire_res} respectively and the deforestation rate is {deforestation_res} respectively. Consider the given data and let me know from which palce should be evacuated to protect the amazon forest and why that palce. Also tell where and how should they be evacuated. Give answer in points with tilte and subtitle.'

    evacuation = evacuation_chain.run(info)
    st.write(evacuation)




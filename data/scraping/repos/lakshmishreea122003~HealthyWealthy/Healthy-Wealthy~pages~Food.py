import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 

st.set_page_config(
    page_title="NutriGuide",
    page_icon="🍽️",
)

st.markdown("<h1 style='color: #3B444B; font-style: italic; font-family: Comic Sans MS; font-size:4rem' >Healthy Wealthy NutriGuide 🥦</h1> <h3 style='color:#54626F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Unleashing the Power of AI for Healthier Choices and Tasty Delights.</h3>", unsafe_allow_html=True)


prompt = st.text_input("Ask anything related to food")


health_template = PromptTemplate(
    input_variables = ['topic'], 
    template='From a health perspective, {topic}'
)

recipes_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'give healthy recipies related to the {topic}'
)

calories_template= PromptTemplate(
    input_variables = ['food'],
    template = 'give the average calories in the food recipie {food}'
)


# Llms
llm = OpenAI(temperature=0.9) 


health_chain = LLMChain(llm=llm, prompt=health_template, output_key='health')
recipes_chain = LLMChain(llm = llm , prompt=recipes_template, output_key = 'recipes')
calories_chain = LLMChain(llm=llm, prompt= calories_template,output_key = 'calories')



# Show stuff to the screen if there's a prompt
if prompt: 
    health = health_chain.run(prompt)
    recipes = recipes_chain.run(health)
    calories = calories_chain.run(recipes)
    
    st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >Health Perspective</p>", unsafe_allow_html=True)
    st.write(health)

    st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >Health Recipies for you</p>", unsafe_allow_html=True)
    st.write(recipes)

    st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >Calories of the above recipy</p>", unsafe_allow_html=True)
    st.write(calories)


 
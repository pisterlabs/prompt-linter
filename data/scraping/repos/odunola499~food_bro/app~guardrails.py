from utils import Func
from langchain.llms import OpenAI
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


func = Func()


st.set_page_config(layout="wide")
st.title("Food Recommendation Engine")
pipe = pipeline('text-classification','odunola/guardrail_distilled')


def greet(question):
    greeting_template = """
    Your name if Chef Lock, you are a highly knowledgeable chef with decades of experience and you are well versed in helping anyone create recipes, reccommend food or cuisine based on what they ask and designing food plans. 
    A user asks you the following question. Respond to this question with full knowledge of who you are an your capability.
    Question: {question}
    Response:
    """
    model = OpenAI(temperature = 0.9, model = 'gpt-3.5-turbo-instruct', max_tokens = 256)
    greeting_prompt = PromptTemplate.from_template(greeting_template)
    chain = LLMChain(llm=model, prompt=greeting_prompt)
    result = chain.run(question = question)
    return result


def unwanted_request(question):
    template = """
    Your name if Chef Lock, you are a highly knowledgeable chef with decades of experience and you are well versed in helping anyone create recipes, reccommend food or cuisine based on what they ask and designing food plans. 
    A user asks you the following question or give you a text. This particular question that no body should ask you and it is unwise of you to answer this. Respectively give a response that doesnt answer the question but turns down the request and ensure you also educatethe user on questions you can actually answer such as food related questions
    Question: {question}
    Response: 
    """
    model = OpenAI(temperature = 0.9, model = 'gpt-3.5-turbo-instruct', max_tokens = 256)
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)
    result = chain.run(question = question)
    return result

def short_question(user_input):
    result = func.RAG_short_content(user_input)
    st.write(result[0])
    with st.expander("See retrieved context"):
        st.write(result[1])



def long_question(user_input):
    result = func.RAG_large_content(user_input)
    st.write(result[0])
    with st.expander("See retrieved context"):
        st.write(result[1])

user_input = st.text_input("Enter Prompt")
if user_input:
    with st.spinner('loading...'):
        result = pipe(user_input)
        if result[0]['label'] == 'LABEL_2':
            st.write(unwanted_request(user_input))
        elif result[0]['label'] == 'LABEL_0':
            short_question(user_input)
        elif result[0]['label'] == 'LABEL_1':
            long_question(user_input)
        elif result[0]['label'] == 'LABEL_3':
            st.write(greet(user_input))




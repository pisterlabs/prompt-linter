from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st

@st.cache_resource
def question_chain():
    question_model_id = "valhalla/t5-base-e2e-qg"
    tokenizer = AutoTokenizer.from_pretrained(question_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(question_model_id)

    q_pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_length=100
    )
    question_model = HuggingFacePipeline(pipeline=q_pipe)
    q_template = "generate questions: {text} </s>"
    q_prompt = PromptTemplate(input_variables=["text"], template=q_template)
    q_chain = LLMChain(llm=question_model, prompt=q_prompt)
    return q_chain

def generate_qn(q_chain, context_list):
    questions = []
    
    for context in context_list:
        questions.append(q_chain.run(context).split("<sep>"))

    questions_flattened = [q for q_list in questions for q in q_list if q]
    # remove duplicates based on similarity in future
    return list(set(questions_flattened))
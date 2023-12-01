import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import  streamlit as st
generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)

prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

def gen_ai(context=None, prompt=None):

    if not context:
        llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
        answer= llm_context_chain.predict(instruction=f"{prompt}", context=context).lstrip()

    else:
        llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
        answer = llm_chain.predict(instruction=f"{prompt}").lstrip()

    return  answer


def side_bar():
    with st.sidebar:
        if st.button('Clear Chat'):
            # Clear the chat history
            st.session_state.messages = []
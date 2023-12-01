import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain import HuggingFacePipeline
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import datetime
from transformers import T5ForConditionalGeneration, AutoTokenizer

huggingfacehub_api_token="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"


def generate_response_starcoder(txt):
    repo_id = "bigcode/starcoder"

    starcoder_llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id, task='text-generation',
                     model_kwargs={"temperature":0,"max_new_tokens":1024, "return_full_text":True, "repetition_penalty":100})
    # print(txt)
    code_template = """Question: {question}

    Answer: """
        
    code_prompt = PromptTemplate(template=code_template, input_variables=["question"])
    
    code_llm_chain = LLMChain(prompt=code_prompt, llm=starcoder_llm)    
    code_output = code_llm_chain.run(txt)
    print(code_output)
    return str(code_output)

def generate_response_codet5(txt):
    repo_id = "Salesforce/codet5p-770m-py"
    device="cpu"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = T5ForConditionalGeneration.from_pretrained(repo_id)
    inputs = tokenizer.encode(txt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, temperature=0.1,max_new_tokens=1024)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return tokenizer.decode(outputs[0])

# Page title
st.set_page_config(page_title='Code Assistance App')
st.title('Code Assistance App')
# Text input
txt_input = st.text_area('Enter your text', '', height=200)
result = []
with st.form('summarize_form', clear_on_submit=True):
    options = st.selectbox(
    'Choose a model:',
    options=["starcoder", "codet5"])
    st.write('You selected:', options)
    submitted = st.form_submit_button('Submit')
    response = ''
    if submitted:
        with st.spinner('Calculating...'):
            st.success(' Processing started', icon="🆗")
        if options=="starcoder":            
            start = datetime.datetime.now()
            response = response + generate_response_starcoder(txt_input)
            stop = datetime.datetime.now() 
            elapsed = stop - start
            st.success(f'Completed in {elapsed}', icon="🆗")
        else:
            start = datetime.datetime.now()
            response = response + generate_response_codet5(txt_input)
            stop = datetime.datetime.now() 
            elapsed = stop - start
            st.success(f'Completed in {elapsed}', icon="🆗")
        
st.info(response)
response = ''
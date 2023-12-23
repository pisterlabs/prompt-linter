import os
import streamlit as st
import langchain
from langchain.document_loaders import Docx2txtLoader
from tempfile import NamedTemporaryFile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


default_prompt = """Your task is two.
first task is to write a summary of the document.
second task is classify the document. The classfication results should related with the 7 labels.
  labels: [policy about water, 
           disaster,
           water quality,
           water and sewer,
           dam, 
           water environment]
           
the document may contains multiple labels. so provide each label's probability.
the summary should be written as Korean.

the output should be a json format.

output example:
dict(
 "summary": "this is a summary",
 "labels": "policy about water": 0.1,
            "disaster": 0.2,
            "water quality": 0.3,
            "water and sewer": 0.4,
            "dam": 0.5,
            "water environment": 0.6
)

Document: {document}
"""


def main(name):
    st.write('Submit Your OpenAI API Key')
    openai_api_key = st.text_input('OpenAI API Key')
        
    uploaded_file = st.file_uploader("Upload a file")
    
    if (uploaded_file is not None) and (openai_api_key != ""):
        path_in = uploaded_file.name        
        suffix = path_in.split('.')[-1]
        with NamedTemporaryFile(dir='tmp', suffix=f'.{suffix}', delete=False) as f:
            f.write(uploaded_file.getbuffer())
            
        loader = Docx2txtLoader(f.name)
        data = loader.load()
        content = data[0].page_content
        
              
        st.text_area('Article', content, disabled=True, height=800)
        
        st.subheader('Default Prompt')
        st.write('The last line of Prompt should be "Document: {document}"')
        st.markdown('---')
        prompt = st.text_area('Prompt', default_prompt, height=800)
        
        prompt = PromptTemplate(template=prompt, input_variables=['document'])
        
        
        if openai_api_key is not None:
            llm = ChatOpenAI(model='gpt-3.5-turbo-16k', 
                    temperature=0.,
                    openai_api_key=openai_api_key)
            chain = LLMChain(llm=llm, prompt=prompt)

            submit = st.button('Submit')
            
            if submit:
                with st.spinner('Generating...'):
                    res = chain.run(document=content)
                st.write(res)
                st.download_button('Download', res, f'{uploaded_file.name}.json', 'application/json')
            os.remove(f.name)
 
    else:
        path_in = None
    
    
    

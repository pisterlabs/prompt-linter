import os
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM




PATH = './models/models--google--flan-t5-large/snapshots/2d6503cbe79448e511312ba3377a9cde16a2135a'

embeddings = HuggingFaceEmbeddings()


# model_id = 'google/flan-t5-large'
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id,
#                                         torch_dtype=torch.float32,
#                                         cache_dir=os.getenv("cache_dir", "./models"))

# model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_id,
#                                             torch_dtype=torch.float32,
#                                             cache_dir=os.getenv("cache_dir", "./models")
#                                             )

tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(PATH, local_files_only=True)

pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_length=512)
local_llm = HuggingFacePipeline(pipeline=pipe)


template = '''Question: {question}

Answer Let's think step by step.'''
prompt = PromptTemplate(template=template, input_variables=['question'])

# llm_chain = LLMChain(prompt=prompt, llm=local_llm)

# print(local_llm('what is the capital of France?'))


st.subheader("PDF QnA")
file_uploads = st.file_uploader('Upload PDF file(s)', type=['pdf'], accept_multiple_files=True)


if file_uploads:

    for pdf_file in file_uploads:
        file_details = {"FileName": pdf_file.name, "FileType": pdf_file.type}
        st.write(file_details)

        with open(os.path.join("./upload/", pdf_file.name),"wb") as f:
            f.write(pdf_file.getbuffer())
            st.success("Saved File")

prompt_qs = st.text_input('Provide your question')

if prompt_qs:
    pdf_folder_path = './upload'
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    print(loaders)
    question = 'What is this research paper about?'

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)


    chain = RetrievalQA.from_chain_type(llm=local_llm,
                                        chain_type="stuff",
                                        retriever=index.vectorstore.as_retriever(),
                                        input_key="question")

    print(f"Prompt qs: {prompt_qs}")
    response = chain.run(prompt_qs)
    print(f"Answer: {response}")
    st.write(response)


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation.qa import QAGenerateChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from langchain.document_loaders import PyPDFium2Loader
from langchain.embeddings import HuggingFaceEmbeddings

import io, requests
import re
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
user_api_key = ''

template = """ You are going to be my assistant.
Use the context below please try to give me the most correct answers to my
question with reasoning for why they are correct.

Context: {context}
Question: {query} 
Answer: """

prompt = PromptTemplate(template=template, input_variables=["context", "query"])

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    @st.cache_resource()
    def create_vector_db(texts, _embeddings):
        return FAISS.from_texts(texts, _embeddings)

    def pdf_to_raw_text(url):
        response = requests.get(url)
        if response.status_code == 200:
            raw_text = ''
            with io.BytesIO(response.content) as open_pdf_file:
                reader = PdfReader(open_pdf_file)
                for _, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        re.sub('<|endoftext|>', '', text)
                        raw_text += text
                num_pages = len(reader.pages)
                print(num_pages)
            return raw_text
        else:
            st.write(f"Error {response.status_code}: Unable to download PDF '{url}'")
            return False
    # Replace 'output_dir' with the path to the directory where you want to save the PDF
    output_dir = "."

    def pdf_to_text(url):
        loader = PyPDFium2Loader(url)
        raw_text = loader.load()

        concatenated_docs = []
        for i in range(len(raw_text)):
            current_doc = ''.join(raw_text[i].page_content)
            concatenated_docs.append(current_doc)
        whole_pdf = ' '.join(concatenated_docs)
        return whole_pdf


    # Streamlit app
    st.title("Talk with your favorite pdf!")

    pdf_url = st.text_input("Enter the URL for a pdf (e.g. https://arxiv.org/pdf/2303.17564.pdf):")
    if len(pdf_url) != 0:
        st.write("## The first time this runs for a new pdf it may take up to a minute to load the model")
        #raw_text = pdf_to_raw_text(pdf_url)
        raw_text = pdf_to_text(pdf_url)
 
        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 950,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        print(len(texts))

        embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=user_api_key)
     
        docsearch = create_vector_db(texts, embeddings)
        retriever = docsearch.as_retriever(search_type="similarity") #, search_kwargs={"k":2})
  
        embeddings_hf = HuggingFaceEmbeddings()
        # embeddings_hf = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
        # cache_folder=None)
      
        docsearch_hf = create_vector_db(texts, embeddings_hf)
        retriever_hf = docsearch_hf.as_retriever(search_type="similarity") #, search_kwargs={"k":2})


        # ####################
        # # evaluation
        # st.write('--------------------------------')
        # st.write('#### Generated questions about document and graded responses with OpenAI')
        # llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key)
        # example_gen_chain = QAGenerateChain.from_llm(llm)
        # examples = example_gen_chain.apply_and_parse([{"doc": t} for t in texts[:5]])

        # qa = RetrievalQA.from_chain_type(
        #     llm=llm, 
        #     chain_type="stuff", 
        #     retriever=retriever, 
        #     verbose=True,
        #     chain_type_kwargs = {
        #         "document_separator": "<<<<>>>>>"}
        # )

        # predictions = qa.apply(examples)

        # eval_chain = QAEvalChain.from_llm(llm)
        # graded_outputs = eval_chain.evaluate(examples, predictions)
        # for i, eg in enumerate(examples):
        #     st.write(f"Example {i}:")
        #     st.write("Question: " + predictions[i]["query"])
        #     st.write("Real Answer: " + predictions[i]["answer"])
        #     st.write("Predicted Answer: " + predictions[i]["result"])
        #     st.write("Predicted Grade: " + graded_outputs[i]["text"])
        #     st.write()
        # st.write('--------------------------------')
        # ####################


        
        
        query = st.text_input("Enter your question (e.g. who are the authors of this paper?):")
        if len(query) != 0:

            # attempt to ask questions with prompt
            docs_hf = docsearch_hf.similarity_search(query)
            docs = docsearch.similarity_search(query)

            _llm=HuggingFaceHub(repo_id='MaRiOrOsSi/t5-base-finetuned-question-answering', 
                               model_kwargs={"temperature":0, "max_length":512})
            _chain = LLMChain(llm=_llm, prompt=prompt)
            inputs = [{"context": doc.page_content, "query": query} for doc in docs_hf]
            print(inputs)
            print('Results')
            print(_chain.apply(inputs))
            ##########################
    

            llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key)
            
            st.write('')
            st.write('### OpenAI')
            st.write('#### with RetrievalQA:')
            qa_stuff = RetrievalQA.from_chain_type(llm=llm,  
                                        chain_type="stuff",  
                                        retriever= retriever,
                                        verbose=False,
                                        return_source_documents=True)
            response = qa_stuff(query)
            st.write(response['result'])
            st.write(response['source_documents'])
            #st.write('--------------------------------')

            st.write('#### with load_qa_chain:')
            chain = load_qa_chain(OpenAI(openai_api_key=user_api_key), chain_type="stuff", )
            st.write(chain.run(input_documents=docs, question=query))
            st.write('--------------------------------')

            
            models = ['declare-lab/flan-alpaca-large', 
                      'MBZUAI/LaMini-Flan-T5-783M', 
                      'IAJw/declare-flan-alpaca-large-18378', 
                      'google/flan-t5-base', 
                      'mrm8488/t5-base-finetuned-common_gen',
                      'lmsys/fastchat-t5-3b-v1.0',
                      #'deepset/roberta-base-squad2',
                      #'distilbert-base-uncased-distilled-squad',
                      #'google/flan-t5-xl',
                      #'Babelscape/rebel-large',
                      #'declare-lab/flan-alpaca-gpt4-xl',
                      #'google/t5-v1_1-base',
                      'google/long-t5-tglobal-base',
                      #'Salesforce/codet5p-770m-py',
                      #'facebook/mbart-large-50'
                      ]

            st.write('### HuggingFace')
            for model in models:
                st.write(f'### Model: {model}:')
                st.write('#### with RetrievalQA')
                llm=HuggingFaceHub(repo_id=model, model_kwargs={"temperature":0, "max_length":512})

                chain = RetrievalQA.from_chain_type(llm=llm,  
                                        chain_type="stuff",  
                                        retriever= retriever_hf,
                                        verbose=False,
                                        return_source_documents=True)
                response = chain(query)
                st.write(response['result'])
                st.write(response['source_documents'])
                st.write('')

                st.write('#### with load_qa_chain:')
                chain = load_qa_chain(llm, chain_type="map_reduce")
                response = chain({"input_documents": docs_hf, "question": query}, return_only_outputs=True)["output_text"]
                st.write(response)
                st.write('--------------------------------')
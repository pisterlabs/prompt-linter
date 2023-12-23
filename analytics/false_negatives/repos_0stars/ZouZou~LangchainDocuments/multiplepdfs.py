import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain.vectorstores import Chroma
import os
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.vectorstores.utils import filter_complex_metadata
import pdfkit

def test_metadata():
    loader = PyPDFLoader(os.path.join('docs', '3-IMS-Business Setup - User Guide - V1.1.pdf'))
    # loader = UnstructuredFileLoader(
    #             os.path.join('docs', '3-IMS-Business Setup - User Guide - V1.1.pdf'),
    #             mode="paged",
    #             post_processors=[clean_extra_whitespace],
    #         )
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(loader.load())
    # chunks = filter_complex_metadata(chunks)
    st.write(chunks)

def get_pdfs(pdf_docs):
    pdfs = []
    for pdf in pdf_docs:
        # st.write(pdf.type)
        # pdf_reader = PdfReader(pdf)
        with open(os.path.join('docs', pdf.name), 'wb') as fp:
            # for page in pdf_reader.pages:
            #     text += page.extract_text()
            fp.write(pdf.getbuffer())
            pass
        # if os.path.isfile(os.path.join('docs', pdf.name)):
        #     # Create and load PDF Loader
        #     if (pdf.type == 'application/pdf'):
        #         loader = PyPDFLoader(os.path.join('docs', pdf.name))
        #     if (pdf.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
        #         loader = Docx2txtLoader(os.path.join('docs', pdf.name))
        #     # loader = UnstructuredFileLoader(
        #     #     os.path.join('docs', pdf.name),
        #     #     mode="paged",
        #     #     post_processors=[clean_extra_whitespace],
        #     # )
        #     # Split pages from pdf 
        #     pdfs.extend(loader.load())
    loader = DirectoryLoader('docs/', glob='**/*.docx', show_progress=True, use_multithreading=True, loader_cls=Docx2txtLoader)
    pdfs.extend(loader.load())    
    loader = DirectoryLoader('docs/', glob='**/*.pdf', show_progress=True, use_multithreading=True, loader_cls=PyPDFLoader)
    pdfs.extend(loader.load())
    return pdfs

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader.pages)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_pdf_chunks(pdf):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(pdf)
    chunks = filter_complex_metadata(chunks)
    # st.write(chunks)
    return chunks

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_chroma_vectorstor(pdf_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Chroma.from_documents(pdf_chunks, embeddings, collection_name='multiplepdfs')
    return vectorstore
    
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore = Chroma.from_documents(pages, embeddings, collection_name='annualreport')
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=1, request_timeout=60, verbose=True)
    print(llm)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    template = """Given the following conversation respond to the best of your ability in a 
    professional voice and act as an insurance expert explaining the answer to a novice
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    template = """Given the following conversation respond as an insurance expert, rephrase 
    the follow up question to be a standalone question and explain 
    clearly the answer to a novice insurance employee and respond in french.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""


    QA_PROMPT_DOCUMENT_CHAT = """You are a helpful and courteous support representative working for an insurance company. 
    Use the following pieces of context to answer the question at the end.
    If the question is not related to the context, politely respond that you are tought to only answer questions that are related to the context.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer. 
    Try to make the title for every answer if it is possible. Answer in markdown.
    Make sure that your answer is always in Markdown.
    {context}
    Question: {question}
    Answer in HTML format:"""

    CONDENSE_PROMPT = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question and respond in english.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    QA_PROMPT = PromptTemplate(
        input_variables=['context', 'question'], 
        template=QA_PROMPT_DOCUMENT_CHAT
    )
    CONDENSED_PROMPT = PromptTemplate(
        input_variables=['chat_history','question'],
        template=CONDENSE_PROMPT
    )

    PROMPT = PromptTemplate(
        input_variables=["chat_history", "question"], 
        template=template
    )

    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True, output_key='answer')
    memory = ConversationBufferWindowMemory(
        k=1, 
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        # condense_question_prompt=PROMPT,
        return_source_documents=True,
        verbose=True,
        condense_question_prompt=CONDENSED_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    # st.write(conversation_chain)
    return conversation_chain


def handle_userinput(user_question):
    chat_history = [] # st.session_state.chat_history
    # print(chat_history)
    bot_message = ''
    references = ''
    response = st.session_state.conversation({'question': user_question, "chat_history": chat_history})
    st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            bot_message = bot_template.replace("{{MSG}}", message.content)
            # if (i == len(st.session_state.chat_history) - 1):
            #     references += '<br><h5>References</h5>'
            #     references += '<ol>'
            #     for source_document in response['source_documents']:
            #         # references += '<li>Page: ' + str(source_document.metadata['page'] + 1) + '</li>'
            #         references += '<li>' + source_document.metadata['source'].replace("docs\\", "")
            #         if 'page' in source_document.metadata:
            #             references += ' (' + str(source_document.metadata['page'] + 1) + ')'
            #         references += '</li>'
            #     references += '</ol>'
            bot_message = bot_message.replace("{{REFERENCES}}", references)
            st.write(bot_message, unsafe_allow_html=True)
    if 'question' in response:
        st.write(user_template.replace(
            "{{MSG}}", response['question']), unsafe_allow_html=True)
    if 'answer' in response:
        bot_message = bot_template.replace("{{MSG}}", response['answer'])
        references += '<br><h5>References</h5>'
        references += '<ol>'
        for source_document in response['source_documents']:
            # references += '<li>Page: ' + str(source_document.metadata['page'] + 1) + '</li>'
            references += '<li>' + source_document.metadata['source'].replace("docs\\", "")
            if 'page' in source_document.metadata:
                references += ' (' + str(source_document.metadata['page'] + 1) + ')'
            references += '</li>'
        references += '</ol>'
        bot_message = bot_message.replace("{{REFERENCES}}", references)
        st.write(bot_message, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # test_metadata()
                # get pdf text
                # raw_text = get_pdf_text(pdf_docs)
                raw_pdf = get_pdfs(pdf_docs)

                # get the text chunks
                #text_chunks = get_text_chunks(raw_text)
                pdf_chunks = get_pdf_chunks(raw_pdf)

                # create vector store
                #vectorstore = get_vectorstore(text_chunks)
                vectorstore = get_chroma_vectorstor(pdf_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()
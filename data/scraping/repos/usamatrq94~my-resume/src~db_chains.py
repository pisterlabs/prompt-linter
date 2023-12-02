import os

from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma

MARKDOWN_PATH = "resume/resume.md"
PRESISIT_DIRECTORY = "docs/chroma/"
LLM_NAME = "gpt-3.5-turbo"


def load_resume(
    markdown_path: str = None, persist_directory: str = None, llm_name: str = None
) -> ConversationalRetrievalChain:
    """
    This function creates a chatbot that can answer questions about a resume.
    """
    markdown_path = markdown_path or MARKDOWN_PATH
    persist_directory = persist_directory or PRESISIT_DIRECTORY
    llm_name = llm_name or LLM_NAME

    # Load the resume
    loader = UnstructuredMarkdownLoader(markdown_path)
    data = loader.load()
    md_file = data[0].page_content

    # Split the resume into sections
    headers_to_split_on = [
        ("##", "Section"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(md_file)

    # Split the sections into chunks
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = r_splitter.split_documents(md_header_splits)

    # Create the vectorstore
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=persist_directory
    )

    # Create the retriever, memory, llm and prompt template
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    template = """The document you have access is a resume.
    You are suppose to assume role of that candidate and try as best as you can to impress the user with your responses. 
    Your goal is to get a job offer from the user.
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer 
    and ask a follow up question to keep the conversation going to understand the user better and mould your answers accordingly.
    {context}
    Question: {question}
    Helpful Answer:"""

    qa_chain_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = OpenAI(model_name=llm_name, temperature=0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the chatbot
    chatbot = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_chain_prompt},
    )

    return chatbot

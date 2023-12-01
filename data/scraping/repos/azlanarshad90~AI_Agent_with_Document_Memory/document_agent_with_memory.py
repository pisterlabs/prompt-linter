import re
import time
from io import BytesIO
from typing import Any, Dict, List
import openai
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import PyPDF2

openai.api_key = "YOUR_API_KEY"

def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    # Indexing
    index = FAISS.from_documents(pages, embeddings)
    return index

pdf_file_path = '/content/filename.pdf'

uploaded_file = PyPDF2.PdfReader(pdf_file_path)

from io import BytesIO
with open(pdf_file_path, 'rb') as file:
    doc = parse_pdf(BytesIO(file.read()))
    pages = text_to_docs(doc)

index = test_embed()

qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai.api_key),
            chain_type="stuff",
            retriever=index.as_retriever(),
        )

tools = [
            Tool(
                name="Tool Name",
                func=qa.run,
                description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
            )
        ]

prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available.
                You have access to a single tool:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

memory = ConversationBufferMemory(
                memory_key="chat_history"
            )

llm_chain = LLMChain(
            llm=OpenAI(
                temperature=0, openai_api_key=openai.api_key, model_name="gpt-4"
            ),
            prompt=prompt,
        )

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=memory
            )

query = "What can you tell about input document?"
response = agent_chain.run(query)

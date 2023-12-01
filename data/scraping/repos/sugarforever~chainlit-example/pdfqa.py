from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from chainlit.types import (
    AskFileResponse
)
import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def store_uploaded_file(uploaded_file: AskFileResponse):
    file_path = f"./tmp/{uploaded_file.name}"
    open(file_path, "wb").write(uploaded_file.content)
    return file_path

@cl.langchain_factory(use_async=True)
async def init():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!", accept=["application/pdf"]
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    file_path = store_uploaded_file(file)
    
    # Load PDF file into documents
    docs = PyMuPDFLoader(file_path).load()

    msg = cl.Message(content=f'You have {len(docs)} document(s) in the PDF file.')
    await msg.send()
    
    # Split the documents into chunks
    split_docs = text_splitter.split_documents(docs)

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        split_docs, embeddings, collection_name=file.name
    )
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    await cl.Message(content=answer).send()

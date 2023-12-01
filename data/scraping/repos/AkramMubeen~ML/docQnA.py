# Import necessary modules from the Langchain and Chainlit libraries
import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse

# Adding OPENAI key
os.environ['OPENAI_API_KEY']= "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"

# Initialize the RecursiveCharacterTextSplitter and OpenAIEmbeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

# Welcome message for the user
welcome_message = """Welcome to the Chainlit PDF QnA! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

# Function to process the uploaded file
def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False) as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs

# Function to create a Chroma vector store from the processed documents
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    cl.user_session.set("docs", docs)

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch

# Define an async function that runs when the chat session starts
@cl.on_chat_start
async def start():
    # Send a welcome message to the user
    await cl.Message(content="You can now chat with your PDFs.").send()
    files = None
    while files is None:
        # Ask the user to upload a file
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Process the uploaded file and create a Chroma vector store
    docsearch = await cl.make_async(get_docsearch)(file)

    # Create a retrieval chain for question answering
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    # Update the message to indicate that processing is complete
    msg.content = f"`{file.name}` has been processed. You can now proceed asking questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

# Define an async function that handles user messages
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()

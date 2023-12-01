import os

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

class CustomNamedTemporaryFile:
    """
    This custom implementation is needed because of the following limitation of tempfile.NamedTemporaryFile:

    > Whether the name can be used to open the file a second time, while the named temporary file is still open,
    > varies across platforms (it can be so used on Unix; it cannot on Windows NT or later).
    """
    def __init__(self, mode='wb', delete=False):
        self._mode = mode
        self._delete = delete

    def __enter__(self):
        # Generate a random temporary file name
        file_name = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        # Ensure the file is created
        open(file_name, "x").close()
        # Open the file in the given mode
        self._tempFile = open(file_name, self._mode)
        return self._tempFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempFile.close()
        if self._delete:
            os.remove(self._tempFile.name)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
embeddings = OpenAIEmbeddings(openai_api_key = OPEN_AI_KEY)

welcome_message = """Welcome to Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    
    with tempfile.CustomNamedTemporaryFile(delete = False) as tempfile:
        tempfile.write(file.content)
        print(tempfile.name)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i,doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        tempfile.close()
        os.unlink(tempfile.name)
        return docs

def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs",docs)

    if not os.path.exists("persists"):
        # Create a unique namespace for the file 
        docsearch = chroma.from_document(
            docs, embeddings
        )
        docsearch.persist()
    return docsearch

@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    await cl.Message(content="Welcome to this space, you can chat with your pdfs").send()
    files = None 
    while files is None:
        files = await cl.AskFileMessage(
            content = welcome_message,
            accept = ["text/plain","application/pdf"],
            max_size_mb = 20,
            timeout = 180,
        ).send()
    
    file = files[0]

    msg = cl.Message(content = f"Processing `{file.name}`...")
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature = 0, streaming=True),
        chain_type = "stuff",
        retriever = docsearch.as_retriever(max_tokens_limit=4096),
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask question!"
    await msg.update()

    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True,
        answer_prefix_tokens = ["FINAL", "ANSWER"],
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the document from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message 
        for source in sources.split(","):
            source_name = source.strip().replace(".","")
            # Get the index of source
            try: 
                index = all_sources.index(source_name)
            except ValueError:
                continue
            
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text,name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"
        
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=message,elements=source_elements).send()
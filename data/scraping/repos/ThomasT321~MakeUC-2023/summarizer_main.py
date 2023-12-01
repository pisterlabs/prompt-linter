from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.llms.gpt4all import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def fileUploadSummarize(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

    except Exception as e:
        raise Exception("Document type not supported")
    
    return summarize(docs)

def summarize(docs):

    local_path = (
        "./models/mistral-7b-instruct-v0.1.Q4_0.gguf"
    )
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

    chain = load_summarize_chain(llm, chain_type="map_reduce")

    results = chain.run(input_documents=docs)
    
    return results
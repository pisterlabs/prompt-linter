from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileIOLoader


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
from chainlit.server import app
from chainlit.types import AskFileResponse
from io import BytesIO,StringIO
import chardet

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
{summaries}

请中文回答
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def process_file(file: AskFileResponse):
    # import tempfile

    # if file.type == "text/plain":
    #     Loader = TextLoader
    # elif file.type == "application/pdf":
    #     Loader = PyPDFLoader

    # with tempfile.NamedTemporaryFile() as tempfile:
    #     tempfile.write(file.content)
    #     loader = Loader(tempfile.name)
    #     documents = loader.load()
    #     docs = text_splitter.split_documents(documents)
    #     for i, doc in enumerate(docs):
    #         doc.metadata["source"] = f"source_{i}"
    #     return docs
    
    result = chardet.detect(file.content)
    pprint(result)
    
    io = BytesIO(file.content)
    loader = UnstructuredFileIOLoader(io, )
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


@cl.on_chat_start
async def init():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="先上传文件", 
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"`{file.name}` 分析中...")
    await msg.send()

    # Decode the file
    docs = process_file(file)

    # Create a metadata for each chunk
    #metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        docs, embeddings
    )
    llm =  ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo-16k")
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("docs", docs)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` 分析完成. 正在总结文档内容..."
    await msg.update()
    
    cl.user_session.set("chain", chain)
    
    template = '''
用30字总结一下这文档内容，列出最多5点主旨
{text}

中文回答
'''
    prompt = PromptTemplate(template=template, input_variables=["text"])
    sumchain = load_summarize_chain(llm=llm, chain_type='stuff', prompt=prompt, verbose=True)
    text = await sumchain.arun(docs)
    await cl.Message(content=text).send()
    
    

from pprint import pprint
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    
    
    source_elements = []

    # Get the metadata and texts from the user session
    docs = cl.user_session.get("docs")
    pprint(res)
    pprint([doc.metadata['source'] for doc in docs])

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                search_docs = [doc for doc in docs if doc.metadata['source']==source_name]
            except ValueError:
                continue
            if len(search_docs)==0:
                continue
            text = search_docs[0].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
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
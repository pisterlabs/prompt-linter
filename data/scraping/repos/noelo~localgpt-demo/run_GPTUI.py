import os
import openai
import chainlit as cl
from chainlit import Message, on_chat_start
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY



SYSTEM_TEMPLATE = """Use the following pieces of context to answer the users question.
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
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@on_chat_start
async def main():
    '''Startup and setup env'''
    await cl.Avatar(
        name="OCP",
        path="./OpenShift-LogoType.svg.png",
    ).send()
    await Message(
        content=f"Ask questions of the OpenShift Documentation", author="OCP"
    ).send()


@cl.langchain_factory(use_async=True)
def load_model():
    '''Load embeddings and embeddings db, setup retriever chain'''
    openai.api_key = os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0.0)
    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(PERSIST_DIRECTORY,embeddings)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


@cl.langchain_postprocess
async def process_response(res):
    '''Format response and make it pretty ;-) '''
    answer = res["result"]
    sources = res["source_documents"]
    elements=[]
    for source in sources:
        src_str = source.metadata['source']
        res_str = src_str.replace("/home/noelo/dev/localGPT/SOURCE_DOCUMENTS/", "")
        final_str = 'Page ' + str(source.metadata['page'])
        elements.append(cl.Text(content=final_str, name=res_str, display="inline"))
 
    await cl.Message(content=answer,elements=elements,author="OCP").send()

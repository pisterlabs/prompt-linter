# Finetuned LLMs and experimentation w vectorstores

Example of finetuning local data on machine and cross-referencing with Chroma vectorstore on performance, using gradio as interface
"""

# install packages

!pip -q install transformers
!pip -q install gradio
!pip -q install langchain==0.0.031
!pip -q install pypdf
!pip -q install wikipedia
!pip -q install openai==0.27.8
!pip -q install docarray
!pip -q install tiktoken
!pip -q install llama-cpp-python
!pip -q install unstructured[local-inference]

!pip install -U git+https://github.com/hwchase17/langchain.git

!pip freeze --local > /content/drive/MyDrive/colab_installed.txt

# load env vars for colab

from google.colab import drive
drive.mount('/content/drive')
import os
!source /content/drive/MyDrive/gc/env_v.sh

# import packages for full-ish template to use from LC

import gradio as gr
import openai
from langchain.llms import LlamaCpp, OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain


from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain, RetrievalQAWithSourcesChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader as ppdf
from langchain.chat_models import ChatOpenAI

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.text_splitter import CharacterTextSplitter


# Additional vectorstore options
from langchain.vectorstores import DocArrayInMemorySearch # compared to chroma for 1 page
from langchain.indexes import VectorstoreIndexCreator
from langchain.retrievers import ContextualCompressionRetriever
from transformers import pipeline

# os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

os.environ['OPENAI_API_KEY'] = OPEN_API_KEY

# Load internal company data
loader = ppdf("/content/drive/MyDrive/gc/our_future.pdf")
page = loader.load()

# llm definition
llm = ChatOpenAI(streaming=True, temperature=0.0, model_name='gpt-4')

# split
text_splitter = CharacterTextSplitter(
    chunk_size=115,
    chunk_overlap=60,
    length_function=len,
    separator = "\n"
)
split = text_splitter.split_documents(page)
len(split)

# embeddings
embedding = OpenAIEmbeddings()#(deployment="text-embedding-ada-002")
vectorstore = Chroma.from_documents(page, embedding)
db = DocArrayInMemorySearch.from_documents(
    split,
    embedding
)

# create memory buffer for chat
#memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
chain1 = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.0),
    vectorstore.as_retriever(),
   # memory = memory
    )

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []

    def user(user_message, history):
        # Get response from QA chain
        response = chain1({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

import os

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, SQLDatabase, SQLDatabaseChain

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
db = SQLDatabase.from_uri(DATABASE_URL)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    max_retries=1,
)

sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader("data/state_of_the_union.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings(max_retries=1)

from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings

model_name = "intfloat_e5-large-v2"
modelpath = f"../model/{model_name}"
embeddings = HuggingFaceInstructEmbeddings(model_name=modelpath)

from langchain.vectorstores import Chroma, Milvus

vectordb = Milvus.from_documents(
    texts,
    embeddings,
    connection_args={"host": "192.168.2.201", "port": "19530"},
)

doc_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever()
)
doc_chain.verbose = True

from langchain.prompts import PromptTemplate

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
gpt_chain = LLMChain(prompt=prompt, llm=llm)

from langchain.agents import AgentType, Tool, initialize_agent, load_tools

tools = []

tools.append(
    Tool(
        name="sql chain",
        func=sql_chain.run,
        description="useful for when you need to answer questions about countries",
    )
)
tools.append(
    Tool(
        name="doc chain",
        func=doc_chain.run,
        description="useful for when you need to answer questions about Ketanji Brown Jackson",
    )
)
tools.append(
    Tool(
        name="chatgpt chain",
        func=gpt_chain.run,
        description="useful for when you need to answer questions don't about countries and Ketanji Brown Jackson",
    )
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

import uuid

import gradio as gr
from loguru import logger

from helper.cache import RedisCache

cache = RedisCache()


async def make_completion(query):
    _prompt_id = str(uuid.uuid4())
    logger.info(f"Prompt ID: {_prompt_id}")
    with get_openai_callback() as cb:
        res = cache.get(query)
        if res is not None:
            logger.info("Using cached response")

        else:
            res = agent.run(query)
            cache.set(query, res)

        print(res)
        print(cb)
        return str(res)


async def predict(input, history):
    """
    Predict the response of the chatbot and complete a running list of chat history.
    """
    history.append(input)
    response = await make_completion(history[-1])
    history.append(response)
    messages = [(history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)]
    return messages, history


css = """
.contain {margin-top: 80px;}
.title > div {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo:
    logger.info("Starting Demo...")
    chatbot = gr.Chatbot(label="Chatbot", elem_classes="title")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(
            show_label=False, placeholder="Enter text and press enter"
        ).style(container=False)
    txt.submit(predict, [txt, state], [chatbot, state])

demo.launch(server_port=18080, share=True, server_name="0.0.0.0")

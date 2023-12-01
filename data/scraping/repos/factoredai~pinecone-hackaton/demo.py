from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from prompts import TEMPLATES

import os
from dotenv import load_dotenv
import pinecone
import chainlit as cl

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),environment=os.getenv("PINECONE_ENV"))
openai_api_key = os.environ["OPENAI_API_KEY"] 


#embeddings = HuggingFaceInstructEmbeddings()
api_key = os.environ["COHERE_API_KEY"]
embeddings = CohereEmbeddings(cohere_api_key = api_key, model = 'embed-english-v2.0')


@cl.action_callback("action_button")
async def on_action(action):
    actions = cl.user_session.get("actions")
    cl.user_session.set("task", action.value)
    if action.value == "prior_art":
        await cl.Message(content="You chose prior art search. What are you looking for?", author = 'Librarian').send()
        cl.user_session.set("botname", "Librarian")
    elif action.value == "draft_patent":
        await cl.Message(content="You chose to draft a patent. What is your invention?", author = 'Drafter').send()
        cl.user_session.set("botname", "Drafter")
    elif action.value == "compare_patent":
        await cl.Message(content="You chose to compare your patent. What is your patent?", author = 'Curator').send()
        cl.user_session.set("botname", "Curator")

    await get_model(action.value)

    for action in actions:
        await action.remove()
    # Optionally remove the action button from the chatbot user interface
    #await action.remove()

@cl.on_chat_start
async def start():
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="action_button", value='prior_art', description="Click me!", label = "Prior art search"),
        cl.Action(name="action_button", value='draft_patent', description="Click me!", label = "Draft your patent"),
        cl.Action(name="action_button", value='compare_patent', description="Click me!", label = "Compare your patent")]
    cl.user_session.set("actions", actions)


    await cl.Message(content="How can PatentBot help you today?", actions=actions).send()

@cl.on_message
async def main(message: str):
    # Your custom logic goes here...
    task = cl.user_session.get("task")
    chain = await get_model(task)
    res = await cl.make_async(chain)(message, callbacks=[cl.ChainlitCallbackHandler()])
    # msg = cl.Message("")
    # async for stream_resp in cl.user_session.get('async_iterator').aiter():
    #     token = stream_resp.get("choices")[0].get("text")
    #     await msg.stream_token(token)
    # await msg.send()
    msg = cl.Message(content="", author = cl.user_session.get("botname"))
    for token in res["answer"].split(" "):
        await cl.sleep(0.1)
        await msg.stream_token(token + " ")
    await msg.send()
    #await cl.Message(content=res["answer"], author = cl.user_session.get("botname")).send()

async def get_model(task: str):
    if cl.user_session.get("chain"):
        return cl.user_session.get("chain")
    
    docsearch = await cl.make_async(Pinecone.from_existing_index)(
        'patentbot', embeddings
    )

    messages = [
        SystemMessagePromptTemplate.from_template(TEMPLATES[task]),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(memory_key="chat_history", output_key = "answer",  return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0, streaming = True, openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        verbose=False,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "summaries"}
    )
    # chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm=ChatOpenAI(temperature=0, streaming = True, openai_api_key=openai_api_key),
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     verbose = True
    # )
    cl.user_session.set("chain", chain)
    return chain
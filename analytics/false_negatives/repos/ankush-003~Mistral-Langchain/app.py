import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
 
model_id = "mistralai/Mistral-7B-v0.1"
mistral_llm = HuggingFaceHub(repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

template = """

You are an AI assistant that provides helpful answers to user queries.

{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])

# llm_chain = LLMChain(llm=mistral_llm, prompt=prompt, verbose=True)


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=mistral_llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()
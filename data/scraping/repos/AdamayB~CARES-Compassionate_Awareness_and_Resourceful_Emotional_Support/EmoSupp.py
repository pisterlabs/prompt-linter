from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv('example.env')



HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})


template = """You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
if the user asks for a suggestion give the answer in bullet points.

{question}

Add an inspirational quote at the end of every answer in the next paragraph.


"""


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()
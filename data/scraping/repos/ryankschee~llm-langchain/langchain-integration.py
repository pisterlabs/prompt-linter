import os
import openai
import chainlit as cl
from langchain import PromptTemplate, OpenAI, LLMChain

os.environ['OPENAI_API_KEY'] = "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"

template = """Question: {question}

Answer: Let's think step by step."""

#print(template.format(question = "What is the capital of France?"))

@cl.on_chat_start
def main(): 
    prompt = PromptTemplate(template = template, input_variables = ["question"])
    llm_chain = LLMChain(prompt = prompt, llm = OpenAI(temperature = 1, streaming = True),  verbose = True)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks = [cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content = res["text"]).send()
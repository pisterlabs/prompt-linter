import chainlit as cl
import os
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

template = """ 
        Question: {question}
        Answer: {answer}
        """


@cl.on_chat_start
def main():
    # Variables to initiate as soon as chainlit UI is deployed
    prompt = PromptTemplate(template=template, input_variables=['question', 'answer'])
    llm_chain = LLMChain(
        prompt=prompt,
        llm=OpenAI(api_key=api_key, temperature=1, streaming=True),
        verbose=True
    )

    cl.user_session.set('llm_chain', llm_chain)


@cl.on_message
async def main_1(message: cl.Message):
    llm_chain = cl.user_session.get('llm_chain')

    # Initial answer
    answer = ''

    # Processing user's input
    result = await llm_chain.acall({'question': message.content, 'answer': ''}, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Get generated answer from result
    if 'text' in result:
        answer = result['text']

    # Send response back to user
    await cl.Message(content=answer).send()

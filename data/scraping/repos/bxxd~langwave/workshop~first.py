import logging
from langwave.memory.volatile import VolatileChatMemory
import asyncio

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from langchain.chat_models import ChatOpenAI

from langchain.llms import OpenAI
from langchain.chains import ConversationChain

from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts.chat import ChatPromptTemplate

from langwave.chains.wave import ChatWave

from langchain.chains import LLMChain

log = logging.getLogger(__name__)


def test_agent():
    chat = ChatOpenAI()

    resp = chat.predict("hi there")

    log.info(f"chat({chat.model_name}): {resp}")


async def test_wave(args):
    log.info(f"test_wave")
    chat = ChatOpenAI(
        streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0
    )

    memory = VolatileChatMemory()

    prompt = ChatPromptTemplate()

    llm = ChatOpenAI(temperature=0.2, verbose=False)

    wave = ChatWave.from_llm(llm)

    # llm = OpenAI(temperature=0)
    # conversation = ConversationChain(
    #     llm=llm,
    #     verbose=True,
    #     memory=ChatMessageHistory()
    # )

    # resp = conversation.predict(input="Hi there!")

    # log.info(f'conversation: {resp}')
    # log.info(f'memory: {conversation.memory}')


async def test_conversation():
    chat = ChatOpenAI(
        streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0
    )

    history = ChatMessageHistory()

    history.add_user_message("hi")
    history.add_user_message("i am brian")
    history.add_user_message("i am a human")
    history.add_user_message("i love you")
    history.add_user_message("what is my name?")

    prompt = ChatPromptTemplate.from_template("{input}")

    chain = LLMChain(llm=chat, prompt=prompt)

    resp = await chat.apredict_messages(history.messages)
    print("\n")

    history.add_message(resp)
    history.add_user_message("i just told you my name?")

    resp = await chat.apredict_messages(history.messages)

    log.info(f"resp: {resp}")

    log.info(f"history messages: {history.messages}")


def test_memory():
    memory = VolatileChatMemory()
    log.info(f"memory: {memory}")


async def main(args):
    log.info("Hello there!")
    # test_memory()
    await test_conversation()
    # await test_wave(args)


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

import asyncio
import logging

import langwave
from langwave.memory import VolatileChatMemory, MixedChatMemory
from langwave.chains.wave import ChatWave

from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    ChatMessagePromptTemplate,
)
from langchain import OpenAI, LLMChain

log = logging.getLogger(__name__)


async def test_history():
    log.info("test_history")

    # memory = VolatileChatMemory()

    llm = OpenAI(verbose=True, temperature=0.2)

    wave = ChatWave.from_llm(llm)

    log.info(f"wave: {wave} memory: {wave.memory}")

    resp = await wave.acall("hi there!")
    log.info(f"resp: {resp}")


async def test_history2():
    log.info("test_history2")
    memory = MixedChatMemory(memory_key="chat_history", return_messages=True)

    chat_history = MessagesPlaceholder(variable_name="chat_history")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and "
                "provides lots of specific details from its context. If the AI does not know the answer to a "
                "question, it truthfully says it does not know."
            ),
            chat_history,
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    llm = OpenAI(verbose=True, temperature=0.2)
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    memory.chat_memory.add_user_message("Hi there!")
    memory.chat_memory.add_ai_message("How are you?")
    resp = await llm_chain.arun(input="who am i talking to?")
    log.info(f"resp: {resp}")
    log.info(f"memory: {llm_chain.memory}")


async def main():
    log.info("hi there!")
    await test_history()


if __name__ == "__main__":
    asyncio.run(main())

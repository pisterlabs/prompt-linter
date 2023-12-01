from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_qianwen import Qwen_v1
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import asyncio


async def use_async_handler(input):
    handler = AsyncIteratorCallbackHandler()
    llm = Qwen_v1(
        model_name="qwen-turbo",
        streaming=True,
        callbacks=[handler], 
    )

    memory = ConversationBufferMemory()
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )

    asyncio.create_task(chain.apredict(input=input))

    return handler.aiter()


async def async_test():
    async_gen = await use_async_handler("hello")
    async for i in async_gen:
        print(i)


if __name__ == "__main__":
    asyncio.run(async_test())

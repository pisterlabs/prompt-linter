import asyncio

from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)


def single_communication():
    with get_openai_callback() as cb:
        llm("what is the square root of 4?")

    print(cb)


def multiple_communications(count=2):
    with get_openai_callback() as cb:
        for _ in range(count):
            llm("what is the square root of 4?")

    print(cb)


async def async_multiple_communication():
    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.agenerate(["What is the square root of 4?"]) for _ in range(3)]
        )

    print(cb)


async def task_multiple_communication():
    task = asyncio.create_task(llm.agenerate(["What is the square root of 4?"]))
    with get_openai_callback() as cb:
        await llm.agenerate(["What is the square root of 4?"])

    await task

    print(cb)


if __name__ == "__main__":
    single_communication()
    multiple_communications()
    # async_multiple_communication()
    # task_multiple_communication()

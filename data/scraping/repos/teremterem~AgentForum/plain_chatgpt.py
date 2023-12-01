# pylint: disable=wrong-import-position
"""Chat with OpenAI ChatGPT using the AgentForum library."""
import asyncio

# noinspection PyUnresolvedReferences
import readline  # pylint: disable=unused-import
import warnings

from dotenv import load_dotenv

load_dotenv()

import promptlayer  # TODO Oleksandr: make this optional

# TODO Oleksandr: get rid of this warning suppression when PromptLayer doesn't produce "Expected Choice but got dict"
#  warning anymore
warnings.filterwarnings("ignore", module="pydantic")

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext
from agentforum.storage import InMemoryStorage

forum = Forum(immutable_storage=InMemoryStorage())
async_openai_client = promptlayer.openai.AsyncOpenAI()


@forum.agent
async def openai_agent(ctx: InteractionContext, **kwargs) -> None:
    """The first agent that uses OpenAI ChatGPT. It sends the full chat history to the OpenAI API."""
    full_chat = await ctx.request_messages.amaterialize_full_history()
    ctx.respond(openai_chat_completion(prompt=full_chat, async_openai_client=async_openai_client, **kwargs))


@forum.agent
async def user_proxy_agent(ctx: InteractionContext) -> None:
    """An agent that acts as a proxy between the user and other agents."""
    async for request in ctx.request_messages:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in request:
            print(token.text, end="", flush=True)
        print("\033[0m")

    user_input = input("\nYOU: ")
    if user_input == "exit":
        raise KeyboardInterrupt
    ctx.respond(user_input)


async def main() -> None:
    """The chat loop."""
    assistant_responses = []
    try:
        while True:
            user_requests = user_proxy_agent.quick_call(assistant_responses)

            # the following line is needed in order to wait until the previous back-and-forth is processed
            # (otherwise back-and-forth-s will be perpetually scheduled but never executed)
            # TODO Oleksandr: how to turn this hack into something more elegant ?
            await user_requests.amaterialize_all()

            assistant_responses = openai_agent.quick_call(
                user_requests,
                # model="gpt-4-1106-preview",
                model="gpt-3.5-turbo-1106",
                stream=True,
            )
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    asyncio.run(main())

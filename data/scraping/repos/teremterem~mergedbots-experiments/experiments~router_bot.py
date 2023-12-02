"""A bot that routes messages to other bots based on the user's intent."""
import json
from typing import AsyncGenerator

from langchain import LLMChain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from mergedbots import MergedMessage, MergedBot

from experiments.active_listener import active_listener
from experiments.common.bot_manager import FAST_GPT_MODEL, bot_manager
from experiments.plain_gpt import plain_gpt

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("HERE IS A CONVERSATION BETWEEN A USER AND AN AI ASSISTANT."),
        SystemMessagePromptTemplate.from_template("{conversation}"),
        SystemMessagePromptTemplate.from_template(
            "AND HERE IS A LIST OF BOTS WHO COULD BE USED TO RESPOND TO THE CONVERSATION ABOVE."
        ),
        SystemMessagePromptTemplate.from_template("{bots}"),
        HumanMessagePromptTemplate.from_template(
            """\
Which of the bots above would you like to use to respond to the LAST message of the conversation above?

BOT NAME: \""""
        ),
    ]
)


@bot_manager.create_bot(handle="RouterBot")
async def router_bot(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    """A bot that routes messages to other bots based on the user's intent."""
    if not message.previous_msg and not message.is_visible_to_bots:
        yield await message.service_followup_as_final_response(bot, "```\nCONVERSATION RESTARTED\n```")
        return

    chat_llm = PromptLayerChatOpenAI(
        model_name=FAST_GPT_MODEL,
        temperature=0.0,
        max_tokens=10,
        model_kwargs={
            "stop": ['"', "\n"],
            "user": str(message.originator.uuid),
        },
        pl_tags=["mb_router"],
    )
    llm_chain = LLMChain(
        llm=chat_llm,
        prompt=ROUTER_PROMPT,
    )

    bots_json = [
        {"name": other_bot.handle, "description": other_bot.description}
        for other_bot in (plain_gpt.bot, active_listener.bot)
    ]
    conversation = await message.get_full_conversion()
    formatted_conv_parts = [
        f"{'USER' if msg.is_sent_by_originator else 'ASSISTANT'}: {msg.content}" for msg in conversation
    ]

    # choose a bot and run it
    chosen_bot_handle = await llm_chain.arun(
        conversation="\n\n".join(formatted_conv_parts), bots=json.dumps(bots_json)
    )
    print(f"ROUTING TO: {chosen_bot_handle}")
    async for msg in bot.manager.fulfill(chosen_bot_handle, message, fallback_bot_handle=plain_gpt.bot.handle):
        yield msg

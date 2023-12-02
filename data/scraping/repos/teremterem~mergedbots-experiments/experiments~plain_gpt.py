"""A bot that uses either GPT-4 or ChatGPT to generate responses without any hidden prompts."""
from typing import AsyncGenerator

from langchain.chat_models import PromptLayerChatOpenAI
from langchain.schema import ChatMessage
from mergedbots import MergedMessage, MergedBot
from mergedbots.ext.langchain_integration import LangChainParagraphStreamingCallback

from experiments.common.bot_manager import SLOW_GPT_MODEL, bot_manager


@bot_manager.create_bot(
    handle="PlainGPT",
    description=(
        "A bot that uses either GPT-4 or ChatGPT to generate responses. Useful when the user seeks information and "
        "needs factual answers."
    ),
)
async def plain_gpt(bot: MergedBot, message: MergedMessage) -> AsyncGenerator[MergedMessage, None]:
    """A bot that uses either GPT-4 or ChatGPT to generate responses without any hidden prompts."""
    if not message.previous_msg and not message.is_visible_to_bots:
        yield await message.service_followup_as_final_response(bot, "```\nCONVERSATION RESTARTED\n```")
        return

    model_name = SLOW_GPT_MODEL
    yield await message.service_followup_for_user(bot, f"`{model_name}`")

    print()
    paragraph_streaming = LangChainParagraphStreamingCallback(bot, message, verbose=True)
    chat_llm = PromptLayerChatOpenAI(
        model_name=model_name,
        temperature=0.0,
        streaming=True,
        callbacks=[paragraph_streaming],
        model_kwargs={"user": str(message.originator.uuid)},
        pl_tags=["mb_plain_gpt"],
    )
    conversation = await message.get_full_conversion()
    async for msg in paragraph_streaming.stream_from_coroutine(
        chat_llm.agenerate(
            [
                [
                    ChatMessage(
                        role="user" if msg.is_sent_by_originator else "assistant",
                        content=msg.content,
                    )
                    for msg in conversation
                ]
            ],
        )
    ):
        yield msg
    print()

# Third-party library imports
import openai

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

# Custom module imports
from handlers.utils import send_thinking_message_async
from extentions.chat_gpt import get_chat_response_async


async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the '/ai' command by simulating a human-like conversation using OpenAI's GPT-X model.
    """

    # Inform the user that the bot is processing their message
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    thinking_message = await send_thinking_message_async(update.message)

    # Get the conversation history for this chat
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.7, client=openai.Completion.create
    )
    memory = context.chat_data.get(
        update.message.chat_id,
        ConversationSummaryBufferMemory(llm=llm, max_token_limit=800),
    )

    # Add the new message to the conversation history
    memory.chat_memory.add_user_message(update.message.text)

    # Generate a thoughtful response using the conversation history
    response = await get_chat_response_async(update.message.text, memory)

    # Respond to the user by editing the thinking message
    await thinking_message.edit_text(text=response)

    # Add the response to the conversation history
    memory.chat_memory.add_ai_message(response)

    # Update conversation history in chat_data
    context.chat_data[update.message.chat_id] = memory

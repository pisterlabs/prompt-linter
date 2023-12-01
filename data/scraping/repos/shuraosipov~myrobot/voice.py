# Standard library imports
from collections import deque
from datetime import datetime
import uuid
import logging

# Related third party imports
import openai
from pydub import AudioSegment
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

# Local application/library specific imports
from extentions.chat_gpt import get_chat_response_async
from extentions.whisper import transcribe_audio
from handlers.utils import send_thinking_message_async

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a voice message by replying with a text message."""
    logger.info("Voice message received")

    # Inform the user that the bot is processing their message
    await context.bot.send_chat_action(chat_id=update.effective_chat.id,action=ChatAction.TYPING)
    thinking_message = await send_thinking_message_async(update.message)

    # Download file to drive
    file_name = await download_file_to_disc(update, context)
    logger.info(f"Audio file downloaded: {file_name}")

    # Convert to mp3
    mp3_file_name = await convert_ogg_to_mp3(file_name)
    logger.info(f"Audio file converted to mp3: {mp3_file_name}")

    # Transcribe audio using OpenAI Whisper
    transcript = await transcribe_audio(mp3_file_name)
    logger.info(f"Audio transcribed")

    # Get the conversation history for this chat
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.7, client=openai.Completion.create
    )
    memory = context.chat_data.get(
        update.message.chat_id,
        ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000),
    )

    # Add the new message to the conversation history
    memory.chat_memory.add_user_message(transcript)

    # Generate a thoughtful response using the conversation history
    response = await get_chat_response_async(transcript, memory)
    logger.info(f"Generated response")
    
    # Respond to the user by editing the thinking message
    await thinking_message.edit_text(text=response)
    logger.info("Voice message processed")

    # Add the response to the conversation history
    memory.chat_memory.add_ai_message(response)

    # Update conversation history in chat_data
    context.chat_data[update.message.chat_id] = memory
    logger.info("Memory updated")


async def convert_ogg_to_mp3(ogg_file_path) -> str:
    """ Convert ogg to mp3 """
    audio = AudioSegment.from_ogg(ogg_file_path)
    mp3_file_path = str(ogg_file_path).replace(".ogg", ".mp3")
    audio.export(mp3_file_path, format="mp3")
    return mp3_file_path


async def download_file_to_disc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Download a file to the local disk."""
    message = update.message
    voice = message.voice

    # download file to drive
    file = await voice.get_file()

    # Generate a UUID
    uuid_str = str(uuid.uuid4())

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Combine the UUID and date/time strings to create the file name
    file_name = await file.download_to_drive(f"audio_{uuid_str}_{date_str}.ogg")

    return file_name
import traceback
from telethon import TelegramClient, events

import config # Custom file containing configuration settings for the bot.

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper # GPTSimpleVectorIndex
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import openai
import sys
import os

# openai.api_key = config.openai_key
os.environ["OPENAI_API_KEY"] = "sk-G2arlkdAq408VAgkHfb8T3BIbkFJeGo9Qwipy3it8nDXHmry"
openai.api_key = "sk-G2arlkdAq408VAgkHfb8T3BIbkFJeGo9Qwipy3it8nDXHmry"
client = TelegramClient(None, config.API_ID, config.API_HASH).start(bot_token=config.BOT_TOKEN)


@client.on(events.NewMessage(pattern="(?i)/start"))
async def handle_start_command(event):
    try:
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        vIndex = load_index_from_storage(storage_context)
        query_engine = vIndex.as_query_engine()

        async with client.conversation(await event.get_chat(), exclusive=True, timeout=600) as conv:
            # history = []
            await conv.send_message("Hey, how's it going?")

            while True:
                user_input = (await conv.get_response()).raw_text
                response = query_engine.query(user_input)
                # print(f"MJ: {response}\n")
                # print(f"MJ: {type(response)}\n")
                await conv.send_message(str(response))
                # engine.say(response)
                # engine.runAndWait()

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    print("Bot Started...")    
    client.run_until_disconnected() # Start the bot!
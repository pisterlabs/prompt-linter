import discord
from discord.ext.commands import Bot
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import KoboldApiLLM, TextGen
from langchain.llms import OpenAI
from helpers.llmstudio import LMStudio
from helpers.custom_memory import CustomBufferWindowMemory
import json
import os
import asyncio
import re
# Add this at the top of your script
CHAT_HISTORY_LINE_LIMIT = 20


# Initialize bot
intents = discord.Intents.all()
bot = Bot(command_prefix="/", intents=intents, help_command=None)

# Load configuration from config.json
with open("config.json", "r") as file:
    config = json.load(file)

bot.token = config["required"]["TOKEN"]
bot.endpoint = config["required"]["ENDPOINT"]
bot.channels = [int(x) for x in config["required"]["CHANNELS"].split(",")]
bot.name = config["required"]["NAME"]
bot.prompt = config["required"]["PROMPT"].replace("{{char}}", bot.name)
bot.mention = config["extras"]["MENTION"].lower()
bot.histories = {}  # Initialize the history dictionary
bot.channel_stop_sequences = {}

# Extras
if config["extras"]["MENTION"].lower() == "t":
    bot.mention = True
bot.stop_sequences = config["extras"]["STOP_SEQUENCES"].split(",")

# Main prompt
TEMPLATE = f"""

{bot.prompt.replace('{{char}}', bot.name)}

<START OF CONVERSATION>
{{history}}
### Instruction:
{{input}}

### Response:
{bot.name}:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=TEMPLATE)


# add llm = OpenAI(openai_api_key=endpoint, api_base='https://api.pawan.krd/v1')
async def endpoint_test(endpoint):
    try:
        llm = KoboldApiLLM(endpoint=endpoint)
        llm("Question: What is the sum of 2 and 2?\nAnswer:")
        return llm
    except:
        print("Kobold API failed, trying TextGen")
    try:
        llm = TextGen(model_url=endpoint)
        llm("Question: What is the sum of 2 and 2?\nAnswer:")
        return llm
    except:
        print("TextGen failed, trying OpenAI")
    try:
        llm = OpenAI(openai_api_key=endpoint,
                     openai_api_base="https://api.pawan.krd/v1")
        llm("Question: What is the sum of 2 and 2?\nAnswer:")
        return llm
    except:
        print("OpenAI failed. Trying LLMStudio")
    try:
        llm = LMStudio(endpoint="http://localhost:1234")
        llm("Question: What is the sum of 2 and 2?\nAnswer:")
        return llm
    except:
        print("LLMStudio failed. Please check your endpoint.")
        exit(1)


async def get_memory_for_channel(channel_id):
    """Get the memory for the channel with the given ID. If no memory exists yet, create one."""
    channel_id = str(channel_id)
    if channel_id not in bot.histories:
        # Create a new memory for the channel

        bot.histories[channel_id] = CustomBufferWindowMemory(
            k=CHAT_HISTORY_LINE_LIMIT, ai_prefix=bot.name
        )
        # Get the last 5 messages from the channel in a list
        messages = await get_messages_by_channel(channel_id)
        # # Exclude the last message using slicing
        messages_to_add = messages[-2::-1]
        messages_to_add_minus_one = messages_to_add[:-1]
        # Add the messages to the memory
        for message in messages_to_add_minus_one:
            # If the message has an attachment

            name = message[0]
            channel_ids = str(message[1])
            message = message[2]
            print(f"{name}: {message}")
            await add_history(name, channel_ids, message)

    # bot.memory = bot.histories[channel_id]
    return bot.histories[channel_id]


async def generate_response(name, channel_id, message_content) -> None:
    name = name
    memory = await get_memory_for_channel(str(channel_id))
    stop_sequence = await get_stop_sequence_for_channel(channel_id, name)
    print(f"stop sequence: {stop_sequence}")
    formatted_message = f"{name}: {message_content}"
    MAIN_TEMPLATE = TEMPLATE
    PROMPT = PromptTemplate(
        input_variables=["history", "input"],
        template=MAIN_TEMPLATE
    )

    # Create a conversation chain using the channel-specific memory
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=bot.llm,
        verbose=True,
        memory=memory,
    )
    input_dict = {"input": formatted_message, "stop": stop_sequence}
    response_text = conversation(input_dict)
    response = response_text["response"]

    return response


async def add_history(name, channel_id, message_content) -> None:
    # get the memory for the channel
    memory = await get_memory_for_channel(str(channel_id))

    formatted_message = f"{name}: {message_content}"

    # add the message to the memory
    print(f"adding message to memory: {formatted_message}")
    memory.add_input_only(formatted_message)
    return None


async def get_stop_sequence_for_channel(channel_id, name):
    name_token = f"\n{name}:"
    if channel_id not in bot.channel_stop_sequences:
        # Assign an empty list or a list with default stop sequences
        bot.channel_stop_sequences[channel_id] = []

    if name_token not in bot.channel_stop_sequences[channel_id]:
        bot.channel_stop_sequences[channel_id].append(name_token)

    return bot.channel_stop_sequences[channel_id]


async def get_messages_by_channel(channel_id):
    channel = bot.get_channel(int(channel_id))
    messages = []

    async for message in channel.history(limit=None):
        # Skip messages that start with '.' or '/'
        if message.content.startswith(".") or message.content.startswith("/"):
            continue

        # If message has an attachment, caption the attachment and make the result the message content. If not, append the messages directly teehee
        if message.attachments:
            attachment = message.attachments[0]
            if attachment.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                image_response = await bot.get_cog("image_caption").image_comment(
                    message, message.clean_content
                )
                messages.append(
                    (
                        message.author.display_name,
                        message.channel.id,
                        image_response.replace("\n", " "),
                    )
                )
        else:
            messages.append(
                (
                    message.author.display_name,
                    message.channel.id,
                    message.clean_content.replace("\n", " "),
                )
            )

        # Break the loop once we have at least 5 non-skipped messages
        if len(messages) >= 10:
            break

    # Return the first 5 non-skipped messages
    return messages[:10]


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    bot.llm = await endpoint_test(bot.endpoint)


async def has_image_attachment(message):
    url_pattern = re.compile(
        r"http[s]?://[^\s/$.?#].[^\s]*\.(jpg|jpeg|png|gif)", re.IGNORECASE
    )
    tenor_pattern = re.compile(r"https://tenor.com/view/[\w-]+")
    for attachment in message.attachments:
        if attachment.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            return True
        # Check if the message content contains a URL that ends with an image ext
    if url_pattern.search(message.content):
        return True
    # Check if the message content contains a Tenor GIF URL
    elif tenor_pattern.search(message.content):
        return True
    else:
        return False


async def is_mentioned(bot, message):
    return bot.mention and bot.name.lower() in message.clean_content.lower()


@bot.event
async def on_message(message):
    # Don't process messages sent by the bot
    if message.content.startswith(".") or message.content.startswith("/"):
        return
    
    if message.author == bot.user:
        return

    # Only process messages in specified channels
    if message.channel.id not in bot.channels:
        return

    print(f"Checking message: {message.clean_content}")  # debug print
    # debug print
    print(f"Is bot mentioned: {await is_mentioned(bot, message)}")
    # debug print
    print(
        f"Bot name: {bot.name.lower()}, Message content: {message.clean_content.lower()}")

    try:
        if await is_mentioned(bot, message):
            async with message.channel.typing():
                # if the message doesn't have an image attachment
                if not await has_image_attachment(message):
                    response = await generate_response(
                        message.author.display_name, message.channel.id, message.clean_content
                    )
                # if the message has an image attachment
                else:
                    image_response = await bot.get_cog("image_caption").image_comment(
                        message, message.clean_content
                    )
                    await add_history(
                        message.author.display_name, message.channel.id, image_response
                    )
                    response = await generate_response(
                        message.author.display_name, message.channel.id, image_response
                    )
                await message.channel.send(response)
        else:
            await add_history(
                message.author.display_name, message.channel.id, message.clean_content
            )
            
    except Exception as e:
        print(f"An error occurred: {e}")


async def load_cogs() -> None:
    for file in os.listdir(f"{os.path.realpath(os.path.dirname(__file__))}/cogs"):
        if file.endswith(".py"):
            extension = file[:-3]
            try:
                await bot.load_extension(f"cogs.{extension}")
            except Exception as e:
                # log the error and continue with the next file
                error_info = (
                    f"Failed to load extension {extension}. {type(e).__name__}: {e}"
                )
                print(error_info)


asyncio.run(load_cogs())
bot.run(bot.token)